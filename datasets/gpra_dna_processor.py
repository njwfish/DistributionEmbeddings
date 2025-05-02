import os
import torch
import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any, Union
import glob
import logging
import random
from collections import defaultdict
import pickle
from tqdm import tqdm
import time
from datasets.hyena_tokenizer import CharacterTokenizer

logger = logging.getLogger(__name__)

class GPRADNAProcessor:
    """
    Processor for GPRA DNA sequences with expression values.
    
    This class handles the processing of DNA sequences from Parquet files and
    creates preprocessed sets that can be used by the GPRADNADataset class.
    """
    
    def __init__(
        self,
        data_dir: str = "/orcd/data/omarabu/001/njwfish/DistributionEmbeddings/data/gpra_processed",  # Directory containing preprocessed Parquet files
        set_size: int = 20,                     # Number of sequences per set
        num_quantiles: int = 100,               # Number of quantiles to divide expression values
        window_width: int = 3,                  # Width of sliding window for selecting sequences
        max_seq_length: int = 129,              # Maximum sequence length
        num_sets: Optional[int] = None,         # If None, create sets to cover all sequences
        seed: Optional[int] = 42,
        processed_dir: str = "/orcd/data/omarabu/001/njwfish/DistributionEmbeddings/data/gpra_conditions",  # Directory to store processed condition files
        force_reprocess: bool = False,          # Force reprocessing even if cache exists
        chunk_size: int = 10000,                # Number of sets per chunk file
    ):
        """
        Initialize the GPRA DNA processor.
        
        Args:
            data_dir: Directory containing preprocessed Parquet files
            set_size: Number of sequences per set
            num_quantiles: Number of quantiles to divide expression values
            window_width: Width of sliding window for selecting sequences
            max_seq_length: Maximum sequence length for both encoder and HyenaDNA
            num_sets: Number of sets to create per condition (if None, calculated automatically)
            seed: Random seed for reproducibility
            processed_dir: Directory to store processed condition files
            force_reprocess: Force reprocessing even if cache exists
            chunk_size: Number of sets per chunk file
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)
        
        self.data_dir = data_dir
        self.set_size = set_size
        self.num_quantiles = num_quantiles
        self.window_width = window_width
        self.max_seq_length = max_seq_length
        self.num_sets = num_sets
        self.processed_dir = processed_dir
        self.force_reprocess = force_reprocess
        self.chunk_size = chunk_size
        
        # Create the DNA vocabulary for one-hot encoding
        self.dna_vocab = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}
        self.vocab_size = len(self.dna_vocab)
        
        # Initialize HyenaDNA tokenizer
        self.hyena_tokenizer = CharacterTokenizer(characters=self.dna_vocab, model_max_length=self.max_seq_length)
        
        # Create processed directory if it doesn't exist
        os.makedirs(self.processed_dir, exist_ok=True)
        
        # Create a conditions index file
        self.index_file = os.path.join(self.processed_dir, "conditions_index.json")
        
    def process_all_conditions(self):
        """Process all conditions from Parquet files and save as processed pickles."""
        start_time = time.time()
        logger.info("Processing GPRA DNA dataset from Parquet files...")
        
        # Find all Parquet files
        parquet_files = glob.glob(os.path.join(self.data_dir, "*.parquet"))
        
        if not parquet_files:
            raise FileNotFoundError(f"No Parquet files found in {self.data_dir}")
        
        logger.info(f"Found {len(parquet_files)} Parquet files")
        
        # Track processed conditions
        processed_conditions = {}
        total_sets = 0
        
        # Create directory for saving processed condition files
        os.makedirs(self.processed_dir, exist_ok=True)
        
        # Create a chunks directory for split files
        chunks_dir = os.path.join(self.processed_dir, "chunks")
        os.makedirs(chunks_dir, exist_ok=True)
        
        # Process each file
        for file_path in tqdm(parquet_files, desc="Processing conditions"):
            # Skip YPD file (as in original code)
            if "YPD" in file_path:
                continue
            
            condition_name = os.path.basename(file_path).split('.')[0]
            logger.info(f"Processing condition: {condition_name}")
            
            # Check if this condition has already been processed
            condition_file = os.path.join(self.processed_dir, f"{condition_name}.pkl")
            if os.path.exists(condition_file) and not self.force_reprocess:
                try:
                    # Load from cache instead of reprocessing
                    logger.info(f"Condition {condition_name} already processed, loading from file")
                    with open(condition_file, 'rb') as f:
                        cached_data = pickle.load(f)
                    
                    # Add to processed conditions
                    processed_conditions[condition_name] = {
                        'file': condition_file,
                        'num_sets': cached_data.get('num_sets', 0),
                        'chunks': cached_data.get('chunks', [])
                    }
                    total_sets += cached_data.get('num_sets', 0)
                    continue
                except Exception as e:
                    logger.warning(f"Failed to load processed condition {condition_name}: {e}. Will reprocess.")
            
            # Process this condition
            try:
                condition_info = self.process_condition(file_path, condition_name, chunks_dir)
                if condition_info:
                    processed_conditions[condition_name] = condition_info
                    total_sets += condition_info.get('num_sets', 0)
            except Exception as e:
                logger.error(f"Error processing condition {condition_name}: {e}")
        
        # Save an index of all processed conditions
        self._save_conditions_index(processed_conditions, total_sets)
        
        logger.info(f"Processed {len(processed_conditions)} conditions with {total_sets} total sets in {time.time() - start_time:.2f} seconds")
        return processed_conditions
    
    def process_condition(self, file_path, condition_name, chunks_dir):
        """Process a single condition file and save the results."""
        try:
            # Load the Parquet file
            df = pd.read_parquet(file_path)
            
            if len(df) < self.set_size:
                logger.warning(f"Condition {condition_name} has only {len(df)} sequences, "
                              f"which is less than the required set_size {self.set_size}. Skipping.")
                return None
            
            # Compute quantiles and group sequences
            quantile_groups = self._compute_quantiles(df)
            
            # Log the size of each quantile group (first 5 and last 5 for brevity)
            quantile_keys = sorted(quantile_groups.keys())
            for q in quantile_keys[:5] + quantile_keys[-5:]:
                logger.info(f"Condition {condition_name}, Quantile {q}: {len(quantile_groups[q])} sequences")
            
            if len(quantile_groups) < self.window_width:
                logger.warning(f"Condition {condition_name} has only {len(quantile_groups)} quantile groups, "
                              f"which is less than the window width {self.window_width}. Skipping.")
                return None
            
            # Create sets using sliding window
            condition_sets = self._create_sets_for_condition(condition_name, quantile_groups)
            
            if not condition_sets:
                logger.warning(f"No sets created for condition {condition_name}")
                return None
            
            # Split into chunks if needed
            chunks = []
            if self.chunk_size > 0 and len(condition_sets) > self.chunk_size:
                chunks = self._split_into_chunks(condition_sets, condition_name, chunks_dir)
            
            # Create condition metadata
            condition_data = {
                'file_path': file_path,
                'num_sets': len(condition_sets),
                'num_quantiles': len(quantile_groups),
                'chunks': chunks,
                'processed_time': time.time()
            }
            
            # Save condition file
            condition_file = os.path.join(self.processed_dir, f"{condition_name}.pkl")
            with open(condition_file, 'wb') as f:
                pickle.dump(condition_data, f)
            
            logger.info(f"Processed condition {condition_name} with {len(condition_sets)} sets")
            
            # Return information about this condition
            return {
                'file': condition_file,
                'num_sets': len(condition_sets),
                'chunks': chunks
            }
            
        except Exception as e:
            logger.error(f"Error processing condition {condition_name}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def _compute_quantiles(self, df):
        """
        Compute quantile groups for a dataframe.
        
        Args:
            df: Pandas dataframe with 'sequence' and 'expression' columns
            
        Returns:
            Dict mapping quantile index to list of sequences
        """
        # Compute quantile for each sequence
        df['rank'] = df['expression'].rank(method='first')
        bin_size = len(df) / self.num_quantiles
        df['quantile'] = (df['rank'] / bin_size).astype(int)
        df.loc[df['quantile'] >= self.num_quantiles, 'quantile'] = self.num_quantiles - 1
        
        # Group sequences by quantile
        quantile_groups = {}
        for q in tqdm(range(self.num_quantiles), desc="Computing quantiles", leave=False):
            q_sequences = df[df['quantile'] == q]['sequence'].tolist()
            if q_sequences:  # Only add non-empty groups
                quantile_groups[q] = q_sequences
        
        return quantile_groups
    
    def _create_sets_for_condition(self, condition_name, quantile_groups):
        """
        Create sets for a specific condition using sliding window over quantiles.
        Fully vectorized implementation using NumPy for parallel processing.
        
        Args:
            condition_name: Name of the condition
            quantile_groups: Dict mapping quantile index to list of sequences
            
        Returns:
            List of sets for this condition
        """
        # Get sorted list of available quantiles
        available_quantiles = sorted(quantile_groups.keys())
        
        if len(available_quantiles) < self.window_width:
            logger.warning(f"Not enough quantile groups for condition {condition_name}. Skipping.")
            return []
            
        # Determine number of sets to create
        total_seqs = sum(len(seqs) for seqs in quantile_groups.values())
        num_sets = self.num_sets
        if num_sets is None:
            # Calculate how many sets we need to approximately cover all sequences
            num_sets = max(1, total_seqs // self.set_size)
        
        logger.info(f"Creating {num_sets} sets for condition {condition_name}")
        
        # Find the valid range for center quantiles
        min_idx = self.window_width // 2
        max_idx = len(available_quantiles) - (self.window_width // 2) - 1
        
        if max_idx < min_idx:
            logger.warning(f"Not enough quantiles for window width {self.window_width} in condition {condition_name}")
            return []
        
        # Step 1: Create a mapping to store all sequences by quantile index
        # This helps with efficient sequence access
        idx_to_quantile = {i: q for i, q in enumerate(available_quantiles)}
        
        # Step 2: Convert sequence lists to a numpy array of pointers for efficient indexing
        all_seqs_by_quantile = []
        seq_counts = []
        # compute minimum seqs per quantile
        min_seqs_per_quantile = min([len(seqs) for seqs in quantile_groups.values()])
        
        # Add progress bar for quantile processing
        for q in tqdm(available_quantiles, desc=f"Processing quantiles for {condition_name}", leave=False):
            all_seqs_by_quantile.append(np.array(quantile_groups[q][:min_seqs_per_quantile], dtype=object))
            seq_counts.append(len(quantile_groups[q]))

        all_seqs_by_quantile = np.array(all_seqs_by_quantile)
        
        # Step 3: Randomly select center indices for all sets at once
        center_indices = np.random.randint(min_idx, max_idx + 1, size=num_sets)
        
        # Step 4: Create window offsets in a vectorized way
        window_offsets = np.arange(-(self.window_width // 2), (self.window_width // 2) + 1)

        # Create sets with pre-tokenized sequences
        sets = []
        for i in tqdm(range(num_sets), desc=f"Creating sets for {condition_name}", leave=False):
            # Select sequences for this set
            sequences = self._select_sequences_for_set(
                center_indices[i], 
                window_offsets, 
                available_quantiles, 
                all_seqs_by_quantile
            )
            
            # Pre-tokenize sequences
            encoder_inputs = []
            for seq in tqdm(sequences, desc=f"Encoding set {i}/{num_sets}", leave=False, mininterval=1.0):
                encoder_inputs.append(self._encode_dna_sequence(seq))
            encoder_inputs = torch.stack(encoder_inputs)
            
            hyena_input_ids = []
            hyena_attention_masks = []
            for seq in tqdm(sequences, desc=f"Tokenizing for HyenaDNA", leave=False, mininterval=1.0):
                input_ids, attention_mask = self._tokenize_for_hyena(seq)
                hyena_input_ids.append(input_ids)
                hyena_attention_masks.append(attention_mask)
            
            hyena_input_ids = torch.stack(hyena_input_ids)
            hyena_attention_masks = torch.stack(hyena_attention_masks)
            
            # Add to dataset
            sets.append({
                "condition": condition_name,
                "center_quantile": idx_to_quantile[center_indices[i]],
                "window_width": self.window_width,
                "sequences": sequences,
                "tokenized": {
                    "encoder_inputs": encoder_inputs,
                    "hyena_input_ids": hyena_input_ids,
                    "hyena_attention_mask": hyena_attention_masks
                }
            })
        
        # Return the sets for this condition
        return sets
    
    def _select_sequences_for_set(self, center_idx, window_offsets, available_quantiles, all_seqs_by_quantile):
        """
        Select sequences for a single set using vectorized operations.
        
        Args:
            center_idx: The center quantile index
            window_offsets: Array of window offsets
            available_quantiles: Sorted list of all available quantiles
            all_seqs_by_quantile: List of sequence lists for each quantile
            
        Returns:
            List of selected sequences
        """
        # Calculate valid quantile indices for this center
        q_indices = center_idx + window_offsets
        valid_indices = (q_indices >= 0) & (q_indices < len(available_quantiles))
        valid_q_indices = q_indices[valid_indices]
        
        # Collect all sequences in the window
        window_seqs = all_seqs_by_quantile[valid_q_indices].flatten()

        # Always sample with replacement for maximum speed
        # This ensures all sets have exactly set_size sequences
        indices = np.random.choice(window_seqs.shape[0], self.set_size, replace=True)
        selected_seqs = window_seqs[indices]
        return selected_seqs
    
    def _encode_dna_sequence(self, sequence):
        """
        One-hot encode a DNA sequence.
        
        Args:
            sequence: DNA sequence string
            
        Returns:
            One-hot encoded tensor
        """
        # Ensure sequence is a string and uppercase
        if not isinstance(sequence, str):
            sequence = str(sequence)
        sequence = sequence.upper()
        
        # Replace any invalid characters with 'N'
        sequence = ''.join(base if base in 'ACGT' else 'N' for base in sequence)
        
        # Truncate if necessary
        sequence = sequence[:self.max_seq_length]
        
        # Convert to indices using vectorized operations
        indices = torch.tensor([self.dna_vocab.get(base, self.dna_vocab["N"]) for base in sequence])
        
        # Pad to max_seq_length
        padded_indices = torch.full((self.max_seq_length,), self.dna_vocab["N"])
        padded_indices[:len(indices)] = indices
        
        # One-hot encode using scatter_
        one_hot = torch.zeros(self.max_seq_length, self.vocab_size)
        one_hot.scatter_(1, padded_indices.unsqueeze(1), 1.0)
            
        return one_hot
    
    def _tokenize_for_hyena(self, sequence):
        """
        Tokenize a DNA sequence for HyenaDNA.
        
        Args:
            sequence: DNA sequence string
            
        Returns:
            Tokenized tensor and attention mask
        """
        # Ensure sequence is a string and uppercase
        if not isinstance(sequence, str):
            sequence = str(sequence)
        sequence = sequence.upper()
        
        # Replace any invalid characters with 'N'
        sequence = ''.join(base if base in 'ACGT' else 'N' for base in sequence)
        
        # Truncate if necessary - leave room for special tokens
        effective_max_length = self.max_seq_length - 2  # -2 for special tokens
        sequence = sequence[:effective_max_length]

        # Tokenize with special tokens
        tokens = self.hyena_tokenizer(
            sequence, 
            padding='max_length', 
            truncation=True, 
            max_length=self.max_seq_length,
            add_special_tokens=True,  # This will add special tokens
            return_tensors='pt'
        )
        
        return tokens.input_ids[0], tokens.attention_mask[0]
    
    def _split_into_chunks(self, sets, condition_name, chunks_dir):
        """
        Split a large number of sets into smaller chunk files.
        
        Args:
            sets: List of all sets for a condition
            condition_name: Name of the condition
            chunks_dir: Directory to save chunk files
            
        Returns:
            List of chunk file paths
        """
        num_chunks = (len(sets) + self.chunk_size - 1) // self.chunk_size
        chunk_files = []
        
        logger.info(f"Splitting {len(sets)} sets into {num_chunks} chunks of size {self.chunk_size}")
        
        for i in range(num_chunks):
            start_idx = i * self.chunk_size
            end_idx = min(start_idx + self.chunk_size, len(sets))
            chunk_sets = sets[start_idx:end_idx]
            
            # Create chunk file name
            chunk_file = os.path.join(chunks_dir, f"{condition_name}_chunk_{i+1}of{num_chunks}.pkl")
            
            # Save chunk
            with open(chunk_file, 'wb') as f:
                pickle.dump({
                    'condition': condition_name,
                    'chunk_number': i+1,
                    'total_chunks': num_chunks,
                    'num_sets': len(chunk_sets),
                    'sets': chunk_sets
                }, f)
            
            chunk_files.append(chunk_file)
            
        return chunk_files
    
    def _save_conditions_index(self, processed_conditions, total_sets):
        """Save an index of all processed conditions."""
        import json
        
        # Create a JSON-serializable version of the index
        index = {
            'total_conditions': len(processed_conditions),
            'total_sets': total_sets,
            'conditions': {},
            'created_time': time.time(),
            'parameters': {
                'set_size': self.set_size,
                'num_quantiles': self.num_quantiles,
                'window_width': self.window_width,
                'max_seq_length': self.max_seq_length
            }
        }
        
        for condition_name, info in processed_conditions.items():
            index['conditions'][condition_name] = {
                'file': os.path.basename(info['file']),
                'num_sets': info['num_sets'],
                'chunks': [os.path.basename(chunk) for chunk in info.get('chunks', [])]
            }
        
        # Save as JSON
        with open(self.index_file, 'w') as f:
            json.dump(index, f, indent=2)
        
        logger.info(f"Saved conditions index to {self.index_file}")


def main():
    """Main function to process GPRA DNA data when script is run directly."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Process GPRA DNA data")
    parser.add_argument("--data_dir", type=str, 
                        default="/orcd/data/omarabu/001/njwfish/DistributionEmbeddings/data/gpra_processed",
                        help="Directory containing preprocessed Parquet files")
    parser.add_argument("--processed_dir", type=str, 
                        default="/orcd/data/omarabu/001/njwfish/DistributionEmbeddings/data/gpra_conditions",
                        help="Directory to store processed condition files")
    parser.add_argument("--set_size", type=int, default=20, 
                        help="Number of sequences per set")
    parser.add_argument("--num_quantiles", type=int, default=100, 
                        help="Number of quantiles to divide expression values")
    parser.add_argument("--window_width", type=int, default=3, 
                        help="Width of sliding window for selecting sequences")
    parser.add_argument("--max_seq_length", type=int, default=129, 
                        help="Maximum sequence length")
    parser.add_argument("--num_sets", type=int, default=None, 
                        help="Number of sets to create per condition (if None, calculated automatically)")
    parser.add_argument("--force_reprocess", action="store_true", 
                        help="Force reprocessing even if cache exists")
    parser.add_argument("--chunk_size", type=int, default=10000, 
                        help="Number of sets per chunk file (0 to disable chunking)")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Initialize the processor
    processor = GPRADNAProcessor(
        data_dir=args.data_dir,
        set_size=args.set_size,
        num_quantiles=args.num_quantiles,
        window_width=args.window_width,
        max_seq_length=args.max_seq_length,
        num_sets=args.num_sets,
        seed=args.seed,
        processed_dir=args.processed_dir,
        force_reprocess=args.force_reprocess,
        chunk_size=args.chunk_size
    )
    
    # Process all conditions
    processor.process_all_conditions()

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    
    main() 