import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from typing import Optional, List, Dict, Any, Union
import glob
import logging
import random
from functools import lru_cache

logger = logging.getLogger(__name__)

class GPRADNADataset(Dataset):
    """
    Dataset for GPRA DNA sequences with expression values.
    
    This dataset loads DNA sequences and their corresponding expression values from preprocessed Parquet files.
    Sequences are grouped based on expression quantiles within each file (condition).
    Sets are created using a sliding window approach over adjacent quantiles.
    """
    
    def __init__(
        self,
        data_dir: str = "data/gpra_processed",  # Directory containing preprocessed Parquet files
        set_size: int = 20,                     # Number of sequences per set
        num_quantiles: int = 100,               # Number of quantiles to divide expression values
        window_width: int = 3,                  # Width of sliding window for selecting sequences
        max_seq_length: int = 129,              # Maximum sequence length
        encoder_tokenizer: str = "dna",         # Tokenizer type for the encoder
        hyena_tokenizer: str = "char",          # Tokenizer type for HyenaDNA
        num_sets: Optional[int] = None,         # If None, create sets to cover all sequences
        seed: Optional[int] = 42,
        cache_quantiles: bool = True,           # Whether to cache quantile groups in memory
        cache_size: int = 1000,                 # Size of LRU cache for tokenized sequences
    ):
        """
        Initialize the GPRA DNA dataset.
        
        Args:
            data_dir: Directory containing preprocessed Parquet files
            set_size: Number of sequences per set
            num_quantiles: Number of quantiles to divide expression values
            window_width: Width of sliding window for selecting sequences
            max_seq_length: Maximum sequence length for both encoder and HyenaDNA
            encoder_tokenizer: Tokenizer type for the encoder
            hyena_tokenizer: Tokenizer type for HyenaDNA
            num_sets: Number of sets to create per condition (if None, calculated automatically)
            seed: Random seed for reproducibility
            cache_quantiles: Whether to cache quantile groups in memory
            cache_size: Size of LRU cache for tokenized sequences
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
        self.encoder_tokenizer = encoder_tokenizer
        self.hyena_tokenizer = hyena_tokenizer
        self.num_sets = num_sets
        self.cache_quantiles = cache_quantiles
        self.cache_size = cache_size
        
        # Create the DNA vocabulary for one-hot encoding
        self.dna_vocab = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}
        self.vocab_size = len(self.dna_vocab)
        
        # Initialize HyenaDNA tokenizer
        self._init_hyena_tokenizer()
        
        # Initialize caches
        self._encode_dna_sequence = lru_cache(maxsize=cache_size)(self._encode_dna_sequence)
        self._tokenize_for_hyena = lru_cache(maxsize=cache_size)(self._tokenize_for_hyena)
        
        # Process the data
        self._process_data()
    
    def _init_hyena_tokenizer(self):
        """Initialize the HyenaDNA tokenizer."""
        from datasets.hyena_tokenizer import CharacterTokenizer
        
        # HyenaDNA uses a character-level tokenizer
        self.hyena_tokenizer = CharacterTokenizer(characters=self.dna_vocab, model_max_length=self.max_seq_length)
    
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
        for q in range(self.num_quantiles):
            q_sequences = df[df['quantile'] == q]['sequence'].tolist()
            if q_sequences:  # Only add non-empty groups
                quantile_groups[q] = q_sequences
        
        return quantile_groups
    
    def _process_data(self):
        """Process all preprocessed GPRA DNA data files."""
        logger.info("Loading GPRA DNA dataset from Parquet files...")
        
        # Find all Parquet files
        parquet_files = glob.glob(os.path.join(self.data_dir, "*.parquet"))
        
        if not parquet_files:
            raise FileNotFoundError(f"No Parquet files found in {self.data_dir}")
        
        logger.info(f"Found {len(parquet_files)} Parquet files")
        
        # Process each file
        self.condition_data = {}
        self.data = []
        
        for file_path in parquet_files:
            if "YPD" in file_path:
                continue
            condition_name = os.path.basename(file_path).split('.')[0]
            logger.info(f"Processing condition: {condition_name}")
            
            # Load the data
            df = pd.read_parquet(file_path)
            
            if len(df) < self.set_size:
                logger.warning(f"Condition {condition_name} has only {len(df)} sequences, "
                               f"which is less than the required set_size {self.set_size}. Skipping.")
                continue
            
            # Compute quantiles and group sequences
            quantile_groups = self._compute_quantiles(df)
            
            # Log the size of each quantile group (first 5 and last 5 for brevity)
            quantile_keys = sorted(quantile_groups.keys())
            for q in quantile_keys[:5] + quantile_keys[-5:]:
                logger.info(f"Condition {condition_name}, Quantile {q}: {len(quantile_groups[q])} sequences")
            
            if len(quantile_groups) < self.window_width:
                logger.warning(f"Condition {condition_name} has only {len(quantile_groups)} quantile groups, "
                               f"which is less than the window width {self.window_width}. Skipping.")
                continue
            
            if self.cache_quantiles:
                # Store quantile groups in memory
                self.condition_data[condition_name] = {
                    'file_path': file_path,
                    'quantile_groups': quantile_groups
                }
            else:
                # Just store the file path and quantile stats to save memory
                self.condition_data[condition_name] = {
                    'file_path': file_path,
                    'quantile_sizes': {q: len(seqs) for q, seqs in quantile_groups.items()}
                }
            
            # Create sets using sliding window
            self._create_sets_for_condition(condition_name, quantile_groups)
        
        logger.info(f"Created a total of {len(self.data)} sequence sets")
    
    def _create_sets_for_condition(self, condition_name, quantile_groups):
        """
        Create sets for a specific condition using sliding window over quantiles.
        
        Args:
            condition_name: Name of the condition
            quantile_groups: Dict mapping quantile index to list of sequences
        """
        # Get sorted list of available quantiles
        available_quantiles = sorted(quantile_groups.keys())
        
        if len(available_quantiles) < self.window_width:
            logger.warning(f"Not enough quantile groups for condition {condition_name}. Skipping.")
            return
            
        # Determine number of sets to create
        total_seqs = sum(len(seqs) for seqs in quantile_groups.values())
        if self.num_sets is None:
            # Calculate how many sets we need to approximately cover all sequences
            self.num_sets = max(1, total_seqs // self.set_size)
        
        logger.info(f"Creating {self.num_sets} sets for condition {condition_name}")
        
        # Find the valid range for center quantiles
        min_idx = self.window_width // 2
        max_idx = len(available_quantiles) - (self.window_width // 2) - 1
        
        if max_idx < min_idx:
            logger.warning(f"Not enough quantiles for window width {self.window_width} in condition {condition_name}")
            return
        
        # Create the sets
        for i in range(self.num_sets):
            # Randomly select center quantile within valid range
            center_idx = random.randint(min_idx, max_idx)
            center_quantile = available_quantiles[center_idx]
            
            # Get sequences from window of quantiles
            window_seqs = []
            for window_offset in range(-(self.window_width // 2), (self.window_width // 2) + 1):
                q_idx = center_idx + window_offset
                if 0 <= q_idx < len(available_quantiles):
                    q = available_quantiles[q_idx]
                    window_seqs.extend(quantile_groups[q])
            
            # Randomly sample set_size sequences (or fewer if not enough available)
            if len(window_seqs) >= self.set_size:
                set_sequences = random.sample(window_seqs, self.set_size)
            else:
                set_sequences = window_seqs
            
            # Add to dataset if we have sequences
            if set_sequences:
                self.data.append({
                    "condition": condition_name,
                    "center_quantile": center_quantile,
                    "window_width": self.window_width,
                    "sequences": set_sequences
                })
    
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

    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        sequences = item["sequences"]
        
        # Ensure we have enough sequences
        if len(sequences) < self.set_size:
            # Pad with duplicates if needed
            sequences = sequences + sequences * (self.set_size // len(sequences) + 1)
            sequences = sequences[:self.set_size]
            
        # Batch process sequences for encoder
        encoder_inputs = torch.stack([self._encode_dna_sequence(seq) for seq in sequences])

        # Batch process sequences for HyenaDNA
        hyena_input_ids = []
        hyena_attention_masks = []
        
        for seq in sequences:
            input_ids, attention_mask = self._tokenize_for_hyena(seq)
            hyena_input_ids.append(input_ids)
            hyena_attention_masks.append(attention_mask)
        
        hyena_input_ids = torch.stack(hyena_input_ids)
        hyena_attention_masks = torch.stack(hyena_attention_masks)

        return {
            'condition': item["condition"],
            'center_quantile': item["center_quantile"],
            'samples': {
                'encoder_inputs': encoder_inputs,
                'hyena_input_ids': hyena_input_ids,
                'hyena_attention_mask': hyena_attention_masks
            },
            'raw_texts': sequences
        } 