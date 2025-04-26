import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from typing import Optional, List, Dict, Any, Union
import glob
import logging
from collections import defaultdict
import re

logger = logging.getLogger(__name__)

class DNADataset(Dataset):
    """Dataset for DNA sequences organized by tissue types."""
    
    def __init__(
        self,
        data_dir: str = "/orcd/data/omarabu/001/gokul/DistributionEmbeddings/data/geo_downloads",
        set_size: int = 20,   # Number of sequences per set
        min_seqs_per_tissue: int = 50,  # Minimum sequences required for a tissue to be included
        max_seq_length: int = 512,  # Maximum sequence length for both encoder and HyenaDNA
        encoder_tokenizer: str = "dna",  # Tokenizer type for the encoder ("dna" for one-hot encoding)
        hyena_tokenizer: str = "char",  # Tokenizer type for HyenaDNA
        download: bool = False,  # Not used but kept for API consistency
        seed: Optional[int] = 42,
    ):
        """
        Initialize the DNA dataset.
        
        Args:
            data_dir: Directory containing the DNA sequence CSV files
            set_size: Number of sequences per set
            min_seqs_per_tissue: Minimum sequences required for a tissue to be included
            max_seq_length: Maximum sequence length for both encoder and HyenaDNA
            encoder_tokenizer: Tokenizer type for the encoder
            hyena_tokenizer: Tokenizer type for HyenaDNA
            download: Not used but kept for API consistency
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        self.data_dir = data_dir
        self.set_size = set_size
        self.min_seqs_per_tissue = min_seqs_per_tissue
        self.max_seq_length = max_seq_length
        self.encoder_tokenizer = encoder_tokenizer
        self.hyena_tokenizer = hyena_tokenizer
        
        # Process the data
        self._process_data()
        
        # Create the DNA vocabulary for one-hot encoding
        self.dna_vocab = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}
        self.vocab_size = len(self.dna_vocab)
        
        # Initialize HyenaDNA tokenizer
        self._init_hyena_tokenizer()
    
    def _init_hyena_tokenizer(self):
        """Initialize the HyenaDNA tokenizer."""
        from datasets.hyena_tokenizer import CharacterTokenizer
        
        # HyenaDNA uses a character-level tokenizer
        vocab = ["A", "C", "G", "T", "N"]
        self.hyena_tokenizer = CharacterTokenizer(characters=vocab, model_max_length=self.max_seq_length)
    
    def _process_data(self):
        """Process the DNA sequence data files."""
        logger.info("Processing DNA sequence data...")
        
        # Get all CSV files containing sequences
        csv_files = glob.glob(os.path.join(self.data_dir, "*_seqs.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No sequence CSV files found in {self.data_dir}")
        
        # Group sequences by tissue type
        tissue_to_sequences = defaultdict(list)
        
        for csv_file in csv_files:
            # Extract tissue type from filename
            # Format: GSM5652176_GSM5652176_Adipocytes-Z000000T7_seqs.csv
            filename = os.path.basename(csv_file)
            match = re.search(r'_([^_]+)-Z', filename)
            if match:
                tissue_type = match.group(1)
            else:
                # Use the first part if we can't extract a specific tissue
                parts = filename.split('_')
                if len(parts) > 2:
                    tissue_type = parts[2].split('-')[0]
                else:
                    continue
            
            # Read the CSV file
            try:
                df = pd.read_csv(csv_file)
                if 'seq' not in df.columns:
                    logger.warning(f"File {csv_file} does not have a 'seq' column. Skipping.")
                    continue
                
                # Add sequences to the corresponding tissue type
                tissue_to_sequences[tissue_type].extend(df['seq'].tolist())
            except Exception as e:
                logger.warning(f"Error reading file {csv_file}: {e}")
        
        # Filter out tissue types with too few sequences
        self.tissue_types = [
            tissue for tissue, sequences in tissue_to_sequences.items()
            if len(sequences) >= self.min_seqs_per_tissue
        ]
        
        # Create data structure for sequences by tissue type
        self.sequences_by_tissue = {
            tissue: tissue_to_sequences[tissue]
            for tissue in self.tissue_types
        }
        
        logger.info(f"Processed {len(self.tissue_types)} tissue types with sufficient sequences.")
        
        # Create final dataset structure
        self.data = []
        for tissue in self.tissue_types:
            tissue_sequences = self.sequences_by_tissue[tissue]
            num_sets = len(tissue_sequences) // self.set_size
            
            for i in range(num_sets):
                set_sequences = tissue_sequences[i*self.set_size:(i+1)*self.set_size]
                self.data.append({
                    "tissue_type": tissue,
                    "sequences": set_sequences
                })
        
        logger.info(f"Created {len(self.data)} sequence sets.")
    
    def _encode_dna_sequence(self, sequence):
        """
        One-hot encode a DNA sequence.
        
        Args:
            sequence: DNA sequence string
            
        Returns:
            One-hot encoded tensor
        """
        # Truncate if necessary
        sequence = sequence[:self.max_seq_length].upper()
        
        # Convert to indices
        indices = [self.dna_vocab.get(base, self.dna_vocab["N"]) for base in sequence]
        
        # Pad to max_seq_length
        padded_indices = indices + [self.dna_vocab["N"]] * (self.max_seq_length - len(indices))
        
        # One-hot encode
        one_hot = torch.zeros(self.max_seq_length, self.vocab_size)
        for i, idx in enumerate(padded_indices):
            one_hot[i, idx] = 1.0
            
        return one_hot
    
    def _tokenize_for_hyena(self, sequence):
        """
        Tokenize a DNA sequence for HyenaDNA.
        
        Args:
            sequence: DNA sequence string
            
        Returns:
            Tokenized tensor and attention mask
        """
        # Truncate if necessary
        sequence = sequence[:self.max_seq_length].upper()
        
        # Tokenize with BOS token
        tokens = self.hyena_tokenizer(
            sequence, 
            padding='max_length', 
            truncation=True, 
            max_length=self.max_seq_length,
            add_special_tokens=True,  # This will add the BOS token
            return_tensors='pt'
        )
        
        return tokens.input_ids[0], tokens.attention_mask[0]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        sequences = item["sequences"]
        
        # One-hot encode sequences for the encoder
        encoder_inputs = torch.stack([self._encode_dna_sequence(seq) for seq in sequences])
        
        # Tokenize sequences for HyenaDNA
        hyena_input_ids = []
        hyena_attention_masks = []
        
        for seq in sequences:
            input_ids, attention_mask = self._tokenize_for_hyena(seq)
            hyena_input_ids.append(input_ids)
            hyena_attention_masks.append(attention_mask)
        
        hyena_input_ids = torch.stack(hyena_input_ids)
        hyena_attention_masks = torch.stack(hyena_attention_masks)
        
        return {
            'tissue_type': item["tissue_type"],
            'samples': {
                'encoder_inputs': encoder_inputs,
                'hyena_input_ids': hyena_input_ids,
                'hyena_attention_mask': hyena_attention_masks
            },
            'raw_texts': sequences
        } 