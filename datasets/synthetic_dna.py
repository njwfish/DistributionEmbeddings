import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Optional, List, Dict, Any, Union
import logging
import random

logger = logging.getLogger(__name__)

class SyntheticDNADataset(Dataset):
    """
    Synthetic DNA dataset with repeating patterns for testing distribution learning.
    Each set contains sequences that are variants of the same repeating pattern,
    with the only difference being the starting position in the pattern.
    """
    
    def __init__(
        self,
        num_sets: int = 100,           # Number of different pattern sets to generate
        set_size: int = 20,            # Number of sequences per set
        pattern_length: int = 20,      # Length of the repeating pattern
        seq_length: int = 200,         # Length of each sequence
        num_patterns_per_set: int = 1, # Number of different patterns within a set (usually 1 for simplicity)
        nucleotides: List[str] = ["A", "C", "G", "T"],
        max_seq_length: int = 512,     # Maximum sequence length for both encoder and HyenaDNA
        seed: Optional[int] = 42,
    ):
        """
        Initialize the synthetic DNA dataset.
        
        Args:
            num_sets: Number of different pattern sets to generate
            set_size: Number of sequences per set
            pattern_length: Length of the basic repeating pattern
            seq_length: Length of each sequence
            num_patterns_per_set: Number of different patterns within a set (usually 1)
            nucleotides: List of nucleotides to use
            max_seq_length: Maximum sequence length for both encoder and HyenaDNA
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)
        
        self.num_sets = num_sets
        self.set_size = set_size
        self.pattern_length = pattern_length
        self.seq_length = min(seq_length, max_seq_length)
        self.num_patterns_per_set = num_patterns_per_set
        self.nucleotides = nucleotides
        self.max_seq_length = max_seq_length
        
        # Generate the dataset
        self._generate_data()
        
        # Create the DNA vocabulary for one-hot encoding
        self.dna_vocab = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}
        self.vocab_size = len(self.dna_vocab)
        
        # Initialize HyenaDNA tokenizer
        self._init_hyena_tokenizer()
    
    def _init_hyena_tokenizer(self):
        """Initialize the HyenaDNA tokenizer."""
        from generator.hyenadna import CharacterTokenizer
        
        # HyenaDNA uses a character-level tokenizer
        vocab = ["A", "C", "G", "T", "N"]
        self.hyena_tokenizer = CharacterTokenizer(characters=vocab, model_max_length=self.max_seq_length)
    
    def _generate_pattern(self):
        """Generate a random DNA pattern of specified length."""
        return ''.join(random.choice(self.nucleotides) for _ in range(self.pattern_length))
    
    def _generate_sequence_from_pattern(self, pattern, start_pos=0):
        """Generate a sequence by repeating a pattern, starting from a given position."""
        # Create a cyclic pattern starting from start_pos
        cyclic_pattern = pattern[start_pos:] + pattern[:start_pos]
        
        # Repeat the pattern to reach desired sequence length
        repetitions = self.seq_length // len(cyclic_pattern) + 1
        full_pattern = cyclic_pattern * repetitions
        
        # Trim to exact sequence length
        return full_pattern[:self.seq_length]
    
    def _generate_data(self):
        """Generate the synthetic DNA dataset."""
        logger.info("Generating synthetic DNA dataset...")
        
        self.data = []
        self.patterns = []  # Store the patterns for reference
        
        for set_idx in range(self.num_sets):
            # Generate patterns for this set
            set_patterns = [self._generate_pattern() for _ in range(self.num_patterns_per_set)]
            self.patterns.append(set_patterns)
            
            # Generate sequences for this set
            set_sequences = []
            
            # For each sequence in the set, pick a pattern and a starting position
            for seq_idx in range(self.set_size):
                pattern_idx = seq_idx % self.num_patterns_per_set
                pattern = set_patterns[pattern_idx]
                
                # Calculate a different starting position for each sequence
                start_pos = seq_idx % len(pattern)
                
                # Generate sequence from the pattern
                sequence = self._generate_sequence_from_pattern(pattern, start_pos)
                set_sequences.append(sequence)
            
            # Add to dataset
            self.data.append({
                "set_id": set_idx,
                "patterns": set_patterns,
                "sequences": set_sequences
            })
        
        logger.info(f"Generated {len(self.data)} sets with {self.set_size} sequences each.")
    
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
            'set_id': item["set_id"],
            'samples': {
                'encoder_inputs': encoder_inputs,
                'hyena_input_ids': hyena_input_ids,
                'hyena_attention_mask': hyena_attention_masks
            },
            'raw_texts': sequences
        } 