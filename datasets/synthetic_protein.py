import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Optional, List, Dict, Any, Union
import logging
import random
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

class SyntheticProteinDataset(Dataset):
    """
    Synthetic protein dataset with repeating motifs for testing distribution learning.
    Each set contains sequences that are variants of the same repeating motif,
    with variations such as substitutions, insertions, or different starting positions.
    """
    
    def __init__(
        self,
        num_sets: int = 100,           # Number of different motif sets to generate
        set_size: int = 20,            # Number of sequences per set
        pattern_length: int = 10,      # Length of the repeating motif
        seq_length: int = 100,         # Length of each sequence
        num_patterns_per_set: int = 1, # Number of different patterns within a set (usually 1 for simplicity)
        amino_acids: List[str] = ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L", 
                                 "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"],
        mutation_rate: float = 0.05,   # Rate at which to mutate amino acids
        max_seq_length: int = 512,     # Maximum sequence length
        esm_name: str = "facebook/esm2_t6_8M_UR50D",  # ESM model for encoder
        progen_name: str = "hugohrban/progen2-medium", # Progen model for generator
        seed: Optional[int] = 42,
    ):
        """
        Initialize the synthetic protein dataset.
        
        Args:
            num_sets: Number of different motif sets to generate
            set_size: Number of sequences per set
            pattern_length: Length of the basic repeating motif
            seq_length: Length of each sequence
            num_patterns_per_set: Number of different patterns within a set (usually 1)
            amino_acids: List of amino acids to use
            mutation_rate: Rate at which to mutate amino acids in the pattern
            max_seq_length: Maximum sequence length
            esm_name: Name of the ESM model to use for tokenization
            progen_name: Name of the Progen model to use for tokenization
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
        self.amino_acids = amino_acids
        self.mutation_rate = mutation_rate
        self.max_seq_length = max_seq_length
        
        # Initialize tokenizers
        logger.info(f"Initializing ESM tokenizer from {esm_name}")
        self.esm_tokenizer = AutoTokenizer.from_pretrained(esm_name, trust_remote_code=True)
        
        logger.info(f"Initializing Progen tokenizer from {progen_name}")
        self.progen_tokenizer = AutoTokenizer.from_pretrained(progen_name, trust_remote_code=True)
        
        self.progen_tokenizer.pad_token = '<|pad|>'
        self.progen_tokenizer.bos_token = '<|bos|>'
        self.progen_tokenizer.eos_token = '<|eos|>'
            
            
        # Log special token information
        logger.info(f"Progen tokenizer special tokens: PAD={self.progen_tokenizer.pad_token}, "
                   f"BOS={self.progen_tokenizer.bos_token}, EOS={self.progen_tokenizer.eos_token}")
        
        # Generate the dataset
        self._generate_data()
    
    def _generate_pattern(self):
        """Generate a random protein motif pattern of specified length."""
        return ''.join(random.choice(self.amino_acids) for _ in range(self.pattern_length))
    
    def _mutate_amino_acid(self, aa):
        """Randomly mutate an amino acid with the defined rate."""
        if random.random() < self.mutation_rate:
            return random.choice(self.amino_acids)
        return aa
    
    def _generate_sequence_from_pattern(self, pattern, start_pos=0, apply_mutations=True):
        """Generate a protein sequence by repeating a pattern with optional mutations."""
        # Create a cyclic pattern starting from start_pos
        cyclic_pattern = pattern[start_pos:] + pattern[:start_pos]
        
        # Repeat the pattern to reach desired sequence length
        repetitions = self.seq_length // len(cyclic_pattern) + 1
        full_pattern = cyclic_pattern * repetitions
        
        # Apply mutations if requested
        if apply_mutations:
            full_pattern = ''.join(self._mutate_amino_acid(aa) for aa in full_pattern)
        
        # Trim to exact sequence length
        return full_pattern[:self.seq_length]
    
    def _generate_data(self):
        """Generate the synthetic protein dataset."""
        logger.info("Generating synthetic protein dataset...")
        
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
                
                # Generate sequence from the pattern with mutations
                sequence = self._generate_sequence_from_pattern(pattern, start_pos, apply_mutations=True)
                set_sequences.append(sequence)
            
            # Add to dataset
            self.data.append({
                "set_id": set_idx,
                "patterns": set_patterns,
                "sequences": set_sequences
            })
        
        logger.info(f"Generated {len(self.data)} sets with {self.set_size} sequences each.")
    
    def _tokenize_for_esm(self, sequence):
        """
        Tokenize a protein sequence for ESM.
        
        Args:
            sequence: Protein sequence string
            
        Returns:
            Tokenized tensor and attention mask
        """
        # ESM tokenizer requires starting with <cls> token
        # Ensure the sequence is not modified with extra spaces or newlines
        sequence = sequence.strip()
        
        # Tokenize with appropriate settings and explicitly add special tokens
        tokens = self.esm_tokenizer(
            sequence, 
            padding='max_length', 
            truncation=True, 
            max_length=self.max_seq_length,
            add_special_tokens=True,  # This will add the CLS token
            return_tensors='pt'
        )
        
        return tokens.input_ids[0], tokens.attention_mask[0]
    
    def _tokenize_for_progen(self, sequence):
        """
        Tokenize a protein sequence for Progen.
        
        Args:
            sequence: Protein sequence string
            
        Returns:
            Tokenized tensor and attention mask
        """
        # Clean the sequence
        sequence = sequence.strip()
        
        # Since the tokenizer isn't automatically adding special tokens,
        # we'll manually add BOS and EOS tokens
        bos_token = self.progen_tokenizer.bos_token
        eos_token = self.progen_tokenizer.eos_token
        
        # Ensure sequence starts with BOS and ends with EOS
        if bos_token and not sequence.startswith(bos_token):
            sequence = bos_token + sequence
        
        if eos_token and not sequence.endswith(eos_token):
            sequence = sequence + eos_token
        
        # Tokenize with appropriate settings
        # Set add_special_tokens=False since we've manually added them
        tokens = self.progen_tokenizer(
            sequence, 
            padding='max_length', 
            truncation=True, 
            max_length=self.max_seq_length,
            add_special_tokens=False,  # Don't add again since we did it manually
            return_tensors='pt'
        )
        
        # Log the first sequence's token IDs for debugging
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Progen tokenized sequence: {tokens.input_ids[0]}")
            logger.debug(f"Progen BOS token ID: {self.progen_tokenizer.convert_tokens_to_ids(bos_token)}")
            logger.debug(f"Progen EOS token ID: {self.progen_tokenizer.convert_tokens_to_ids(eos_token)}")
        
        return tokens.input_ids[0], tokens.attention_mask[0]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        sequences = item["sequences"]
        
        # Tokenize sequences for ESM (encoder)
        esm_input_ids = []
        esm_attention_masks = []
        
        for seq in sequences:
            input_ids, attention_mask = self._tokenize_for_esm(seq)
            esm_input_ids.append(input_ids)
            esm_attention_masks.append(attention_mask)
        
        esm_input_ids = torch.stack(esm_input_ids)
        esm_attention_masks = torch.stack(esm_attention_masks)
        
        # Tokenize sequences for Progen (generator)
        progen_input_ids = []
        progen_attention_masks = []
        
        for seq in sequences:
            input_ids, attention_mask = self._tokenize_for_progen(seq)
            progen_input_ids.append(input_ids)
            progen_attention_masks.append(attention_mask)
        
        progen_input_ids = torch.stack(progen_input_ids)
        progen_attention_masks = torch.stack(progen_attention_masks)
        
        return {
            'set_id': item["set_id"],
            'samples': {
                'esm_input_ids': esm_input_ids,
                'esm_attention_mask': esm_attention_masks,
                'progen_input_ids': progen_input_ids,
                'progen_attention_mask': progen_attention_masks
            },
            'raw_texts': sequences
        } 