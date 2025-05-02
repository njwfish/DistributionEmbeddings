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
from collections import defaultdict
import pickle
from tqdm import tqdm
import time
import json

logger = logging.getLogger(__name__)

class GPRADNADataset(Dataset):
    """
    Dataset for GPRA DNA sequences with expression values.
    
    This dataset loads DNA sequences and their corresponding expression values from preprocessed files.
    Sequences are grouped based on expression quantiles within each file (condition).
    Sets are created using a sliding window approach over adjacent quantiles.
    """
    
    def __init__(
        self,
        processed_dir: str = "/orcd/data/omarabu/001/njwfish/DistributionEmbeddings/data/gpra_conditions",  # Directory containing processed condition files
        max_conditions: Optional[int] = None,   # Maximum number of conditions to load (None for all)
        max_sets_per_condition: Optional[int] = None,  # Maximum sets to load per condition (None for all)
        use_chunks: bool = True,                # Whether to load data from chunk files when available
        cache_size: int = 1000,                 # Size of LRU cache for any dynamic tokenization
        seed: Optional[int] = 42,
    ):
        """
        Initialize the GPRA DNA dataset.
        
        Args:
            processed_dir: Directory containing processed condition files
            max_conditions: Maximum number of conditions to load (None for all)
            max_sets_per_condition: Maximum sets to load per condition (None for all)
            use_chunks: Whether to load data from chunk files when available
            cache_size: Size of LRU cache for any dynamic tokenization
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)
        
        self.processed_dir = processed_dir
        self.max_conditions = max_conditions
        self.max_sets_per_condition = max_sets_per_condition
        self.use_chunks = use_chunks
        self.cache_size = cache_size
        
        # Initialize caches for any potential on-the-fly processing
        # (should rarely be used since data is pre-processed)
        self._encode_dna_sequence = lru_cache(maxsize=cache_size)(self._encode_dna_sequence)
        self._tokenize_for_hyena = lru_cache(maxsize=cache_size)(self._tokenize_for_hyena)
        
        # Load dataset index and data
        self._load_dataset()
    
    def _load_dataset(self):
        """Load the processed GPRA DNA dataset."""
        start_time = time.time()
        logger.info(f"Loading GPRA DNA dataset from {self.processed_dir}")
        
        # Try to load the condition index file
        index_file = os.path.join(self.processed_dir, "conditions_index.json")
        if os.path.exists(index_file):
            # Load from index
            self._load_from_index(index_file)
        else:
            # No index file, scan directory for condition files
            self._load_by_scanning_directory()
        
        logger.info(f"Loaded {len(self.data)} sets from {len(self.condition_data)} conditions in {time.time() - start_time:.2f} seconds")

    def _load_from_index(self, index_file):
        """Load dataset using the condition index file."""
        try:
            with open(index_file, 'r') as f:
                index = json.load(f)
            
            logger.info(f"Found index with {index.get('total_conditions', 0)} conditions and {index.get('total_sets', 0)} total sets")
            
            # Initialize data storage
            self.data = []
            self.condition_data = {}
            
            # Get parameters from index
            parameters = index.get('parameters', {})
            self.set_size = parameters.get('set_size', 20)
            self.max_seq_length = parameters.get('max_seq_length', 129)
            self.num_quantiles = parameters.get('num_quantiles', 100)
            self.window_width = parameters.get('window_width', 3)
            
            # Determine which conditions to load
            conditions = list(index.get('conditions', {}).items())
            if self.max_conditions is not None and self.max_conditions < len(conditions):
                # Randomly select conditions if we have a limit
                conditions = random.sample(conditions, self.max_conditions)
            
            # Load each condition
            for condition_name, condition_info in tqdm(conditions, desc="Loading conditions"):
                # Check if there are chunk files
                chunks = condition_info.get('chunks', [])
                
                if self.use_chunks and chunks:
                    # Load from chunk files (more memory efficient)
                    self._load_condition_from_chunks(condition_name, chunks)
                else:
                    # Load from main condition file
                    condition_file = os.path.join(self.processed_dir, condition_info.get('file'))
                    self._load_condition_file(condition_file, condition_name)
        
        except Exception as e:
            logger.error(f"Error loading from index: {e}")
            # Fall back to scanning directory
            self._load_by_scanning_directory()
    
    def _load_by_scanning_directory(self):
        """Load dataset by scanning the processed directory for condition files."""
        logger.info(f"No index file found. Scanning directory for condition files.")
        
        # Initialize data storage
        self.data = []
        self.condition_data = {}
        
        # Find all condition files (excluding chunks directory)
        condition_files = glob.glob(os.path.join(self.processed_dir, "*.pkl"))
        condition_files = [f for f in condition_files if os.path.isfile(f)]
        
        if not condition_files:
            raise FileNotFoundError(f"No condition files found in {self.processed_dir}")
        
        logger.info(f"Found {len(condition_files)} condition files")
        
        # Randomly sample conditions if we have a limit
        if self.max_conditions is not None and self.max_conditions < len(condition_files):
            condition_files = random.sample(condition_files, self.max_conditions)
        
        # Load each condition file
        for condition_file in tqdm(condition_files, desc="Loading conditions"):
            condition_name = os.path.basename(condition_file).split('.')[0]
            self._load_condition_file(condition_file, condition_name)
    
    def _load_condition_file(self, condition_file, condition_name):
        """Load data from a condition file."""
        try:
            with open(condition_file, 'rb') as f:
                condition_data = pickle.load(f)
            
            # Check if this is a full condition file with metadata or a chunk file with sets
            if 'sets' in condition_data:
                # This is a chunk file with sets
                sets = condition_data['sets']
                
                # Apply max_sets_per_condition limit if needed
                if (self.max_sets_per_condition is not None and 
                    self.max_sets_per_condition < len(sets)):
                    sets = random.sample(sets, self.max_sets_per_condition)
                
                # Add sets to the dataset
                self.data.extend(sets)
                
                # Add condition metadata
                self.condition_data[condition_name] = {
                    'file_path': condition_file,
                    'num_sets': len(sets)
                }
                
                logger.info(f"Loaded {len(sets)} sets from condition {condition_name}")
            
            else:
                # This is a condition file with metadata but no sets
                # Check if it has chunk files
                chunks = condition_data.get('chunks', [])
                
                if self.use_chunks and chunks:
                    # Load from chunk files
                    self._load_condition_from_chunks(condition_name, chunks)
                else:
                    logger.warning(f"Condition file {condition_file} doesn't contain sets and has no chunks")
        
        except Exception as e:
            logger.warning(f"Error loading condition file {condition_file}: {e}")
    
    def _load_condition_from_chunks(self, condition_name, chunk_paths):
        """Load condition data from chunk files."""
        # If relative paths, make them absolute
        chunks_dir = os.path.join(self.processed_dir, "chunks")
        chunk_files = [p if os.path.isabs(p) else os.path.join(chunks_dir, p) for p in chunk_paths]
        
        # Verify which chunk files exist
        existing_chunks = [f for f in chunk_files if os.path.exists(f)]
        
        if not existing_chunks:
            logger.warning(f"No chunk files found for condition {condition_name}")
            return
        
        logger.info(f"Loading {len(existing_chunks)} chunks for condition {condition_name}")
        
        # Randomly select chunks if we have a max_sets_per_condition limit
        total_sets = 0
        
        for chunk_file in tqdm(existing_chunks, desc=f"Loading chunks for {condition_name}", leave=False):
            try:
                with open(chunk_file, 'rb') as f:
                    chunk_data = pickle.load(f)
                
                chunk_sets = chunk_data.get('sets', [])
                
                # Check if we've reached our sets limit
                if self.max_sets_per_condition is not None:
                    remaining_sets = self.max_sets_per_condition - total_sets
                    if remaining_sets <= 0:
                        break  # We've reached our limit
                    
                    if len(chunk_sets) > remaining_sets:
                        # Randomly sample the remaining sets we need
                        chunk_sets = random.sample(chunk_sets, remaining_sets)
                
                # Add sets to the dataset
                self.data.extend(chunk_sets)
                total_sets += len(chunk_sets)
                
            except Exception as e:
                logger.warning(f"Error loading chunk file {chunk_file}: {e}")
        
        # Add condition metadata
        self.condition_data[condition_name] = {
            'num_sets': total_sets,
            'chunks': existing_chunks
        }
        
        logger.info(f"Loaded {total_sets} sets from condition {condition_name} chunks")
    
    def _encode_dna_sequence(self, sequence):
        """
        One-hot encode a DNA sequence.
        This is a fallback method and should rarely be used since data is pre-processed.
        
        Args:
            sequence: DNA sequence string
            
        Returns:
            One-hot encoded tensor
        """
        # Create the DNA vocabulary for one-hot encoding
        dna_vocab = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}
        vocab_size = len(dna_vocab)
        
        # Ensure sequence is a string and uppercase
        if not isinstance(sequence, str):
            sequence = str(sequence)
        sequence = sequence.upper()
        
        # Replace any invalid characters with 'N'
        sequence = ''.join(base if base in 'ACGT' else 'N' for base in sequence)
        
        # Truncate if necessary
        sequence = sequence[:self.max_seq_length]
        
        # Convert to indices using vectorized operations
        indices = torch.tensor([dna_vocab.get(base, dna_vocab["N"]) for base in sequence])
        
        # Pad to max_seq_length
        padded_indices = torch.full((self.max_seq_length,), dna_vocab["N"])
        padded_indices[:len(indices)] = indices
        
        # One-hot encode using scatter_
        one_hot = torch.zeros(self.max_seq_length, vocab_size)
        one_hot.scatter_(1, padded_indices.unsqueeze(1), 1.0)
            
        return one_hot
    
    def _tokenize_for_hyena(self, sequence):
        """
        Tokenize a DNA sequence for HyenaDNA.
        This is a fallback method and should rarely be used since data is pre-processed.
        
        Args:
            sequence: DNA sequence string
            
        Returns:
            Tokenized tensor and attention mask
        """
        # Lazy-load the tokenizer only if needed
        if not hasattr(self, 'hyena_tokenizer'):
            from datasets.hyena_tokenizer import CharacterTokenizer
            dna_vocab = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}
            self.hyena_tokenizer = CharacterTokenizer(characters=dna_vocab, model_max_length=self.max_seq_length)
        
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
        
        # Return pre-tokenized data if available
        if 'tokenized' in item:
            return {
                'condition': item["condition"],
                'center_quantile': item["center_quantile"],
                'samples': {
                    'encoder_inputs': item["tokenized"]["encoder_inputs"],
                    'hyena_input_ids': item["tokenized"]["hyena_input_ids"],
                    'hyena_attention_mask': item["tokenized"]["hyena_attention_mask"]
                },
                'raw_texts': item["sequences"]
            }
        
        # Fall back to on-the-fly tokenization if needed
        # This should be rare since all data should be pre-tokenized
        logger.warning("Using fallback on-the-fly tokenization. This should be rare!")
        sequences = item["sequences"]
            
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
        
    def split_train_eval(self, k=5, eval_ratio=0.2, seed=None):
        """
        Split the dataset into training and evaluation sets.
        
        For evaluation, filter out sets containing data from the top k quantiles.
        
        Args:
            k: Number of top quantiles to reserve for evaluation
            eval_ratio: Ratio of evaluation data (out of eligible sets)
            seed: Random seed for reproducibility
            
        Returns:
            train_indices, eval_indices: Lists of indices for training and evaluation
        """
        if seed is not None:
            np.random.seed(seed)
            
        # Get all available quantiles from the data
        all_quantiles = set()
        for item in self.data:
            all_quantiles.add(item["center_quantile"])
        
        # Sort quantiles and identify the top k
        sorted_quantiles = sorted(list(all_quantiles))
        if len(sorted_quantiles) <= k:
            logger.warning(f"Not enough quantiles ({len(sorted_quantiles)}) to reserve top {k}. Using half.")
            k = max(1, len(sorted_quantiles) // 2)
            
        top_k_quantiles = sorted_quantiles[-k:]
        logger.info(f"Using top {k} quantiles for evaluation: {top_k_quantiles}")
        
        # Find sets eligible for evaluation (having center in top k)
        eval_eligible_indices = []
        train_indices = []
        
        for i, item in enumerate(self.data):
            if item["center_quantile"] in top_k_quantiles:
                eval_eligible_indices.append(i)
            else:
                train_indices.append(i)
        
        # Determine the number of evaluation sets
        num_eval = min(int(len(eval_eligible_indices) * eval_ratio), len(eval_eligible_indices))
        
        # Randomly select evaluation sets from eligible indices
        if num_eval > 0:
            eval_indices = np.random.choice(eval_eligible_indices, num_eval, replace=False).tolist()
        else:
            eval_indices = []
            
        logger.info(f"Split dataset: {len(train_indices)} training sets, {len(eval_indices)} evaluation sets")
        
        return train_indices, eval_indices
    
    def get_train_eval_subsets(self, k=5, eval_ratio=0.2, seed=None):
        """
        Create training and evaluation subset datasets.
        
        Args:
            k: Number of top quantiles to reserve for evaluation
            eval_ratio: Ratio of evaluation data (out of eligible sets)
            seed: Random seed for reproducibility
            
        Returns:
            train_dataset, eval_dataset: Subset datasets for training and evaluation
        """
        from torch.utils.data import Subset
        
        train_indices, eval_indices = self.split_train_eval(k, eval_ratio, seed)
        train_dataset = Subset(self, train_indices)
        eval_dataset = Subset(self, eval_indices)
        
        return train_dataset, eval_dataset 