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

from datasets.hyena_tokenizer import CharacterTokenizer

logger = logging.getLogger(__name__)

class GPRADNADataset(Dataset):
    """
    Dataset for GPRA DNA sequences with expression values.
    
    This simplified version loads pre-processed data directly from a cache directory.
    """
    
    def __init__(
        self,
        cache_dir: str = "/orcd/data/omarabu/001/njwfish/DistributionEmbeddings/data/gpra_conditions",
        max_seq_length: int = 129,              # Maximum sequence length
        seed: Optional[int] = 42,
        **kwargs
    ):
        """
        Initialize the simplified GPRA DNA dataset that loads from cache.
        
        Args:
            cache_dir: Directory containing cached condition files
            max_seq_length: Maximum sequence length
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        self.cache_dir = cache_dir
        self.max_seq_length = max_seq_length
        
        # Create the DNA vocabulary for one-hot encoding
        self.dna_vocab = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}
        self.vocab_size = len(self.dna_vocab)
        
        # Initialize HyenaDNA tokenizer
        self._init_hyena_tokenizer()
        
        # Load cached data
        self._load_cached_data()
    
    def _init_hyena_tokenizer(self):
        """Initialize the HyenaDNA tokenizer."""
        # HyenaDNA uses a character-level tokenizer
        self.hyena_tokenizer = CharacterTokenizer(characters=self.dna_vocab, model_max_length=self.max_seq_length)
    
    def _load_cached_data(self):
        """Load all cached condition data."""
        start_time = time.time()
        logger.info(f"Loading dataset from cache directory: {self.cache_dir}")
        
        # Find all pickle files in the cache directory
        cached_files = glob.glob(os.path.join(self.cache_dir, "*.pkl"))
        
        if not cached_files:
            raise FileNotFoundError(f"No cached files found in {self.cache_dir}")
        
        logger.info(f"Found {len(cached_files)} cached condition files")
        
        # Load all data
        self.data = []
        self.condition_data = {}
        
        for file_path in cached_files:
            condition_name = os.path.basename(file_path).split('.')[0]
            logger.info(f"Loading condition: {condition_name}")
            
            try:
                with open(file_path, 'rb') as f:
                    cached_data = pickle.load(f)
                
                self.condition_data[condition_name] = cached_data.get('condition_data', {})
                self.data.extend(cached_data.get('sets', []))
                logger.info(f"Loaded {len(cached_data.get('sets', []))} sets for condition {condition_name}")
            except Exception as e:
                logger.warning(f"Failed to load condition {condition_name}: {e}")
        
        logger.info(f"Loaded a total of {len(self.data)} sequence sets in {time.time() - start_time:.2f} seconds")
        
        if not self.data:
            # Check for chunked data (for large datasets)
            chunks_dir = os.path.join(self.cache_dir, "chunks")
            if os.path.exists(chunks_dir):
                logger.info(f"Checking for chunked data in {chunks_dir}")
                chunk_files = glob.glob(os.path.join(chunks_dir, "*.pkl"))
                
                if chunk_files:
                    logger.info(f"Found {len(chunk_files)} chunk files")
                    for chunk_file in chunk_files:
                        try:
                            with open(chunk_file, 'rb') as f:
                                chunk_data = pickle.load(f)
                            self.data.extend(chunk_data)
                            logger.info(f"Loaded {len(chunk_data)} sets from chunk {os.path.basename(chunk_file)}")
                        except Exception as e:
                            logger.warning(f"Failed to load chunk {os.path.basename(chunk_file)}: {e}")
            
            if not self.data:
                raise ValueError(f"No valid data found in cache directory: {self.cache_dir}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Return pre-tokenized data if available
        return {
            'condition': item.get("condition", "unknown"),
            'center_quantile': item.get("center_quantile", 0),
            'samples': {
                'encoder_inputs': item["tokenized"]["encoder_inputs"],
                'hyena_input_ids': torch.flip(item["tokenized"]["hyena_input_ids"], [1]),
                'hyena_attention_mask': torch.flip(item["tokenized"]["hyena_attention_mask"], [1])
            },
            'raw_texts': item.get("sequences", []).tolist()
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
            all_quantiles.add(item.get("center_quantile", 0))
        
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
            if item.get("center_quantile", 0) in top_k_quantiles:
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