import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, Subset
from typing import Optional, List, Dict, Any, Union, Tuple
import glob
import logging
from collections import defaultdict
import re
import pickle
import time
import random
from tqdm import tqdm

logger = logging.getLogger(__name__)

class DNADataset(Dataset):
    """Dataset for DNA sequences organized by tissue types and samples."""
    
    def __init__(
        self,
        processed_data_dir: str = "/orcd/data/omarabu/001/njwfish/DistributionEmbeddings/data/processed_dna",  # Directory containing processed DNA files
        set_size: int = 20,  # Number of sequences per set
        min_seqs_per_tissue: int = 50,  # Minimum sequences required for a tissue to be included
        min_samples_per_tissue: int = 2,  # Minimum samples required for a tissue to be included
        max_seq_length: int = 128,  # Maximum sequence length
        seed: Optional[int] = 42,
        p_sample_level: float = 0.7,  # Probability of sampling within the same sample
        data_dir: str = "data/processed_dna",
        encoder_tokenizer: str = "dna_encoder_tokenizer.json",
        hyena_tokenizer: str = "hyena_tokenizer.json",
        download: bool = False,
    ):
        """
        Initialize the DNA dataset.
        
        Args:
            processed_data_dir: Directory containing processed DNA files
            set_size: Number of sequences per set
            min_seqs_per_tissue: Minimum sequences required for a tissue to be included
            min_samples_per_tissue: Minimum samples required for a tissue to be included
            max_seq_length: Maximum sequence length
            seed: Random seed for reproducibility
            p_sample_level: Probability of sampling within the same sample
        """
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            random.seed(seed)
        
        self.processed_data_dir = processed_data_dir
        self.set_size = set_size
        self.min_seqs_per_tissue = min_seqs_per_tissue
        self.min_samples_per_tissue = min_samples_per_tissue
        self.max_seq_length = max_seq_length
        self.p_sample_level = p_sample_level
        
        # Load the processed data
        self._load_processed_data()
        
        # Initialize tissue and sample maps for sampling
        self._create_tissue_sample_maps()
    
    def _load_processed_data(self):
        """Load the processed DNA dataset from individual pickle files."""
        start_time = time.time()
        logger.info(f"Loading processed DNA data from {self.processed_data_dir}")
        
        # Find all tissue directories
        tissue_dirs = [d for d in os.listdir(self.processed_data_dir) 
                      if os.path.isdir(os.path.join(self.processed_data_dir, d))]
        
        logger.info(f"Found {len(tissue_dirs)} tissue directories")
        
        # Track statistics
        tissue_samples = defaultdict(list)
        tissue_sequence_counts = defaultdict(int)
        tissue_set_counts = defaultdict(int)
        all_sets = []
        sample_info = {}
        
        # Load all sample files across all tissues
        for tissue in tqdm(tissue_dirs, desc="Loading tissues"):
            tissue_dir = os.path.join(self.processed_data_dir, tissue)
            pickle_files = glob.glob(os.path.join(tissue_dir, "*.pkl"))
            
            tissue_samples[tissue].extend([os.path.basename(p).split('.')[0] for p in pickle_files])
            
            # Load each sample file
            for pickle_file in pickle_files:
                try:
                    with open(pickle_file, 'rb') as f:
                        data = pickle.load(f)
                    
                    sample_id = data.get("sample_id")
                    sets = data.get("sets", [])
                    
                    # Track statistics
                    num_sequences = data.get("num_sequences", 0)
                    tissue_sequence_counts[tissue] += num_sequences
                    tissue_set_counts[tissue] += len(sets)
                    
                    # Add sets to the dataset
                    all_sets.extend(sets)
                    
                    # Add sample info
                    sample_info[sample_id] = {
                        'tissue': tissue,
                        'file': pickle_file,
                        'num_sequences': num_sequences,
                        'num_sets': len(sets)
                    }
                    
                except Exception as e:
                    logger.warning(f"Error loading {pickle_file}: {e}")
        
        # Filter tissues with too few samples
        valid_tissues = [
            tissue for tissue, samples in tissue_samples.items()
            if (len(samples) >= self.min_samples_per_tissue and 
                tissue_sequence_counts[tissue] >= self.min_seqs_per_tissue)
        ]
        
        # Filter sets to only include valid tissues
        self.data = [s for s in all_sets if s["tissue_type"] in valid_tissues]
        self.tissue_types = valid_tissues
        self.sample_info = sample_info
        
        # Print summary statistics
        logger.info(f"Loaded {len(self.data)} sets from {len(valid_tissues)} tissues in {time.time() - start_time:.2f} seconds")
        logger.info(f"Total samples: {len(sample_info)}")
        
        # Print tissue statistics (first 10)
        for tissue in sorted(valid_tissues)[:10]:
            logger.info(f"Tissue {tissue}: {len(tissue_samples[tissue])} samples, {tissue_sequence_counts[tissue]} sequences, {tissue_set_counts[tissue]} sets")
        
        if len(valid_tissues) > 10:
            logger.info(f"... and {len(valid_tissues) - 10} more tissues")
    
    def _create_tissue_sample_maps(self):
        """Create maps for efficient sampling by tissue and sample."""
        # Map from tissue to samples
        self.tissue_to_samples = defaultdict(set)
        # Map from tissue to all sets
        self.tissue_to_sets = defaultdict(list)
        # Map from sample to all sets
        self.sample_to_sets = defaultdict(list)
        
        for i, item in enumerate(self.data):
            tissue = item["tissue_type"]
            sample = item["sample_id"]
            
            self.tissue_to_samples[tissue].add(sample)
            self.tissue_to_sets[tissue].append(i)
            self.sample_to_sets[sample].append(i)
    
    def split_train_eval(self, eval_ratio=0.2, seed=None):
        """
        Split the dataset into training and evaluation sets.
        
        This ensures that evaluation sets come from tissues that are also
        represented in the training set.
        
        Args:
            eval_ratio: Ratio of samples per tissue to use for evaluation
            seed: Random seed for reproducibility
            
        Returns:
            train_indices, eval_indices: Lists of indices for training and evaluation
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        train_indices = []
        eval_indices = []
        
        # Group sets by tissue
        for tissue in self.tissue_types:
            # Get all samples for this tissue
            samples = list(self.tissue_to_samples[tissue])
            
            if len(samples) <= 1:
                # If only one sample, put it in training
                for idx in self.tissue_to_sets[tissue]:
                    train_indices.append(idx)
            else:
                # Randomly select samples for evaluation
                num_eval_samples = max(1, int(len(samples) * eval_ratio))
                eval_samples = random.sample(samples, num_eval_samples)
                
                # Assign sets to train or eval based on their sample
                for idx in self.tissue_to_sets[tissue]:
                    sample = self.data[idx]["sample_id"]
                    if sample in eval_samples:
                        eval_indices.append(idx)
                    else:
                        train_indices.append(idx)
        
        logger.info(f"Split dataset: {len(train_indices)} training sets, {len(eval_indices)} evaluation sets")
        return train_indices, eval_indices
    
    def get_train_eval_subsets(self, eval_ratio=0.2, seed=None):
        """
        Create training and evaluation subset datasets.
        
        Args:
            eval_ratio: Ratio of samples per tissue to use for evaluation
            seed: Random seed for reproducibility
            
        Returns:
            train_dataset, eval_dataset: Subset datasets for training and evaluation
        """
        train_indices, eval_indices = self.split_train_eval(eval_ratio, seed)
        train_dataset = DNASubset(self, train_indices)
        eval_dataset = DNASubset(self, eval_indices)
        
        return train_dataset, eval_dataset
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Return pre-tokenized data
        return {
            'tissue_type': item["tissue_type"],
            'sample_id': item["sample_id"],
            'samples': {
                'encoder_inputs': item["tokenized"]["encoder_inputs"],
                'hyena_input_ids': item["tokenized"]["hyena_input_ids"],
                'hyena_attention_mask': item["tokenized"]["hyena_attention_mask"]
            },
            'raw_texts': item["sequences"]
        }
    
    def sample_by_strategy(self, tissue_type=None):
        """
        Sample a set using the specified strategy.
        
        Args:
            tissue_type: If provided, sample from this tissue. Otherwise, randomly select a tissue.
            
        Returns:
            index: Index of the sampled set
        """
        # If no tissue specified, randomly select one
        if tissue_type is None:
            tissue_type = random.choice(self.tissue_types)
        
        # Decide whether to sample at sample level or tissue level
        if random.random() < self.p_sample_level:
            # Sample level: get sets from the same sample
            # First, randomly select a sample from this tissue
            sample = random.choice(list(self.tissue_to_samples[tissue_type]))
            # Then, randomly select a set from this sample
            return random.choice(self.sample_to_sets[sample])
        else:
            # Tissue level: randomly select any set from this tissue
            return random.choice(self.tissue_to_sets[tissue_type])


class DNASubset(Dataset):
    """
    Subset of DNADataset that maintains the tissue/sample sampling capability.
    """
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
        self.p_sample_level = dataset.p_sample_level
        
        # Create subset-specific maps
        self._create_subset_maps()
    
    def _create_subset_maps(self):
        """Create maps for efficient sampling within the subset."""
        # Map from tissue to samples in the subset
        self.tissue_to_samples = defaultdict(set)
        # Map from tissue to all sets in the subset
        self.tissue_to_sets = defaultdict(list)
        # Map from sample to all sets in the subset
        self.sample_to_sets = defaultdict(list)
        # Tissue types in the subset
        self.tissue_types = set()
        
        for i, idx in enumerate(self.indices):
            item = self.dataset.data[idx]
            tissue = item["tissue_type"]
            sample = item["sample_id"]
            
            self.tissue_types.add(tissue)
            self.tissue_to_samples[tissue].add(sample)
            self.tissue_to_sets[tissue].append(i)  # Use subset index
            self.sample_to_sets[sample].append(i)  # Use subset index
        
        self.tissue_types = list(self.tissue_types)
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]
    
    def sample_by_strategy(self, tissue_type=None):
        """
        Sample a set using the specified strategy.
        
        Args:
            tissue_type: If provided, sample from this tissue. Otherwise, randomly select a tissue.
            
        Returns:
            index: Index within the subset of the sampled set
        """
        # If no tissue specified, randomly select one
        if tissue_type is None or tissue_type not in self.tissue_types:
            tissue_type = random.choice(self.tissue_types)
        
        # Decide whether to sample at sample level or tissue level
        if random.random() < self.p_sample_level:
            # Sample level: get sets from the same sample
            # First, randomly select a sample from this tissue
            samples = list(self.tissue_to_samples[tissue_type])
            if not samples:
                # Fall back to tissue level if no samples for this tissue
                return random.choice(self.tissue_to_sets[tissue_type])
                
            sample = random.choice(samples)
            # Then, randomly select a set from this sample
            set_indices = self.sample_to_sets[sample]
            if not set_indices:
                # Fall back to tissue level if no sets for this sample
                return random.choice(self.tissue_to_sets[tissue_type])
                
            return random.choice(set_indices)
        else:
            # Tissue level: randomly select any set from this tissue
            set_indices = self.tissue_to_sets[tissue_type]
            if not set_indices:
                # Fall back to random set if no sets for this tissue
                return random.randint(0, len(self.indices) - 1)
                
            return random.choice(set_indices) 