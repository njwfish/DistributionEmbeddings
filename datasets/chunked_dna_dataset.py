import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, Subset
from typing import Optional, List, Dict, Any, Union, Tuple
import glob
import logging
from collections import defaultdict
import json
import pickle
import time
import random
from tqdm import tqdm

logger = logging.getLogger(__name__)

class ChunkedDNADataset(Dataset):
    """
    Dataset for DNA sequences organized by tissue types and samples,
    optimized to load from chunked files for memory efficiency.
    """
    
    def __init__(
        self,
        chunks_dir: str,  # Directory containing chunked DNA files
        max_chunks: Optional[int] = None,  # Maximum number of chunks to load (None for all)
        max_sets_per_chunk: Optional[int] = None,  # Maximum sets to load per chunk (None for all)
        lazy_loading: bool = True,  # Whether to use lazy loading for chunks
        seed: Optional[int] = 42,
        p_sample_level: float = 0.7,  # Probability of sampling within the same sample
    ):
        """
        Initialize the chunked DNA dataset.
        
        Args:
            chunks_dir: Directory containing chunked DNA files
            max_chunks: Maximum number of chunks to load (None for all)
            max_sets_per_chunk: Maximum sets to load per chunk (None for all)
            lazy_loading: Whether to use lazy loading for chunks
            seed: Random seed for reproducibility
            p_sample_level: Probability of sampling within the same sample
        """
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            random.seed(seed)
        
        self.chunks_dir = chunks_dir
        self.max_chunks = max_chunks
        self.max_sets_per_chunk = max_sets_per_chunk
        self.lazy_loading = lazy_loading
        self.p_sample_level = p_sample_level
        
        # Load the chunked data
        self._load_chunked_data()
        
        # Initialize tissue and sample maps for sampling
        self._create_tissue_sample_maps()
    
    def _load_chunked_data(self):
        """Load the chunked DNA dataset."""
        start_time = time.time()
        logger.info(f"Loading chunked DNA data from {self.chunks_dir}")
        
        # Try to load the chunks index file
        index_file = os.path.join(self.chunks_dir, "chunks_index.json")
        if os.path.exists(index_file):
            # Load from index
            self._load_from_index(index_file)
        else:
            # No index file, scan directory for chunk files
            self._load_by_scanning_directory()
        
        if not self.lazy_loading:
            logger.info(f"Loaded {len(self.data)} sets in {time.time() - start_time:.2f} seconds")
        else:
            logger.info(f"Prepared to load {self.total_sets} sets from {len(self.chunk_files)} chunks in {time.time() - start_time:.2f} seconds")
    
    def _load_from_index(self, index_file):
        """Load dataset using the chunk index file."""
        try:
            with open(index_file, 'r') as f:
                index = json.load(f)
            
            # Initialize data storage
            self.data = []
            self.tissue_types = set()
            self.sample_info = {}
            
            # Get parameters from index
            metadata = index.get('metadata', {})
            self.set_size = metadata.get('set_size', 20)
            self.max_seq_length = metadata.get('max_seq_length', 128)
            
            # Get chunk information
            chunks_info = index.get('chunks', [])
            total_sets = index.get('total_sets', 0)
            self.total_sets = total_sets
            
            logger.info(f"Found index with {len(chunks_info)} chunks and {total_sets} total sets")
            
            # Randomly select chunks if we have a limit
            if self.max_chunks is not None and self.max_chunks < len(chunks_info):
                chunks_info = random.sample(chunks_info, self.max_chunks)
                logger.info(f"Randomly selected {self.max_chunks} chunks")
            
            # Track chunk files
            self.chunk_files = []
            
            if self.lazy_loading:
                # In lazy loading mode, just store paths to chunk files
                for chunk_info in chunks_info:
                    chunk_file = os.path.join(self.chunks_dir, chunk_info.get('file'))
                    if os.path.exists(chunk_file):
                        self.chunk_files.append((chunk_file, chunk_info.get('num_sets', 0)))
                
                # Create a mapping from index to chunk and offset
                self._create_index_mapping()
                
                # Store tissue types if available
                self.tissue_set_counts = defaultdict(int)
                for chunk_info in chunks_info:
                    for tissue in chunk_info.get('tissues', []):
                        self.tissue_types.add(tissue)
            else:
                # Load all chunk data upfront
                for chunk_info in tqdm(chunks_info, desc="Loading chunks"):
                    chunk_file = os.path.join(self.chunks_dir, chunk_info.get('file'))
                    if os.path.exists(chunk_file):
                        self._load_chunk_file(chunk_file)
                
                # Create a list of unique tissues
                self.tissue_types = list(self.tissue_types)
        
        except Exception as e:
            logger.error(f"Error loading from index: {e}")
            # Fall back to scanning directory
            self._load_by_scanning_directory()
    
    def _load_by_scanning_directory(self):
        """Load dataset by scanning the chunks directory for chunk files."""
        logger.info(f"No index file found. Scanning directory for chunk files.")
        
        # Initialize data storage
        self.data = []
        self.tissue_types = set()
        self.sample_info = {}
        
        # Find all chunk files
        chunk_files = glob.glob(os.path.join(self.chunks_dir, "dna_chunk_*.pkl"))
        if not chunk_files:
            raise FileNotFoundError(f"No chunk files found in {self.chunks_dir}")
        
        logger.info(f"Found {len(chunk_files)} chunk files")
        
        # Randomly select chunks if we have a limit
        if self.max_chunks is not None and self.max_chunks < len(chunk_files):
            chunk_files = random.sample(chunk_files, self.max_chunks)
            logger.info(f"Randomly selected {self.max_chunks} chunks")
        
        if self.lazy_loading:
            # In lazy loading mode, just store paths to chunk files
            self.chunk_files = []
            
            # Load chunk sizes
            for chunk_file in tqdm(chunk_files, desc="Scanning chunks"):
                try:
                    # Read just enough to get the number of sets in the chunk
                    with open(chunk_file, 'rb') as f:
                        chunk = pickle.load(f)
                    num_sets = chunk.get('num_sets', 0)
                    self.chunk_files.append((chunk_file, num_sets))
                    
                    # Collect tissue types
                    for tissue in chunk.get('tissues', []):
                        self.tissue_types.add(tissue)
                except Exception as e:
                    logger.warning(f"Error scanning chunk file {chunk_file}: {e}")
            
            # Store total number of sets
            self.total_sets = sum(num_sets for _, num_sets in self.chunk_files)
            
            # Create a mapping from index to chunk and offset
            self._create_index_mapping()
        else:
            # Load all chunk data upfront
            for chunk_file in tqdm(chunk_files, desc="Loading chunks"):
                self._load_chunk_file(chunk_file)
            
            # Create a list of unique tissues
            self.tissue_types = list(self.tissue_types)
    
    def _create_index_mapping(self):
        """Create a mapping from dataset index to chunk file and internal offset."""
        self.index_mapping = []
        self.cumulative_sizes = [0]
        
        total = 0
        for chunk_idx, (chunk_file, num_sets) in enumerate(self.chunk_files):
            num_to_use = num_sets
            if self.max_sets_per_chunk is not None:
                num_to_use = min(num_to_use, self.max_sets_per_chunk)
            
            for i in range(num_to_use):
                self.index_mapping.append((chunk_idx, i))
            
            total += num_to_use
            self.cumulative_sizes.append(total)
        
        # Update total sets if we're using a subset
        self.total_sets = total
    
    def _load_chunk_file(self, chunk_file):
        """Load data from a chunk file."""
        try:
            with open(chunk_file, 'rb') as f:
                chunk = pickle.load(f)
            
            chunk_sets = chunk.get('data', [])
            
            # Apply max_sets_per_chunk limit if needed
            if self.max_sets_per_chunk is not None and self.max_sets_per_chunk < len(chunk_sets):
                chunk_sets = random.sample(chunk_sets, self.max_sets_per_chunk)
            
            # Add sets to the dataset
            self.data.extend(chunk_sets)
            
            # Collect tissue types and sample info
            for item in chunk_sets:
                tissue = item.get("tissue_type")
                if tissue:
                    self.tissue_types.add(tissue)
                
                sample_id = item.get("sample_id")
                if sample_id and sample_id not in self.sample_info:
                    self.sample_info[sample_id] = {
                        'tissue': tissue,
                        'file': chunk_file
                    }
        
        except Exception as e:
            logger.warning(f"Error loading chunk file {chunk_file}: {e}")
    
    def _load_set_from_chunk(self, chunk_idx, offset):
        """Load a single set from a chunk file."""
        chunk_file, _ = self.chunk_files[chunk_idx]
        
        try:
            with open(chunk_file, 'rb') as f:
                chunk = pickle.load(f)
            
            chunk_sets = chunk.get('data', [])
            if offset < len(chunk_sets):
                return chunk_sets[offset]
            else:
                logger.warning(f"Offset {offset} out of range for chunk {chunk_file}")
                return None
        
        except Exception as e:
            logger.warning(f"Error loading set from chunk {chunk_file}: {e}")
            return None
    
    def _create_tissue_sample_maps(self):
        """Create maps for efficient sampling by tissue and sample."""
        # Map from tissue to samples
        self.tissue_to_samples = defaultdict(set)
        # Map from tissue to all sets
        self.tissue_to_sets = defaultdict(list)
        # Map from sample to all sets
        self.sample_to_sets = defaultdict(list)
        
        if self.lazy_loading:
            # In lazy loading mode, we need to scan all chunk files
            for chunk_idx, (chunk_file, _) in enumerate(self.chunk_files):
                try:
                    with open(chunk_file, 'rb') as f:
                        chunk = pickle.load(f)
                    
                    chunk_sets = chunk.get('data', [])
                    
                    # For each set in the chunk, update the maps
                    for offset, item in enumerate(chunk_sets):
                        if self.max_sets_per_chunk is not None and offset >= self.max_sets_per_chunk:
                            break
                            
                        tissue = item.get("tissue_type")
                        sample = item.get("sample_id")
                        
                        if tissue and sample:
                            # Calculate the dataset index for this set
                            dataset_idx = self.cumulative_sizes[chunk_idx] + offset
                            
                            self.tissue_to_samples[tissue].add(sample)
                            self.tissue_to_sets[tissue].append(dataset_idx)
                            self.sample_to_sets[sample].append(dataset_idx)
                
                except Exception as e:
                    logger.warning(f"Error creating maps for chunk {chunk_file}: {e}")
            
            # Convert tissue types to list
            self.tissue_types = list(self.tissue_types)
        else:
            # When all data is loaded upfront, create maps from loaded data
            for i, item in enumerate(self.data):
                tissue = item.get("tissue_type")
                sample = item.get("sample_id")
                
                if tissue and sample:
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
                    if self.lazy_loading:
                        # In lazy loading mode, load the set to get its sample_id
                        chunk_idx, offset = self.index_mapping[idx]
                        item = self._load_set_from_chunk(chunk_idx, offset)
                        sample = item.get("sample_id") if item else None
                    else:
                        sample = self.data[idx].get("sample_id")
                    
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
        train_dataset = Subset(self, train_indices)
        eval_dataset = Subset(self, eval_indices)
        
        return train_dataset, eval_dataset
    
    def __len__(self):
        if self.lazy_loading:
            return self.total_sets
        else:
            return len(self.data)
    
    def __getitem__(self, idx):
        if self.lazy_loading:
            # Find which chunk and offset this index corresponds to
            if idx < 0 or idx >= len(self.index_mapping):
                raise IndexError(f"Index {idx} out of range for dataset of size {len(self.index_mapping)}")
            
            chunk_idx, offset = self.index_mapping[idx]
            item = self._load_set_from_chunk(chunk_idx, offset)
            
            if item is None:
                raise IndexError(f"Failed to load item at index {idx}")
        else:
            item = self.data[idx]
        
        # Return the data
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
                return random.randint(0, len(self) - 1)
                
            return random.choice(set_indices)


# Create a helper class for DNA subset with additional sampling functionality
class ChunkedDNASubset(Subset):
    """
    Subset of ChunkedDNADataset that maintains the tissue/sample sampling capability.
    """
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        self.p_sample_level = dataset.p_sample_level
        
        # Copy dataset attributes
        if isinstance(dataset, ChunkedDNADataset):
            self.tissue_types = dataset.tissue_types
            
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
        
        for i, idx in enumerate(self.indices):
            try:
                # Get the item
                item = self.dataset[idx]
                
                tissue = item["tissue_type"]
                sample = item["sample_id"]
                
                self.tissue_to_samples[tissue].add(sample)
                self.tissue_to_sets[tissue].append(i)  # Use subset index
                self.sample_to_sets[sample].append(i)  # Use subset index
            except Exception as e:
                logger.warning(f"Error creating subset maps for index {idx}: {e}")
    
    def sample_by_strategy(self, tissue_type=None):
        """
        Sample a set using the specified strategy.
        
        Args:
            tissue_type: If provided, sample from this tissue. Otherwise, randomly select a tissue.
            
        Returns:
            index: Index within the subset of the sampled set
        """
        if not hasattr(self, 'tissue_types'):
            # Fall back to random sampling if no tissue info is available
            return random.randint(0, len(self) - 1)
            
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
                return random.choice(self.tissue_to_sets[tissue_type]) if self.tissue_to_sets[tissue_type] else random.randint(0, len(self) - 1)
                
            sample = random.choice(samples)
            # Then, randomly select a set from this sample
            set_indices = self.sample_to_sets[sample]
            if not set_indices:
                # Fall back to tissue level if no sets for this sample
                return random.choice(self.tissue_to_sets[tissue_type]) if self.tissue_to_sets[tissue_type] else random.randint(0, len(self) - 1)
                
            return random.choice(set_indices)
        else:
            # Tissue level: randomly select any set from this tissue
            set_indices = self.tissue_to_sets[tissue_type]
            if not set_indices:
                # Fall back to random set if no sets for this tissue
                return random.randint(0, len(self) - 1)
                
            return random.choice(set_indices) 