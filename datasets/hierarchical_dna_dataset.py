import os
import torch
import numpy as np
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

class HierarchicalDNADataset(Dataset):
    """
    Dataset for DNA sequences organized by tissue types and samples,
    with a hierarchical directory structure: tissue/sample/chunks
    """
    
    def __init__(
        self,
        data_dir: str,  # Directory containing the hierarchical DNA data
        max_tissues: Optional[int] = None,  # Maximum number of tissues to load
        max_samples_per_tissue: Optional[int] = None,  # Maximum samples per tissue
        max_sets_per_sample: Optional[int] = None,  # Maximum sets per sample
        lazy_loading: bool = True,  # Whether to use lazy loading
        seed: Optional[int] = 42,
        p_sample_level: float = 0.7,  # Probability of sampling within the same sample
    ):
        """
        Initialize the hierarchical DNA dataset.
        
        Args:
            data_dir: Directory containing the hierarchical DNA data
            max_tissues: Maximum number of tissues to load (None for all)
            max_samples_per_tissue: Maximum samples per tissue (None for all)
            max_sets_per_sample: Maximum sets per sample (None for all)
            lazy_loading: Whether to use lazy loading
            seed: Random seed for reproducibility
            p_sample_level: Probability of sampling within the same sample
        """
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            random.seed(seed)
        
        self.data_dir = data_dir
        self.max_tissues = max_tissues
        self.max_samples_per_tissue = max_samples_per_tissue
        self.max_sets_per_sample = max_sets_per_sample
        self.lazy_loading = lazy_loading
        self.p_sample_level = p_sample_level
        
        # Load the dataset index and structure
        self._load_structure()
        
        # Initialize tissue and sample maps for sampling
        self._create_maps()
    
    def _load_structure(self):
        """Load the dataset structure from the index file or by scanning directories."""
        start_time = time.time()
        logger.info(f"Loading DNA data structure from {self.data_dir}")
        
        # Try to load the index file
        index_file = os.path.join(self.data_dir, "dataset_index.json")
        if os.path.exists(index_file):
            self._load_from_index(index_file)
        else:
            self._scan_directory_structure()
        
        # Log loading time
        logger.info(f"Loaded dataset structure in {time.time() - start_time:.2f} seconds")
        logger.info(f"Found {len(self.tissues)} tissues, {self.total_samples} samples")
        
        if self.lazy_loading:
            logger.info(f"Using lazy loading. Dataset prepared to load {self.total_sets} sets.")
        else:
            logger.info(f"Loaded {len(self.data)} sets in eagerly loaded mode.")
    
    def _load_from_index(self, index_file):
        """Load the dataset structure from the index file."""
        try:
            with open(index_file, 'r') as f:
                index = json.load(f)
            
            # Extract metadata
            self.metadata = index.get('metadata', {})
            self.set_size = self.metadata.get('set_size', 20)
            self.max_seq_length = self.metadata.get('max_seq_length', 128)
            
            # Get structure information
            structure = index.get('structure', [])
            
            # Process tissues, potentially limiting the number
            self._process_structure(structure)
                
        except Exception as e:
            logger.error(f"Error loading from index: {e}")
            self._scan_directory_structure()
    
    def _process_structure(self, structure):
        """Process the structure data, applying any limits on tissues/samples/sets."""
        # Initialize containers
        self.tissues = []
        self.tissue_samples = {}
        self.sample_chunks = {}
        self.sample_stats = {}
        self.chunk_paths = {}
        
        # Possibly limit the number of tissues
        if self.max_tissues is not None and len(structure) > self.max_tissues:
            structure = random.sample(structure, self.max_tissues)
        
        # Process each tissue
        for tissue_info in structure:
            tissue = tissue_info.get('tissue')
            self.tissues.append(tissue)
            
            # Get sample information
            samples = tissue_info.get('samples', [])
            
            # Possibly limit samples per tissue
            if self.max_samples_per_tissue is not None and len(samples) > self.max_samples_per_tissue:
                samples = random.sample(samples, self.max_samples_per_tissue)
            
            # Store samples for this tissue
            self.tissue_samples[tissue] = []
            
            # Process each sample
            for sample_info in samples:
                sample_id = sample_info.get('sample_id')
                self.tissue_samples[tissue].append(sample_id)
                
                # Store sample stats
                self.sample_stats[sample_id] = {
                    'tissue': tissue,
                    'num_chunks': sample_info.get('num_chunks', 0),
                    'num_sets': sample_info.get('num_sets', 0)
                }
                
                # Process chunk information
                chunks = sample_info.get('chunks', [])
                self.sample_chunks[sample_id] = []
                
                for chunk_info in chunks:
                    chunk_file = chunk_info.get('file')
                    num_sets = chunk_info.get('num_sets', 0)
                    
                    # Create the full path to the chunk file
                    chunk_path = os.path.join(self.data_dir, tissue, sample_id, chunk_file)
                    
                    # Store chunk information
                    chunk_key = f"{tissue}/{sample_id}/{chunk_file}"
                    self.sample_chunks[sample_id].append((chunk_key, num_sets))
                    self.chunk_paths[chunk_key] = chunk_path
        
        # Calculate total statistics
        self.total_samples = sum(len(samples) for samples in self.tissue_samples.values())
        self.total_chunks = sum(len(chunks) for chunks in self.sample_chunks.values())
        
        # If lazy loading, create indices mapping
        if self.lazy_loading:
            self._create_indices_mapping()
        else:
            self._load_all_data()
    
    def _scan_directory_structure(self):
        """Scan the directory structure to build the dataset metadata."""
        logger.info("No index file found. Scanning directory structure...")
        
        # Initialize containers
        self.tissues = []
        self.tissue_samples = {}
        self.sample_chunks = {}
        self.sample_stats = {}
        self.chunk_paths = {}
        self.metadata = {}
        
        # Find all tissue directories
        tissue_dirs = [d for d in os.listdir(self.data_dir) 
                      if os.path.isdir(os.path.join(self.data_dir, d))]
        
        # Possibly limit the number of tissues
        if self.max_tissues is not None and len(tissue_dirs) > self.max_tissues:
            tissue_dirs = random.sample(tissue_dirs, self.max_tissues)
        
        # Process each tissue directory
        for tissue in tissue_dirs:
            tissue_path = os.path.join(self.data_dir, tissue)
            
            # Find all sample directories
            sample_dirs = [d for d in os.listdir(tissue_path) 
                          if os.path.isdir(os.path.join(tissue_path, d))]
            
            # Possibly limit samples per tissue
            if self.max_samples_per_tissue is not None and len(sample_dirs) > self.max_samples_per_tissue:
                sample_dirs = random.sample(sample_dirs, self.max_samples_per_tissue)
            
            self.tissues.append(tissue)
            self.tissue_samples[tissue] = []
            
            # Process each sample directory
            for sample_id in sample_dirs:
                sample_path = os.path.join(tissue_path, sample_id)
                
                # Find all chunk files
                chunk_files = glob.glob(os.path.join(sample_path, "chunk_*.pkl"))
                
                if chunk_files:  # Only add sample if it has chunks
                    self.tissue_samples[tissue].append(sample_id)
                    self.sample_chunks[sample_id] = []
                    
                    # Process each chunk file
                    total_sets_in_sample = 0
                    
                    for chunk_file in chunk_files:
                        try:
                            # Read the chunk to get metadata
                            with open(chunk_file, 'rb') as f:
                                chunk = pickle.load(f)
                            
                            num_sets = chunk.get('num_sets', 0)
                            total_sets_in_sample += num_sets
                            
                            # Store chunk information
                            chunk_key = f"{tissue}/{sample_id}/{os.path.basename(chunk_file)}"
                            self.sample_chunks[sample_id].append((chunk_key, num_sets))
                            self.chunk_paths[chunk_key] = chunk_file
                            
                        except Exception as e:
                            logger.warning(f"Error reading chunk file {chunk_file}: {e}")
                    
                    # Store sample stats
                    self.sample_stats[sample_id] = {
                        'tissue': tissue,
                        'num_chunks': len(chunk_files),
                        'num_sets': total_sets_in_sample
                    }
        
        # Calculate total statistics
        self.total_samples = sum(len(samples) for samples in self.tissue_samples.values())
        self.total_chunks = sum(len(chunks) for chunks in self.sample_chunks.values())
        
        # If lazy loading, create indices mapping
        if self.lazy_loading:
            self._create_indices_mapping()
        else:
            self._load_all_data()
    
    def _create_indices_mapping(self):
        """Create a mapping from dataset indices to chunk files and internal indices."""
        self.index_mapping = []
        self.cumulative_sizes = [0]
        
        total_sets = 0
        
        # Process each sample
        for tissue in self.tissues:
            for sample_id in self.tissue_samples[tissue]:
                # Get chunks for this sample
                chunks = self.sample_chunks[sample_id]
                
                for chunk_key, num_sets in chunks:
                    # Calculate how many sets to use from this chunk
                    num_to_use = num_sets
                    if self.max_sets_per_sample is not None:
                        # Check if we've already used the maximum for this sample
                        sets_used_in_sample = sum(min(n, self.max_sets_per_sample) 
                                               for _, n in self.sample_chunks[sample_id])
                        if sets_used_in_sample >= self.max_sets_per_sample:
                            num_to_use = 0
                        else:
                            remaining = self.max_sets_per_sample - sets_used_in_sample
                            num_to_use = min(num_to_use, remaining)
                    
                    # Add mappings for each set
                    for i in range(num_to_use):
                        self.index_mapping.append((chunk_key, i))
                    
                    total_sets += num_to_use
                    self.cumulative_sizes.append(total_sets)
        
        # Update total sets
        self.total_sets = total_sets
    
    def _load_all_data(self):
        """Load all data into memory for non-lazy mode."""
        self.data = []
        self.chunk_cache = {}
        
        # Process each tissue and sample
        for tissue in tqdm(self.tissues, desc="Loading tissues"):
            for sample_id in tqdm(self.tissue_samples[tissue], desc=f"Loading samples for {tissue}", leave=False):
                # Get chunks for this sample
                chunks = self.sample_chunks[sample_id]
                
                # Count sets loaded for this sample
                sets_loaded = 0
                
                for chunk_key, num_sets in chunks:
                    # Check if we've reached the max for this sample
                    if self.max_sets_per_sample is not None and sets_loaded >= self.max_sets_per_sample:
                        break
                    
                    # Load the chunk
                    chunk_path = self.chunk_paths[chunk_key]
                    try:
                        with open(chunk_path, 'rb') as f:
                            chunk = pickle.load(f)
                        
                        # Get the sets from this chunk
                        chunk_sets = chunk.get('data', [])
                        
                        # Determine how many sets to use
                        num_to_use = len(chunk_sets)
                        if self.max_sets_per_sample is not None:
                            remaining = self.max_sets_per_sample - sets_loaded
                            num_to_use = min(num_to_use, remaining)
                        
                        # Add sets to the dataset
                        self.data.extend(chunk_sets[:num_to_use])
                        sets_loaded += num_to_use
                        
                    except Exception as e:
                        logger.warning(f"Error loading chunk {chunk_path}: {e}")
        
        logger.info(f"Loaded {len(self.data)} sets in non-lazy mode")
    
    def _load_set_from_chunk(self, chunk_key, index):
        """Load a specific set from a chunk file."""
        # Check if the chunk is already in cache
        if chunk_key in self.chunk_cache:
            chunk_sets = self.chunk_cache[chunk_key]
        else:
            # Load the chunk
            chunk_path = self.chunk_paths[chunk_key]
            try:
                with open(chunk_path, 'rb') as f:
                    chunk = pickle.load(f)
                
                chunk_sets = chunk.get('data', [])
                
                # Add to cache
                self.chunk_cache[chunk_key] = chunk_sets
                
            except Exception as e:
                logger.warning(f"Error loading chunk {chunk_path}: {e}")
                return None
        
        # Get the specific set
        if index < len(chunk_sets):
            return chunk_sets[index]
        else:
            logger.warning(f"Index {index} out of range for chunk {chunk_key}")
            return None
    
    def _create_maps(self):
        """Create maps for efficient sampling by tissue and sample."""
        # Map from tissue to samples
        self.tissue_to_samples = self.tissue_samples
        
        # Map from tissue to all sets
        self.tissue_to_sets = defaultdict(list)
        # Map from sample to all sets
        self.sample_to_sets = defaultdict(list)
        
        # Initialize chunk cache for lazy loading
        if self.lazy_loading:
            self.chunk_cache = {}  # Cache for loaded chunks
            
            # Create maps based on the index mapping
            for i, (chunk_key, _) in enumerate(self.index_mapping):
                # Extract tissue and sample from chunk_key
                parts = chunk_key.split('/')
                if len(parts) >= 2:
                    tissue = parts[0]
                    sample = parts[1]
                    
                    self.tissue_to_sets[tissue].append(i)
                    self.sample_to_sets[sample].append(i)
        else:
            # Create maps based on loaded data
            for i, item in enumerate(self.data):
                tissue = item.get("tissue_type")
                sample = item.get("sample_id")
                
                if tissue and sample:
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
        for tissue in self.tissues:
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
                        # In lazy loading mode, extract sample from index mapping
                        chunk_key, _ = self.index_mapping[idx]
                        parts = chunk_key.split('/')
                        sample = parts[1] if len(parts) >= 2 else None
                    else:
                        # In non-lazy mode, get sample from data
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
        train_dataset = HierarchicalDNASubset(self, train_indices)
        eval_dataset = HierarchicalDNASubset(self, eval_indices)
        
        return train_dataset, eval_dataset
    
    def __len__(self):
        if self.lazy_loading:
            return len(self.index_mapping)
        else:
            return len(self.data)
    
    def __getitem__(self, idx):
        if self.lazy_loading:
            # Get chunk key and internal index
            if idx < 0 or idx >= len(self.index_mapping):
                raise IndexError(f"Index {idx} out of range for dataset of size {len(self.index_mapping)}")
            
            chunk_key, internal_idx = self.index_mapping[idx]
            item = self._load_set_from_chunk(chunk_key, internal_idx)
            
            if item is None:
                raise IndexError(f"Failed to load item at index {idx}")
        else:
            item = self.data[idx]
        
        # Return structured data
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
        if tissue_type is None or tissue_type not in self.tissues:
            tissue_type = random.choice(self.tissues)
        
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


class HierarchicalDNASubset(Subset):
    """
    Subset of HierarchicalDNADataset that maintains the tissue/sample sampling capability.
    """
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        self.p_sample_level = dataset.p_sample_level
        
        # Copy dataset attributes
        if isinstance(dataset, HierarchicalDNADataset):
            self.tissues = dataset.tissues
            
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
        if not hasattr(self, 'tissues'):
            # Fall back to random sampling if no tissue info is available
            return random.randint(0, len(self) - 1)
        
        # Get available tissues in the subset
        available_tissues = list(self.tissue_to_sets.keys())
        if not available_tissues:
            return random.randint(0, len(self) - 1)
            
        # If no tissue specified or specified tissue not in subset, randomly select one
        if tissue_type is None or tissue_type not in available_tissues:
            tissue_type = random.choice(available_tissues)
        
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