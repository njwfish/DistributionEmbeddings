import os
import torch
import numpy as np
import random
import logging
import pickle
import time
from torch.utils.data import Dataset
from typing import Optional, Dict, Set, List
import glob
from collections import defaultdict
from tqdm import tqdm
logger = logging.getLogger(__name__)

class DNADataset(Dataset):
    """Simplified dataset for DNA sequences organized by tissue types and samples."""
    
    def __init__(
        self,
        processed_data_dir: str = "/orcd/scratch/bcs/001/njwfish/data/processed_dna/",
        p_sample_level: float = 0.0,  # Probability of sampling within the same sample
        eval_ratio: float = 0.3,      # Portion of samples to hold out for evaluation
        seed: Optional[int] = 42,
        split: str = "train",          # 'train' or 'eval'
        max_seq_length: int = 128, 
        max_sets_per_sample: int = 20_000,
        num_classes: int = 83
    ):
        """
        Initialize the DNA dataset.
        
        Args:
            processed_data_dir: Directory containing processed DNA files
            min_samples_per_tissue: Minimum samples required for a tissue to be included
            p_sample_level: Probability of sampling within the same sample
            eval_ratio: Portion of samples to hold out for evaluation
            seed: Random seed for reproducibility
            split: Dataset split ('train' or 'eval')
        """
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            random.seed(seed)
        
        self.processed_data_dir = processed_data_dir
        self.p_sample_level = p_sample_level
        self.eval_ratio = eval_ratio
        self.split = split
        self.max_sets_per_sample = max_sets_per_sample
        
        # Load data (this will handle the train/eval split)
        self._load_data()

        self.tissue_types_oh = torch.eye(num_classes)
        self.tissue_types_oh = {tissue_type: self.tissue_types_oh[i] for i, tissue_type in enumerate(self.tissue_types)}
    
    def _load_data(self):
        """Load DNA data and create train/eval split during loading."""
        start_time = time.time()
        logger.info(f"Loading DNA data from {self.processed_data_dir} for {self.split} split")
        
        # Step 1: Identify all tissue directories
        tissue_dirs = [d for d in os.listdir(self.processed_data_dir) 
                      if os.path.isdir(os.path.join(self.processed_data_dir, d))]
        
        logger.info(f"Found {len(tissue_dirs)} tissue directories")
        
        # Step 2: Map tissues to their samples
        tissue_to_samples: Dict[str, List[str]] = defaultdict(list)
        for tissue in tissue_dirs:
            tissue_dir = os.path.join(self.processed_data_dir, tissue)
            sample_files = glob.glob(os.path.join(tissue_dir, "*.pkl"))
            sample_ids = [os.path.basename(p).split('.')[0] for p in sample_files]
            tissue_to_samples[tissue] = sample_ids
        
        
        # Step 4: Create train/eval split of samples within each tissue
        train_samples: Set[str] = set()
        eval_samples: Set[str] = set()
        
        for tissue in tissue_dirs:
            samples = tissue_to_samples[tissue]
            num_eval_samples = int(len(samples) * self.eval_ratio)
            
            # Hold out some samples for evaluation
            random.shuffle(samples)
            eval_samples_for_tissue = set(samples[:num_eval_samples])
            train_samples_for_tissue = set(samples[num_eval_samples:])
            
            eval_samples.update([f"{tissue}_{s}" for s in eval_samples_for_tissue])
            train_samples.update([f"{tissue}_{s}" for s in train_samples_for_tissue])
        
        # Step 5: Load only the data for the requested split
        target_samples = train_samples if self.split == "train" else eval_samples
        
        # Data structures for sampling
        self.data = []
        self.tissue_to_indices = defaultdict(list)
        self.sample_to_indices = defaultdict(list)
        self.tissue_types = set()
        
        # Step 6: Load the actual data
        for tissue in tqdm(tissue_dirs, desc="Loading tissues"):
            tissue_dir = os.path.join(self.processed_data_dir, tissue)
            sample_files = glob.glob(os.path.join(tissue_dir, "*.pkl"))
            
            for sample_file in sample_files:
                sample_id = os.path.basename(sample_file).split('.')[0]
                combined_id = f"{tissue}_{sample_id}"
                
                # Skip if this sample is not in the target split
                if combined_id not in target_samples:
                    continue
                
                # Load the sample data
                try:
                    with open(sample_file, 'rb') as f:
                        sample_data = pickle.load(f)
                    
                    sets = sample_data.get("sets", [])
                    if not sets:
                        continue
                    if len(sets) > self.max_sets_per_sample:
                        sets = sets[:self.max_sets_per_sample]
                    
                    # Add each set to the dataset
                    for set_data in sets:
                        idx = len(self.data)
                        self.data.append(set_data)
                        self.tissue_to_indices[tissue].append(idx)
                        self.sample_to_indices[sample_id].append(idx)
                        self.tissue_types.add(tissue)
                
                except Exception as e:
                    logger.warning(f"Error loading {sample_file}: {e}")
        
        self.tissue_types = list(self.tissue_types)
        
        logger.info(f"Loaded {len(self.data)} sets for {self.split} split "
                   f"from {len(self.tissue_types)} tissues "
                   f"in {time.time() - start_time:.2f} seconds")
    
    def __len__(self):
        return len(self.data) // 100
    
    def __getitem__(self, idx):
        """
        Get an item by index, or perform hierarchical sampling if idx is None.
        
        When idx is None, samples using the hierarchical sampling strategy:
        1. Randomly select a tissue
        2. With probability p_sample_level, sample from the same sample
            Otherwise, sample from the tissue level
        
        Args:
            idx: Index of the item to get, or None to perform hierarchical sampling
            
        Returns:
            The sampled item
        """
        # Hierarchical sampling
        # First, randomly select a tissue
        tissue = random.choice(self.tissue_types)

        # Tissue level: randomly select any set from this tissue
        tissue_indices = self.tissue_to_indices[tissue]

        sets = {
            'tissue_type': tissue,
            'samples': {
                'encoder_inputs': [],
                'hyena_input_ids': [],
                'hyena_attention_mask': []
            },
            'raw_texts': [],
            'classes': []
        }

        for i in range(350):
            idx = random.choice(tissue_indices)
            
            # Return the data at the selected index
            item = self.data[idx]

            # Fix the hyena input ids
            toks = torch.flip(item["tokenized"]["hyena_input_ids"], [1])
            sep_idx = (toks == 0)
            cls_idx = (toks == 1)
            toks[sep_idx] = 1
            toks[cls_idx] = 0
            item["tokenized"]["hyena_input_ids"] = toks
            item["tokenized"]["hyena_attention_mask"] = torch.flip(item["tokenized"]["hyena_attention_mask"], [1])

            sets['samples']['encoder_inputs'].append(item["tokenized"]["encoder_inputs"])
            sets['samples']['hyena_input_ids'].append(item["tokenized"]["hyena_input_ids"])
            sets['samples']['hyena_attention_mask'].append(item["tokenized"]["hyena_attention_mask"])
            sets['classes'].append(self.tissue_types_oh[item["tissue_type"]].repeat(toks.shape[0], 1))
            sets['raw_texts'] += item["sequences"]

        sets['samples']['encoder_inputs'] = torch.cat(sets['samples']['encoder_inputs'], dim=0)
        sets['samples']['hyena_input_ids'] = torch.cat(sets['samples']['hyena_input_ids'], dim=0)
        sets['samples']['hyena_attention_mask'] = torch.cat(sets['samples']['hyena_attention_mask'], dim=0)
        sets['classes'] = torch.cat(sets['classes'], dim=0)

        return sets