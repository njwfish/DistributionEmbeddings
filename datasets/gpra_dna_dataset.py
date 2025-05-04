import os
import torch
import numpy as np
import pickle
import glob
import logging
import random
from torch.utils.data import Dataset
from typing import Optional, Dict

from datasets.hyena_tokenizer import CharacterTokenizer

logger = logging.getLogger(__name__)

class GPRADNADataset(Dataset):
    """Simplified dataset for GPRA DNA sequences organized by condition and quantile."""
    
    def __init__(
        self,
        data_dir: str = "/orcd/scratch/bcs/001/njwfish/data/gpra_by_quantile",
        max_seq_length: int = 129,
        seed: Optional[int] = 42,
        top_k_to_exclude: int = 5,
        **kwargs
    ):
        """
        Initialize the dataset with condition and quantile organized GPRA DNA data.
        
        Args:
            data_dir: Directory containing quantile-organized data files
            max_seq_length: Maximum sequence length
            seed: Random seed for reproducibility
            top_k_to_exclude: Number of top quantiles to exclude from training
        """
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            random.seed(seed)
        
        self.data_dir = data_dir
        self.max_seq_length = max_seq_length
        self.top_k_to_exclude = top_k_to_exclude
        
        # Create the DNA vocabulary
        self.dna_vocab = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}
        
        # Initialize HyenaDNA tokenizer
        self.hyena_tokenizer = CharacterTokenizer(characters=self.dna_vocab, model_max_length=self.max_seq_length)
        
        # Load metadata and identify available quantiles
        self.metadata = self._load_metadata()
        
        # Load and organize data
        self.data_by_condition = self._load_data()
        
        # Determine which quantiles are available for training
        self._setup_quantiles()
    
    def _load_metadata(self):
        """Load metadata about available quantiles and conditions."""
        metadata_path = os.path.join(self.data_dir, "metadata.pkl")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        logger.info(f"Loaded metadata with {metadata.get('num_conditions', 0)} conditions and {metadata.get('num_quantiles', 0)} quantiles")
        return metadata
    
    def _setup_quantiles(self):
        """Setup quantile information and build index of valid samples."""
        # Get all quantiles from metadata
        all_quantiles = sorted(self.metadata.get("quantiles", []))
        if not all_quantiles:
            raise ValueError("No quantile information found in metadata")
        
        # Determine which quantiles to exclude
        k = min(self.top_k_to_exclude, len(all_quantiles))
        self.excluded_quantiles = set(all_quantiles[-k:]) if k > 0 else set()
        
        # Log excluded quantiles
        if self.excluded_quantiles:
            logger.info(f"Excluding top {k} quantiles: {self.excluded_quantiles}")
        
        # Build flat index of valid samples
        self.valid_samples = []
        
        for condition, condition_data in self.data_by_condition.items():
            for quantile, items in condition_data.items():
                if quantile not in self.excluded_quantiles:
                    for i, item in enumerate(items):
                        self.valid_samples.append((condition, quantile, i))
        
        logger.info(f"Dataset contains {len(self.valid_samples)} valid samples for training")
    
    def _load_data(self):
        """Load data organized by condition and quantile."""
        data_by_condition = {}
        
        # Get all conditions from metadata or directory structure
        conditions = self.metadata.get("conditions", [])
        if not conditions:
            conditions = [d for d in os.listdir(self.data_dir) 
                         if os.path.isdir(os.path.join(self.data_dir, d)) and d != "quantiles"]
        
        if not conditions:
            raise ValueError(f"No conditions found in {self.data_dir}")
        
        # Load data for each condition
        for condition in conditions:
            condition_dir = os.path.join(self.data_dir, condition)
            if not os.path.isdir(condition_dir):
                continue
            
            data_by_condition[condition] = {}
            quantile_files = glob.glob(os.path.join(condition_dir, "quantile_*.pkl"))
            
            for file_path in quantile_files:
                try:
                    quantile = int(os.path.basename(file_path).split('_')[1].split('.')[0])
                    with open(file_path, 'rb') as f:
                        items = pickle.load(f)
                    
                    data_by_condition[condition][quantile] = items
                    logger.info(f"Loaded {len(items)} items for condition {condition}, quantile {quantile}")
                except Exception as e:
                    logger.warning(f"Failed to load {file_path}: {e}")
        
        return data_by_condition
    
    def _get_next_set(self, condition, quantile):
        """Get a sample from the next higher quantile for the same condition."""
        condition_data = self.data_by_condition.get(condition, {})
        all_quantiles = sorted(condition_data.keys())

        current_idx = all_quantiles.index(quantile)
        if current_idx + 1 < len(all_quantiles):
            next_quantile = all_quantiles[current_idx + 1]
            items = condition_data[next_quantile]
            if items:
                return random.choice(items), next_quantile
        else:
            raise ValueError(f"No next quantile found for condition {condition} and quantile {quantile}, you are probably not holding out enough quantiles")

    
    def __len__(self):
        return len(self.valid_samples)
    
    def __getitem__(self, idx):
        # Get item from valid samples
        condition, quantile, item_idx = self.valid_samples[idx]
        item = self.data_by_condition[condition][quantile][item_idx]
        
        # Always try to get next set from higher quantile
        next_item, next_quantile = self._get_next_set(condition, quantile)
                
        # Fix the hyena input ids
        toks = torch.flip(item["tokenized"]["hyena_input_ids"], [1])
        sep_idx = (toks == 0)
        cls_idx = (toks == 1)
        toks[sep_idx] = 1
        toks[cls_idx] = 0
        item["tokenized"]["hyena_input_ids"] = toks
        item["tokenized"]["hyena_attention_mask"] = torch.flip(item["tokenized"]["hyena_attention_mask"], [1])

        # Prepare result
        result = {
            'condition': condition,
            'center_quantile': quantile,
            'samples': {
                'encoder_inputs': item["tokenized"]["encoder_inputs"],
                'hyena_input_ids': item["tokenized"]["hyena_input_ids"],
                'hyena_attention_mask': item["tokenized"]["hyena_attention_mask"]
            },
            'raw_texts': item.get("sequences", []).tolist()
        }
        
        # Always include next set if available
        if next_item is not None:
            result['next_set_samples'] = {
                'encoder_inputs': next_item["tokenized"]["encoder_inputs"]
            }
            result['next_quantile'] = next_quantile
        
        return result 