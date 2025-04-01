import numpy as np
import torch
import anndata
import pandas as pd
from torch.utils.data import Dataset
from typing import Optional, Tuple, List, Any

class EssentialGenesDataset(Dataset):
    """Dataset for essential genes perturbseq data."""
    
    def __init__(
        self,
        h5ad_file: str = "data/essential_genes/essential_gene_knockouts_raw.h5ad",
        set_size: int = 100,
        data_shape: List[int] = [11907],
        seed: Optional[int] = None,
    ):
        """
        Args:
            h5ad_file: Path to the h5ad file containing the essential genes data
            set_size: Number of samples per cell type x perturbation type
            data_shape: Shape of the feature vector
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        # Load the h5ad file
        self.adata = anndata.read_h5ad(h5ad_file)
        # convert to dense float32
        self.X = self.adata.X.toarray()
        print(self.X.shape)
        
        # Get unique cell types and perturbation types
        self.cell_types = self.adata.obs['cell_type'].unique()
        self.perturbation_types = self.adata.obs['gene'].unique()
        
        # Create sets based on cell type x perturbation type combinations
        self.sets = []
        self.set_indices = []
        
        for cell_type in self.cell_types:
            for perturbation in self.perturbation_types:
                # Get indices for this combination
                mask = (self.adata.obs['cell_type'] == cell_type) & (self.adata.obs['gene'] == perturbation)
                indices = np.where(mask)[0]
                
                if len(indices) > 0:
                    self.sets.append((cell_type, perturbation))
                    self.set_indices.append(indices)
        
        self.n_sets = len(self.sets)
        self.set_size = set_size
        self.data_shape = data_shape
        
        print(f"Loaded {self.n_sets} sets (cell type x perturbation combinations)")
        
    def __len__(self):
        return self.n_sets
    
    def __getitem__(self, idx):
        indices = self.set_indices[idx]
        
        # If we have fewer samples than set_size, sample with replacement
        if len(indices) < self.set_size:
            sampled_indices = np.random.choice(indices, size=self.set_size, replace=True)
        else:
            sampled_indices = np.random.choice(indices, size=self.set_size, replace=False)
        
        # Get the expression vectors for the sampled cells
        samples = self.X[sampled_indices]

        return {
            'samples': torch.tensor(samples, dtype=torch.float),
            'cell_type': self.sets[idx][0],
            'perturbation': self.sets[idx][1]
        }
    
    def get_metadata(self):
        """Return metadata about the dataset."""
        return {
            'n_sets': self.n_sets,
            'set_size': self.set_size,
            'data_shape': self.data_shape,
            'cell_types': list(self.cell_types),
            'perturbation_types': list(self.perturbation_types),
            'sets': self.sets
        } 