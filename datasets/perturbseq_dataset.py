import numpy as np
import torch
import anndata
import pandas as pd
from torch.utils.data import Dataset
from typing import Optional, Tuple, List, Any, Dict, Union, Literal
from sklearn.decomposition import PCA

class PerturbseqDataset(Dataset):
    """Dataset for perturbseq data."""
    
    def __init__(
        self,
        adata_path: str,
        pert_embedding_path: Optional[str] = None,
        set_size: int = 100,
        data_shape: List[int] = [11907],
        seed: Optional[int] = None,
        pert_key: str = "gene",
        cell_key: str = "cell_type",
        control_pert: str = "non-targeting",
        pca_components: int = 10,
        split_mode: Literal["ood", "iid"] = "ood",
        heldout_perts: List[str] = [],
        heldout_cell_types: List[str] = [],
    ):
        """
        Args:
            adata_path: Path to the h5ad file containing the essential genes data
            pert_embedding_path: Path to the perturbation embeddings (.pt file)
            set_size: Number of samples per cell type x perturbation type
            data_shape: Shape of the feature vector
            seed: Random seed for reproducibility
            pert_key: Key in adata.obs for perturbation information
            cell_key: Key in adata.obs for cell type information
            control_pert: Identifier for control perturbation
            pca_components: Number of PCA components for perturbation embeddings
            use_cellxgene_genes: Whether to use cellxgene genes 
            heldout_perts: List of perturbations to hold out for evaluation (deprecated, use eval_sets)
            heldout_cell_types: List of cell types to hold out for evaluation (deprecated, use eval_sets)
            eval_sets: List of (cell_type, perturbation) tuples that belong to the evaluation set
        """
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        self.split_mode = split_mode
        self.pert_key = pert_key
        self.cell_key = cell_key
        self.control_pert = control_pert
        
        # Load the h5ad file
        self.adata = anndata.read_h5ad(adata_path)
        # convert to dense float32
        self.X = self.adata.X.toarray()

        self.n_cells = self.X.shape[0]
        
        # Get unique cell types and perturbation types
        self.cell_types = self.adata.obs[cell_key].unique()
        self.perturbation_types = self.adata.obs[pert_key].unique()
        
        # Process perturbation embeddings if provided
        self.pert_embeddings = None
        if pert_embedding_path is not None:
            self._load_and_process_pert_embeddings(pert_embedding_path, pca_components, seed)
        
        # Pre-calculate eval_sets if not provided directly

        self.eval_sets = {}
        self.sets = {}
        for cell_type in self.cell_types:
            self.eval_sets[cell_type] = {}
            self.sets[cell_type] = {}
            for perturbation in self.perturbation_types:
                # Get indices for this combination
                mask = (self.adata.obs[cell_key] == cell_type) & (self.adata.obs[pert_key] == perturbation)
                indices = np.where(mask)[0]
                
                if len(indices) > 0:
                    eval_set = ((perturbation in heldout_perts or cell_type in heldout_cell_types) and perturbation != self.control_pert) if self.split_mode == "ood" \
                                else (perturbation in heldout_perts and cell_type in heldout_cell_types)
                    if not eval_set:
                        self.sets[cell_type][perturbation] = indices
                    if eval_set or (perturbation == self.control_pert):
                        self.eval_sets[cell_type][perturbation] = indices
                        
        
        self.set_order = [(cell_type, perturbation) for cell_type in self.sets for perturbation in self.sets[cell_type]]
        self.n_sets = len(self.set_order)
        self.set_size = set_size
        self.data_shape = data_shape
        
        print(f"Loaded {self.n_sets} sets ({cell_key} x {pert_key} combinations)")
        
    def _load_and_process_pert_embeddings(self, pert_embedding_path: str, pca_components: int, seed: Optional[int] = None):
        """Load and process perturbation embeddings with deterministic PCA."""
        # Load perturbation embeddings
        self.raw_pert_embeddings = torch.load(pert_embedding_path)
        
        # Extract the embeddings and keys
        # Assume it's already a tensor
        embeddings = self.raw_pert_embeddings['embedding'].numpy()
        pert_names = self.raw_pert_embeddings['pert_names']
        
        # Apply deterministic PCA if requested
        if pca_components > 0 and pca_components < embeddings.shape[1]:
            print(f"Applying PCA with {pca_components} components")
            # Initialize PCA with a fixed random state for determinism
            random_state = 42 if seed is None else seed
            pca = PCA(n_components=pca_components, random_state=random_state)
            embeddings = pca.fit_transform(embeddings)
            
            # Print the explained variance ratio
            explained_variance = np.sum(pca.explained_variance_ratio_)
            print(f"PCA with {pca_components} components explains {explained_variance:.4f} of variance")
        else:
            print(f"No PCA applied, using {embeddings.shape[1]} components")
        
        # Store as a dictionary
        self.pert_embeddings = {pert_names[i]: embeddings[i] for i in range(len(pert_names))}
        self.pert_embedding_shape = embeddings[0].shape
    
    def __len__(self):
        return self.n_cells // self.set_size
    
    def __getitem__(self, idx):
        idx = idx % self.n_sets
        (cell_type, perturbation) = self.set_order[idx]
        indices = self.sets[cell_type][perturbation] 
        
        # If we have fewer samples than set_size, sample with replacement
        if len(indices) < self.set_size:
            sampled_indices = np.random.choice(indices, size=self.set_size, replace=True)
        else:
            sampled_indices = np.random.choice(indices, size=self.set_size, replace=False)

        ctrl_indices = np.random.choice(self.sets[cell_type][self.control_pert], size=self.set_size, replace=False)
        
        # Get the expression vectors for the sampled cells
        samples = self.X[sampled_indices]

        result = {
            'samples': torch.tensor(samples, dtype=torch.float),
            'cell_type': cell_type,
            'perturbation': perturbation,
            'is_control': perturbation == self.control_pert,
            'ctrl_samples': torch.tensor(self.X[ctrl_indices], dtype=torch.float)
        }
        
        # Add perturbation embedding if available
        if self.pert_embeddings is not None:
            if perturbation in self.pert_embeddings:
                result['pert_embedding'] = torch.tensor(
                    self.pert_embeddings[perturbation], dtype=torch.float
                )
            else:
                result['pert_embedding'] = torch.zeros(
                    self.pert_embedding_shape, dtype=torch.float
                )
        
        return result
    
    def get_metadata(self):
        """Return metadata about the dataset."""
        return {
            'n_sets': self.n_sets,
            'set_size': self.set_size,
            'data_shape': self.data_shape,
            'cell_types': list(self.cell_types),
            'perturbation_types': list(self.perturbation_types),
            'sets': self.sets,
            'eval_sets': self.eval_sets,
        } 