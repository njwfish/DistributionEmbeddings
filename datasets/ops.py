from collections import defaultdict
import os
import numpy as np
from glob import glob
import pickle
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

class OPSDataset(Dataset):
    def __init__(
            self, 
            set_size: int = 100,
            prob_spatial: float = 0.5,
            spatial_kernel_width: float = 10,
            replace: bool = False,
            seed: int = 42,
            data_dir: str = "/orcd/data/omarabu/001/njwfish/DistributionEmbeddings/data/ops",
            data_shape: list[int] = [2, 104, 104],
        ):
        
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.set_size = set_size
        self.prob_spatial = prob_spatial
        self.replace = replace
        self.data_dir = data_dir
        self.spatial_kernel_width = spatial_kernel_width

        tiles = sorted(glob(f"{data_dir}/processed/*/*/*"))

        self.images = []
        self.pert_indices = defaultdict(list)
        self.dist_matrices = []
        self.tile_to_idx = []

        image_index = 0
        cumulative_index = 0
        for tile_path in tqdm(tiles):
            if os.path.exists(f"{tile_path}/gdf.parquet"):
                self.images.append(np.load(f"{tile_path}/single_cell_images.npy"))
                
                pert_indices = pickle.load(open(f"{tile_path}/pert_indices.pkl", "rb"))
                for k in pert_indices:
                    pert_indices[k] += cumulative_index
                    self.pert_indices[k].extend(pert_indices[k])
                
                self.dist_matrices.append(np.load(f"{tile_path}/dist_matrix.npy"))
                self.tile_to_idx.append((cumulative_index, self.images[-1].shape[0]))
                image_index += 1
                cumulative_index += len(self.images[-1])

        self.images = np.concatenate(self.images)
        for k in self.pert_indices:
            self.pert_indices[k] = np.array(self.pert_indices[k])
    
    def __len__(self):
        return self.images.shape[0] // self.set_size
    

    def _sample_spatial(self):
        # sample a random tile
        tile_idx = np.random.randint(0, len(self.tile_to_idx))
        tile_start, tile_len = self.tile_to_idx[tile_idx]
        center_cell_idx = np.random.randint(0, tile_len)
        
        dist_matrix = self.dist_matrices[tile_idx][center_cell_idx]
        prob_dist = np.exp(-self.spatial_kernel_width * dist_matrix)
        prob_dist /= prob_dist.sum()
        cell_idx = np.random.choice(
            np.arange(len(prob_dist)), size=self.set_size, p=prob_dist, 
            replace=self.replace or self.set_size > tile_len
        )
        return cell_idx + tile_start
    
    def _sample_pert(self):
        # sample a random perturbation
        pert_to_sample = np.random.choice(list(self.pert_indices.keys()))
        pert_idx = np.random.choice(self.pert_indices[pert_to_sample], size=self.set_size, replace=self.replace or self.set_size > len(self.pert_indices[pert_to_sample]))
        return pert_idx, pert_to_sample
    
    def __getitem__(self, idx):
        sample_spatial = np.random.binomial(1, self.prob_spatial)
        if sample_spatial:
            idx = self._sample_spatial()
            # might be nice to save the pert so i can easily return that here 
            return {
                'samples': torch.tensor(self.images[idx]).float(),
                # 'pert': np.repeat(-1, self.set_size)
            }
        else:
            # sample a random image
            idx, pert_to_sample = self._sample_pert()
            return {
                'samples': torch.tensor(self.images[idx]).float(),
                # 'pert': np.repeat(pert_to_sample, self.set_size)
            }