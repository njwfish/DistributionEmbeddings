from collections import defaultdict
import os
import numpy as np
from glob import glob
import pickle
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from functools import lru_cache
import json
import bisect
from sklearn.decomposition import PCA

def get_tile_size(tile_path):
    """Get the size of a tile without loading the full data."""
    with open(f"{tile_path}/single_cell_images.npy", 'rb') as f:
        # Read the shape from the numpy file header
        version = np.lib.format.read_magic(f)
        shape, fortran_order, dtype = np.lib.format._read_array_header(f, version)
        return shape[0]  # Return number of images

def generate_metadata(data_dir: str, output_file: str = "ops_metadata.json"):
    """Generate and save metadata for OPS dataset."""
    print("Generating metadata for OPS dataset...")
    tiles = sorted(glob(f"{data_dir}/processed/*/*/*"))
    print(f"Found {len(tiles)} potential tiles")
    
    metadata = {
        'tile_paths': [],
        'tile_sizes': [],
        'pert_indices': defaultdict(list),
        'tile_to_idx': [],
        'version': '1.1',  # Add version for future compatibility
        'total_images': 0,
        'perturbation_stats': defaultdict(lambda: {'tile_count': 0, 'total_cells': 0})
    }
    
    cumulative_index = 0
    valid_tiles = 0
    for tile_idx, tile_path in enumerate(tqdm(tiles)):
        if os.path.exists(f"{tile_path}/gdf.parquet"):
            try:
                # Get number of images without loading the full data
                n_images = get_tile_size(tile_path)
                
                # Store perturbation indices
                pert_indices = pickle.load(open(f"{tile_path}/pert_indices.pkl", "rb"))
                
                # Validate perturbation indices
                for k, indices in pert_indices.items():
                    if not isinstance(indices, np.ndarray):
                        indices = np.array(indices)
                    if indices.max() >= n_images:
                        print(f"Warning: Invalid indices in tile {tile_path} for perturbation {k}")
                        print(f"Max index {indices.max()} >= tile size {n_images}")
                        continue
                        
                    # Store local indices for each perturbation
                    metadata['pert_indices'][k].append({
                        'tile_idx': tile_idx,
                        'local_indices': indices.tolist()  # Keep indices local to tile
                    })
                    metadata['perturbation_stats'][k]['tile_count'] += 1
                    metadata['perturbation_stats'][k]['total_cells'] += len(indices)
                
                metadata['tile_to_idx'].append([cumulative_index, n_images])
                metadata['tile_paths'].append(tile_path)
                metadata['tile_sizes'].append(n_images)
                metadata['total_images'] += n_images
                cumulative_index += n_images
                valid_tiles += 1
                
            except Exception as e:
                print(f"Error processing tile {tile_path}: {str(e)}")
                continue
    
    print(f"\nMetadata generation complete:")
    print(f"Processed {valid_tiles} valid tiles out of {len(tiles)} total tiles")
    print(f"Total images: {metadata['total_images']}")
    print("\nPerturbation statistics:")
    for pert, stats in metadata['perturbation_stats'].items():
        print(f"{pert}: {stats['tile_count']} tiles, {stats['total_cells']} total cells")
    
    # Save metadata
    with open(output_file, 'w') as f:
        json.dump(metadata, f)
    print(f"\nMetadata saved to {output_file}")
    return metadata

class OPSDataset(Dataset):
    def __init__(
            self, 
            set_size: int = 100,
            prob_spatial: float = 0.5,
            spatial_kernel_width: float = 10,
            replace: bool = True,
            seed: int = 42,
            data_dir: str = "/orcd/scratch/bcs/001/njwfish/data/ops", # "/orcd/data/omarabu/001/njwfish/DistributionEmbeddings/data/ops",
            pert_repr: str = 'genept',
            pert_embedding_dim: int = 16,
            data_shape: list[int] = [2, 104, 104],
            cache_size: int = 1000,  # Number of tiles to cache
            metadata_file: str = "ops_metadata.json",
            control_pert: str = 'nontargeting',
            holdout_perturbations: list[str] = ['BIRC5', 'DONSON', 'UBE2I', 'RACGAP1', 'PCNA', 'RPA1', 'MAD2L1', 'RRM1'],
        ):
        if pert_repr == 'genept':
            self.pert_indices = pickle.load(open(f"{data_dir}/genept_gen_rep.pkl", "rb"))
        else:
            raise ValueError(f"Perturbation representation {pert_repr} not supported")
        
        # compute pca for pert_embedding_dim
        self.pert_embedding_dim = pert_embedding_dim
        pert_embedding = np.vstack([self.pert_indices[k] for k in self.pert_indices.keys()])
        self.pert_embedding = PCA(n_components=pert_embedding_dim).fit_transform(pert_embedding)
        self.pert_embedding_dict = {k: self.pert_embedding[i] for i, k in enumerate(self.pert_indices.keys())}

        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.set_size = set_size
        self.prob_spatial = prob_spatial
        self.replace = replace
        self.data_dir = data_dir
        self.spatial_kernel_width = spatial_kernel_width
        self.holdout_perturbations = set(holdout_perturbations)  # Convert to set for O(1) lookup

        # Load metadata
        metadata_path = os.path.join(data_dir, metadata_file)
        if not os.path.exists(metadata_path):
            print(f"Metadata file {metadata_file} not found. Generating...")
            metadata = generate_metadata(data_dir, metadata_path)
        else:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                
            # Validate metadata version
            if metadata.get('version', '1.0') != '1.1':
                print(f"Warning: Metadata version mismatch. Expected 1.1, got {metadata.get('version', '1.0')}")
                print("Regenerating metadata...")
                metadata = generate_metadata(data_dir, metadata_path)
        
        self.metadata = metadata
        self.tile_paths = metadata['tile_paths']
        self.tile_to_idx = metadata['tile_to_idx']
        
        self.control_pert = control_pert
        # Filter out holdout perturbations
        self.pert_indices = {
            k: v for k, v in metadata['pert_indices'].items() 
            if (k not in self.holdout_perturbations) and (k in self.pert_embedding_dict or k == control_pert) 
        }
        self.heldout_pert_indices = {
            k: v for k, v in metadata['pert_indices'].items() 
            if (k in self.holdout_perturbations) and (k in self.pert_embedding_dict)
        }
        self.total_images = metadata['total_images']
        
        # Print dataset statistics
        print(f"\nDataset initialized:")
        print(f"Total tiles: {len(self.tile_paths)}")
        print(f"Total images: {self.total_images}")
        print(f"Available perturbations: {len(self.pert_indices)}")
        print(f"Holdout perturbations: {self.holdout_perturbations}")
        
        # Set up caching
        self._load_images = lru_cache(maxsize=cache_size)(self._load_images)
        self._load_dist_matrix = lru_cache(maxsize=cache_size)(self._load_dist_matrix)
    
    def _load_images(self, tile_path):
        """Load just the images from a tile."""
        return np.load(f"{tile_path}/single_cell_images.npy")
    
    def _load_dist_matrix(self, tile_path):
        """Load just the distance matrix from a tile."""
        return np.load(f"{tile_path}/dist_matrix.npy")
    
    def __len__(self):
        return sum(n_images for _, n_images in self.tile_to_idx) // self.set_size
    
    def _sample_spatial(self):
        # sample a random tile
        tile_idx = np.random.randint(0, len(self.tile_to_idx))
        tile_start, tile_len = self.tile_to_idx[tile_idx]
        center_cell_idx = np.random.randint(0, tile_len)
        
        # Load only the distance matrix
        dist_matrix = self._load_dist_matrix(self.tile_paths[tile_idx])
        
        prob_dist = np.exp(-self.spatial_kernel_width * dist_matrix[center_cell_idx])
        prob_dist /= prob_dist.sum()
        cell_idx = np.random.choice(
            np.arange(len(prob_dist)), size=self.set_size, p=prob_dist, 
            replace=self.replace or self.set_size > tile_len
        )
        
        # Load only the selected images
        images = self._load_images(self.tile_paths[tile_idx])
        return cell_idx + tile_start, images[cell_idx]
    
    def _sample_pert_across_tiles(self):
        # Sample a random perturbation
        pert_to_sample = np.random.choice(list(self.pert_indices.keys()))
        pert_data = self.pert_indices[pert_to_sample]
        
        # Calculate how many samples we need from each tile
        total_available = sum(len(tile_data['local_indices']) for tile_data in pert_data)
        samples_needed = self.set_size
        
        if total_available < samples_needed and not self.replace:
            samples_needed = total_available  # Can't sample more than available without replacement
        
        # Distribute samples proportionally across tiles containing this perturbation
        selected_images = []
        remaining = samples_needed
        
        # Shuffle tile order for randomness
        tile_indices = np.random.permutation(len(pert_data))
        
        for idx in tile_indices:
            tile_data = pert_data[idx]
            tile_idx = tile_data['tile_idx']
            local_indices = np.array(tile_data['local_indices'])
            
            # Determine how many to sample from this tile
            if self.replace:
                # With replacement, we can sample as many as needed from any tile
                n_from_this = min(remaining, len(local_indices))
            else:
                # Without replacement, allocate proportionally to available samples
                n_from_this = min(remaining, len(local_indices))
            
            if n_from_this <= 0:
                continue
            
            # Sample from this tile
            selected_local = np.random.choice(
                local_indices, 
                size=n_from_this, 
                replace=self.replace
            )
            
            # Load only the selected images from this tile
            images = self._load_images(self.tile_paths[tile_idx])
            selected_images.append(images[selected_local])
            
            remaining -= n_from_this
            if remaining <= 0:
                break
        
        # Concatenate samples from different tiles
        all_images = np.concatenate(selected_images, axis=0)
        
        # If we couldn't get enough samples, handle accordingly
        if all_images.shape[0] < self.set_size and self.replace:
            # With replacement, just sample randomly from what we have
            indices = np.random.choice(all_images.shape[0], size=self.set_size, replace=True)
            all_images = all_images[indices]
        
        return np.array([0]), pert_to_sample, all_images  # Return format matching _sample_pert
    
    def __getitem__(self, idx):
        sample_spatial = np.random.binomial(1, self.prob_spatial)
        if sample_spatial:
            idx, images = self._sample_spatial()
            return {
                'samples': torch.tensor(images).float(),
            }
        else:
            # Use the new cross-tile sampling method instead
            _, pert_to_sample, images = self._sample_pert_across_tiles()
            
            if self.prob_spatial > 0:
                return {
                    'samples': torch.tensor(images).float(),
                }
            else:
                return {
                    'samples': torch.tensor(images).float(),
                    'is_control': pert_to_sample == self.control_pert,
                    'pert_embedding': torch.tensor(self.pert_embedding_dict[pert_to_sample]).float() if pert_to_sample != self.control_pert else torch.zeros(self.pert_embedding_dim)
                }