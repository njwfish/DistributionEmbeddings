import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, Tuple
from torchvision import transforms
from torchvision.datasets import CIFAR10


class CIFAR10Dataset(Dataset):
    """Dataset for CIFAR-10 with pure class sets (one class per set)."""
    
    def __init__(
        self,
        set_size: int = 100,
        n_sets: int = 10_000,
        n_classes: int = 10,
        data_shape: Tuple[int, ...] = (3, 32, 32),  # CIFAR-10 has RGB images of size 32x32
        root: str = './data',
        train: bool = True,
        download: bool = True,
        seed: Optional[int] = None,
    ):
        """
        Args:
            set_size: Number of samples per set
            n_sets: Total number of sets to generate (will be multiple of n_classes)
            n_classes: Number of classes to use (max 10 for CIFAR-10)
            data_shape: Shape of each data sample
            root: Root directory for CIFAR-10 data
            train: Whether to use training or test data
            download: Whether to download the dataset if not found
            seed: Random seed for reproducibility

        notes:
            n_sets will be multiple of n_classes
        """
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            
        self.set_size = set_size
        self.n_classes = n_classes
        self.n_sets = n_sets
        # Load CIFAR-10 dataset
        transform = transforms.Compose([transforms.ToTensor()])
        self.cifar10 = CIFAR10(root=root, train=train, download=download, transform=transform)
        
        # Generate pure sets
        self.data, self.metadata = self.generate_pure_sets()
    
    def generate_pure_sets(self):
        """Generate sets with one class per set."""
        # Group indices by label
        label_to_indices = {i: [] for i in range(10)}
        for idx, label in enumerate(self.cifar10.targets):
            label_to_indices[label].append(idx)
        
        # Convert to tensors for easier indexing
        for label in label_to_indices:
            label_to_indices[label] = torch.tensor(label_to_indices[label])
        
        sets = []
        metadata = []
        
        # Create one set for each class
        for class_idx in range(self.n_classes):
            for _ in range(self.n_sets // self.n_classes):
                # Sample set_size images from this class
                indices = torch.randperm(len(label_to_indices[class_idx]))[:self.set_size]
                
                # Get the actual data indices
                data_indices = label_to_indices[class_idx][indices]
                
                # CIFAR-10 data is stored differently than MNIST - gather the images
                images = torch.stack([self.cifar10[idx.item()][0] for idx in data_indices])  
                
                sets.append(images)
                metadata.append(class_idx)

        sets = torch.stack(sets).float()
        metadata = torch.tensor(metadata)
        self.n_sets = sets.shape[0]
        
        return sets, metadata
    
    def __len__(self):
        return self.n_sets 
    
    def __getitem__(self, idx):
        return {
            'samples': self.data[idx],  # Shape: [set_size, channels, height, width]
            'metadata': self.metadata[idx]  # Class information
        } 