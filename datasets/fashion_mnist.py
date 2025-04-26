import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, Tuple
from torchvision import transforms
from torchvision.datasets import FashionMNIST


class FashionMNISTDataset(Dataset):
    """Dataset for FashionMNIST with pure class sets (one class per set)."""
    
    def __init__(
        self,
        set_size: int = 100,
        n_sets: int = 10_000,
        n_classes: int = 10,
        data_shape: Tuple[int, ...] = (1, 28, 28),
        root: str = './data',
        train: bool = True,
        download: bool = True,
        seed: Optional[int] = None,
    ):
        """
        Args:
            set_size: Number of samples per set
            n_sets: Total number of sets to generate (will be multiple of n_classes)
            n_classes: Number of classes to use (max 10 for FashionMNIST)
            data_shape: Shape of each data sample
            root: Root directory for FashionMNIST data
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
        # Load FashionMNIST dataset
        transform = transforms.Compose([transforms.ToTensor()])
        self.fashion_mnist = FashionMNIST(root=root, train=train, download=download, transform=transform)
        
        # Generate pure sets
        self.data, self.metadata = self.generate_pure_sets()
    
    def generate_pure_sets(self):
        """Generate sets with one class per set."""
        # Group indices by label
        label_to_indices = {i: torch.where(self.fashion_mnist.targets == i)[0] for i in range(10)}
        
        sets = []
        metadata = []
        
        # Create one set for each class
        for class_idx in range(self.n_classes):
            for _ in range(self.n_sets // self.n_classes):
                # Sample set_size images from this class
                indices = torch.randperm(len(label_to_indices[class_idx]))[:self.set_size]
                images = self.fashion_mnist.data[label_to_indices[class_idx][indices]].float().unsqueeze(1) / 255.0
                
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