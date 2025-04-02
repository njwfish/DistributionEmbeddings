import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, List, Any, Dict, Callable
from torchvision import datasets, transforms
from torchvision.datasets import MNIST
import random


class SetDataset(Dataset):
    """Base class for datasets that contain sets of elements."""
    
    def __len__(self):
        raise NotImplementedError
    
    def __getitem__(self, idx):
        """
        Returns:
            Dict containing at least 'samples' and 'metadata' keys
        """
        raise NotImplementedError


class MNISTDataset(SetDataset):
    """Dataset for MNIST with pure digit classes per set (one digit class per set)."""
    
    def __init__(
        self,
        set_size: int = 100,
        root: str = './data',
        train: bool = True,
        download: bool = True,
        seed: Optional[int] = None,
    ):
        """
        Args:
            set_size: Number of samples per set
            root: Root directory for MNIST data
            train: Whether to use training or test data
            download: Whether to download the dataset if not found
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            
        self.set_size = set_size
        
        # Load MNIST dataset
        transform = transforms.Compose([transforms.ToTensor()])
        self.mnist = MNIST(root=root, train=train, download=download, transform=transform)
        
        # Generate pure sets
        self.data, self.metadata = self.generate_pure_sets()
    
    def generate_pure_sets(self):
        """Generate sets with one digit class per set."""
        # Group indices by label
        label_to_indices = {i: torch.where(torch.tensor(self.mnist.targets) == i)[0] for i in range(10)}
        
        sets = []
        metadata = []
        
        # Create one set for each digit (0-9)
        for digit in range(10):
            # Sample set_size images from this class
            indices = torch.randperm(len(label_to_indices[digit]))[:self.set_size]
            images = self.mnist.data[label_to_indices[digit][indices]].float().unsqueeze(1) / 255.0
            
            sets.append(images)
            metadata.append(digit)
        
        return torch.stack(sets).float(), metadata
    
    def __len__(self):
        return len(self.metadata)  # 10 sets, one for each digit
    
    def __getitem__(self, idx):
        # Return samples for a specific set and its class
        return {
            'samples': self.data[idx],  # Shape: [set_size, channels, height, width]
            'metadata': self.metadata[idx]  # The digit class (0-9)
        }


class SetMixingAugmentation:
    """
    Simple set mixing augmentation that can be applied to batches of sets.
    It randomly selects pairs of sets and mixes them based on a random mixture weight.
    """
    
    def __init__(
        self, 
        mix_probability: float = 0.5,
        seed: Optional[int] = None
    ):
        """
        Args:
            mix_probability: Probability of applying mixing to a batch
            seed: Random seed for reproducibility
        """
        self.mix_probability = mix_probability
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
    
    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply set mixing to a batch of sets.
        
        Args:
            batch: Dictionary with 'samples' tensor of shape [batch_size, set_size, ...] 
                  and 'metadata' list of metadata for each set
                  
        Returns:
            Dictionary with mixed samples and metadata
        """
        # If random value is above mix_probability, return the original batch
        if random.random() > self.mix_probability:
            return batch
            
        samples = batch['samples']
        metadata = batch['metadata']
        
        batch_size = samples.shape[0]
        
        # Need at least 2 sets to mix
        if batch_size < 2:
            return batch
            
        # Get the size of each set
        set_size = samples.shape[1]
        
        # Create pairs of sets to mix
        # For simplicity, we'll just create batch_size//2 pairs
        # If batch_size is odd, the last set will remain unmixed
        mixed_samples = []
        mixed_metadata = []
        
        for i in range(0, batch_size - 1, 2):
            # Get two sets to mix
            set1 = samples[i]
            set2 = samples[i + 1]
            meta1 = metadata[i]
            meta2 = metadata[i + 1]
            
            # Generate a random mixture weight
            alpha = random.random()
            
            # Calculate samples per set
            samples_from_set1 = int(alpha * set_size)
            samples_from_set2 = set_size - samples_from_set1
            
            # Randomly shuffle each set
            indices1 = torch.randperm(set_size)[:samples_from_set1]
            indices2 = torch.randperm(set_size)[:samples_from_set2]
            
            # Select samples from each set
            selected_samples1 = set1[indices1]
            selected_samples2 = set2[indices2]
            
            # Combine samples into a mixed set
            mixed_set = torch.cat([selected_samples1, selected_samples2], dim=0)
            
            # Store mixed set and metadata
            mixed_samples.append(mixed_set)
            mixed_metadata.append([meta1, meta2, alpha])
            
        # If batch_size is odd, keep the last set unchanged
        if batch_size % 2 == 1:
            mixed_samples.append(samples[-1])
            mixed_metadata.append(metadata[-1])
            
        # Return the mixed batch
        return {
            'samples': torch.stack(mixed_samples),
            'metadata': mixed_metadata
        }


def set_mixing_collate_fn(
    batch_items: List[Dict[str, Any]], 
    augmentation: Optional[SetMixingAugmentation] = None
) -> Dict[str, Any]:
    """
    Collate function that first creates a batch from individual items, 
    then optionally applies set mixing augmentation.
    
    Args:
        batch_items: List of dictionaries, each with 'samples' and 'metadata'
        augmentation: Optional SetMixingAugmentation to apply
        
    Returns:
        Dictionary with batched samples and metadata
    """
    # Standard collation - stack samples and create list of metadata
    samples = torch.stack([item['samples'] for item in batch_items])
    metadata = [item['metadata'] for item in batch_items]
    
    batch = {
        'samples': samples,
        'metadata': metadata
    }
    
    # Apply augmentation if provided
    if augmentation is not None:
        batch = augmentation(batch)
        
    return batch


def get_mnist_dataloader(
    set_size: int = 100,
    batch_size: int = 4,
    mix_probability: float = 0.5,
    shuffle: bool = True,
    seed: Optional[int] = None,
    root: str = './data',
    train: bool = True,
    download: bool = True,
) -> DataLoader:
    """
    Creates a DataLoader for MNIST pure sets with optional set mixing augmentation.
    
    Args:
        set_size: Number of samples per set
        batch_size: Number of sets per batch
        mix_probability: Probability of applying mixing to a batch
        shuffle: Whether to shuffle the datasets
        seed: Random seed for reproducibility
        root: Root directory for MNIST data
        train: Whether to use training or test data
        download: Whether to download the dataset if not found
        
    Returns:
        DataLoader with set mixing augmentation
    """
    # Create the dataset
    dataset = MNISTDataset(
        set_size=set_size,
        root=root,
        train=train,
        download=download,
        seed=seed
    )
    
    # Create the augmentation
    augmentation = SetMixingAugmentation(
        mix_probability=mix_probability,
        seed=seed
    )
    
    # Create a collate_fn with the augmentation
    collate_fn = lambda batch: set_mixing_collate_fn(batch, augmentation)
    
    # Create and return the DataLoader
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn
    ) 