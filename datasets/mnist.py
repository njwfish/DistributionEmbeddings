import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, Tuple, List, Any
from torchvision import datasets, transforms
from torchvision.datasets import MNIST


class MNISTMixedSetsDataset(Dataset):
    """Dataset for MNIST with mixed digit classes per set."""
    
    def __init__(
        self,
        n_sets: int = 10,
        set_size: int = 100,
        classes_per_set: int = 2,
        root: str = './data',
        train: bool = True,
        download: bool = True,
        seed: Optional[int] = None,
        data_shape: Tuple[int, int, int] = (1, 28, 28),
    ):
        """
        Args:
            n_sets: Number of sets to generate
            set_size: Number of samples per set
            classes_per_set: Number of digit classes to mix in each set
            root: Root directory for MNIST data
            train: Whether to use training or test data
            download: Whether to download the dataset if not found
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            
        self.n_sets = n_sets
        self.set_size = set_size
        self.classes_per_set = classes_per_set
        
        # Load MNIST dataset
        transform = transforms.Compose([transforms.ToTensor()])
        self.mnist = MNIST(root=root, train=train, download=download, transform=transform)
        
        # Generate mixed sets
        self.data, self.metadata = self.generate_mixed_sets()
    
    def generate_mixed_sets(self):
        """Generate sets with mixed digit classes."""
        # Group indices by label
        label_to_indices = {i: torch.where(torch.tensor(self.mnist.targets) == i)[0] for i in range(10)}
        
        sets = []
        metadata = []
        
        for _ in range(self.n_sets):
            # Generate random mixture weights
            mixture_weights = torch.rand(self.classes_per_set)
            mixture_weights = mixture_weights / mixture_weights.sum()
            
            # Determine number of samples per class
            set_sizes_per_class = np.random.multinomial(self.set_size, mixture_weights)
            
            set_per_class = []
            metadata_per_class = []
            
            # Sample from each selected class
            for set_size_per_class in set_sizes_per_class:
                # Random label from 0-9
                label = torch.randint(0, 10, (1,)).item()
                
                # Sample set_size_per_class images from this class
                indices = torch.randperm(len(label_to_indices[label]))[:set_size_per_class]
                images = self.mnist.data[label_to_indices[label][indices]].float().unsqueeze(1) / 255.0
                
                set_per_class.append(images)
                metadata_per_class.append(label)
            
            # Combine all classes into one set
            sets.append(torch.cat(set_per_class, dim=0))
            metadata.append(metadata_per_class)
        
        print(torch.stack(sets).float().shape)
        return torch.stack(sets).float(), metadata
    
    def __len__(self):
        return self.n_sets
    
    def __getitem__(self, idx):
        # Return samples for a specific set and its distances to other sets
        return {
            'samples': self.data[idx],  # Shape: [set_size, channels, height, width]
            'metadata': self.metadata[idx]  # Class information
        }

class MNISTBernoulliDataset(Dataset):
    # mnist sets with dirichlet-mixed 0s and 1s! :)
    
    def __init__(
        self,
        n_sets: int = 10,
        set_size: int = 100,
        root: str = './data',
        train: bool = True,
        download: bool = True,
        seed: Optional[int] = None,
        alpha: float = 1.0,  # dirichlet concentration
        data_shape: Tuple[int, int, int] = (1, 28, 28),
    ):
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        self.n_sets = n_sets
        self.set_size = set_size
        self.alpha = alpha
        self.data_shape = data_shape

        # load mnist
        transform = transforms.Compose([transforms.ToTensor()])
        self.mnist = MNIST(root=root, train=train, download=download, transform=transform)
        
        # filter for digits 0 and 1 only
        targets = torch.tensor(self.mnist.targets)
        self.indices = {
            0: torch.where(targets == 0)[0],
            1: torch.where(targets == 1)[0],
        }

        # generate the sets!
        self.data, self.metadata, self.prob = self.make_sets()
    
    def make_sets(self):
        sets, metadata, probs = [], [], []
        
        for _ in range(self.n_sets):
            # dirichlet sample for [p0, p1]
            mix = np.random.dirichlet([self.alpha, self.alpha])
            probs.append(mix.tolist())

            # multinomial count from mixture
            counts = np.random.multinomial(self.set_size, mix)

            set_imgs, set_labels = [], []

            for digit, count in zip([0, 1], counts):
                idx = torch.randperm(len(self.indices[digit]))[:count]
                imgs = self.mnist.data[self.indices[digit][idx]].float().unsqueeze(1) / 255.0
                set_imgs.append(imgs)
                set_labels += [digit] * count

            sets.append(torch.cat(set_imgs, dim=0))
            metadata.append(set_labels)

        return torch.stack(sets), metadata, probs

    def __len__(self):
        return self.n_sets
    
    def fisher_rao_distance(self, probs):
        return np.arccos(np.sqrt(probs[:, None, :] * probs[None, :, :]).sum(axis=2))

    def __getitem__(self, idx):
        return {
            'samples': self.data[idx],       # shape: [set_size, 1, 28, 28]
            'metadata': self.metadata[idx],  # list of 0s and 1s
        }
