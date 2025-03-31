import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, Tuple, List, Any
from torchvision import datasets, transforms
from torchvision.datasets import MNIST


class NormalDistributionDataset(Dataset):
    """Dataset for multivariate normal distributions."""
    
    def __init__(
        self,
        n_sets: int = 10,
        set_size: int = 100,
        data_shape: Tuple[int, ...] = (5,),
        custom_mu: Optional[np.ndarray] = None,
        custom_var: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
    ):
        """
        Args:
            n_sets: Number of parameter sets to generate
            set_size: Number of samples per parameter set
            data_shape: Shape of each sample
            custom_mu: Optional custom mean parameters
            custom_var: Optional custom variance parameters
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)
            
        # Generate or use provided parameters
        if custom_mu is None or custom_var is None:
            self.mu, self.var = self.generate_params(n_sets, data_shape)
        else:
            self.mu = custom_mu
            self.var = custom_var
            
        # Generate samples
        self.data = self.sample(self.mu, self.var, n_sets, set_size, data_shape)
        
    
    def generate_params(self, n_sets, data_shape):
        mu = np.random.randn(n_sets, *data_shape)
        var = np.random.randn(n_sets, *data_shape)**2
        return mu.squeeze(), var.squeeze()
    
    def sample(self, mu, var, n_sets, set_size, data_shape):
        if mu.ndim == 1 and var.ndim == 1:
            return np.random.randn(n_sets, set_size, *data_shape) * np.sqrt(var)[None, None, :] + mu[None, None, :]
        elif mu.ndim == 2 and var.ndim == 2:
            return np.random.randn(n_sets, set_size, *data_shape) * np.sqrt(var)[:, None, :] + mu[:, None, :]
        else:
            raise ValueError("mu and var must have the same number of dimensions")
    
    def fisher_rao_distance(self, mu, var):
        diff_mu = (mu[:, None, :] - mu[None, :, :])**2
        diff_var = (var[:, None, :] - var[None, :, :])**2
        sum_var = (var[:, None, :] + var[None, :, :])**2
        return np.linalg.norm(
            np.arctanh((diff_mu + 2 * diff_var)/(diff_mu + 2 * sum_var)), axis=2
        )
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        return {
            'samples': torch.tensor(self.data[idx], dtype=torch.float)
        }


class PoissonDistributionDataset(Dataset):
    """Dataset for Poisson distributions."""
    
    def __init__(
        self,
        n_sets: int = 10,
        set_size: int = 100,
        data_shape: Tuple[int, ...] = (5,),
        rate_range: Tuple[float, float] = (0, 10),
        custom_rate: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
    ):
        """
        Args:
            n_sets: Number of parameter sets to generate
            set_size: Number of samples per parameter set
            data_shape: Shape of each sample
            rate_range: Range of rate parameters to generate
            custom_rate: Optional custom rate parameters
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)
            
        # Generate or use provided parameters
        if custom_rate is None:
            self.rate = self.generate_params(n_sets, data_shape, rate_range)
        else:
            self.rate = custom_rate
            
        # Generate samples
        self.data = self.sample(self.rate, n_sets, set_size, data_shape)
        
    
    def generate_params(self, n_sets, data_shape, rate_range):
        rate = np.random.uniform(rate_range[0], rate_range[1], (n_sets, *data_shape))
        return rate.squeeze()
    
    def sample(self, rate, n_sets, set_size, data_shape):
        if rate.ndim == 1:
            return np.random.poisson(rate[None, None, :], (n_sets, set_size, *data_shape))
        elif rate.ndim == 2:
            return np.random.poisson(rate[:, None, :], (n_sets, set_size, *data_shape))
        else:
            raise ValueError("rate must have the same number of dimensions as n_features")
    
    def fisher_rao_distance(self, rate):
        return np.linalg.norm(np.sqrt(rate[:, None, :]) - np.sqrt(rate[None, :, :]), axis=2)
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        return {
            'samples': torch.tensor(self.data[idx], dtype=torch.float)
        }


class MultinomialDistributionDataset(Dataset):
    """Dataset for multinomial distributions."""
    
    def __init__(
        self,
        n_sets: int = 10,
        set_size: int = 100,
        data_shape: Tuple[int, ...] = (5,),
        n_per_multinomial: int = 10,
        custom_probs: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
    ):
        """
        Args:
            n_sets: Number of parameter sets to generate
            set_size: Number of samples per parameter set
            data_shape: Shape of each sample (categories)
            n_per_multinomial: Number of trials for each multinomial
            custom_probs: Optional custom probability parameters
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)
            
        # Generate or use provided parameters
        if custom_probs is None:
            self.probs = self.generate_params(n_sets, data_shape)
        else:
            self.probs = custom_probs
            
        self.n_per_multinomial = n_per_multinomial
            
        # Generate samples
        self.data = self.sample(self.probs, n_per_multinomial, n_sets, set_size, data_shape)
        
    
    def generate_params(self, n_sets, data_shape):
        # Generate probability vectors for each set
        probs = np.random.rand(n_sets, *data_shape)
        probs = probs / np.sum(probs, axis=1, keepdims=True)
        return probs.squeeze()
    
    def sample(self, probs, n_per_multinomial, n_sets, set_size, data_shape):
        if probs.ndim == 1:
            probs = np.tile(probs, (n_sets, 1))
        elif probs.ndim == 2:
            pass
        else:
            raise ValueError("probs must have the same number of dimensions as n_features")
            
        # Initialize array to store samples
        x = np.zeros((n_sets, set_size, *data_shape))
        for i in range(n_sets):
            x[i] = np.random.multinomial(n_per_multinomial, probs[i], size=set_size)
        return x
    
    def fisher_rao_distance(self, probs):
        return np.arccos(np.sqrt(probs[:, None, :] * probs[None, :, :]).sum(axis=2))
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        return {
            'samples': torch.tensor(self.data[idx], dtype=torch.float)
        }   
