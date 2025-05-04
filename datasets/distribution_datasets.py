import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, Tuple, List, Any
from torchvision import datasets, transforms
from torchvision.datasets import MNIST
import scipy as sp


class NormalDistributionDataset(Dataset):
    """Dataset for multivariate normal distributions."""
    
    def __init__(
        self,
        n_sets: int = 10,
        set_size: int = 100,
        data_shape: Tuple[int, ...] = (5,),
        prior: str = 'normal',
        prior_params: Tuple[float, float] = (0, 1),
        seed: Optional[int] = None,
    ):
        """
        Args:
            n_sets: Number of parameter sets to generate
            set_size: Number of samples per parameter set
            data_shape: Shape of each sample
            prior: Prior distribution for the mean
            prior_params: Parameters for the prior distribution
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)
            
        # Generate or use provided parameters
        self.mu, self.var = self.generate_params(n_sets, data_shape, prior, prior_params)

        # Generate samples
        self.data = self.sample(self.mu, self.var, n_sets, set_size, data_shape)
        
    
    def generate_params(self, n_sets, data_shape, prior, prior_params):
        if prior == 'normal':
            mu = np.random.normal(prior_params[0], prior_params[1], (n_sets, *data_shape))
            var = np.random.normal(prior_params[0], prior_params[1], (n_sets, *data_shape))**2
        elif prior == 'uniform':
            mu = np.random.uniform(prior_params[0], prior_params[1], (n_sets, *data_shape))
            var = np.random.uniform(0, prior_params[1], (n_sets, *data_shape))
        return mu.squeeze(), var.squeeze()
    
    def sample(self, mu, var, n_sets, set_size, data_shape):
        if mu.ndim == 1 and var.ndim == 1:
            return np.random.randn(n_sets, set_size, *data_shape) * np.sqrt(var)[None, None, :] + mu[None, None, :]
        elif mu.ndim == 2 and var.ndim == 2:
            return np.random.randn(n_sets, set_size, *data_shape) * np.sqrt(var)[:, None, :] + mu[:, None, :]
        else:
            raise ValueError("mu and var must have the same number of dimensions")
    
    def fisher_rao_distance(self, mu, var):
        sigma = np.sqrt(var)

        diff_mu = (mu[:, None, :] - mu[None, :, :])**2
        diff_sigma = (sigma[:, None, :] - sigma[None, :, :])**2

        sigma_prod = sigma[:, None, :] * sigma[None, :, :]

        cosh_arg = 1 + (diff_mu + diff_sigma) / (2 * sigma_prod)
        dist = np.sqrt(2) * np.arccosh(cosh_arg)

        return np.linalg.norm(dist, axis=2)
    
    def wasserstein_distance(self,mu, var):
        mean_dist = np.linalg.norm(mu[:, None, :] - mu[None, :, :], axis=2)
        var_matrices = np.eye(var.shape[1])[None, :] * var[:, None]
        var_dist = np.linalg.trace(var_matrices[None, :, :] + var_matrices[:, None, :] - 2 * np.sqrt(np.sqrt(var_matrices[None, :, :]) @ var_matrices[:, None, :] @ np.sqrt(var_matrices[None, :, :])))
        return mean_dist + var_dist
    
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
        spike: float = 1.,
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

        self.spike = spike

        # Generate or use provided parameters
        if custom_probs is None:
            self.probs = self.generate_params(n_sets, data_shape)
        else:
            self.probs = custom_probs
            
        self.n_per_multinomial = n_per_multinomial
            
        # Generate samples
        self.data = self.sample(self.probs, n_per_multinomial, n_sets, set_size, data_shape)
            
    def generate_params(self, n_sets, data_shape):
        dim = np.prod(data_shape)
        alphas = np.ones(dim)
        alphas[0] = self.spike
        # instead of np.ones(dim)*self.spike
        probs = np.random.dirichlet(alpha=alphas, size=n_sets)
        return probs.reshape(n_sets, *data_shape)

    
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

class MultivariateNormalDistributionDataset(Dataset):
    """Dataset for multivariate normal distributions."""
    
    def __init__(
            self, 
            n_sets: int = 10_000, 
            set_size: int = 100, 
            data_shape: Tuple[int, ...] = (2,),     
            prior_mu: Tuple[float, float] = (0, 1),
            prior_cov_df: int = 10,
            prior_cov_scale: float = 1.,
            seed: Optional[int] = None,
            ):
        """
        Args:
            n_sets: Number of parameter sets to generate
            set_size: Number of samples per parameter set
            data_shape: Shape of each sample
            seed: Random seed for reproducibility
        """
        self.n_sets = n_sets
        
        if seed is not None:
            np.random.seed(seed)
            
        self.mu = np.random.uniform(prior_mu[0], prior_mu[1], (n_sets, data_shape[0]))
        # sample covariance matrix from inverse wishart distribution
        self.cov = np.linalg.inv(sp.stats.wishart.rvs(
            df=prior_cov_df, scale=prior_cov_scale*np.eye(data_shape[0]), size=n_sets
        ))
        
        self.data = self.sample(self.mu, self.cov, n_sets, set_size, data_shape)
        
    def sample(self, mu, cov, n_sets, set_size, data_shape):
        """
        Vectorized sampling from a multivariate normal distribution.
        
        Args:
            mu: Mean of the distributions (n_sets, n_features)
            cov: Covariance matrix of the distributions (n_sets, n_features, n_features)
            n_sets: Number of sets to sample from
            set_size: Number of samples per set
            data_shape: Shape of each sample
            
        Returns:
            z: Samples from the multivariate normal distribution (n_sets, set_size, n_features)
        """
        z = np.random.randn(n_sets, set_size, data_shape[0])

        L_cov = np.linalg.cholesky(cov)
        # transform the normal to a multivariate normal
        # z = np.einsum('ijk,ikl->ijl', z, L_cov)
        z = np.matmul(z, L_cov.transpose(0, 2, 1))
        z = z + mu[:, None, :]
        return z
    
    def fisher_rao_distance(self, mu, cov):
        raise NotImplementedError("Fisher-Rao distance not implemented for multivariate normal distribution")
    
    def wasserstein_distance(self, mu, cov):
        mean_dist = np.linalg.norm(mu[:, None, :] - mu[None, :, :], axis=2)
        var_dist = np.zeros((mu.shape[0], mu.shape[0]))
        for i, cov_i in enumerate(cov):
            for j, cov_j in enumerate(cov):
                var_dist[i, j] = np.trace(cov_i + cov_j - 2 * sp.linalg.sqrtm(sp.linalg.sqrtm(cov_i) @ cov_j @ sp.linalg.sqrtm(cov_i)))
        return mean_dist + var_dist

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        return {
            'samples': torch.tensor(self.data[idx], dtype=torch.float),
            'mean': torch.tensor(self.mu[idx], dtype=torch.float),
            'cov': torch.tensor(self.cov[idx], dtype=torch.float)
        }

class GaussianMixtureModelDataset(Dataset):
    """Dataset for Gaussian Mixture Models."""
    
    def __init__(
            self, 
            n_sets: int = 10_000, 
            set_size: int = 100, 
            data_shape: Tuple[int, ...] = (2,),
            n_components: int = 3,
            prior_mu: Tuple[float, float] = (0, 1),
            prior_cov_df: int = 10,
            prior_cov_scale: float = 1.,
            dirichlet_alpha: float = 1.,
            seed: Optional[int] = None,
            ):
        """
        Args:
            n_sets: Number of parameter sets to generate
            set_size: Number of samples per parameter set
            data_shape: Shape of each sample
            n_components: Number of mixture components
            prior_mu: Range for uniform prior on component means
            prior_cov_df: Degrees of freedom for Wishart prior on precision matrices
            prior_cov_scale: Scale matrix for Wishart prior
            dirichlet_alpha: Concentration parameter for Dirichlet prior on mixture weights
            seed: Random seed for reproducibility
        """
        self.n_sets = n_sets
        self.n_components = n_components
        
        if seed is not None:
            np.random.seed(seed)
            
        # Sample mixture weights from Dirichlet distribution
        self.weights = np.random.dirichlet(
            alpha=[dirichlet_alpha] * n_components, 
            size=n_sets
        )  # shape: (n_sets, n_components)
        
        # Sample component means
        self.mu = np.random.uniform(
            prior_mu[0], 
            prior_mu[1], 
            (n_sets, n_components, data_shape[0])
        )  # shape: (n_sets, n_components, data_dim)
        
        # Sample component covariances from inverse Wishart
        self.cov = np.zeros((n_sets, n_components, data_shape[0], data_shape[0]))
        for i in range(n_sets):
            for j in range(n_components):
                self.cov[i, j] = np.linalg.inv(sp.stats.wishart.rvs(
                    df=prior_cov_df, 
                    scale=prior_cov_scale*np.eye(data_shape[0])
                ))
        
        self.data = self.sample(
            self.weights, self.mu, self.cov, 
            n_sets, set_size, data_shape
        )
        
    def sample(self, weights, mu, cov, n_sets, set_size, data_shape):
        """
        Vectorized sampling from Gaussian Mixture Models.
        
        Args:
            weights: Mixture weights (n_sets, n_components)
            mu: Component means (n_sets, n_components, n_features)
            cov: Component covariances (n_sets, n_components, n_features, n_features)
            n_sets: Number of sets to sample from
            set_size: Number of samples per set
            data_shape: Shape of each sample
            
        Returns:
            z: Samples from the GMM (n_sets, set_size, n_features)
        """
        # Generate component assignments using cumsum trick
        # Shape: (n_sets, set_size)
        rand_uniform = np.random.rand(n_sets, set_size)  # shape: (n_sets, set_size)
        cumsum_weights = np.cumsum(weights, axis=1)  # shape: (n_sets, n_components)
        
        # Expand dimensions for broadcasting
        rand_uniform = rand_uniform[..., None]  # shape: (n_sets, set_size, 1)
        cumsum_weights = cumsum_weights[:, None, :]  # shape: (n_sets, 1, n_components)
        
        # Compare and sum to get assignments
        assignments = np.sum(rand_uniform > cumsum_weights[..., :-1], axis=2)
        
        # Generate standard normal samples for all components at once
        # Shape: (n_sets, n_components, set_size, data_dim)
        z = np.random.randn(n_sets, self.n_components, set_size, data_shape[0])
        
        # Compute Cholesky decomposition for all covariance matrices
        # Shape: (n_sets, n_components, data_dim, data_dim)
        L_cov = np.linalg.cholesky(cov)
        
        # Transform standard normal to multivariate normal for all components
        # Shape after matmul: (n_sets, n_components, set_size, data_dim)
        z = np.matmul(z, L_cov.transpose(0, 1, 3, 2))
        
        # Add means for all components
        # Shape: (n_sets, n_components, set_size, data_dim)
        z = z + mu[:, :, None, :]
        
        # Create indices for selecting samples
        batch_idx = np.arange(n_sets)[:, None]
        sample_idx = np.arange(set_size)[None, :]
        
        # Select samples according to assignments
        # Shape: (n_sets, set_size, data_dim)
        result = z[batch_idx, assignments, sample_idx]
        
        return result
    
    def wasserstein_distance(self, weights1, mu1, cov1, weights2, mu2, cov2):
        """
        Approximate Wasserstein distance between two GMMs.
        This is a simplified version that treats each component independently.
        """
        raise NotImplementedError(
            "Wasserstein distance not implemented for GMM. "
            "Computing exact Wasserstein between GMMs is challenging "
            "and typically requires numerical approximations."
        )

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        return {
            'samples': torch.tensor(self.data[idx], dtype=torch.float)
        }