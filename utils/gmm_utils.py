import numpy as np
from sklearn.mixture import GaussianMixture
from typing import Tuple, List
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.colors as mcolors
import torch
import torch.autograd.functional as F
from ot.utils import proj_SDP
from ot.gmm import gmm_ot_loss

def fit_gmm_batch(samples: np.ndarray,
                 init_means: np.ndarray,
                 init_covs: np.ndarray,
                 init_weights: np.ndarray,
                 n_iter: int = 100,
                 use_kmeans_init: bool = False,
                 tol: float = 1e-6) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit multiple GMMs to batches of samples using sklearn's GaussianMixture.
    
    Args:
        samples: Array of shape (num_mixtures, num_samples, dim) containing the data points
        init_means: Initial means of shape (num_mixtures, n_components, dim)
        init_covs: Initial covariance matrices of shape (num_mixtures, n_components, dim, dim)
        init_weights: Initial mixing weights of shape (num_mixtures, n_components)
        n_iter: Maximum number of EM iterations
        tol: Convergence tolerance for log-likelihood
        
    Returns:
        means: Final means of shape (num_mixtures, n_components, dim)
        covs: Final covariance matrices of shape (num_mixtures, n_components, dim, dim)
        weights: Final mixing weights of shape (num_mixtures, n_components)
    """
    num_mixtures, num_samples, dim = samples.shape
    n_components = init_means.shape[1]
    
    # Initialize output arrays
    means = np.zeros_like(init_means)
    covs = np.zeros_like(init_covs)
    weights = np.zeros_like(init_weights)
    
    # Fit GMM for each mixture independently
    for mix_idx in range(num_mixtures):
        # Initialize GMM with given parameters
        # double cast weights to normalize
        weights_init_double = init_weights[mix_idx].astype(np.float64)
        weights_init_double = weights_init_double / np.sum(weights_init_double)
        if not use_kmeans_init:
            gmm = GaussianMixture(
                n_components=n_components,
                max_iter=n_iter,
                tol=tol,
                covariance_type='full',
                weights_init=weights_init_double,
                means_init=init_means[mix_idx],
                precisions_init=np.linalg.inv(init_covs[mix_idx]),
                warm_start=True,
                random_state=None
            )
        else:
            gmm = GaussianMixture(
                n_components=n_components,
                max_iter=n_iter,
                tol=tol,
                covariance_type='full',
                random_state=None
            )
        
        # Fit the model
        gmm.fit(samples[mix_idx])
        
        # Store results
        covs[mix_idx] = gmm.covariances_
        weights[mix_idx] = gmm.weights_
        # print("Fit GMM for mixture", mix_idx)
    
    return means, covs, weights 

def plot_gmm_trajectory(means: np.ndarray, 
                       covs: np.ndarray, 
                       weights: np.ndarray,
                       figsize: Tuple[int, int] = (12, 8),
                       alpha: float = 0.3,
                       n_std: float = 2,
                       title: str = None) -> plt.Figure:
    """
    Plot the trajectory of Gaussian components over time with a simplex inset for weights.
    
    Args:
        means: Array of shape (num_timesteps, n_components, dim)
        covs: Array of shape (num_timesteps, n_components, dim, dim)
        weights: Array of shape (num_timesteps, n_components)
        figsize: Figure size (width, height) for the plot
        alpha: Transparency of ellipses
        n_std: Number of standard deviations for ellipse size
        title: Optional title for the trajectory plot
    
    Returns:
        fig: Figure object containing the plot
    """
    if means.shape[-1] != 2:
        raise ValueError("This visualization only works for 2D Gaussian mixtures")
        
    num_timesteps, n_components, _ = means.shape
    
    # Create figure with a single main plot
    fig, ax_traj = plt.subplots(figsize=figsize)
    
    # Set up colors for components
    component_colors = plt.cm.Set2(np.linspace(0, 1, n_components))
    time_colors = plt.cm.viridis(np.linspace(0, 1, num_timesteps))
    
    # Plot trajectories and ellipses
    for k in range(n_components):
        trajectory = means[:, k]
        
        # Plot mean trajectory
        ax_traj.plot(trajectory[:, 0], trajectory[:, 1], '--', 
                    color=component_colors[k], alpha=0.5,
                    label=f'Component {k}')
        
        # Plot ellipses and means at each timestep
        for t in range(num_timesteps):
            # Calculate eigenvalues and eigenvectors
            eigenvals, eigenvecs = np.linalg.eigh(covs[t, k])
            
            # Calculate angle and axes lengths for ellipse
            angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
            width, height = 2 * n_std * np.sqrt(eigenvals)
            
            # Create and add ellipse
            ellipse = Ellipse(xy=means[t, k],
                            width=width,
                            height=height,
                            angle=angle,
                            facecolor=time_colors[t],
                            alpha=alpha,
                            edgecolor=component_colors[k],
                            linewidth=1)
            ax_traj.add_patch(ellipse)
            
            # Plot mean point
            ax_traj.plot(means[t, k, 0], means[t, k, 1], 'o',
                        color=component_colors[k], markersize=4)
    
    # Configure trajectory plot
    ax_traj.set_aspect('equal')
    ax_traj.grid(True, alpha=0.3)
    ax_traj.set_xlabel('x')
    ax_traj.set_ylabel('y')
    ax_traj.set_xlim(-1, 5)
    ax_traj.set_ylim(-1, 5)
    if title:
        ax_traj.set_title(title)
    
    # Add colorbar to show time progression
    # sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis,
    #                           norm=plt.Normalize(vmin=0, vmax=num_timesteps-1))
    # plt.colorbar(sm, ax=ax_traj, label='Time step')
    
    # Create smaller inset axes for the weight simplex in the bottom left corner
    # with white background and black outline
    ax_inset = fig.add_axes([0.16, 0.1, 0.23, 0.17], facecolor='white')
    ax_inset.spines['top'].set_color('black')
    ax_inset.spines['right'].set_color('black')
    ax_inset.spines['bottom'].set_color('black')
    ax_inset.spines['left'].set_color('black')
    ax_inset.spines['top'].set_linewidth(1)
    ax_inset.spines['right'].set_linewidth(1)
    ax_inset.spines['bottom'].set_linewidth(1)
    ax_inset.spines['left'].set_linewidth(1)
    
    # Plot weights evolution in probability simplex in the inset (without labels)
    plot_weight_simplex_inset(weights, ax_inset, component_colors)
    
    plt.tight_layout()
    return fig

def plot_weight_simplex_inset(weights: np.ndarray, 
                             ax: plt.Axes, 
                             colors: np.ndarray) -> None:
    """
    Plot the evolution of mixture weights in a probability simplex inset without labels.
    
    Args:
        weights: Array of shape (num_timesteps, n_components)
        ax: Matplotlib axes to plot on
        colors: Array of colors for each component
    """
    num_timesteps, n_components = weights.shape
    
    if n_components == 2:
        # For 2 components, plot as a 1D line
        times = np.arange(num_timesteps)
        ax.plot(times, weights[:, 0], '-o', color=colors[0],
               markersize=3, markeredgecolor='white', markeredgewidth=0.5)
        ax.plot(times, weights[:, 1], '-o', color=colors[1],
               markersize=3, markeredgecolor='white', markeredgewidth=0.5)
        ax.grid(True, alpha=0.3)
        # No labels or title
        
    elif n_components == 3:
        # For 3 components, plot in triangular simplex
        # Convert to barycentric coordinates
        triangle = np.array([[0, 0], [1, 0], [0.5, np.sqrt(0.75)]])
        barycentric = weights @ triangle
        
        # Draw the simplex triangle
        ax.plot([0, 1], [0, 0], 'k-', alpha=0.3, linewidth=1)
        ax.plot([0, 0.5], [0, np.sqrt(0.75)], 'k-', alpha=0.3, linewidth=1)
        ax.plot([1, 0.5], [0, np.sqrt(0.75)], 'k-', alpha=0.3, linewidth=1)
        
        # Add grid lines inside the simplex
        n_lines = 4  # reduced number of grid lines
        
        for i in range(1, n_lines):
            t = i / n_lines
            
            # Lines parallel to bottom edge
            start = np.array([0, 0]) * (1-t) + np.array([0.5, np.sqrt(0.75)]) * t
            end = np.array([1, 0]) * (1-t) + np.array([0.5, np.sqrt(0.75)]) * t
            ax.plot([start[0], end[0]], [start[1], end[1]], 'gray', alpha=0.2, linewidth=0.5)
            
            # Lines parallel to left edge
            start = np.array([0, 0]) * (1-t) + np.array([1, 0]) * t
            end = np.array([0.5, np.sqrt(0.75)]) - (np.array([0.5, np.sqrt(0.75)]) - np.array([0, 0])) * (1-t)
            ax.plot([start[0], end[0]], [start[1], end[1]], 'gray', alpha=0.2, linewidth=0.5)
            
            # Lines parallel to right edge
            start = np.array([1, 0]) * (1-t) + np.array([0.5, np.sqrt(0.75)]) * t
            end = np.array([0, 0]) + (np.array([1, 0]) - np.array([0, 0])) * (1-t)
            ax.plot([start[0], end[0]], [start[1], end[1]], 'gray', alpha=0.2, linewidth=0.5)
        
        # Plot trajectory with gradient color
        points = barycentric.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        time_colors = plt.cm.viridis(np.linspace(0, 1, num_timesteps-1))
        
        # Plot trajectory line
        for i, (p1, p2) in enumerate(segments):
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], '-',
                   color=time_colors[i], linewidth=1.5, alpha=0.7)
        
        # Plot points with white edge for better visibility
        for t in range(num_timesteps):
            color = plt.cm.viridis(t / (num_timesteps-1))
            ax.plot(barycentric[t, 0], barycentric[t, 1], 'o',
                   color=color, markersize=4,
                   markeredgecolor='white', markeredgewidth=0.5)
        
        # Add small dots at vertices with component colors (smaller than before)
        vertex_size = 40
        ax.scatter([0, 1, 0.5], [0, 0, np.sqrt(0.75)], 
                  c=colors[:3], s=vertex_size, zorder=10,
                  edgecolor='white', linewidth=1)
        
        ax.set_aspect('equal')
        ax.axis('off')
        
        # No title or labels
        
        # Set limits with padding
        pad = 0.05
        ax.set_xlim(-pad, 1+pad)
        ax.set_ylim(-pad, np.sqrt(0.75)+pad)
        
    else:
        # For >3 components, show weight evolution over time
        times = np.arange(num_timesteps)
        for k in range(n_components):
            ax.plot(times, weights[:, k], '-o', 
                   color=colors[k],
                   markersize=3, markeredgecolor='white', markeredgewidth=0.5)
        ax.grid(True, alpha=0.3)
        # No labels or title

def plot_weight_simplex(weights: np.ndarray, 
                       ax: plt.Axes, 
                       colors: np.ndarray) -> None:
    """
    Plot the evolution of mixture weights in a probability simplex.
    
    Args:
        weights: Array of shape (num_timesteps, n_components)
        ax: Matplotlib axes to plot on
        colors: Array of colors for each component
    """
    num_timesteps, n_components = weights.shape
    
    if n_components == 2:
        # For 2 components, plot as a 1D line
        times = np.arange(num_timesteps)
        ax.plot(times, weights[:, 0], '-o', color=colors[0], label='Component 0',
               markersize=4, markeredgecolor='white', markeredgewidth=1)
        ax.plot(times, weights[:, 1], '-o', color=colors[1], label='Component 1',
               markersize=4, markeredgecolor='white', markeredgewidth=1)
        ax.set_xlabel('Time step')
        ax.set_ylabel('Weight')
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=True, fancybox=True, shadow=True)
        ax.set_title('Weight Evolution')
        
    elif n_components == 3:
        # For 3 components, plot in triangular simplex
        # Convert to barycentric coordinates
        triangle = np.array([[0, 0], [1, 0], [0.5, np.sqrt(0.75)]])
        barycentric = weights @ triangle
        
        # Draw the simplex triangle
        ax.plot([0, 1], [0, 0], 'k-', alpha=0.3, linewidth=1)
        ax.plot([0, 0.5], [0, np.sqrt(0.75)], 'k-', alpha=0.3, linewidth=1)
        ax.plot([1, 0.5], [0, np.sqrt(0.75)], 'k-', alpha=0.3, linewidth=1)
        
        # Add grid lines inside the simplex
        n_lines = 6  # number of grid lines per edge
        
        for i in range(1, n_lines):
            t = i / n_lines
            
            # Lines parallel to bottom edge
            start = np.array([0, 0]) * (1-t) + np.array([0.5, np.sqrt(0.75)]) * t
            end = np.array([1, 0]) * (1-t) + np.array([0.5, np.sqrt(0.75)]) * t
            ax.plot([start[0], end[0]], [start[1], end[1]], 'gray', alpha=0.2, linewidth=0.5)
            
            # Lines parallel to left edge
            start = np.array([0, 0]) * (1-t) + np.array([1, 0]) * t
            end = np.array([0.5, np.sqrt(0.75)]) - (np.array([0.5, np.sqrt(0.75)]) - np.array([0, 0])) * (1-t)
            ax.plot([start[0], end[0]], [start[1], end[1]], 'gray', alpha=0.2, linewidth=0.5)
            
            # Lines parallel to right edge
            start = np.array([1, 0]) * (1-t) + np.array([0.5, np.sqrt(0.75)]) * t
            end = np.array([0, 0]) + (np.array([1, 0]) - np.array([0, 0])) * (1-t)
            ax.plot([start[0], end[0]], [start[1], end[1]], 'gray', alpha=0.2, linewidth=0.5)
        
        # Plot trajectory with gradient color
        points = barycentric.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        time_colors = plt.cm.viridis(np.linspace(0, 1, num_timesteps-1))
        
        # Plot trajectory line
        for i, (p1, p2) in enumerate(segments):
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], '-',
                   color=time_colors[i], linewidth=2, alpha=0.7)
        
        # Plot points with white edge for better visibility
        for t in range(num_timesteps):
            color = plt.cm.viridis(t / (num_timesteps-1))
            ax.plot(barycentric[t, 0], barycentric[t, 1], 'o',
                   color=color, markersize=6,
                   markeredgecolor='white', markeredgewidth=1)
        
        # Add component labels with better positioning
        label_pad = 0.05
        ax.text(0, -label_pad, 'Component 0', ha='center', va='top')
        ax.text(1, -label_pad, 'Component 1', ha='center', va='top')
        ax.text(0.5, np.sqrt(0.75)+label_pad, 'Component 2', ha='center', va='bottom')
        
        # Add small dots at vertices with component colors
        vertex_size = 80
        ax.scatter([0, 1, 0.5], [0, 0, np.sqrt(0.75)], 
                  c=colors[:3], s=vertex_size, zorder=10,
                  edgecolor='white', linewidth=1.5)
        
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Add title with adjusted position
        ax.set_title('Weight Simplex Trajectory', pad=20)
        
        # Set limits with padding
        pad = 0.1
        ax.set_xlim(-pad, 1+pad)
        ax.set_ylim(-pad, np.sqrt(0.75)+pad)
        
    else:
        # For >3 components, show weight evolution over time
        times = np.arange(num_timesteps)
        for k in range(n_components):
            ax.plot(times, weights[:, k], '-o', 
                   color=colors[k], label=f'Component {k}',
                   markersize=4, markeredgecolor='white', markeredgewidth=1)
        ax.set_xlabel('Time step')
        ax.set_ylabel('Weight')
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=True, fancybox=True, shadow=True)
        ax.set_title('Weight Evolution')

def optimize_gmm(m_s: torch.Tensor,
                m_t: torch.Tensor,
                C_s: torch.Tensor,
                C_t: torch.Tensor,
                w_s: torch.Tensor,
                w_t: torch.Tensor,
                n_steps: int = 100,
                lr: float = 0.01,
                min_cov: float = 1e-6,
                use_natural_gradient: bool = False,
                damping: float = 1e-4) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[float]]:
    """
    Optimize GMM parameters using torch's functional autograd API with optional natural gradient.
    
    Args:
        m_s: Source means tensor
        m_t: Target means tensor
        C_s: Source covariance tensor
        C_t: Target covariance tensor
        w_s: Source weights tensor (logits)
        w_t: Target weights tensor
        n_steps: Number of optimization steps
        lr: Learning rate
        min_cov: Minimum eigenvalue for covariance matrices
        use_natural_gradient: Whether to use natural gradient descent
        damping: Damping factor for Hessian inversion
    
    Returns:
        means_list: List of mean arrays through optimization
        covs_list: List of covariance arrays through optimization
        weights_list: List of weight arrays through optimization
        loss_list: List of loss values through optimization
    """
    # Ensure inputs are on CPU and detached
    m_t, C_t, w_t = map(lambda x: x.detach().cpu().requires_grad_(False), (m_t, C_t, w_t))
    
    # Initialize parameters
    means = m_s.clone().detach().requires_grad_(True)
    covs = C_s.clone().detach().requires_grad_(True)
    logits = w_s.clone().detach().requires_grad_(True)
    params = (means, covs, logits)
    shapes = [p.shape for p in params]
    
    def flatten_params(params):
        flat = torch.cat([p.reshape(-1) for p in params])
        flat.requires_grad_(True)
        return flat
    
    def unflatten_params(flat_params):
        result = []
        idx = 0
        for shape in shapes:
            n_params = np.prod(shape)
            param = flat_params[idx:idx + n_params].reshape(shape)
            param.requires_grad_(True)
            result.append(param)
            idx += n_params
        return tuple(result)
    
    def loss_fn(params_flat):
        """Loss function that takes flattened parameters."""
        means, covs, logits = unflatten_params(params_flat)
        weights = torch.softmax(logits, dim=0)
        covs = proj_SDP(covs)
        return gmm_ot_loss(means, m_t, covs, C_t, weights, w_t)
    
    # Initialize trajectory storage
    means_list = [means.detach().cpu().numpy()]
    covs_list = [covs.detach().cpu().numpy()]
    weights_list = [torch.softmax(logits, dim=0).detach().cpu().numpy()]
    loss_list = []
    
    params_flat = flatten_params(params)
    
    for _ in range(n_steps):
        # Compute loss and gradients
        loss = loss_fn(params_flat)
        grad_flat = torch.autograd.grad(loss, params_flat, create_graph=True)[0]
        
        if use_natural_gradient:
            # Compute Fisher Information Matrix as outer product of gradients
            F = grad_flat.unsqueeze(1) @ grad_flat.unsqueeze(0)
            
            # Add damping and solve for natural gradient
            F_damped = F + damping * torch.eye(F.shape[0], dtype=F.dtype)
            nat_grad_flat = torch.linalg.solve(F_damped, grad_flat)
            
            grad_flat = nat_grad_flat
        
        # Update parameters
        with torch.no_grad():
            params_flat = params_flat - lr * grad_flat
            params_flat.requires_grad_(True)
            params = unflatten_params(params_flat)
            means, covs, logits = params
            
        # Store current state
        means_list.append(means.detach().cpu().numpy())
        covs_list.append(proj_SDP(covs).clone().detach().cpu().numpy())
        weights_list.append(torch.softmax(logits, dim=0).detach().cpu().numpy())
        loss_list.append(loss.item())
    
    return np.array(means_list), np.array(covs_list), np.array(weights_list), np.array(loss_list) 