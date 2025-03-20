import seaborn as sns
import matplotlib.pyplot as plt
import torch

def plot_latent_pairs_overlaid(
        dist_ae, sample_fn, 
        sample_sizes=[50, 100, 500, 1000, 5000], n_sets=1000
):
    """Plot scatter plots of pairs of latent dimensions with different sample sizes overlaid.
    
    Args:
        dist_ae: Trained autoencoder model
        mu_test: Mean parameters for test data
        var_test: Variance parameters for test data
        sample_sizes: List of sample sizes to test
    """
    
    # Create figure with 2 subplots (one for each pair of dimensions)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Use different colors for different sample sizes
    colors = ['purple','blue', 'green', 'yellow', 'red']
    
    pairs = [(0,1), (2,3)]
    for n_samples, color in zip(sample_sizes, colors):
        # Generate test data
        x = sample_fn(n_sets, n_samples)
        x = torch.from_numpy(x).float()

        # Get latent representations
        with torch.no_grad():
            lat, _ = dist_ae(x)
        
        # Convert to numpy for plotting
        latent_data = lat.numpy()  # Shape: (n_samples, latent_dim)
        
        # Plot pairs of dimensions
        for col, (dim1, dim2) in enumerate(pairs):
            # Create scatter plot
            sns.scatterplot(
                x=latent_data[:, dim1],
                y=latent_data[:, dim2],
                alpha=0.5,
                ax=axes[col],
                label=f'n={n_samples}',
                color=color
            )
            
            # Add labels
            axes[col].set_xlabel(f'Dimension {dim1+1}')
            axes[col].set_ylabel(f'Dimension {dim2+1}')
            axes[col].set_title(f'Dims {dim1+1} vs {dim2+1}')
            axes[col].legend()
    
    plt.tight_layout()
    plt.show()
