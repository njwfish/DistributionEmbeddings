import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from generator.losses import sliced_wasserstein_distance, mmd, sink_D_batched


def compute_distribution_metrics(distribution_pairs, metrics_to_compute=None, batch_size=10, sample_batch_size=10_000):
    """
    Compute distribution metrics (MMD, SW, Sinkhorn) between pairs of distributions.
    
    Args:
        distribution_pairs: List of tuples (X, Y) where X and Y are distributions to compare
        metrics_to_compute: List of metrics to compute (default: ['mmd', 'sliced_wasserstein', 'sinkhorn'])
        batch_size: Number of pairs to process in a batch
        sample_batch_size: Maximum number of samples to use per distribution
    
    Returns:
        Dictionary with computed metrics for each pair
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if metrics_to_compute is None:
        metrics_to_compute = ['mmd', 'sliced_wasserstein', 'sinkhorn']
        
    # Set up metric functions
    metric_functions = {
        'sinkhorn': lambda x, y: compute_sinkhorn_distance(x, y, sample_batch_size),
        'mmd': lambda x, y: compute_mmd_distance(x, y, sample_batch_size),
        'sliced_wasserstein': lambda x, y: compute_sw_distance(x, y, sample_batch_size, n_projections=100)
    }
    
    # Initialize results
    results = {metric: [] for metric in metrics_to_compute}
    
    # Process in batches
    n_pairs = len(distribution_pairs)
    for i in tqdm(range(0, n_pairs, batch_size), desc="Computing distribution metrics"):
        batch_pairs = distribution_pairs[i:i+batch_size]
        
        # Prepare batched tensors
        X_batch = torch.stack([
            torch.tensor(pair[0][:sample_batch_size], dtype=torch.float32) 
            for pair in batch_pairs
        ]).to(device)
        
        Y_batch = torch.stack([
            torch.tensor(pair[1][:sample_batch_size], dtype=torch.float32) 
            for pair in batch_pairs
        ]).to(device)
        
        # Compute each metric
        for metric in metrics_to_compute:
            if metric in metric_functions:
                with torch.no_grad():
                    metric_values = metric_functions[metric](X_batch, Y_batch).cpu().numpy()
                    results[metric].extend(metric_values.tolist())
        
        # Clear CUDA cache
        torch.cuda.empty_cache()
    
    return results

def compute_sinkhorn_distance(original_samples, resamples, batch_size=10_000, reg=1.0, maxiter=100):
    """
    Compute the Sinkhorn distance between original samples and resamples.
    
    Args:
        original_samples: Original input samples (torch tensor)
        resamples: Generated samples from latents (torch tensor) 
        batch_size: Number of samples to use for calculation
        reg: Regularization parameter
        maxiter: Maximum number of iterations
        
    Returns:
        Tensor of Sinkhorn distances [batch_size]
    """
    with torch.no_grad():
        # Make sure inputs have the right shape (add batch dimension if needed)
        if original_samples.dim() == 2:
            original_samples = original_samples.unsqueeze(0)
        if resamples.dim() == 2:
            resamples = resamples.unsqueeze(0)
            
        # Limit sample counts for memory efficiency
        original_samples = original_samples[:, :batch_size, :]
        resamples = resamples[:, :batch_size, :]
        
        sink = sink_D_batched(
            original_samples,
            resamples,
            reg=reg,
            maxiter=maxiter
        )
        
        return sink

def compute_mmd_distance(original_samples, resamples, batch_size=10_000, gamma=None, p=2):
    """
    Compute the Maximum Mean Discrepancy (MMD) between original samples and resamples.
    Supports batched inputs using torch.vmap.
    
    Args:
        original_samples: Original input samples (torch tensor) [batch_size, n_samples, dim]
        resamples: Generated samples from latents (torch tensor) [batch_size, n_samples, dim]
        batch_size: Number of samples to use for calculation
        gamma: RBF kernel bandwidth parameter (if None, computed using median heuristic)
        p: Power of distance metric
        
    Returns:
        Tensor of MMD distances [batch_size]
    """
    with torch.no_grad():
        # Ensure 3D tensors (batch_size, n_samples, dim)
        if original_samples.dim() == 2:
            original_samples = original_samples.unsqueeze(0)
        if resamples.dim() == 2:
            resamples = resamples.unsqueeze(0)
            
        # Limit sample counts for memory efficiency
        original_samples = original_samples[:, :batch_size, :]
        resamples = resamples[:, :batch_size, :]

        # Use torch.vmap to compute MMD for each pair in the batch
        vmapped_mmd = torch.vmap(mmd, randomness='different')
        
        # Reshape for vmap: from [batch, n_samples, dim] to list of [n_samples, dim]
        batch_size = original_samples.size(0)
        results = []
        
        # For larger batches, process in chunks to avoid OOM
        chunk_size = 32  # Adjust based on available memory
        for i in range(0, batch_size, chunk_size):
            chunk_orig = original_samples[i:i+chunk_size]
            chunk_resample = resamples[i:i+chunk_size]
            
            # Apply vmapped MMD
            chunk_results = vmapped_mmd(chunk_orig, chunk_resample, gamma=gamma, p=p)
            results.append(chunk_results)
        
        # Combine results
        if len(results) > 1:
            return torch.cat(results)
        else:
            return results[0]

def compute_sw_distance(original_samples, resamples, batch_size=10_000, n_projections=100, p=2):
    """
    Compute the Sliced Wasserstein distance between original samples and resamples.
    Supports batched inputs using torch.vmap.
    
    Args:
        original_samples: Original input samples (torch tensor) [batch_size, n_samples, dim]
        resamples: Generated samples from latents (torch tensor) [batch_size, n_samples, dim]
        batch_size: Number of samples to use for calculation
        n_projections: Number of random projections
        p: Power of distance metric
        
    Returns:
        Tensor of SW distances [batch_size]
    """
    with torch.no_grad():
        # Ensure 3D tensors (batch_size, n_samples, dim)
        if original_samples.dim() == 2:
            original_samples = original_samples.unsqueeze(0)
        if resamples.dim() == 2:
            resamples = resamples.unsqueeze(0)
            
        # Limit sample counts for memory efficiency
        original_samples = original_samples[:, :batch_size, :]
        resamples = resamples[:, :batch_size, :]

        # Use torch.vmap to compute SW for each pair in the batch
        vmapped_sw = torch.vmap(sliced_wasserstein_distance, randomness='different')
        
        # Reshape for vmap: from [batch, n_samples, dim] to list of [n_samples, dim]
        batch_size = original_samples.size(0)
        results = []
        
        # For larger batches, process in chunks to avoid OOM
        chunk_size = 32  # Adjust based on available memory
        for i in range(0, batch_size, chunk_size):
            chunk_orig = original_samples[i:i+chunk_size]
            chunk_resample = resamples[i:i+chunk_size]
            
            # Apply vmapped SW
            chunk_results = vmapped_sw(chunk_orig, chunk_resample, n_projections=n_projections, p=p)
            results.append(chunk_results)
        
        # Combine results
        if len(results) > 1:
            return torch.cat(results)
        else:
            return results[0]

def compute_latent_reconstruction_errors(results):
    """
    Compute latent reconstruction errors for a list of results.
    
    Args:
        results: List of dictionaries with 'latent' and 'relatent' keys
        
    Returns:
        List of reconstruction errors for each result
    """
    errors = []
    latents = np.stack([result['latent'] for result in results])
    relatents = np.stack([result['relatent'] for result in results])
    errors = np.mean(np.linalg.norm(latents - relatents, axis=-1), axis=-1)
    return errors

def batch_compute_metrics(results, metrics_to_compute=None, sample_batch_size=10_000, eval_batch_size=10):
    """
    Compute evaluation metrics for distribution embeddings using batched computation.
    
    Args:
        results: List of dictionaries with keys:
            - 'original_samples': Original input samples
            - 'latent': Encoded latent vectors
            - 'resample': Generated samples from latents
            - 'relatent': Re-encoded latent vectors from resamples
        metrics_to_compute: List of metric names to compute (default: all)
        sample_batch_size: Maximum number of samples to use per distribution
        eval_batch_size: Number of sets to evaluate in a single batch
    
    Returns:
        Dictionary with computed metrics
    """
    if metrics_to_compute is None:
        metrics_to_compute = ['latent_recon_error', 'sinkhorn', 'mmd', 'sliced_wasserstein']
    
    metrics = {}
    
    # Compute latent reconstruction errors if needed
    if 'latent_recon_error' in metrics_to_compute:
        errors = compute_latent_reconstruction_errors(results)
        metrics['latent_recon_error'] = {
            'mean': np.mean(errors),
            'std': np.std(errors),
            'per_set': np.array(errors)
        }
    
    # Prepare distribution pairs for distribution metrics
    distribution_metrics = set(metrics_to_compute) - {'latent_recon_error'}
    if distribution_metrics:
        distribution_pairs = [(result['original_samples'], result['resample']) for result in results]
        metric_results = compute_distribution_metrics(
            distribution_pairs, 
            metrics_to_compute=distribution_metrics,
            batch_size=eval_batch_size,
            sample_batch_size=sample_batch_size
        )
        
        # Add distribution metrics to results
        for metric_name, values in metric_results.items():
            metrics[metric_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'per_set': np.array(values)
            }
    
    return metrics

def compute_metrics(results, metrics_to_compute=None, batch_size=10_000):
    """
    Compute evaluation metrics for distribution embeddings.
    This is a wrapper around batch_compute_metrics with legacy behavior.
    
    Args:
        results: List of dictionaries with keys:
            - 'original_samples': Original input samples
            - 'latent': Encoded latent vectors
            - 'resample': Generated samples from latents
            - 'relatent': Re-encoded latent vectors from resamples
        metrics_to_compute: List of metric names to compute (default: all)
        batch_size: Batch size for computing distribution metrics
    
    Returns:
        Dictionary with computed metrics
    """
    return batch_compute_metrics(
        results,
        metrics_to_compute=metrics_to_compute,
        sample_batch_size=batch_size,
        eval_batch_size=10  # Default to processing 10 sets at a time
    )

def batch_encode_samples(enc, sample_sets, device, max_encode_samples=50_000, batch_size=10):
    """
    Encode multiple sets of samples in batches.
    This function can be used for both initial encoding of input samples and re-encoding of generated samples.
    
    Args:
        enc: Encoder model
        sample_sets: List of sample sets (numpy arrays)
        device: Device to run model on
        max_encode_samples: Maximum samples to encode per set
        batch_size: Number of sets to encode in a batch (0 = all sets at once)
    
    Returns:
        List of encoded latents (numpy arrays)
    """
    # Process all sets at once if batch_size is 0
    if batch_size <= 0:
        batch_size = len(sample_sets)
    
    all_latents = []
    
    # Process in batches
    for i in tqdm(range(0, len(sample_sets), batch_size), desc="Encoding samples"):
        # Get batch of sets
        batch_samples = sample_sets[i:i+batch_size]
        
        # Create tensor for batch encoding
        # Shape: [batch_size, max_encode_samples, feature_dim]
        batch_tensor = torch.stack([
            torch.tensor(s[:max_encode_samples], dtype=torch.float32) 
            for s in batch_samples
        ]).to(device)
        
        # Encode batch
        with torch.no_grad():
            batch_latents = enc(batch_tensor).float()
            batch_latents_cpu = batch_latents.cpu().detach().numpy()
        
        # Store latents
        all_latents.extend([lat for lat in batch_latents_cpu])
        
        # Clear CUDA cache
        torch.cuda.empty_cache()
    
    return all_latents

def batch_generate_samples(gen, latents, device, num_resamples=100_000, batch_size=10):
    """
    Generate samples from multiple latent vectors in batches.
    
    Args:
        gen: Generator model
        latents: List of latent vectors (numpy arrays)
        device: Device to run model on
        num_resamples: Number of samples to generate per latent
        batch_size: Number of latents to process in a batch (0 = all at once)
    
    Returns:
        List of generated samples (numpy arrays)
    """
    # Process all latents at once if batch_size is 0
    if batch_size <= 0:
        batch_size = len(latents)
    
    all_samples = []
    
    # Process in batches
    for i in tqdm(range(0, len(latents), batch_size), desc="Generating samples"):
        # Get batch of latents
        batch_latents = latents[i:i+batch_size]
        
        # Create tensor for batch generation
        batch_tensor = torch.tensor(batch_latents, dtype=torch.float32).to(device)
        
        # Generate samples
        with torch.no_grad():
            batch_samples = gen.sample(batch_tensor, num_resamples)
            batch_samples_cpu = batch_samples.cpu().detach().numpy()
        
        # Store samples
        all_samples.extend([samples for samples in batch_samples_cpu])
        
        # Clear CUDA cache
        torch.cuda.empty_cache()
    
    return all_samples

def compute_encodings_and_resamples(
    enc,
    gen,
    sample_sets,
    device,
    max_encode_samples=50_000,
    num_resamples=100_000,
    encode_batch_size=0,
    resample_batch_size=10,
    reencode_batch_size=10
):
    """
    Process multiple distributions with efficient batching of sets.
    The process follows three sequential steps:
    1. Encode original samples to latent vectors
    2. Generate new samples from latent vectors
    3. Re-encode generated samples back to latent space
    
    Args:
        enc: Encoder model
        gen: Generator model
        sample_sets: List of sample sets
        device: Device to run models on
        max_encode_samples: Maximum samples to encode per set
        num_resamples: Number of samples to generate per set
        encode_batch_size: Number of sets to encode in a batch (0 = all sets at once)
        resample_batch_size: Number of sets to resample in a batch (0 = all sets at once)
        reencode_batch_size: Number of sets to re-encode in a batch (0 = all sets at once)
        
    Returns:
        List of dictionaries with results for each set
    """
    # Prepare sample data
    processed_samples = []
    for samples in sample_sets:
        if isinstance(samples, dict) and 'samples' in samples:
            samples = samples['samples'].squeeze()
        processed_samples.append(samples)
    
    # Step 1: Encode all sample sets in batches
    print("Step 1/3: Encoding original samples")
    latents = batch_encode_samples(
        enc, 
        processed_samples, 
        device, 
        max_encode_samples, 
        encode_batch_size
    )
    
    # Step 2: Generate samples from all latents in batches
    print("Step 2/3: Generating samples from latents")
    resamples = batch_generate_samples(
        gen, 
        latents, 
        device, 
        num_resamples, 
        resample_batch_size
    )
    
    # Step 3: Re-encode generated samples in batches
    print("Step 3/3: Re-encoding generated samples")
    relatents = batch_encode_samples(
        enc, 
        resamples, 
        device, 
        max_encode_samples, 
        reencode_batch_size
    )
    
    # Combine results
    results = []
    for i in range(len(processed_samples)):
        results.append({
            'original_samples': processed_samples[i],
            'latent': latents[i],
            'resample': resamples[i],
            'relatent': relatents[i]
        })
    
    return results

def compute_metrics_single(result, metrics_to_compute=None, batch_size=10_000):
    """
    Compute metrics for a single sample set.
    
    Args:
        result: Dictionary with keys:
            - 'original_samples': Original input samples
            - 'latent': Encoded latent vectors
            - 'resample': Generated samples from latents
            - 'relatent': Re-encoded latent vectors from resamples
        metrics_to_compute: List of metric names to compute (default: all)
        batch_size: Batch size for computing distribution metrics
    
    Returns:
        Dictionary with computed metrics for the single set
    """
    return compute_metrics([result], metrics_to_compute, batch_size)