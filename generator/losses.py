import torch
import warnings

def sliced_wasserstein_distance(X1, X2, n_projections=100, p=2):
    """
    Computes the Sliced Wasserstein Distance (SWD) between two batches using POT.

    Args:
        X1: Tensor of shape (N, d) - First batch of points.
        X2: Tensor of shape (M, d) - Second batch of points.
        n_projections: Number of random projections (default: 100).
        p: Power of distance metric (default: 2).

    Returns:
        SWD (scalar tensor).
    """
    device = X1.device
    d = X1.shape[1]  # Feature dimension

    # Generate random projection vectors
    projections = torch.randn((n_projections, d), device=device)
    projections = projections / torch.norm(projections, dim=1, keepdim=True)  # Normalize

    # Project both distributions onto 1D subspaces
    X1_proj = X1 @ projections.T  # Shape: (N, n_projections)
    X2_proj = X2 @ projections.T  # Shape: (M, n_projections)

    # Sort projections along each 1D slice
    X1_proj_sorted, _ = torch.sort(X1_proj, dim=0)
    X2_proj_sorted, _ = torch.sort(X2_proj, dim=0)

    # Compute 1D Wasserstein distance per projection (L_p norm)
    SW_dist = torch.mean(torch.abs(X1_proj_sorted - X2_proj_sorted) ** p) ** (1/p)

    return SW_dist

def compute_gamma(X: torch.Tensor, Y: torch.Tensor, p = 2) -> float:
    """
    Compute gamma using the median heuristic.
    Args:
        X: (n, d) tensor - samples from distribution P
        Y: (m, d) tensor - samples from distribution Q
    Returns:
        gamma: RBF kernel bandwidth parameter
    """
    # Combine samples
    Z = torch.cat([X, Y], dim=0)
    
    # Compute pairwise squared distances
    D = torch.cdist(Z, Z, p=p).pow(p)  # (n+m, n+m)
    
    # Extract upper triangular (excluding diagonal)
    upper_tri = torch.triu(D, diagonal=1)
    
    # Compute sqrt distances directly 
    distances_sqrt = torch.sqrt(upper_tri)
    
    # This is a hack to get the median of the distances
    # because we cannot use binary indexing since we 
    # are using torch.vmap to wrap the function
    flattened_distances = distances_sqrt.flatten()
    flattened_distances = (flattened_distances / flattened_distances) * flattened_distances
    sigma = torch.nanmedian(flattened_distances, dim=0)[0]
    
    # Avoid division by zero (though we handled the all-zeros case already)
    sigma = torch.maximum(sigma, torch.tensor(1e-8, device=sigma.device))
    
    gamma = 1.0 / (2 * sigma ** 2)
    return gamma

def mmd(X, Y, gamma=None, p: int = 2) -> torch.Tensor:
    """
    Biased MMD² estimator with RBF kernel (includes diagonal terms)
    Compatible with PyTorch gradients

    Args:
        X: (n, d) tensor - samples from distribution P
        Y: (m, d) tensor - samples from distribution Q
        gamma: RBF kernel bandwidth parameter (1/(2σ²))

    Returns:
        Scalar tensor containing MMD² (biased)
    """
    if gamma is None:
        gamma = compute_gamma(X, Y, p)
    
    # Compute pairwise squared distances
    XX = torch.cdist(X, X, p=p).pow(p)  # (n, n)
    YY = torch.cdist(Y, Y, p=p).pow(p)  # (m, m)
    XY = torch.cdist(X, Y, p=p).pow(p)  # (n, m)

    # Compute RBF kernels
    K_XX = torch.exp(-gamma * XX)
    K_YY = torch.exp(-gamma * YY)
    K_XY = torch.exp(-gamma * XY)

    # Compute biased MMD² (includes diagonal terms)
    mmd_squared = K_XX.mean() + K_YY.mean() - 2 * K_XY.mean()
    
    return mmd_squared

def sinkhorn(
    X,
    Y,
    reg=1.,
    max_iter=50,
    eps=1e-16,
    p=2
):
    device = X.device
    dtype = X.dtype
    
    n = X.shape[0]
    m = Y.shape[0]
    a = torch.ones(n, 1, device=device, dtype=dtype) / n  # Proper uniform distribution
    b = torch.ones(m, 1, device=device, dtype=dtype) / m   # Proper uniform distribution

    # Compute pairwise cost matrix
    M = torch.cdist(X, Y, p=p)**p
    K = torch.exp(-M / (reg + eps))  # (n, m)

    # Initialize dual variables
    u = torch.ones_like(a)
    v = torch.ones_like(b)
    
    for _ in range(max_iter):
        # Update u and v
        u = a / (K @ v + eps)
        v = b / (K.T @ u + eps)
    
    # Compute transport plan
    P = u * K * v.T  # diag(u) @ K @ diag(v)
    
    # Compute loss
    loss = torch.sum(P * M)
    
    return loss

def sinkhorn_loss(X, Y, reg=1., max_iter=100, eps=1e-16, p=2):
    """
    Compute Sinkhorn divergence between two sets of samples.
    This is a proper distance metric (positive definite).
    """
    XY = sinkhorn(X, Y, reg, max_iter, eps, p)
    XX = sinkhorn(X, X, reg, max_iter, eps, p)
    YY = sinkhorn(Y, Y, reg, max_iter, eps, p)
    return XY - 0.5 * (XX + YY)  # Note the 0.5 factor instead of 2

def pairwise_sinkhorn(X, reg=1., max_iter=100, eps=1e-1, p=2):
    """
    X: (b, n, d) tensor
    returns: (b, b) sinkhorn loss matrix
    """
    b = X.shape[0]
    
    # expand X to compare all pairs
    X1 = X.unsqueeze(1).expand(-1, b, -1, -1)  # (b, b, n, d)
    X2 = X.unsqueeze(0).expand(b, -1, -1, -1)  # (b, b, n, d)
    
    # flatten for vmap
    X1_flat = X1.reshape(-1, *X.shape[1:])
    X2_flat = X2.reshape(-1, *X.shape[1:])
    
    # compute all pairs
    all_pairs = torch.vmap(lambda x, y: sinkhorn_loss(x, y, reg, max_iter, eps, p))(X1_flat, X2_flat)
    
    # reshape back
    loss_matrix = all_pairs.reshape(b, b)
    
    return loss_matrix