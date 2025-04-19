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
    reg = 1.,
    max_iter=50,
    eps = 1e-16,
    p = 2
):
    """
    X: (n, d) tensor of source samples
    Y: (m, d) tensor of target samples
    reg: regularization parameter
    Returns: Sinkhorn loss between empirical distributions of X and Y
    """
    # Device and dtype setup
    device = X.device
    dtype = X.dtype
    
  
    # Create uniform distributions
    n = X.shape[0]
    m = Y.shape[0]
    a = torch.ones(n, 1, device=device, dtype=dtype) 
    b = torch.ones(m, 1, device=device, dtype=dtype) 

    # Compute pairwise cost matrix (squared Euclidean)

    M = torch.cdist(X, Y, p=p)**p
    #M = pairwise_cosine_distance(X,Y)
    #reg = 0.1 * torch.median(M)
    #reg = 0.1 * torch.median(M)

    # Compute kernel matrix with numerical stability
    K = torch.exp(-M / (reg + eps))  # (n, m)

    # Sinkhorn iterations
    for ii in range(max_iter):
        # uprev = u.clone()
        # vprev = v.clone()

        # Update v then u
        a = 1 / (torch.mm(K, b) + eps)  # (m, 1)
        b = 1 / (torch.mm(K.t(), a) + eps)  # (n, 1)

        # have to comment this because we want to
        # vmap which cannot be done through control flow
        # Check for numerical issues
        # if (torch.any(Ktu.abs() < stop_thresh) or 
        #     torch.any(torch.isnan(u)) or 
        #     torch.any(torch.isnan(v)) or 
        #     torch.any(torch.isinf(u)) or 
        #     torch.any(torch.isinf(v))
        # ):
        #    u = uprev
        #    v = vprev
        #    break

    # Compute transport plan and loss
    loss = torch.einsum('ij,i,j,ij->', M, a.flatten(), b.flatten(), K)
    
    return loss.squeeze()

def sinkhorn_loss(X, Y, reg=1., max_iter=50, eps=1e-16, p=2):
    """
    Compute Sinkhorn loss between two sets of samples.
    
    Args:
        X: (n, d) tensor of source samples
        Y: (m, d) tensor of target samples
        reg: regularization parameter
        max_iter: maximum number of Sinkhorn iterations
        eps: small constant for numerical stability
        p: power parameter for the distance metric
    """
    return 2 * sinkhorn(X, Y, reg, max_iter, eps, p) - sinkhorn(X, X, reg, max_iter, eps, p) - sinkhorn(Y, Y, reg, max_iter, eps, p)
