import torch
import warnings
from torch.autograd import Variable

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

def sink_stab_batched(m, reg, maxiter=1000, tau=1e2, eps=1e-2, warm=None, pstep=20, device='cuda'):
    """
    stabilized sinkhorn
    m: (b, n, m) tensor of cost matrices
    Returns: (b) tensor of sinkhorn distances
    """
    b, na, nb = m.shape
    
    # init marginals
    a = Variable(torch.ones(b, na, device=device) / na)
    b_marg = Variable(torch.ones(b, nb, device=device) / nb)
    
    # init dual
    if warm is None:
        alpha = Variable(torch.zeros(b, na, device=device))
        beta = Variable(torch.zeros(b, nb, device=device))
    else:
        alpha, beta = warm
    
    u = Variable(torch.ones(b, na, device=device) / na)
    v = Variable(torch.ones(b, nb, device=device) / nb)
    
    def get_k(alpha, beta, M):
        return torch.exp(-(M - alpha.unsqueeze(2) - beta.unsqueeze(1)) / reg)
    
    def get_gamma(alpha, beta, u, v):
        return torch.exp(-(m - alpha.unsqueeze(2) - beta.unsqueeze(1)) / reg + 
                        torch.log(u.unsqueeze(2)) + torch.log(v.unsqueeze(1)))
    
    k = get_k(alpha, beta, m)
    transp = k
    loop = True
    cpt = 0
    err = torch.ones(b, device=device)
    
    while loop:
        uprev, vprev = u, v
        
        # batched sinkhorn step
        v = b_marg / (torch.bmm(k.transpose(1, 2), u.unsqueeze(2)).squeeze(2) + 1e-16)
        u = a / (torch.bmm(k, v.unsqueeze(2)).squeeze(2) + 1e-16)
        
        # rescale if too big
        mask = (torch.max(torch.abs(u), dim=1)[0] > tau) | (torch.max(torch.abs(v), dim=1)[0] > tau)
        if mask.any():
            alpha = alpha.clone()
            beta = beta.clone()
            u = u.clone()
            v = v.clone()
            k = k.clone()

            alpha[mask] = alpha[mask] + reg * torch.log(u[mask])
            beta[mask] = beta[mask] + reg * torch.log(v[mask])
            u[mask] = torch.ones_like(u[mask]) / na
            v[mask] = torch.ones_like(v[mask]) / nb
            k[mask] = get_k(alpha[mask], beta[mask], m[mask])

        
        if cpt % pstep == 0:
            transp = get_gamma(alpha, beta, u, v)
            err = (torch.sum(transp, dim=(1, 2)) - 1).abs().pow(2)  # Assuming marginals sum to 1
        
        if (err.max() <= eps) or (cpt >= maxiter):
            loop = False
        
        cpt += 1
    
    return torch.sum(get_gamma(alpha, beta, u, v) * m, dim=(1, 2))

def sink_D_batched(X, Y, reg=1, maxiter=20, tau=1e2, eps=1e-2, warm=None, pstep=20, device='cuda', p=2):
    XX = torch.cdist(X, X, p=p) 
    YY = torch.cdist(Y, Y, p=p)  
    XY = torch.cdist(X, Y, p=p)  
    
    sink_XX = sink_stab_batched(XX, reg, maxiter, tau, eps, warm, pstep, device)
    sink_YY = sink_stab_batched(YY, reg, maxiter, tau, eps, warm, pstep, device)
    sink_XY = sink_stab_batched(XY, reg, maxiter, tau, eps, warm, pstep, device)
    
    return sink_XY - 0.5 * (sink_XX + sink_YY)

def pairwise_sinkhorn(X, reg=1., max_iter=100, eps=1e-2, p=2, device='cuda'):
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
    all_pairs = sink_D_batched(X1_flat, X2_flat, reg=reg, maxiter=max_iter, eps=eps, p=p, device=device)
    
    # reshape back
    loss_matrix = all_pairs.reshape(b, b)
    
    return loss_matrix