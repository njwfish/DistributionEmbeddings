import numpy as np

def sample_normal(mu, var, n_sets, set_size, n_features):
    if mu.ndim == 1 and var.ndim == 1:
        return np.random.randn(n_sets, set_size, n_features) * np.sqrt(var)[None, None, :] + mu[None, None, :]
    elif mu.ndim == 2 and var.ndim == 2:
        return np.random.randn(n_sets, set_size, n_features) * np.sqrt(var)[:, None, :] + mu[:, None, :]
    else:
        raise ValueError("mu and var must have the same number of dimensions")

def generate_normal_params(n_sets, n_features):
    mu = np.random.randn(n_sets, n_features)
    var = np.random.randn(n_sets, n_features)**2
    return mu.squeeze(), var.squeeze()

def fr_dist_normal(params):
    mu, var = params
    diff_mu = (mu[:, None, :] - mu[None, :, :])**2
    diff_var = (var[:, None, :] - var[None, :, :])**2
    sum_var = (var[:, None, :] + var[None, :, :])**2
    return np.linalg.norm(
        np.arctanh((diff_mu + 2 * diff_var)/(diff_mu + 2 * sum_var)), axis=2
    )

def sample_poisson(rate, n_sets, set_size, n_features):
    if rate.ndim == 1:
        return np.random.poisson(1, (n_sets, set_size, n_features)) * rate[None, None, :]
    elif rate.ndim == 2:
        return np.random.poisson(1, (n_sets, set_size, n_features)) * rate[:, None, :]
    else:
        raise ValueError("rate must have the same number of dimensions as n_features")
# generate poisson data
def generate_poisson_params(n_sets, n_features, rate_range = (0, 100)):
    rate = np.random.uniform(rate_range[0], rate_range[1], (n_sets, n_features))
    return (rate.squeeze(),)

def fr_dist_poisson(params):
    rate, = params
    return np.linalg.norm(np.sqrt(rate[:, None, :]) - np.sqrt(rate[None, :, :]), axis=2)

# generate multinomial data
def sample_multinomial(probs, n_per_multinomial, n_sets, set_size, n_features):
    if probs.ndim == 1:
        probs = np.tile(probs, (n_sets, 1))
    elif probs.ndim == 2:
        pass
    else:
        raise ValueError("probs must have the same number of dimensions as n_features")
    # Initialize array to store samples
    x = np.zeros((n_sets, set_size, n_features))
    for i in range(n_sets):
        x[i] = np.random.multinomial(n_per_multinomial, probs[i], size=set_size)
    return x

def generate_multinomial_params(n_sets, n_features, n_per_multinomial = 10):
    # Generate probability vectors for each set
    prob = np.random.rand(n_sets, n_features)
    prob = prob / np.sum(prob, axis=1, keepdims=True)
    return prob.squeeze(), n_per_multinomial

def fr_dist_multinomial(params):
    probs, n_per_multinomial = params
    return np.arccos(np.sqrt(probs[:, None, :] * probs[None, :, :]).sum(axis=2))

distns = {
    'normal': {
        'sample': sample_normal,
        'generate_params': generate_normal_params,
        'fr_dist': fr_dist_normal
    },
    'poisson': {
        'sample': sample_poisson,
        'generate_params': generate_poisson_params,
        'fr_dist': fr_dist_poisson
    },
    'multinomial': {
        'sample': sample_multinomial,
        'generate_params': generate_multinomial_params,
        'fr_dist': fr_dist_multinomial
    }
}