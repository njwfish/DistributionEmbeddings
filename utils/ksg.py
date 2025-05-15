from scipy.special import digamma
from sklearn.neighbors import BallTree, KDTree
import numpy as np
import numpy.linalg as la

############# adapted from NPEET ############
### https://github.com/gregversteeg/NPEET ###
#############################################

def add_noise(x, intens=1e-10):
    # small noise to break degeneracy
    return x + intens * np.random.random_sample(x.shape)


def query_neighbors(tree, x, k):
    return tree.query(x, k=k + 1)[0][:, k]


def count_neighbors(tree, x, r):
    return tree.query_radius(x, r, count_only=True)

def avgdigamma(points, dvec):
    """
    tweaked this to return list of psi(nx) for pointwise estimate
    """
    tree = build_tree(points)
    dvec = dvec - 1e-15
    num_points = count_neighbors(tree, points, dvec)
    return digamma(num_points)


def build_tree(points):
    if points.shape[1] >= 20:
        return BallTree(points, metric="chebyshev")
    return KDTree(points, metric="chebyshev")

def lnc_correction(tree, points, k, alpha):
    """
    in my experience this doesn't work in practice for N_dims > 1
    """
    e = 0
    n_sample = points.shape[0]
    for point in points:
        # Find k-nearest neighbors in joint space, p=inf means max norm
        knn = tree.query(point[None, :], k=k + 1, return_distance=False)[0]
        knn_points = points[knn]
        # Substract mean of k-nearest neighbor points
        knn_points = knn_points - knn_points[0]
        # Calculate covariance matrix of k-nearest neighbor points, obtain eigen vectors
        covr = knn_points.T @ knn_points / k
        _, v = la.eig(covr)
        # Calculate PCA-bounding box using eigen vectors
        V_rect = np.log(np.abs(knn_points @ v).max(axis=0)).sum()
        # Calculate the volume of original box
        log_knn_dist = np.log(np.abs(knn_points).max(axis=0)).sum()

        # Perform local non-uniformity checking and update correction term
        if V_rect < log_knn_dist + np.log(alpha):
            print(V_rect)
            print(log_knn_dist + np.log(alpha))
            print()
            e += (log_knn_dist - V_rect) / n_sample
    return e

def entropy(x, k=3, base=2):
    """The classic K-L k-nearest neighbor continuous entropy estimator
    x should be a list of vectors, e.g. x = [[1.3], [3.7], [5.1], [2.4]]
    if x is a one-dimensional scalar and we have four samples
    """
    assert k <= len(x) - 1, "Set k smaller than num. samples - 1"
    x = np.asarray(x)
    n_elements, n_features = x.shape
    x = add_noise(x)
    tree = build_tree(x)
    nn = query_neighbors(tree, x, k)
    const = digamma(n_elements) - digamma(k) + n_features * np.log(2)
    return (const + n_features * np.log(nn).mean()) / np.log(base)

def mi(x, y, z=None, k=3, base=2, alpha=0):
    """Mutual information of x and y (conditioned on z if z is not None)
    x, y should be a list of vectors, e.g. x = [[1.3], [3.7], [5.1], [2.4]]
    if x is a one-dimensional scalar and we have four samples
    """
    
    assert len(x) == len(y), "Arrays should have same length"
    assert k <= len(x) - 1, "Set k smaller than num. samples - 1"
    
    x, y = np.asarray(x), np.asarray(y)
    x, y = x.reshape(x.shape[0], -1), y.reshape(y.shape[0], -1)
    
    x = add_noise(x)
    y = add_noise(y)
    points = [x, y]
    
    if z is not None:
        z = np.asarray(z)
        z = z.reshape(z.shape[0], -1)
        points.append(z)
    points = np.hstack(points)
    # Find nearest neighbors in joint space, p=inf means max-norm
    tree = build_tree(points)
    dvec = query_neighbors(tree, points, k)
    if z is None:
        a, b, c, d = (
            avgdigamma(x, dvec),
            avgdigamma(y, dvec),
            digamma(k),
            digamma(len(x)),
        )
        if alpha > 0:
            d += lnc_correction(tree, points, k, alpha)
    else:
        xz = np.c_[x, z]
        yz = np.c_[y, z]
        a, b, c, d = (
            avgdigamma(xz, dvec),
            avgdigamma(yz, dvec),
            avgdigamma(z, dvec),
            digamma(k),
        )

#     mi_estimate = c + d - np.mean(a) - np.mean(b)
#     nxys = a + b
#     return mi_estimate, nxys
    return (c + d - a - b) / np.log(base)

# DISCRETE ESTIMATORS
def entropyd(sx, base=2):
    """Discrete entropy estimator
    sx is a list of samples
    """
    unique, count = np.unique(sx, return_counts=True, axis=0)
    # Convert to float as otherwise integer division results in all 0 for proba.
    proba = count.astype(float) / len(sx)
    # Avoid 0 division; remove probabilities == 0.0 (removing them does not change the entropy estimate as 0 * log(1/0) = 0.
    proba = proba[proba > 0.0]
    return np.sum(proba * np.log(1.0 / proba)) / np.log(base)

def centropyd(x, y, base=2):
    """The classic K-L k-nearest neighbor continuous entropy estimator for the
    entropy of X conditioned on Y.
    """
    xy = np.c_[x, y]
    return entropyd(xy, base) - entropyd(y, base)


def midd(x, y, base=2):
    """Discrete mutual information estimator
    Given a list of samples which can be any hashable object
    """
    assert len(x) == len(y), "Arrays should have same length"
    return entropyd(x, base) - centropyd(x, y, base)


def out_of_sample_avgdigamma(obs_points, est_points, dvec):
    """
    here we compute psi(nx) for query points that are not
    necessarily part of observed samples

    :param obs_points:
    :param est_points:
    :param dvec:
    """

    tree = build_tree(obs_points)
    dvec = dvec - 1e-15
    num_points = count_neighbors(tree, est_points, dvec)
    return digamma(num_points)

def out_of_sample_pmi(obs_x, obs_y, est_x, est_y, k=3, base=2):
    """
    
    
    """
    obs_x = add_noise(obs_x)
    obs_y = add_noise(obs_y)
    
    est_x = add_noise(est_x)
    est_y = add_noise(est_y)
    
    obs_points = np.hstack([obs_x, obs_y])
    est_points = np.hstack([est_x, est_y])
    tree = build_tree(obs_points)
    dvec = query_neighbors(tree, est_points, k)
    
    a = out_of_sample_avgdigamma(obs_x, est_x, dvec)
    b = out_of_sample_avgdigamma(obs_y, est_y, dvec)
    c = digamma(k)
    d = digamma(len(obs_x))
    
    return (c+d-a-b) / np.log(base)