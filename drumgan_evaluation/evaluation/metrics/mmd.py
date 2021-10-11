import numpy as np
from sklearn.metrics.pairwise import manhattan_distances, cosine_distances, euclidean_distances

def mmd(x, y, distance='manhattan'):
    """
    Args:
        x, y: matrix of embeddings (n_samples * embedding_size)
        distance: distance metric used to compute mmd
    """
    assert distance in ['manhattan', 'euclidean', 'cosine']
    assert x.shape == y.shape

    n_samples = x.shape[0]

    if distance == 'manhattan':
        xy = manhattan_distances(x, y, sum_over_features=True)
        xx = manhattan_distances(x, sum_over_features=True)
        yy = manhattan_distances(y, sum_over_features=True)
    elif distance == 'euclidean':
        xy = euclidean_distances(x, y, squared=False)
        xx = euclidean_distances(x, squared=False)
        yy = euclidean_distances(y, squared=False)
    elif distance == 'cosine':
        xy = cosine_distances(x, y)
        xx = cosine_distances(x)
        yy = cosine_distances(y)

    mmd_ = (1/n_samples**2) * (2*np.sum(xy) - np.sum(xx) - np.sum(yy))
    return mmd_