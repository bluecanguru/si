import numpy as np

def tanimoto_similarity(x, y):
    """
    Compute the Tanimoto similarity between a single binary sample and multiple binary samples.

    Parameters
    ----------
    x : array-like, shape (n_features,)
        A single binary sample.

    y : array-like, shape (n_samples, n_features)
        Multiple binary samples, where each row is a sample.

    Returns
    -------
    similarities : ndarray, shape (n_samples,)
        An array containing the Tanimoto similarities between x and each sample in y.
    """
    x = np.asarray(x, dtype=bool)
    y = np.asarray(y, dtype=bool)

    # Compute dot products (intersection)
    dot_products = np.dot(y, x)

    # Compute squared norms
    x_norm_sq = np.sum(x)
    y_norm_sq = np.sum(y, axis=1)

    # Compute Tanimoto similarity
    similarities = dot_products / (x_norm_sq + y_norm_sq - dot_products)

    return similarities