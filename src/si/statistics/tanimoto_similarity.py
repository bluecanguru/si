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
    x = np.asarray(x, dtype=bool).reshape(-1)
    y = np.asarray(y, dtype=bool)

    # intersection = count of features where both are 1
    intersection = np.logical_and(y, x).sum(axis=1)
    
    # union = count of features where either is 1
    union = np.logical_or(y, x).sum(axis=1)
    
    similarities = intersection / union

    # avoid division by zero
    similarities[np.isnan(similarities)] = 0.0    
    
    return similarities