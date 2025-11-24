import numpy as np
from si.base.estimator import Estimator
from si.data.dataset import Dataset
from si.statistics.euclidean_distance import euclidean_distance

class KNNRegressor(Estimator):
    """
    K-Nearest Neighbors Regressor.

    Parameters
    ----------
    k : int, default=5
        Number of neighbors to consider.
    """

    def __init__(self, k: int = 5):
        super().__init__()
        self.k = k
        self.X_train = None
        self.y_train = None

    def _fit(self, dataset: Dataset) -> 'KNNRegressor':
        """
        Fit the KNNRegressor to the training data.

        Parameters
        ----------
        dataset : Dataset
            The training dataset.

        Returns
        -------
        self : KNNRegressor
            The fitted regressor.
        """
        self.X_train = dataset.X
        self.y_train = dataset.y
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the target values for the input samples.

        Parameters
        ----------
        X : np.ndarray
            The input samples.

        Returns
        -------
        y_pred : np.ndarray
            The predicted target values.
        """
        if self.X_train is None or self.y_train is None:
            raise RuntimeError("The model has not been fitted yet.")

        X = np.asarray(X)
        y_pred = np.zeros(X.shape[0])
        for i, x in enumerate(X):
            # vectorized distance computation to all training samples
            distances = np.linalg.norm(self.X_train - x, axis=1)

            # Get indices of the k nearest neighbors
            nearest_indices = np.argsort(distances)[:self.k]

            # Predict the target as the mean of the k nearest neighbors
            y_pred[i] = np.mean(self.y_train[nearest_indices])

        return y_pred