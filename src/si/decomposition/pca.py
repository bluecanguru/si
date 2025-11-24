import numpy as np
from si.base.transformer import Transformer
from si.data.dataset import Dataset

class PCA(Transformer):
    """
    Principal Component Analysis (PCA) using eigenvalue decomposition of the covariance matrix.

    Parameters
    ----------
    n_components : int
        Number of principal components to keep.
    """

    def __init__(self, n_components=2):
        super().__init__()
        self.n_components = n_components
        self.mean = None
        self.components = None
        self.explained_variance = None

    def _fit(self, dataset: Dataset) -> 'PCA':
        """
        Fit the PCA model to the dataset.

        Parameters
        ----------
        dataset : Dataset
            The dataset to fit the transformer to.

        Returns
        -------
        self : PCA
            The fitted transformer.
        """
        X = dataset.X

        # 1. Center the data
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # 2. Calculate the covariance matrix and perform eigenvalue decomposition
        cov_matrix = np.cov(X_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        # 3. Infer the principal components (top n_components eigenvectors)
        # Sort eigenvalues and eigenvectors in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        self.components = eigenvectors[:, :self.n_components].T
        self.explained_variance = eigenvalues[:self.n_components] / np.sum(eigenvalues)

        return self

    def _transform(self, dataset: Dataset) -> Dataset:
        """
        Transform the dataset using the principal components.

        Parameters
        ----------
        dataset : Dataset
            The dataset to transform.

        Returns
        -------
        transformed_dataset : Dataset
            The transformed dataset with reduced dimensions.
        """
        if self.mean is None or self.components is None:
            raise RuntimeError("The transformer has not been fitted yet.")

        # 1. Center the data
        X_centered = dataset.X - self.mean

        # 2. Calculate the reduced X
        X_reduced = np.dot(X_centered, self.components.T)

        # Create a new Dataset object
        features = [f"PC_{i+1}" for i in range(self.n_components)]
        transformed_dataset = Dataset(X_reduced, y=dataset.y, features=features, label=dataset.label)

        return transformed_dataset