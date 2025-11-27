import numpy as np
from si.base.estimator import Estimator
from si.data.dataset import Dataset
from si.metrics.mse import mse

class RidgeRegressionLeastSquares(Estimator):
    """
    Ridge Regression using Least Squares with L2 regularization.

    Parameters
    ----------
    l2_penalty : float, default=1.0
        L2 regularization parameter.

    scale : bool, default=True
        Whether to scale the data.
    """

    def __init__(self, l2_penalty: float = 1.0, scale: bool = True):
        """
        Initializes the Ridge Regression Least Squares estimator.

        Parameters
        ----------
        l2_penalty : float, default=1.0
            L2 regularization parameter (lambda). Controls the strength of the penalty.
        
        scale : bool, default=True
            Whether to standardize the input features (X) by removing the mean and scaling to unit variance
            during fitting and prediction.
        """
        super().__init__()
        self.l2_penalty = l2_penalty
        self.scale = scale
        self.theta = None
        self.theta_zero = None
        self.mean = None
        self.std = None

    def _fit(self, dataset: Dataset) -> 'RidgeRegressionLeastSquares':
        """
        Fit the Ridge Regression model to the dataset.

        Parameters
        ----------
        dataset : Dataset
            The dataset to fit the model to.

        Returns
        -------
        self : RidgeRegressionLeastSquares
            The fitted model.
        """
        X = dataset.X
        y = dataset.y

        if self.scale:
            self.mean = np.mean(X, axis=0)
            self.std = np.std(X, axis=0)
            X = (X - self.mean) / self.std
        else:
            self.mean = np.zeros(X.shape[1])
            self.std = np.ones(X.shape[1])

        X = np.c_[np.ones(X.shape[0]), X]

        penalty_matrix = self.l2_penalty * np.eye(X.shape[1])
        penalty_matrix[0, 0] = 0  

        XTX = X.T.dot(X)
        XTX_plus_penalty = XTX + penalty_matrix
        XTX_inv = np.linalg.inv(XTX_plus_penalty)
        XTy = X.T.dot(y)
        thetas = XTX_inv.dot(XTy)

        self.theta_zero = thetas[0]
        self.theta = thetas[1:]

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the dependent variable (y) using the estimated coefficients.

        Parameters
        ----------
        X : np.ndarray
            The input samples.

        Returns
        -------
        y_pred : np.ndarray
            The predicted target values.
        """
        if self.scale:
            X = (X - self.mean) / self.std

        X = np.c_[np.ones(X.shape[0]), X]

        thetas = np.r_[self.theta_zero, self.theta]
        y_pred = X.dot(thetas)

        return y_pred

    def score(self, dataset: Dataset) -> float:
        """
        Calculate the MSE score between the real and predicted y values.

        Parameters
        ----------
        dataset : Dataset
            The dataset to score the model on.

        Returns
        -------
        mse_score : float
            The mean squared error of the model.
        """
        y_pred = self.predict(dataset.X)
        return mse(dataset.y, y_pred)