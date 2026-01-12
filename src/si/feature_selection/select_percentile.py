import numpy as np
from si.base.transformer import Transformer
from si.statistics.f_classification import f_classification
from si.data.dataset import Dataset

class SelectPercentile(Transformer):
    """
    Select features according to a percentile of the highest scores.

    Feature ranking is performed by computing the scores of each feature using a scoring function:
        - f_classification: ANOVA F-value between label/feature for classification tasks.
        - f_regression: F-value for regression tasks.

    Parameters
    ----------
    percentile : int or float, default=10
        If int, number of top features to select.
        If float, fraction of top features to select.
    score_func : callable, default=f_classification
        Function taking dataset and returning (scores, p_values).
    """

    def __init__(self, percentile=10, score_func=f_classification):
        """
        Initialize SelectPercentile.

        Parameters
        ----------
        percentile : int or float, default=10
            Number or fraction of top features to select.
        score_func : callable, default=f_classification
            Function taking dataset and returning (scores, p_values).
        """
        super().__init__()
        self.percentile = percentile
        self.score_func = score_func
        self.F_ = None
        self.p_ = None

    def _fit(self, dataset: Dataset) -> 'SelectPercentile':
        """
        Fit the SelectPercentile model using the dataset.

        Compute the scores and p-values for all features.

        Parameters
        ----------
        dataset : Dataset
            The dataset to fit the transformer to.

        Returns
        -------
        self : SelectPercentile
            The fitted transformer.
        """
        self.F_, self.p_ = self.score_func(dataset)
        return self

    def _transform(self, dataset: Dataset) -> Dataset:
        """
        Transform the dataset by selecting the highest scoring features
        according to the percentile value.

        Parameters
        ----------
        dataset : Dataset
            A labeled dataset.

        Returns
        -------
        dataset : Dataset
            Dataset with the selected features.
        """
        if self.F_ is None:
            raise RuntimeError("The transformer has not been fitted yet.")

        n_features = dataset.X.shape[1]

        if self.percentile == 0:
            X_transformed = np.empty((dataset.X.shape[0], 0))
            transformed_dataset = Dataset(X_transformed, dataset.y, features=[], label=dataset.label)
            return transformed_dataset

        if isinstance(self.percentile, float):
            n_selected = int(self.percentile * n_features)
        else:
            n_selected = self.percentile

        n_selected = min(n_selected, n_features)

        if n_selected <= 0:
            X_transformed = np.empty((dataset.X.shape[0], 0))
            transformed_dataset = Dataset(X_transformed, dataset.y, features=[], label=dataset.label)
            return transformed_dataset

        sorted_F = np.sort(self.F_)
        threshold = sorted_F[-n_selected]

        mask = self.F_ >= threshold

        if np.sum(mask) > n_selected:
            sorted_indices = np.argsort(self.F_)
            selected_indices = sorted_indices[-n_selected:]
        else:
            selected_indices = np.where(mask)[0]

        X_transformed = dataset.X[:, selected_indices]
        y_transformed = dataset.y

        features = [dataset.features[i] for i in selected_indices]
        transformed_dataset = Dataset(X_transformed, y_transformed, features=features, label=dataset.label)

        return transformed_dataset
