import numpy as np
from si.base.transformer import Transformer
from si.statistics.f_classification import f_classification
from si.data.dataset import Dataset

class SelectPercentile(Transformer):
    """
    Select features based on a percentile of the highest scores.

    Parameters
    ----------
    percentile : int or float
        The percentile of features to keep. If int, it is interpreted as the number of features.
        If float, it represents the fraction of features to keep.

    score_func : callable, default=f_classification
        Function to compute the feature scores.
    """

    # Added score_func to __init__
    def __init__(self, percentile=10, score_func=f_classification):
        super().__init__()
        self.percentile = percentile
        self.score_func = score_func # Storing the score function
        self.F_ = None  # F values for each feature
        self.p_ = None  # p values for each feature

    def _fit(self, dataset: Dataset) -> 'SelectPercentile':
        """
        Fit the SelectPercentile model using the dataset.

        Parameters
        ----------
        dataset : Dataset
            The dataset to fit the transformer to.

        Returns
        -------
        self : SelectPercentile
            The fitted transformer.
        """
        # Call the stored score_func with the Dataset object (assumes score_func takes a Dataset)
        self.F_, self.p_ = self.score_func(dataset)
        return self

    def _transform(self, dataset: Dataset) -> Dataset:
        # The _transform method is fine as it was
        if self.F_ is None:
            raise RuntimeError("The transformer has not been fitted yet.")

        # Calculate the number of features to select
        n_features = dataset.X.shape[1]
        if isinstance(self.percentile, float):
            # Use np.ceil or round for robust conversion, or int() for simple truncation
            n_selected = int(self.percentile * n_features) 
        else:
            n_selected = self.percentile

        # Get the threshold F-value
        sorted_F = np.sort(self.F_)
        # Ensures n_selected is within bounds
        n_selected = min(n_selected, n_features) 
        
        # Handle the case where n_selected is 0 or less
        if n_selected <= 0:
            # Select no features (empty dataset)
            selected_indices = np.array([], dtype=int) 
            threshold = -np.inf # Set threshold to select nothing
        else:
            # Get the F-value that marks the cutoff for the top n_selected features
            threshold = sorted_F[-n_selected]

            # Handle ties at the threshold
            mask = self.F_ >= threshold
            selected_indices = np.where(mask)[0]

            # Ensure the correct number of features is selected by only taking the indices 
            # corresponding to the largest n_selected F-values
            if len(selected_indices) > n_selected:
                # Get the indices of the largest n_selected F values directly
                top_indices = np.argsort(self.F_)[-n_selected:]
                selected_indices = top_indices
        
        # Create the transformed dataset
        X_transformed = dataset.X[:, selected_indices]
        y_transformed = dataset.y

        # Create a new Dataset object
        features = [dataset.features[i] for i in selected_indices]
        transformed_dataset = Dataset(X_transformed, y_transformed, features=features, label=dataset.label)

        return transformed_dataset