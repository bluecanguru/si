import numpy as np
from typing import List, Tuple
from si.base.model import Model
from si.models.decision_tree_classifier import DecisionTreeClassifier
from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy


class RandomForestClassifier(Model):
    """
    Random Forest Classifier.

    Random Forest is an ensemble learning method that operates by constructing multiple decision trees
    during training and outputting the mode of the classes (classification) of the individual trees.

    Parameters
    ----------
    n_estimators : int, default=10
        The number of trees in the forest.
    max_features : int, optional
        The number of features to consider when looking for the best split.
        If None, then `max_features = sqrt(n_features)`.
    min_sample_split : int, default=2
        The minimum number of samples required to split an internal node.
    max_depth : int, optional
        The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure.
    mode : str, default="gini"
        The function to measure the quality of a split. Supported criteria are "gini" for the Gini impurity
        and "entropy" for the information gain.
    seed : int, default=42
        Seed for the random number generator.
    """

    def __init__(self, n_estimators: int = 10, max_features: int = None, min_sample_split: int = 2,
                 max_depth: int = None, mode: str = "gini", seed: int = 42):
        super().__init__()
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_sample_split = min_sample_split
        self.max_depth = max_depth
        self.mode = mode
        self.seed = seed

        # model parameters
        self.trees: List[Tuple[np.ndarray, DecisionTreeClassifier]] = []  # (feature_indices, tree)

    def _fit(self, dataset: Dataset) -> "RandomForestClassifier":
        """
        Fit the Random Forest model according to the given training data.

        Parameters
        ----------
        dataset : Dataset
            The dataset used to train the model.

        Returns
        -------
        self : RandomForestClassifier
            The fitted model.
        """
        
        np.random.seed(self.seed)

        n_samples, n_features = dataset.X.shape
        if self.max_features is None:
            self.max_features = int(np.sqrt(n_features))

        self.trees = []
        for _ in range(self.n_estimators):
            # bootstrap samples
            idx_samples = np.random.choice(n_samples, size=n_samples, replace=True)
            # random subset of features
            idx_features = np.random.choice(n_features, size=self.max_features, replace=False)

            X_boot = dataset.X[idx_samples][:, idx_features]
            y_boot = dataset.y[idx_samples]
            boot_dataset = Dataset(X_boot, y_boot)

            tree = DecisionTreeClassifier(
                min_sample_split=self.min_sample_split,
                max_depth=self.max_depth,
                mode=self.mode
            ).fit(boot_dataset)

            self.trees.append((idx_features, tree))

        return self

    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predict class for X in the dataset.

        The predicted class of an input sample is computed as the majority vote of the
        predictions of the trees in the forest.

        Parameters
        ----------
        dataset : Dataset
            The dataset for which to predict the class.

        Returns
        -------
        y_pred : np.ndarray
            The predicted classes.
        """
        tree_predictions = []
        for feature_indices, tree in self.trees:
            X_subset = dataset.X[:, feature_indices]
            preds = tree.predict(Dataset(X_subset))
            tree_predictions.append(preds)

        tree_predictions = np.array(tree_predictions)  # (n_estimators, n_samples)

        # majority vote per sample
        final_pred = []
        for i in range(dataset.X.shape[0]):
            votes = tree_predictions[:, i]
            values, counts = np.unique(votes, return_counts=True)
            final_pred.append(values[np.argmax(counts)])

        return np.array(final_pred)


    def _score(self, dataset: Dataset, predictions: np.ndarray) -> float:
        """
        Return the accuracy score on the given dataset and predictions.

        Parameters
        ----------
        dataset : Dataset
            The dataset containing the true labels.
        predictions : np.ndarray
            The predicted labels.

        Returns
        -------
        score : float
            The accuracy score.
        """
        return accuracy(dataset.y, predictions)
