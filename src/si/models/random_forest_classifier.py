import numpy as np
from si.base.model import Model
from si.data.dataset import Dataset
from si.models.decision_tree_classifier import DecisionTreeClassifier
from si.metrics.accuracy import accuracy

class RandomForestClassifier(Model):
    """
    Random Forest Classifier.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of decision trees to use.

    max_features : int, optional
        Maximum number of features to use per tree. If None, it is set to sqrt(n_features).

    min_sample_split : int, default=2
        Minimum number of samples required to split a node.

    max_depth : int, optional
        Maximum depth of the trees. If None, nodes are expanded until all leaves are pure.

    mode : str, default='gini'
        Impurity calculation mode ('gini' or 'entropy').

    seed : int, optional
        Random seed for reproducibility.
    """

    def __init__(self, n_estimators: int = 100, max_features: int = None, min_sample_split: int = 2,
                 max_depth: int = None, mode: str = 'gini', seed: int = None):
        super().__init__()
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_sample_split = min_sample_split
        self.max_depth = max_depth
        self.mode = mode
        self.seed = seed
        self.trees = []  # List to store tuples of (features, tree)

    def _fit(self, dataset: Dataset) -> 'RandomForestClassifier':
        """
        Train the decision trees of the random forest.

        Parameters
        ----------
        dataset : Dataset
            The dataset to fit the model to.

        Returns
        -------
        self : RandomForestClassifier
            The fitted model.
        """
        if self.seed is not None:
            np.random.seed(self.seed)

        n_samples, n_features = dataset.X.shape

        # Set max_features to sqrt(n_features) if None
        if self.max_features is None:
            self.max_features = int(np.sqrt(n_features))

        # Train each tree
        for _ in range(self.n_estimators):
            # Create a bootstrap dataset (with replacement)
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_bootstrap = dataset.X[indices]
            y_bootstrap = dataset.y[indices]

            # Randomly select features (without replacement)
            feature_indices = np.random.choice(n_features, self.max_features, replace=False)
            X_bootstrap = X_bootstrap[:, feature_indices]

            # Create and train a decision tree
            # Set max_depth to a large number if it is None
            current_max_depth = 1000 if self.max_depth is None else self.max_depth
            tree = DecisionTreeClassifier(
                min_sample_split=self.min_sample_split,
                max_depth=current_max_depth,
                mode=self.mode
            )
            tree.fit(Dataset(X_bootstrap, y_bootstrap))

            # Append the features and tree to the list
            self.trees.append((feature_indices, tree))

        return self

    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predict the labels using the ensemble models.

        Parameters
        ----------
        dataset : Dataset
            The dataset to predict.

        Returns
        -------
        y_pred : np.ndarray
            The predicted labels.
        """
        n_samples = dataset.X.shape[0]
        predictions = np.zeros((n_samples, len(self.trees)))

        for i, (feature_indices, tree) in enumerate(self.trees):
            X_subset = dataset.X[:, feature_indices]
            predictions[:, i] = tree.predict(X_subset)

        # Get the most common predicted class for each sample
        y_pred = np.array([np.bincount(row.astype(int)).argmax() for row in predictions])

        return y_pred

    def _score(self, dataset: Dataset) -> float:
        """
        Compute the accuracy between predicted and real labels.

        Parameters
        ----------
        dataset : Dataset
            The dataset to score the model on.

        Returns
        -------
        score : float
            The accuracy score.
        """
        y_pred = self._predict(dataset)
        return accuracy(dataset.y, y_pred)