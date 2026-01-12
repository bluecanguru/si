import numpy as np
from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy

class StackingClassifier(Model):
    """
    Stacking Classifier.

    Parameters
    ----------
    models : list of Model
        The initial set of models.
    final_model : Model
        The model to make the final predictions.
    """
    def __init__(self, models, final_model):
        super().__init__()
        self.models = models
        self.final_model = final_model

    def _fit(self, dataset: Dataset) -> 'StackingClassifier':
        """
        Fit the StackingClassifier model to the dataset.

        Parameters
        ----------
        dataset : Dataset
            The dataset to fit the model to.

        Returns
        -------
        self : StackingClassifier
            The fitted model.
        """
        X = dataset.X
        y = dataset.y

        for model in self.models:
            model.fit(dataset)

        predictions = np.zeros((X.shape[0], len(self.models)))
        for i, model in enumerate(self.models):
            predictions[:, i] = model.predict(dataset)

        predictions_dataset = Dataset(predictions, y)

        self.final_model.fit(predictions_dataset)

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
        X = dataset.X

        predictions = np.zeros((X.shape[0], len(self.models)))
        for i, model in enumerate(self.models):
            predictions[:, i] = model.predict(dataset)

        predictions_dataset = Dataset(predictions, None)

        y_pred = self.final_model.predict(predictions_dataset)

        return y_pred

    def _score(self, dataset: Dataset, predictions=None) -> float:
        """
        Calculate the accuracy score between the real and predicted labels.

        Parameters
        ----------
        dataset : Dataset
            The dataset to score the model on.
        predictions : np.ndarray, optional
            The predicted labels. If not provided, they will be computed.

        Returns
        -------
        score : float
            The accuracy score.
        """
        if predictions is None:
            predictions = self._predict(dataset)
        return accuracy(dataset.y, predictions)