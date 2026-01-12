import unittest
import os
import numpy as np
from si.data.dataset import Dataset
from si.ensemble.stacking_classifier import StackingClassifier
from si.io.csv_file import read_csv
from si.model_selection.split import train_test_split
from si.models.knn_classifier import KNNClassifier
from si.models.logistic_regression import LogisticRegression
from si.models.decision_tree_classifier import DecisionTreeClassifier

class TestStackingClassifier(unittest.TestCase):
    """
    Unit tests for the StackingClassifier ensemble model.

    This class contains tests to verify the correct implementation and behavior of the
    StackingClassifier, including its fitting, prediction, and scoring capabilities.
    """

    @classmethod
    def setUpClass(cls):
        """
        Set up the test class with dataset.

        Loads the breast-bin dataset or creates a synthetic dataset if the file is not found.
        Splits the dataset into training and testing sets for use in the tests.
        """
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        data_path = os.path.join(base_dir, 'datasets', 'breast-bin', 'breast-bin.csv')

        if os.path.exists(data_path):
            data = read_csv(data_path, sep=',', features=True, label=True)
            cls.dataset = Dataset(data.X, data.y, features=data.features, label=data.label)
        else:
            print(f"Dataset file not found at: {data_path}. Creating a synthetic dataset for testing.")
            np.random.seed(42)
            X = np.random.rand(100, 5)
            y = np.random.randint(0, 2, 100)
            features = [f'feature_{i}' for i in range(5)]
            cls.dataset = Dataset(X, y, features=features, label='target')

        cls.train_data, cls.test_data = train_test_split(cls.dataset, test_size=0.2, random_state=42)

    def test_stacking_classifier_fit(self):
        """
        Test the fit method of the StackingClassifier.

        Verifies that the StackingClassifier can be fitted to the training data and that
        all component models are properly trained.
        """
        knn = KNNClassifier(k=3)
        logistic_regression = LogisticRegression(learning_rate=0.01, max_iter=1000)
        decision_tree = DecisionTreeClassifier(max_depth=5)

        final_knn = KNNClassifier(k=3)

        stacking_classifier = StackingClassifier(
            models=[knn, logistic_regression, decision_tree],
            final_model=final_knn
        )

        stacking_classifier.fit(self.train_data)

        for model in stacking_classifier.models:
            predictions = model.predict(self.train_data)
            self.assertIsNotNone(predictions)

        predictions = np.zeros((self.train_data.X.shape[0], len(stacking_classifier.models)))
        for i, model in enumerate(stacking_classifier.models):
            predictions[:, i] = model.predict(self.train_data)
        predictions_dataset = Dataset(predictions, self.train_data.y)
        final_predictions = stacking_classifier.final_model.predict(predictions_dataset)
        self.assertIsNotNone(final_predictions)

    def test_stacking_classifier_predict(self):
        """
        Test the predict method of the StackingClassifier.

        Verifies that the StackingClassifier can make predictions on new data and that
        the predictions have the correct format and length.
        """
        knn = KNNClassifier(k=3)
        logistic_regression = LogisticRegression(learning_rate=0.01, max_iter=1000)
        decision_tree = DecisionTreeClassifier(max_depth=5)

        final_knn = KNNClassifier(k=3)

        stacking_classifier = StackingClassifier(
            models=[knn, logistic_regression, decision_tree],
            final_model=final_knn
        )

        stacking_classifier.fit(self.train_data)

        y_pred = stacking_classifier.predict(self.test_data)

        self.assertIsNotNone(y_pred)
        self.assertEqual(len(y_pred), len(self.test_data.y))

    def test_stacking_classifier_score(self):
        """
        Test the score method of the StackingClassifier.

        Verifies that the StackingClassifier can compute an accuracy score on the test set
        and that the score is a valid value between 0 and 1.
        """
        knn = KNNClassifier(k=3)
        logistic_regression = LogisticRegression(learning_rate=0.01, max_iter=1000)
        decision_tree = DecisionTreeClassifier(max_depth=5)

        final_knn = KNNClassifier(k=3)

        stacking_classifier = StackingClassifier(
            models=[knn, logistic_regression, decision_tree],
            final_model=final_knn
        )

        stacking_classifier.fit(self.train_data)

        score = stacking_classifier.score(self.test_data)
        print(f"Test set accuracy: {score:.4f}")

        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1)

if __name__ == '__main__':
    unittest.main()