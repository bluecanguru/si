import unittest
import os
from si.data.dataset import Dataset
from si.models.knn_regressor import KNNRegressor
from si.io.csv_file import read_csv
from si.model_selection.split import train_test_split
from si.metrics.mse import mse

class TestKNNRegressor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Construct the absolute path to the cpu dataset
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        data_path = os.path.join(base_dir, 'datasets', 'cpu', 'cpu.csv')

        # Check if the file exists
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"The dataset file was not found at: {data_path}")

        data = read_csv(data_path, sep=',', features=True, label=True)
        cls.dataset = Dataset(data.X, data.y, features=data.features, label=data.label)
        cls.train_data, cls.test_data = train_test_split(cls.dataset, test_size=0.2, random_state=42)

    def test_initialization(self):
        # Test initialization with default parameters
        knn = KNNRegressor()
        self.assertEqual(knn.k, 5)
        self.assertIsNone(knn.X_train)
        self.assertIsNone(knn.y_train)

        # Test initialization with custom parameters
        knn = KNNRegressor(k=10)
        self.assertEqual(knn.k, 10)

    def test_fit(self):
        knn = KNNRegressor(k=5)
        knn.fit(self.train_data)
        self.assertIsNotNone(knn.X_train)
        self.assertIsNotNone(knn.y_train)
        self.assertEqual(knn.X_train.shape, self.train_data.X.shape)
        self.assertEqual(knn.y_train.shape, self.train_data.y.shape)

    def test_predict(self):
        knn = KNNRegressor(k=5)
        knn.fit(self.train_data)
        predictions = knn.predict(self.test_data.X)
        self.assertIsNotNone(predictions)
        self.assertEqual(predictions.shape[0], self.test_data.X.shape[0])

    def test_score(self):
        knn = KNNRegressor(k=5)
        knn.fit(self.train_data)
        predictions = knn.predict(self.test_data.X)
        score = mse(self.test_data.y, predictions)
        self.assertGreaterEqual(score, 0)

if __name__ == '__main__':
    unittest.main()