import unittest
import os
from si.data.dataset import Dataset
from si.feature_selection.select_percentile import SelectPercentile
from si.io.csv_file import read_csv

class TestSelectPercentile(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Construct the absolute path to the iris dataset
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        data_path = os.path.join(base_dir, 'datasets', 'iris', 'iris.csv')

        # Check if the file exists
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"The dataset file was not found at: {data_path}")

        data = read_csv(data_path, sep=',', features=True, label=True)
        cls.dataset = Dataset(data.X, data.y, features=data.features, label=data.label)

    def test_initialization(self):
        # Test initialization with default parameters
        selector = SelectPercentile(percentile=10)
        self.assertEqual(selector.percentile, 10)
        self.assertIsNone(selector.F_)
        self.assertIsNone(selector.p_)

        # Test initialization with custom parameters
        selector = SelectPercentile(percentile=20)
        self.assertEqual(selector.percentile, 20)

    def test_fit(self):
        selector = SelectPercentile(percentile=10)
        selector.fit(self.dataset)

        # Check if F_ and p_ are computed
        self.assertIsNotNone(selector.F_)
        self.assertIsNotNone(selector.p_)

        # Check if the shapes are correct
        self.assertEqual(len(selector.F_), self.dataset.X.shape[1])
        self.assertEqual(len(selector.p_), self.dataset.X.shape[1])

    def test_transform_with_float_percentile(self):
        selector = SelectPercentile(percentile=0.5)  # Select 50% of features
        selector.fit(self.dataset)
        transformed_dataset = selector.transform(self.dataset)

        # Check if the transformed dataset has the correct shape
        n_features = self.dataset.X.shape[1]
        expected_n_features = int(0.5 * n_features)
        self.assertEqual(transformed_dataset.X.shape[1], expected_n_features)
        self.assertEqual(transformed_dataset.X.shape[0], self.dataset.X.shape[0])

        # Check if the features are correctly selected
        self.assertEqual(len(transformed_dataset.features), expected_n_features)

    def test_transform_with_int_percentile(self):
        selector = SelectPercentile(percentile=2)  # Select top 2 features
        selector.fit(self.dataset)
        transformed_dataset = selector.transform(self.dataset)

        # Check if the transformed dataset has the correct shape
        self.assertEqual(transformed_dataset.X.shape[1], 2)
        self.assertEqual(transformed_dataset.X.shape[0], self.dataset.X.shape[0])

        # Check if the features are correctly selected
        self.assertEqual(len(transformed_dataset.features), 2)

    def test_transform_with_zero_percentile(self):
        selector = SelectPercentile(percentile=0)  # Select 0 features
        selector.fit(self.dataset)
        transformed_dataset = selector.transform(self.dataset)

        # Check if the transformed dataset has the correct shape
        self.assertEqual(transformed_dataset.X.shape[1], 0)
        self.assertEqual(transformed_dataset.X.shape[0], self.dataset.X.shape[0])

        # Check if the features are correctly selected
        self.assertEqual(len(transformed_dataset.features), 0)

    def test_transform_with_all_features(self):
        selector = SelectPercentile(percentile=1.0)  # Select all features
        selector.fit(self.dataset)
        transformed_dataset = selector.transform(self.dataset)

        # Check if the transformed dataset has the correct shape
        self.assertEqual(transformed_dataset.X.shape[1], self.dataset.X.shape[1])
        self.assertEqual(transformed_dataset.X.shape[0], self.dataset.X.shape[0])

        # Check if the features are correctly selected
        self.assertEqual(len(transformed_dataset.features), self.dataset.X.shape[1])

    def test_fit_transform(self):
        selector = SelectPercentile(percentile=0.5)  # Select 50% of features
        transformed_dataset = selector.fit_transform(self.dataset)

        # Check if the transformed dataset has the correct shape
        n_features = self.dataset.X.shape[1]
        expected_n_features = int(0.5 * n_features)
        self.assertEqual(transformed_dataset.X.shape[1], expected_n_features)
        self.assertEqual(transformed_dataset.X.shape[0], self.dataset.X.shape[0])

        # Check if the features are correctly selected
        self.assertEqual(len(transformed_dataset.features), expected_n_features)

if __name__ == '__main__':
    unittest.main()
