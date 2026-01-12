import unittest
import os
import numpy as np
from si.data.dataset import Dataset
from si.feature_selection.select_percentile import SelectPercentile
from si.io.csv_file import read_csv

class TestSelectPercentile(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        data_path = os.path.join(base_dir, 'datasets', 'iris', 'iris.csv')

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"The dataset file was not found at: {data_path}")

        data = read_csv(data_path, sep=',', features=True, label=True)
        cls.dataset = Dataset(data.X, data.y, features=data.features, label=data.label)

    def test_initialization(self):
        selector = SelectPercentile(percentile=10)
        self.assertEqual(selector.percentile, 10)
        self.assertIsNone(selector.F_)
        self.assertIsNone(selector.p_)

        selector = SelectPercentile(percentile=20)
        self.assertEqual(selector.percentile, 20)

    def test_fit(self):
        selector = SelectPercentile(percentile=10)
        selector.fit(self.dataset)

        self.assertIsNotNone(selector.F_)
        self.assertIsNotNone(selector.p_)

        self.assertEqual(len(selector.F_), self.dataset.X.shape[1])
        self.assertEqual(len(selector.p_), self.dataset.X.shape[1])

    def test_transform_with_float_percentile(self):
        selector = SelectPercentile(percentile=0.5)  
        selector.fit(self.dataset)
        transformed_dataset = selector.transform(self.dataset)

        n_features = self.dataset.X.shape[1]
        expected_n_features = int(0.5 * n_features)
        self.assertEqual(transformed_dataset.X.shape[1], expected_n_features)
        self.assertEqual(transformed_dataset.X.shape[0], self.dataset.X.shape[0])

        self.assertEqual(len(transformed_dataset.features), expected_n_features)

    def test_transform_with_int_percentile(self):
        selector = SelectPercentile(percentile=2)  
        selector.fit(self.dataset)
        transformed_dataset = selector.transform(self.dataset)

        self.assertEqual(transformed_dataset.X.shape[1], 2)
        self.assertEqual(transformed_dataset.X.shape[0], self.dataset.X.shape[0])

        self.assertEqual(len(transformed_dataset.features), 2)

    def test_transform_with_zero_percentile(self):
        selector = SelectPercentile(percentile=0)  
        selector.fit(self.dataset)
        transformed_dataset = selector.transform(self.dataset)

        self.assertEqual(transformed_dataset.X.shape[1], 0)
        self.assertEqual(transformed_dataset.X.shape[0], self.dataset.X.shape[0])

        self.assertEqual(len(transformed_dataset.features), 0)

    def test_transform_with_all_features(self):
        selector = SelectPercentile(percentile=1.0)  
        selector.fit(self.dataset)
        transformed_dataset = selector.transform(self.dataset)

        self.assertEqual(transformed_dataset.X.shape[1], self.dataset.X.shape[1])
        self.assertEqual(transformed_dataset.X.shape[0], self.dataset.X.shape[0])

        self.assertEqual(len(transformed_dataset.features), self.dataset.X.shape[1])

    def test_fit_transform(self):
        selector = SelectPercentile(percentile=0.5)  
        transformed_dataset = selector.fit_transform(self.dataset)

        n_features = self.dataset.X.shape[1]
        expected_n_features = int(0.5 * n_features)
        self.assertEqual(transformed_dataset.X.shape[1], expected_n_features)
        self.assertEqual(transformed_dataset.X.shape[0], self.dataset.X.shape[0])

        self.assertEqual(len(transformed_dataset.features), expected_n_features)

    def test_transform_with_ties(self):
        X = np.array([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]])
        y = np.array([0, 1])
        features = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5']
        mock_dataset = Dataset(X, y, features=features, label='target')

        selector = SelectPercentile(percentile=0.6)  
        selector.F_ = np.array([1.0, 3.0, 3.0, 3.0, 5.0])  
        selector.p_ = np.array([0.5, 0.3, 0.3, 0.3, 0.1])

        transformed_dataset = selector.transform(mock_dataset)

        self.assertEqual(transformed_dataset.X.shape[1], 3)
        self.assertEqual(transformed_dataset.X.shape[0], mock_dataset.X.shape[0])

    def test_fit_transform_chaining(self):
        selector1 = SelectPercentile(percentile=0.5)
        dataset1 = selector1.fit(self.dataset).transform(self.dataset)

        selector2 = SelectPercentile(percentile=0.5)
        dataset2 = selector2.fit_transform(self.dataset)

        np.testing.assert_array_equal(dataset1.X, dataset2.X)
        self.assertEqual(dataset1.features, dataset2.features)

    def test_transform_before_fit(self):
        selector = SelectPercentile(percentile=0.5)
        with self.assertRaises(RuntimeError):
            selector.transform(self.dataset)

if __name__ == '__main__':
    unittest.main()
