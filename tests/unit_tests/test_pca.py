import unittest
import os
import numpy as np
from si.data.dataset import Dataset
from si.decomposition.pca import PCA
from si.io.csv_file import read_csv

class TestPCA(unittest.TestCase):
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
        pca = PCA(n_components=2)
        self.assertEqual(pca.n_components, 2)
        self.assertIsNone(pca.mean)
        self.assertIsNone(pca.components)
        self.assertIsNone(pca.explained_variance)

    def test_fit(self):
        pca = PCA(n_components=2)
        pca.fit(self.dataset)

        # Check if mean, components, and explained_variance are computed
        self.assertIsNotNone(pca.mean)
        self.assertIsNotNone(pca.components)
        self.assertIsNotNone(pca.explained_variance)

        # Check if the shapes are correct
        self.assertEqual(pca.mean.shape[0], self.dataset.X.shape[1])
        self.assertEqual(pca.components.shape, (pca.n_components, self.dataset.X.shape[1]))
        self.assertEqual(len(pca.explained_variance), pca.n_components)

    def test_transform(self):
        pca = PCA(n_components=2)
        pca.fit(self.dataset)
        transformed_dataset = pca.transform(self.dataset)

        # Check if the transformed dataset has the correct shape
        self.assertEqual(transformed_dataset.X.shape[1], pca.n_components)
        self.assertEqual(transformed_dataset.X.shape[0], self.dataset.X.shape[0])

        # Check if the features are correctly named
        expected_features = [f"PC_{i+1}" for i in range(pca.n_components)]
        self.assertEqual(transformed_dataset.features, expected_features)

    def test_fit_transform(self):
        pca = PCA(n_components=2)
        transformed_dataset = pca.fit_transform(self.dataset)

        # Check if the transformed dataset has the correct shape
        self.assertEqual(transformed_dataset.X.shape[1], pca.n_components)
        self.assertEqual(transformed_dataset.X.shape[0], self.dataset.X.shape[0])

        # Check if the features are correctly named
        expected_features = [f"PC_{i+1}" for i in range(pca.n_components)]
        self.assertEqual(transformed_dataset.features, expected_features)

    def test_explained_variance(self):
        pca = PCA(n_components=2)
        pca.fit(self.dataset)

        # Check if explained variance is between 0 and 1
        self.assertTrue(all(0 <= var <= 1 for var in pca.explained_variance))

        # Check if the sum of explained variance is less than or equal to 1
        self.assertLessEqual(np.sum(pca.explained_variance), 1)

if __name__ == '__main__':
    unittest.main()
