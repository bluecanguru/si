import unittest

import numpy as np

from si.data.dataset import Dataset


class TestDataset(unittest.TestCase):

    def test_dataset_construction(self):

        X = np.array([[1, 2, 3], [4, 5, 6]])
        y = np.array([1, 2])

        features = np.array(['a', 'b', 'c'])
        label = 'y'
        dataset = Dataset(X, y, features, label)

        self.assertEqual(2.5, dataset.get_mean()[0])
        self.assertEqual((2, 3), dataset.shape())
        self.assertTrue(dataset.has_label())
        self.assertEqual(1, dataset.get_classes()[0])
        self.assertEqual(2.25, dataset.get_variance()[0])
        self.assertEqual(1, dataset.get_min()[0])
        self.assertEqual(4, dataset.get_max()[0])
        self.assertEqual(2.5, dataset.summary().iloc[0, 0])

    def test_dataset_from_random(self):
        dataset = Dataset.from_random(10, 5, 3, features=['a', 'b', 'c', 'd', 'e'], label='y')
        self.assertEqual((10, 5), dataset.shape())
        self.assertTrue(dataset.has_label())
        
    def test_dropna(self):
        X = np.array([[1, 2, np.nan], [4, 5, 6], [np.nan, 1, 2]])
        y = np.array([0, 1, 0])
        dataset = Dataset(X, y)
        dataset = dataset.dropna()
        np.testing.assert_array_equal(dataset.X, np.array([[4, 5, 6]]))
        np.testing.assert_array_equal(dataset.y, np.array([1]))

    def test_fillna_with_value(self):
        X = np.array([[1, np.nan], [np.nan, 3]])
        y = np.array([0, 1])
        dataset = Dataset(X, y)
        dataset = dataset.fillna(0)
        np.testing.assert_array_equal(dataset.X, np.array([[1, 0], [0, 3]]))

    def test_fillna_with_mean(self):
        X = np.array([[1, np.nan], [3, 5]])
        y = np.array([0, 1])
        dataset = Dataset(X, y)
        dataset = dataset.fillna("mean")
        expected = np.array([[1, 5], [3, 5]])  # mean of col2 = 5
        np.testing.assert_array_equal(dataset.X, expected)

    def test_remove_by_index(self):
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([0, 1, 2])
        dataset = Dataset(X, y)
        dataset = dataset.remove_by_index(1)
        np.testing.assert_array_equal(dataset.X, np.array([[1, 2], [5, 6]]))
        np.testing.assert_array_equal(dataset.y, np.array([0, 2]))