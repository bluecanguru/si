from unittest import TestCase
import os
import numpy as np
from si.io.csv_file import read_csv
from si.model_selection.split import stratified_train_test_split

# Define the path to the datasets directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATASETS_PATH = os.path.join(BASE_DIR, 'datasets')

class TestStratifiedTrainTestSplit(TestCase):
    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'iris', 'iris.csv')
        if not os.path.exists(self.csv_file):
            raise FileNotFoundError(f"The dataset file was not found at: {self.csv_file}")
        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)

    def test_stratified_train_test_split(self):
        train, test = stratified_train_test_split(self.dataset, test_size=0.2, random_state=123)

        # Check if the sizes of train and test sets are correct
        test_samples_size = int(self.dataset.shape()[0] * 0.2)
        self.assertEqual(test.shape()[0], test_samples_size)
        self.assertEqual(train.shape()[0], self.dataset.shape()[0] - test_samples_size)

        # Check if the class distribution is preserved in both train and test sets
        unique_labels, label_counts = np.unique(self.dataset.y, return_counts=True)
        test_unique_labels, test_label_counts = np.unique(test.y, return_counts=True)
        train_unique_labels, train_label_counts = np.unique(train.y, return_counts=True)

        # Check if all classes are represented in both train and test sets
        np.testing.assert_array_equal(np.sort(unique_labels), np.sort(test_unique_labels))
        np.testing.assert_array_equal(np.sort(unique_labels), np.sort(train_unique_labels))

        # Check if the proportions of classes are approximately preserved
        original_proportions = label_counts / label_counts.sum()
        test_proportions = test_label_counts / test_label_counts.sum()
        train_proportions = train_label_counts / train_label_counts.sum()

        np.testing.assert_allclose(original_proportions, test_proportions, rtol=0.2)
        np.testing.assert_allclose(original_proportions, train_proportions, rtol=0.2)

if __name__ == '__main__':
    import unittest
    unittest.main()