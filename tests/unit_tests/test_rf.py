import unittest
import numpy as np
import os
from si.data.dataset import Dataset
from si.models.random_forest_classifier import RandomForestClassifier
from si.io.csv_file import read_csv
from si.model_selection.split import train_test_split

class TestRandomForestClassifier(unittest.TestCase):
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
        cls.train_data, cls.test_data = train_test_split(cls.dataset, test_size=0.2, random_state=42)

    def test_initialization(self):
        rf = RandomForestClassifier(n_estimators=10, max_features=2, min_sample_split=2, max_depth=5, mode='gini')
        self.assertEqual(rf.n_estimators, 10)
        self.assertEqual(rf.max_features, 2)
        self.assertEqual(rf.min_sample_split, 2)
        self.assertEqual(rf.max_depth, 5)
        self.assertEqual(rf.mode, 'gini')
        self.assertEqual(len(rf.trees), 0)

    def test_fit(self):
        rf = RandomForestClassifier(n_estimators=10, max_features=2, min_sample_split=2, max_depth=5, mode='gini')
        rf.fit(self.train_data)
        self.assertEqual(len(rf.trees), 10)

    def test_predict(self):
        rf = RandomForestClassifier(n_estimators=10, max_features=2, min_sample_split=2, max_depth=5, mode='gini')
        rf.fit(self.train_data)
        predictions = rf.predict(self.test_data)
        self.assertEqual(len(predictions), len(self.test_data.y))

    def test_score(self):
        rf = RandomForestClassifier(n_estimators=10, max_features=2, min_sample_split=2, max_depth=5, mode='gini')
        rf.fit(self.train_data)
        score = rf.score(self.test_data)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1)

    def test_reproducibility(self):
        rf1 = RandomForestClassifier(n_estimators=10, max_features=2, min_sample_split=2, max_depth=5, mode='gini', seed=42)
        rf1.fit(self.train_data)
        preds1 = rf1.predict(self.test_data)

        rf2 = RandomForestClassifier(n_estimators=10, max_features=2, min_sample_split=2, max_depth=5, mode='gini', seed=42)
        rf2.fit(self.train_data)
        preds2 = rf2.predict(self.test_data)

        np.testing.assert_array_equal(preds1, preds2)

if __name__ == '__main__':
    unittest.main()