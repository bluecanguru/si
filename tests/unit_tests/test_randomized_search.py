import unittest
import numpy as np
from si.io.csv_file import read_csv
from si.models.logistic_regression import LogisticRegression
from si.model_selection.randomized_search import randomized_search_cv

class TestRandomizedSearch(unittest.TestCase):
    
    def setUp(self):
        self.dataset = read_csv('datasets/breast_bin/breast-bin.csv', label=True)

        self.model = LogisticRegression()

    def test_randomized_search_cv(self):
        parameter_grid = {
            'l2_penalty': np.linspace(1, 10, 10),  
            'alpha': np.linspace(0.001, 0.0001, 100),  
            'max_iter': np.linspace(1000, 2000, 200).astype(int)  
        }

        results = randomized_search_cv(
            model=self.model,
            dataset=self.dataset,
            hyperparameter_grid=parameter_grid,
            cv=3,
            n_iter=10
        )

        self.assertEqual(len(results['scores']), 10,
                         "Should return scores for each of the 10 iterations")
        self.assertEqual(len(results['hyperparameters']), 10,
                         "Should return hyperparameters for each of the 10 iterations")
        self.assertIsNotNone(results['best_score'],
                            "Should return a best score from the search")
        self.assertIn('l2_penalty', results['best_hyperparameters'],
                     "Best hyperparameters should include l2_penalty")

        print(f"\nBest Score found: {results['best_score']}")
        print(f"Best Hyperparameters: {results['best_hyperparameters']}")

if __name__ == '__main__':
    unittest.main()