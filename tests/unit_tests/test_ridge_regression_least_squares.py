import unittest
import numpy as np
from si.data.dataset import Dataset
from si.models.ridge_regression_least_squares import RidgeRegressionLeastSquares

class TestRidgeRegressionLeastSquares(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a simple synthetic dataset for testing
        np.random.seed(42)
        X = np.random.rand(100, 3)
        y = np.dot(X, np.array([1.5, -2.0, 0.5])) + np.random.rand(100) * 0.1
        cls.dataset = Dataset(X, y)

    def test_initialization(self):
        # Test initialization with default parameters
        rr = RidgeRegressionLeastSquares()
        self.assertEqual(rr.l2_penalty, 1.0)
        self.assertEqual(rr.scale, True)
        self.assertIsNone(rr.theta)
        self.assertIsNone(rr.theta_zero)
        self.assertIsNone(rr.mean)
        self.assertIsNone(rr.std)

        # Test initialization with custom parameters
        rr = RidgeRegressionLeastSquares(l2_penalty=0.5, scale=False)
        self.assertEqual(rr.l2_penalty, 0.5)
        self.assertEqual(rr.scale, False)

    def test_fit(self):
        rr = RidgeRegressionLeastSquares(l2_penalty=1.0, scale=True)
        rr.fit(self.dataset)

        # Check if theta and theta_zero are computed
        self.assertIsNotNone(rr.theta)
        self.assertIsNotNone(rr.theta_zero)
        self.assertIsNotNone(rr.mean)
        self.assertIsNotNone(rr.std)

        # Check if the shapes are correct
        self.assertEqual(rr.theta.shape[0], self.dataset.X.shape[1])
        self.assertEqual(rr.mean.shape[0], self.dataset.X.shape[1])
        self.assertEqual(rr.std.shape[0], self.dataset.X.shape[1])

    def test_predict(self):
        rr = RidgeRegressionLeastSquares(l2_penalty=1.0, scale=True)
        rr.fit(self.dataset)
        predictions = rr.predict(self.dataset.X)

        # Check if predictions are computed
        self.assertIsNotNone(predictions)
        self.assertEqual(predictions.shape[0], self.dataset.X.shape[0])

    def test_score(self):
        rr = RidgeRegressionLeastSquares(l2_penalty=1.0, scale=True)
        rr.fit(self.dataset)
        mse_score = rr.score(self.dataset)

        # Check if MSE score is computed and is a non-negative value
        self.assertIsNotNone(mse_score)
        self.assertGreaterEqual(mse_score, 0)

    def test_reproducibility(self):
        rr1 = RidgeRegressionLeastSquares(l2_penalty=1.0, scale=True)
        rr1.fit(self.dataset)
        preds1 = rr1.predict(self.dataset.X)

        rr2 = RidgeRegressionLeastSquares(l2_penalty=1.0, scale=True)
        rr2.fit(self.dataset)
        preds2 = rr2.predict(self.dataset.X)

        # Check if predictions are the same without a seed (since there is no randomness in Ridge Regression)
        np.testing.assert_allclose(preds1, preds2, rtol=1e-10)

if __name__ == '__main__':
    unittest.main()