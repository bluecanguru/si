import unittest
import numpy as np
from si.neural_networks.losses import CategoricalCrossEntropy

class TestCategoricalCrossEntropy(unittest.TestCase):

    def test_loss(self):
        
        y_true = np.array([[1, 0, 0], [0, 1, 0]])
        y_pred = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])

        cce = CategoricalCrossEntropy()
        loss = cce.loss(y_true, y_pred)

        expected = - (np.log(0.7) + np.log(0.8)) / 2
        self.assertAlmostEqual(loss, expected,
                              msg="Categorical Cross Entropy loss should match the expected value")

    def test_derivative_shape(self):
        
        y_true = np.array([[1, 0, 0]])
        y_pred = np.array([[0.7, 0.2, 0.1]])

        cce = CategoricalCrossEntropy()
        deriv = cce.derivative(y_true, y_pred)

        self.assertEqual(deriv.shape, y_pred.shape,
                        msg="Derivative shape should match input predictions shape")

if __name__ == '__main__':
    unittest.main()