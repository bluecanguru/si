import unittest
import numpy as np
from si.neural_networks.losses import CategoricalCrossEntropy

class TestCategoricalCrossEntropy(unittest.TestCase):
    """
    Unit tests for the Categorical Cross Entropy loss function.

    This class contains tests to verify the correct implementation of the Categorical Cross Entropy
    loss function, which is commonly used for multi-class classification problems. The tests
    validate both the loss calculation and the shape of its derivative.
    """

    def test_loss(self):
        """
        Test the Categorical Cross Entropy loss calculation.

        Verifies that the loss function correctly computes the cross entropy between true labels
        and predicted probabilities. The test uses a simple example with two samples and three
        classes, where each sample belongs to a different class.

        The expected loss is calculated as the negative log likelihood of the true class
        probabilities, averaged across all samples. For this specific case, the expected loss
        is: - (1*log(0.7) + 1*log(0.8)) / 2, where 0.7 and 0.8 are the predicted probabilities
        for the true classes of the two samples.
        """
        y_true = np.array([[1, 0, 0], [0, 1, 0]])
        y_pred = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])

        cce = CategoricalCrossEntropy()
        loss = cce.loss(y_true, y_pred)

        expected = - (np.log(0.7) + np.log(0.8)) / 2
        self.assertAlmostEqual(loss, expected,
                              msg="Categorical Cross Entropy loss should match the expected value")

    def test_derivative_shape(self):
        """
        Test the shape of the Categorical Cross Entropy derivative.

        Verifies that the derivative of the Categorical Cross Entropy loss has the same shape
        as the input predictions. This is important for proper gradient computation during
        backpropagation in neural networks.

        The test uses a single sample with three classes to check that the derivative maintains
        the same dimensionality as the input.
        """
        y_true = np.array([[1, 0, 0]])
        y_pred = np.array([[0.7, 0.2, 0.1]])

        cce = CategoricalCrossEntropy()
        deriv = cce.derivative(y_true, y_pred)

        self.assertEqual(deriv.shape, y_pred.shape,
                        msg="Derivative shape should match input predictions shape")

if __name__ == '__main__':
    unittest.main()