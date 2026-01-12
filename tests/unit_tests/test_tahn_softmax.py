import unittest
import numpy as np
from si.neural_networks.activation import TanhActivation, SoftmaxActivation

class TestActivations(unittest.TestCase):
    """
    Unit tests for activation functions in neural networks.

    This class contains tests to verify the correct implementation and behavior of
    Tanh and Softmax activation functions, including their forward propagation and
    derivative calculations.
    """

    def setUp(self):
        """
        Set up test data for activation functions.

        Creates a sample input array that will be used to test the activation functions.
        """
        self.input_data = np.array([[0.5, 1.0, -0.5],
                                   [1.5, -1.0, 0.0]])

    def test_tanh_forward(self):
        """
        Test if Tanh squashes values between -1 and 1.

        Verifies that the Tanh activation function correctly maps input values to the
        range (-1, 1) and that specific known values (like tanh(0) = 0) are correct.
        """
        layer = TanhActivation()
        output = layer.forward_propagation(self.input_data, training=False)

        # Check if all values are in the range (-1, 1)
        self.assertTrue(np.all(output > -1) and np.all(output < 1))

        # Specific value test: tanh(0) should be 0
        zero_input = np.array([[0.0]])
        self.assertEqual(layer.forward_propagation(zero_input, training=False), 0.0)

    def test_tanh_derivative(self):
        """
        Test the derivative of Tanh: f'(x) = 1 - f(x)^2.

        Verifies that the derivative of the Tanh function is correctly calculated,
        including that the maximum derivative value is 1.0 (when input is 0) and
        that all derivative values are positive and <= 1.
        """
        layer = TanhActivation()
        # Forward pass with training=True to store input for derivative calculation
        layer.forward_propagation(self.input_data, training=True)
        deriv = layer.derivative(self.input_data)

        # The derivative of tanh is maximum (1.0) when the input is 0
        zero_deriv = layer.derivative(np.array([[0.0]]))
        self.assertEqual(zero_deriv, 1.0)

        # The derivative should always be positive and <= 1
        self.assertTrue(np.all(deriv > 0) and np.all(deriv <= 1))

    def test_softmax_forward_probabilities(self):
        """
        Test if Softmax produces a valid probability distribution (sum = 1).

        Verifies that the Softmax activation function produces output values that
        sum to 1 for each sample (row), forming a valid probability distribution,
        and that all values are non-negative.
        """
        layer = SoftmaxActivation()
        output = layer.forward_propagation(self.input_data, training=False)

        # Each row (sample) should sum approximately to 1.0
        sums = np.sum(output, axis=-1)
        np.testing.assert_allclose(sums, np.ones(self.input_data.shape[0]), atol=1e-7)

        # All values should be non-negative
        self.assertTrue(np.all(output >= 0))

    def test_softmax_numerical_stability(self):
        """
        Test if Softmax handles very large values without returning NaN (overflow).

        Verifies that the Softmax function is numerically stable and can handle
        large input values without resulting in NaN values due to overflow.
        """
        large_input = np.array([[1000.0, 999.0, 998.0]])
        layer = SoftmaxActivation()
        output = layer.forward_propagation(large_input, training=False)

        # If not stable, np.exp(1000) would result in inf and the sum would result in NaN
        self.assertFalse(np.any(np.isnan(output)))
        # The largest input value should have the highest probability
        self.assertTrue(output[0, 0] > output[0, 1])

    def test_softmax_derivative_shape(self):
        """
        Test if the Softmax derivative maintains the input shape.

        Verifies that the derivative of the Softmax function returns an output
        with the same shape as the input data.
        """
        layer = SoftmaxActivation()
        layer.forward_propagation(self.input_data, training=True)
        deriv = layer.derivative(self.input_data)

        # The derivative should have the same shape as the input
        self.assertEqual(deriv.shape, self.input_data.shape)

if __name__ == '__main__':
    unittest.main()