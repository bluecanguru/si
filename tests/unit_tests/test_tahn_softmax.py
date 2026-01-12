import unittest
import numpy as np
from si.neural_networks.activation import TanhActivation, SoftmaxActivation

class TestActivations(unittest.TestCase):
   
    def setUp(self):
        self.input_data = np.array([[0.5, 1.0, -0.5],
                                   [1.5, -1.0, 0.0]])

    def test_tanh_forward(self):
        layer = TanhActivation()
        output = layer.forward_propagation(self.input_data, training=False)

        self.assertTrue(np.all(output > -1) and np.all(output < 1))

        zero_input = np.array([[0.0]])
        self.assertEqual(layer.forward_propagation(zero_input, training=False), 0.0)

    def test_tanh_derivative(self):
        layer = TanhActivation()
        layer.forward_propagation(self.input_data, training=True)
        deriv = layer.derivative(self.input_data)

        zero_deriv = layer.derivative(np.array([[0.0]]))
        self.assertEqual(zero_deriv, 1.0)

        self.assertTrue(np.all(deriv > 0) and np.all(deriv <= 1))

    def test_softmax_forward_probabilities(self):
        
        layer = SoftmaxActivation()
        output = layer.forward_propagation(self.input_data, training=False)

        sums = np.sum(output, axis=-1)
        np.testing.assert_allclose(sums, np.ones(self.input_data.shape[0]), atol=1e-7)

        self.assertTrue(np.all(output >= 0))

    def test_softmax_numerical_stability(self):
        large_input = np.array([[1000.0, 999.0, 998.0]])
        layer = SoftmaxActivation()
        output = layer.forward_propagation(large_input, training=False)

        self.assertFalse(np.any(np.isnan(output)))
        self.assertTrue(output[0, 0] > output[0, 1])

    def test_softmax_derivative_shape(self):
        layer = SoftmaxActivation()
        layer.forward_propagation(self.input_data, training=True)
        deriv = layer.derivative(self.input_data)

        self.assertEqual(deriv.shape, self.input_data.shape)

if __name__ == '__main__':
    unittest.main()