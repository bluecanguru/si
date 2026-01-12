import unittest
import numpy as np
from si.neural_networks.layers import Dropout

class TestDropout(unittest.TestCase):
    """
    Unit tests for the Dropout layer class in neural networks.

    This class tests the functionality of the Dropout layer, which is used for regularization
    in neural networks. Dropout randomly sets a fraction of input units to 0 during training
    to prevent overfitting. The tests verify correct behavior during both training and
    inference phases, as well as proper handling of backward propagation and shape information.
    """

    def setUp(self):
        """
        Set up the testing environment.

        Creates a sample input array and initializes a Dropout layer with a 50% dropout rate.
        The input shape is set to match the dimensions of the sample input data.
        """
        self.input_data = np.array([[1.0, 2.0, 3.0],
                                   [4.0, 5.0, 6.0]])
        self.prob = 0.5
        self.layer = Dropout(probability=self.prob)
        self.layer.set_input_shape((3,))

    def test_forward_inference(self):
        """
        Test forward propagation during inference mode.

        Verifies that the Dropout layer acts as an identity function during inference
        (when training=False), meaning it should not modify the input data in any way.
        This is a critical property of Dropout layers during model evaluation.
        """
        output = self.layer.forward_propagation(self.input_data, training=False)
        np.testing.assert_array_equal(output, self.input_data,
                                     err_msg="Dropout should not change data during inference.")

    def test_forward_training(self):
        """
        Test forward propagation during training mode.

        Verifies that during training:
        1. The layer correctly zeros out approximately the specified fraction of elements
        2. The non-zero elements are scaled by 1/(1-probability) to maintain the expected value
        3. The actual dropout rate is close to the specified probability

        Uses a large input array to ensure statistical significance in the dropout rate test.
        """
        # Using a large array for statistical significance
        large_input = np.ones((100, 100))
        large_layer = Dropout(probability=0.4)

        output = large_layer.forward_propagation(large_input, training=True)

        # Check scaling: non-zero values should be 1 / (1 - 0.4) = 1.666...
        expected_scale = 1 / (1 - 0.4)
        non_zero_vals = output[output != 0]
        if non_zero_vals.size > 0:
            self.assertTrue(np.allclose(non_zero_vals, expected_scale),
                           "Non-zero values should be scaled by 1/(1-probability)")

        # Check dropout rate: should be roughly 40%
        actual_drop_rate = np.sum(output == 0) / output.size
        self.assertAlmostEqual(actual_drop_rate, 0.4, delta=0.05,
                              msg="Actual dropout rate should be close to the specified probability")

    def test_backward_propagation(self):
        """
        Test backward propagation through the Dropout layer.

        Verifies that during backpropagation:
        1. The layer correctly applies the mask generated during the forward pass
        2. The error is zeroed out where the mask was zero during forward pass
        3. The error is preserved where the mask was one during forward pass

        This ensures that gradients are properly propagated through the Dropout layer.
        """
        # Forward pass to generate mask
        self.layer.forward_propagation(self.input_data, training=True)
        mask = self.layer.mask

        # Dummy error gradient
        output_error = np.random.rand(2, 3)
        input_error = self.layer.backward_propagation(output_error)

        # Verification: Error should be 0 where mask was 0, and same where mask was 1
        expected = output_error * mask
        np.testing.assert_array_equal(input_error, expected,
                                    err_msg="Backward propagation should apply the forward mask to errors")

    def test_output_shape(self):
        """
        Test the output_shape method of the Dropout layer.

        Verifies that the output shape matches the input shape, as Dropout layers
        do not change the dimensionality of the data.
        """
        self.assertEqual(self.layer.output_shape(), (3,),
                        "Output shape should match input shape")

    def test_parameters(self):
        """
        Test the parameters method of the Dropout layer.

        Verifies that the Dropout layer correctly reports having zero trainable parameters,
        as it is a regularization technique rather than a learnable layer.
        """
        self.assertEqual(self.layer.parameters(), 0,
                        "Dropout layer should have no trainable parameters")

if __name__ == '__main__':
    unittest.main()