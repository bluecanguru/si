import unittest
import numpy as np
from si.neural_networks.layers import Dropout

class TestDropout(unittest.TestCase):

    def setUp(self):
        
        self.input_data = np.array([[1.0, 2.0, 3.0],
                                   [4.0, 5.0, 6.0]])
        self.prob = 0.5
        self.layer = Dropout(probability=self.prob)
        self.layer.set_input_shape((3,))

    def test_forward_inference(self):
        
        output = self.layer.forward_propagation(self.input_data, training=False)
        np.testing.assert_array_equal(output, self.input_data,
                                     err_msg="Dropout should not change data during inference.")

    def test_forward_training(self):
        
        large_input = np.ones((100, 100))
        large_layer = Dropout(probability=0.4)

        output = large_layer.forward_propagation(large_input, training=True)

        expected_scale = 1 / (1 - 0.4)
        non_zero_vals = output[output != 0]
        if non_zero_vals.size > 0:
            self.assertTrue(np.allclose(non_zero_vals, expected_scale),
                           "Non-zero values should be scaled by 1/(1-probability)")

        actual_drop_rate = np.sum(output == 0) / output.size
        self.assertAlmostEqual(actual_drop_rate, 0.4, delta=0.05,
                              msg="Actual dropout rate should be close to the specified probability")

    def test_backward_propagation(self):
        
        self.layer.forward_propagation(self.input_data, training=True)
        mask = self.layer.mask

        output_error = np.random.rand(2, 3)
        input_error = self.layer.backward_propagation(output_error)

        expected = output_error * mask
        np.testing.assert_array_equal(input_error, expected,
                                    err_msg="Backward propagation should apply the forward mask to errors")

    def test_output_shape(self):
        
        self.assertEqual(self.layer.output_shape(), (3,),
                        "Output shape should match input shape")

    def test_parameters(self):
        
        self.assertEqual(self.layer.parameters(), 0,
                        "Dropout layer should have no trainable parameters")

if __name__ == '__main__':
    unittest.main()