import unittest
import numpy as np
from si.neural_networks.neural_network import NeuralNetwork
from si.neural_networks.layers import DenseLayer
from si.neural_networks.activation import ReLUActivation, SigmoidActivation
from si.neural_networks.optimizers import SGD
from si.neural_networks.losses import BinaryCrossEntropy
from si.metrics.accuracy import accuracy
from si.data.dataset import Dataset

class TestExercise16(unittest.TestCase):
    """
    Unit tests for the neural network integration in Exercise 16.

    This class contains tests to verify the correct implementation and behavior of a neural network
    with dense layers, activation functions, and training procedures.
    """

    def setUp(self):
        """
        Set up the test fixtures.

        This method is called before each test method. It prepares a synthetic dataset and
        initializes a neural network with a specific architecture for testing purposes.
        """
        self.X = np.random.rand(64, 32)
        self.y = np.random.randint(0, 2, 64)
        self.dataset = Dataset(self.X, self.y)

        self.net = NeuralNetwork(
            epochs=10,
            batch_size=8,
            optimizer=SGD,
            learning_rate=0.01,
            loss=BinaryCrossEntropy,
            metric=accuracy
        )

        self.net.add(DenseLayer(16, (32,)))  
        self.net.add(ReLUActivation())       
        self.net.add(DenseLayer(8))          
        self.net.add(ReLUActivation())       
        self.net.add(DenseLayer(1))          
        self.net.add(SigmoidActivation())    

    def test_layer_shapes(self):
        """
        Test if the layer chaining respects the expected dimensionality reduction.

        This test verifies that each layer in the network produces the correct output shape,
        ensuring that the architecture is correctly defined and that the layers are properly connected.
        """
        # Layer 1 (Dense): 32 -> 16
        self.assertEqual(self.net.layers[0].output_shape(), (16,))
        # Layer 3 (Dense 2): 16 -> 8
        self.assertEqual(self.net.layers[2].output_shape(), (8,))
        # Layer 5 (Dense 3): 8 -> 1
        self.assertEqual(self.net.layers[4].output_shape(), (1,))

    def test_model_fit(self):
        """
        Test if the fit method completes training and generates a history.

        This test verifies that the neural network can be trained for the specified number of epochs
        and that the training history is correctly recorded, including loss values for each epoch.
        """
        self.net.fit(self.dataset)

        self.assertEqual(len(self.net.history), 10)

        self.assertIn('loss', self.net.history[1])

    def test_predict_output_range(self):
        """
        Test if the final Sigmoid activation maintains output values between 0 and 1.

        This test verifies that after training, the network's predictions are within the valid range
        for a sigmoid activation function, which is essential for binary classification tasks.
        """
        self.net.fit(self.dataset)

        predictions = self.net.predict(self.dataset)

        self.assertTrue(np.all(predictions >= 0) and np.all(predictions <= 1))

if __name__ == '__main__':
    unittest.main()