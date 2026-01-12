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

    def setUp(self):
       
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
        
        # Layer 1 (Dense): 32 -> 16
        self.assertEqual(self.net.layers[0].output_shape(), (16,))
        # Layer 3 (Dense 2): 16 -> 8
        self.assertEqual(self.net.layers[2].output_shape(), (8,))
        # Layer 5 (Dense 3): 8 -> 1
        self.assertEqual(self.net.layers[4].output_shape(), (1,))

    def test_model_fit(self):
        
        self.net.fit(self.dataset)

        self.assertEqual(len(self.net.history), 10)

        self.assertIn('loss', self.net.history[1])

    def test_predict_output_range(self):
        
        self.net.fit(self.dataset)

        predictions = self.net.predict(self.dataset)

        self.assertTrue(np.all(predictions >= 0) and np.all(predictions <= 1))

if __name__ == '__main__':
    unittest.main()