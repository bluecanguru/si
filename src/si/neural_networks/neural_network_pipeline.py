import numpy as np
#from si.io.csv_file import read_csv
from si.data.dataset import Dataset
from si.neural_networks.neural_network import NeuralNetwork
from si.neural_networks.layers import DenseLayer
from si.neural_networks.activation import ReLUActivation, SigmoidActivation
from si.neural_networks.optimizers import SGD
from si.neural_networks.losses import BinaryCrossEntropy
from si.metrics.accuracy import accuracy
from si.model_selection.cross_validate import k_fold_cross_validation

#dataset = read_csv('../datasets/breast_bin/breast-bin.csv', label=True)
# random dataset with 32 features
X_random = np.random.randn(100, 32)
y_random = np.random.randint(0, 2, 100)
dataset = Dataset(X_random, y_random)

net = NeuralNetwork(
    epochs=100, 
    batch_size=16, 
    optimizer=SGD, 
    learning_rate=0.01, 
    verbose=True,
    loss=BinaryCrossEntropy, 
    metric=accuracy
)

# add Layers reducing units to half
# layer 1: 32 inputs -> 16 units
net.add(DenseLayer(n_units=16, input_shape=(32,)))
net.add(ReLUActivation())

# layer 2: 16 units -> 8 units
net.add(DenseLayer(n_units=8))
net.add(ReLUActivation())

# output layer: 8 units -> 1 unit - Binary
net.add(DenseLayer(n_units=1))
net.add(SigmoidActivation())

scores = k_fold_cross_validation(model=net, dataset=dataset, cv=5)

print("-" * 30)
print(f"K-Fold Cross Validation Scores: {scores}")
print(f"Mean Accuracy: {np.mean(scores):.4f}")
print(f"Standard Deviation: {np.std(scores):.4f}")