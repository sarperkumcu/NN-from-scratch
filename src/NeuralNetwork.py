"""

 NeuralNetworkClass.py  (author: Anson Wong / git: ankonzoid)

"""
import numpy as np
import random
import math

class NeuralNetwork:

    def __init__(self, input_dim=None, output_dim=None, hidden_layers=None):
        if (input_dim is None) or (output_dim is None) or (hidden_layers is None):
            raise Exception("Invalid arguments given to Neural Network")
        self.input_dim = input_dim # number of input nodes
        self.output_dim = output_dim # number of output nodes
        self.hidden_layers = hidden_layers # number of hidden nodes @ each layer
        self.network = self._build_network()

    # Train network
    def train(self, X, y, eta=0.5, n_epochs=1000):
        for epoch in range(n_epochs):
            for (xi, yi) in zip(X, y):
                self._forward_pass(xi) # forward pass (update node "output")
                yi_onehot = self._one_hot_encoding(yi, self.output_dim) # one-hot target
                self._backward_pass(yi_onehot) # backward pass error (update node "delta")
                self._update_weights(xi, eta=eta) # update weights (update node "weight")

    # Predict using argmax of logits
    def predict(self, X):
        ypred = [np.argmax(self._forward_pass(xi)) for xi in X]
        return np.array(ypred, dtype=np.int)

    # ==============================
    #
    # Internal functions
    #
    # ==============================

    # Build fully-connected neural network (no bias terms)
    def _build_network(self):

        def _layer(input_dim, output_dim):
            # Create a single fully-connected layer
            layer = []
            for i in range(output_dim):
                weights = [random.random() for j in range(input_dim)] # sample N(0,1)
                node = {"weights": weights, # list of scalars
                        "output": None, # scalar
                        "delta": None} # scalar
                layer.append(node)
            return layer

        # Stack layers (input -> hidden -> output)
        network = []
        if len(self.hidden_layers) == 0:
            network.append(_layer(self.input_dim, self.output_dim))
        else:
            network.append(_layer(self.input_dim, self.hidden_layers[0]))
            for i in range(1, len(self.hidden_layers)):
                network.append(_layer(self.hidden_layers[i-1], self.hidden_layers[i]))
            network.append(_layer(self.hidden_layers[-1], self.output_dim))

        return network

    # Forward-pass (updates node['output'])
    def _forward_pass(self, x):
        activation_fn = self._sigmoid
        x_in = x
        for layer in self.network:
            x_out = []
            for node in layer:
                node['output'] = activation_fn(self._dotprod(node['weights'], x_in))
                x_out.append(node['output'])
            x_in = x_out # set output as next input
        return x_in

    # Backward-pass (updates node['delta'], L2 loss is assumed)
    def _backward_pass(self, target_onehot):
        activation_fn_derivative = self._sigmoid_derivative # sig' = f(sig)
        n_layers = len(self.network)
        for i in reversed(range(n_layers)): # traverse backwards
            if i == n_layers - 1:
                # Difference between logits and one-hot target
                for j, node in enumerate(self.network[i]):
                    err = node['output'] - target_onehot[j]
                    node['delta'] = err * activation_fn_derivative(node['output'])
            else:
                # Weighted sum of deltas from upper layer
                for j, node in enumerate(self.network[i]):
                    err = sum([node_['weights'][j] * node_['delta'] for node_ in self.network[i+1]])
                    node['delta'] = err * activation_fn_derivative(node['output'])

    # Update weights (updates node['weight'])
    def _update_weights(self, x, eta=0.3):
        for i, layer in enumerate(self.network):
            # Choose previous layer output to update current layer weights
            if i == 0:
                inputs = x
            else:
                inputs = [node_['output'] for node_ in self.network[i-1]]
            # Update weights
            for node in layer:
                for j, input in enumerate(inputs):
                    # dW = - learning_rate * errors * transfer' * input
                    node['weights'][j] += - eta * node['delta'] * input

    # Dot product
    def _dotprod(self, a, b):
        x = sum([ai * bi for (ai, bi) in zip(a, b)])
        return x

    # Sigmoid (activation function)
    def _sigmoid(self, x):
        return 1.0/(1.0+math.exp(-x))

    # Sigmoid derivative
    def _sigmoid_derivative(self, sigmoid):
        return sigmoid*(1.0-sigmoid)

    # One-hot encoding
    def _one_hot_encoding(self, idx, output_dim):
        x = np.zeros(output_dim, dtype=np.int)
        x[idx] = 1
        return x