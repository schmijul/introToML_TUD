import numpy as np


class LinearLayer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size)
        self.biases = np.random.randn(output_size, 1)

    def forward(self, x):
        return np.dot(self.weights, x) + self.biases
    def update_parameters(self, learning_rate, grad_weights, grad_biases):
        self.weights -= learning_rate * grad_weights
        self.biases -= learning_rate * grad_biases
