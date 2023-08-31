import math
import numpy as np
import matplotlib.pyplot as plt

class LinearLayer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size)
        self.biases = np.random.randn(output_size, 1)
        
    def forward(self, input_data):
        output_data = np.dot(self.weights, input_data) + self.biases
        return output_data
    
    def backward(self, input_data, output_error, learning_rate):
        input_error = np.dot(self.weights.T, output_error)
        weights_error = np.dot(output_error, input_data.T)
        biases_error = np.sum(output_error, axis=1, keepdims=True)
        
        self.weights -= learning_rate * weights_error
        self.biases -= learning_rate * biases_error
        
        return input_error

class NeuralNetwork:
    def __init__(self, layer_sizes):
        self.layers = []
        for i in range(len(layer_sizes)-1):
            self.layers.append(LinearLayer(layer_sizes[i], layer_sizes[i+1]))
        
    def forward(self, input_data):
        output_data = input_data
        for layer in self.layers:
            output_data = layer.forward(output_data)
        return output_data
    
    def backward(self, input_data, output_error, learning_rate):
        error = output_error
        for layer in reversed(self.layers):
            error = layer.backward(input_data, error, learning_rate)
        return error
    
    
#-----------------------------------------------------------------------------
# specify functions to be approximated
funSelect = 3
def funToApprox(x):
    functions = {
        1: np.power(x,2), # quadratic function
        2: 7/2 *np.cos(math.pi / 4 * x)**2, # raised cosine of previous problem
        3: np.exp(x), # exponential function
        4: np.cos(math.pi * x)**2
    }
    return functions.get(funSelect)

#-----------------------------------------------------------------------------
# data points for training 
size_data_set = 41
#size_data_set = 121 #for funSelect = 4
x_train = np.linspace(-2,2,size_data_set).reshape(1,size_data_set)
# data points for validation
size_data_set_ext = 51
#size_data_set_ext = 151 #for funSelect = 4
x_test = np.linspace(-2.5,2.5,size_data_set_ext)

#-----------------------------------------------------------------------------
# calculate 'true' output values
y = funToApprox(x_train)

#-----------------------------------------------------------------------------
# number of hidden notes
N1 = 3
# number of epochs for training
epochs = 50000

learning_rate = 1e-2

# initialize neural network
network = NeuralNetwork([1, N1, 1])

#-----------------------------------------------------------------------------
# train the network by updating weights according to gradient descent method
for t in range(epochs):
    # forward pass
    y1 = np.tanh(network.forward(x_train))

    # calculate loss
    loss = ((y-y1)**2).sum() / size_data_set
    if t % 1000 == 0:
        print(t, loss)

    # backward pass
    output_error = 2*(y1-y)/size_data_set
    network.backward(x_train, output_error, learning_rate)

# print learned weights/biases and empirical loss after training    
for i, layer in enumerate(network.layers):
    print(f'Learned weights/biases for layer {i+1}:')
    print(f' weights: {layer.weights}')
    print(f' biases: {layer.biases}')

print(f'Empirical loss: {loss}')

#-----------------------------------------------------------------------------
# calculate 'true' output and network output for learned weights/biases
# for a range larger than the range used for training

# 'true' output for validation data set
y = funToApprox(x_test)

# network output for validation data set
y1 = np.tanh(network.forward(x_test)).reshape(size_data_set_ext)

#output empirical loss for validation data set
loss = ((y-y1)**2).sum() / size_data_set_ext
print(f'Empirical loss for extended x-range: {loss}')

# plot 'true' output y(x) and approximated output yhat(x) for optimal weights
plt.figure()
plt.plot(x_test,y1)
plt.plot(x_test,y)

# set axes labels
plt.ylabel('$y(x)$,$\\hat{y}(x)$')
plt.xlabel('x')
plt.grid(True)

#show plot
plt.show()

