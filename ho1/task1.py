import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


       
class LinearLayer:
    def __init__(self, input_dim, output_dim):
        self.weights = np.random.randn(input_dim, output_dim)
        self.biases = np.random.randn(output_dim)

    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = np.dot(inputs, self.weights) + self.biases
        return self.outputs

    def backward(self, d_outputs):
        self.d_weights = np.dot(self.inputs.T, d_outputs)
        self.d_biases = np.sum(d_outputs, axis=0, keepdims=False)
        self.d_inputs = np.dot(d_outputs, self.weights.T)
        return self.d_inputs

    def update(self, lr):
        self.weights -= lr * self.d_weights
        self.biases -= lr * self.d_biases

def target_fct(x1, x2):
    return ((x1**2 + x2**2)**(7/12)).reshape(-1, 1)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def lossfct(y_hat, y_true):
    return np.sum((y_hat - y_true)**2) / y_true.shape[0]
 

def plot_results(x, y_true, y_pred):
    fig = plt.figure(figsize=(12, 6))

    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(x[:, 0], x[:, 1], y_true)
    ax1.set_title("True Function")

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(x[:, 0], x[:, 1], y_pred)
    ax2.set_title("NN Aproximation")

    plt.show()

def train_network(x_train, y_train, layer1, layer2, init_lr, epochs, target_loss=None):
    lr = init_lr
    best_loss = np.inf
    best_weights = None
    best_biases = None
    losses = []
    last_lr_checkpoint = 0
    for epoch in range(epochs):
        # Forward pass
        layer1_outputs = sigmoid(layer1.forward(x_train))
        layer2_outputs = layer2.forward(layer1_outputs)

        y_pred = layer2_outputs       
        
        # Compute the loss
        loss = lossfct(y_pred,y_train)
        losses.append(loss)

        if epoch % 1000 == 0:
            print(f"Epoch {epoch},Current loss: {np.round(loss,6)}, Best loss: {np.round(best_loss,6)}, Learning rate: {lr}")
            
            
        if (epoch % 3000 == 0) and (epoch != 0):
            # If the loss has not decreased for 3000 epochs, decrease the learning rate
            if np.min(losses[last_lr_checkpoint:epoch]) <= best_loss:
                last_lr_checkpoint = epoch
                lr *= 0.09
                print(f"Learning rate decreased to {lr}")
                                    
        # If the loss is less than the best loss, update the best loss and save the weights and biases
        if loss < best_loss:
            best_loss = loss
            best_weights = [layer1.weights.copy(), layer2.weights.copy()]
            best_biases = [layer1.biases.copy(), layer2.biases.copy()]
           

        # If loss is less than the threshold, stop training
        if target_loss:
            if best_loss <= target_loss:
                print(f"Training stopped at epoch {epoch}, loss: {loss}")
                return best_weights, best_biases, losses

        # Backward pass
        d_layer2_outputs = 2 * (y_pred - y_train) / y_train.shape[0]
        d_layer1_outputs = layer2.backward(d_layer2_outputs)
        layer1.backward(sigmoid_derivative(layer1_outputs) * d_layer1_outputs)

        # Update the weights and biases
        layer1.update(lr)
        layer2.update(lr)

    return best_weights, best_biases, losses

# Define the main function
def main():
    lr = 1e-1
    # Create a dataset
    x1 = np.arange(-5, 5, 0.1)
    x2 = np.arange(-5, 5, 0.1)
    x_train = np.array(np.meshgrid(x1, x2)).T.reshape(-1, 2)
    y_train = target_fct(x_train[:, 0], x_train[:, 1])

    # Number of nodes in the hidden layer
    N1 = 50

    # Create the layers
    layer1 = LinearLayer(2, N1)
    layer2 = LinearLayer(N1, 1)

    # Train the network
    best_weights, best_biases, losses = train_network(x_train=x_train, 
                                                      y_train=y_train, 
                                                      layer1=layer1,
                                                      layer2=layer2, 
                                                      init_lr=lr, 
                                                      epochs=100000,
                                                      target_loss=0.002)

    # Use the best weights and biases to make predictions
    layer1.weights, layer2.weights = best_weights
    layer1.biases, layer2.biases = best_biases
    layer1_outputs = sigmoid(layer1.forward(x_train))
    layer2_outputs = layer2.forward(layer1_outputs)

    # Plot the results
    plot_results(x_train, y_train, layer2_outputs)

# Run the main function
if __name__ == "__main__":
    main()