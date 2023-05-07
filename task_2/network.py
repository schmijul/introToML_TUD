import numpy as np
from linearlayer import LinearLayer

class Network:
    def __init__(self, input_size, hidden_size, output_size):
        self.hidden_layer = LinearLayer(input_size, hidden_size)
        self.output_layer = LinearLayer(hidden_size, output_size)

    def forward(self, x):
        hidden_output = np.tanh(self.hidden_layer.forward(x))
        return self.output_layer.forward(hidden_output)

    def train(self, x_train, y_train, epochs, learning_rate):
        
        
        loss_history = []
        for t in range(epochs):
            # Forward pass
            y1 = np.tanh(self.hidden_layer.forward(x_train))
            yhat = self.output_layer.forward(y1)

            # Calculate loss
            emp_loss = np.sum((y_train - yhat) ** 2) / len(x_train)
            if t % 1000 == 999:
                print(t, emp_loss)

            # Backward pass
            delta2 = 2 * (yhat - y_train)
            grad_w2 = np.dot(delta2, y1.T) / len(x_train)
            grad_b2 = delta2.sum() / len(x_train)

            delta1 = np.dot(self.output_layer.weights.T, delta2) * (1 - y1 ** 2)
            grad_w1 = np.dot(delta1, x_train.T) / len(x_train)
            grad_b1 = delta1.sum() / len(x_train)

            # Update weights and biases
            self.output_layer.update_parameters(learning_rate, grad_w2, grad_b2)
            self.hidden_layer.update_parameters(learning_rate, grad_w1, grad_b1)

            loss_history.append(emp_loss)  # Add this line to store the loss in the list

        return loss_history  # 
    
    def predict(self, x):
        y1 = np.tanh(self.hidden_layer.forward(x))
        return self.output_layer.forward(y1)
    


    def get_weights(self):
        return {
        'w1': self.hidden_layer.weights,
        'w2': self.output_layer.weights
    }
        
    def get_biases(self):
        return {
        'b1': self.hidden_layer.biases,
        'b2': self.output_layer.biases
    }