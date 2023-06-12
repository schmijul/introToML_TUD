import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math


class LinearLayer:
    def __init__(self, input_dim, output_dim):
        self.weights = np.random.randn(output_dim, input_dim)
        self.biases = np.random.randn(output_dim, 1)

    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = np.matmul(self.weights, inputs) + self.biases
        return self.outputs


class SigmoidLayer:
    def forward(self, inputs):
        self.outputs = 1 / (1 + np.exp(-inputs))
        return self.outputs


class SoftmaxLayer:
    def forward(self, inputs):
        shifted_inputs = inputs - np.max(inputs, axis=0, keepdims=True)
        exp_values = np.exp(shifted_inputs)
        self.outputs = exp_values / np.sum(exp_values, axis=0, keepdims=True)
        return self.outputs


class TwoLayerNet:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.linear1 = LinearLayer(input_dim, hidden_dim)
        self.sigmoid = SigmoidLayer()
        self.linear2 = LinearLayer(hidden_dim, output_dim)
        self.softmax = SoftmaxLayer()

    def forward(self, inputs):
        hidden = self.linear1.forward(inputs)
        activated = self.sigmoid.forward(hidden)
        outputs = self.linear2.forward(activated)
        y_hat = self.softmax.forward(outputs)
        return y_hat


def calculate_loss_accuracy(y_hat, y_T, labels):
    size_data_set = y_hat.shape[1]
    labels_hat = np.argmax(y_hat, axis=0)
    emp_loss = (-y_T * np.log(y_hat)).sum() / size_data_set
    acc = (labels_hat == labels).sum() / size_data_set
    return acc, emp_loss


def backpropagation(model, x_train, y_train, learning_rate):
    size_data_set = x_train.shape[0]
    y_hat = model.forward(x_train)
    delta2 = y_hat - y_train
    grad_w2 = np.matmul(delta2, model.sigmoid.outputs.T) / size_data_set
    grad_b2 = np.mean(delta2, axis=1, keepdims=True)  # Compute mean along axis 1
    delta1 = np.matmul(model.linear2.weights.T, delta2) * (model.sigmoid.outputs * (1 - model.sigmoid.outputs))
    grad_w1 = np.matmul(delta1, x_train.T) / size_data_set
    grad_b1 = np.mean(delta1, axis=1, keepdims=True)
    model.linear2.weights -= learning_rate * grad_w2
    model.linear2.biases -= learning_rate * grad_b2
    model.linear1.weights -= learning_rate * grad_w1
    model.linear1.biases -= learning_rate * grad_b1
    return model.linear1.weights, model.linear1.biases, model.linear2.weights, model.linear2.biases


def train(model, learning_rate, x_T, y_T, labels, size_data_set, epochs, print_loss_every, batch_size):
    best_loss = np.inf
    best_weights_biases = None
    loss_history = []
    acc_history = []

    for t in range(epochs):
        for i in range(0, size_data_set, batch_size):
            x_batch = x_T[:, i:i+batch_size]
            y_batch = y_T[:, i:i+batch_size]

            y_hat = model.forward(x_batch)
            model.linear1.weights, model.linear1.biases, model.linear2.weights, model.linear2.biases = backpropagation(
                model, x_batch, y_batch, learning_rate)

        if (t + 1) % print_loss_every == 0:
            acc, emp_loss = calculate_loss_accuracy(y_hat, y_batch, labels[i:i+batch_size])
            print("Epoch:", t + 1, "Model accuracy:", acc * 100, "%", "Emp. loss:", emp_loss)
            loss_history.append(emp_loss)
            acc_history.append(acc)

            if emp_loss < best_loss:
                best_loss = emp_loss
                best_weights_biases = {
                    'linear1_weights': model.linear1.weights,
                    'linear1_biases': model.linear1.biases,
                    'linear2_weights': model.linear2.weights,
                    'linear2_biases': model.linear2.biases
                }

    final_acc, final_loss = calculate_loss_accuracy(y_hat, y_batch, labels[i:i+batch_size])
    print("Final model accuracy:", final_acc * 100, "%", "Final emp. loss:", final_loss)

    return best_weights_biases, loss_history, acc_history


def main():
    labeled_data_pd = pd.read_csv("labeled-dataset-3d-rings.txt", delimiter=',', dtype=np.float32)
    labeled_data_np = np.array(labeled_data_pd)
    size_data_set = len(labeled_data_np)
    np.random.shuffle(labeled_data_np)
    x = labeled_data_np[:, 0:3].T
    labels = labeled_data_np[:, 3:].astype(int).reshape((-1,))
    N_cl = np.max(labels).astype(int) + 1
    y = np.zeros((size_data_set, N_cl))
    y[np.arange(size_data_set), labels] = 1
    y_T = y.T

    input_dim = x.shape[0]
    hidden_dim = 13
    output_dim = N_cl
    epochs = 100
    print_loss_every = 10
    learning_rate = 1
    batch_size = 5*32

    model = TwoLayerNet(input_dim, hidden_dim, output_dim)

    best_weights_biases, loss_history, acc_history = train(
        model, learning_rate, x, y_T, labels, size_data_set, epochs, print_loss_every, batch_size)

    model.linear1.weights = best_weights_biases['linear1_weights']
    model.linear1.biases = best_weights_biases['linear1_biases']
    model.linear2.weights = best_weights_biases['linear2_weights']
    model.linear2.biases = best_weights_biases['linear2_biases']

    # Test the trained model
    test_data_pd = pd.read_csv("labeled-dataset-3d-rings.txt", delimiter=',', dtype=np.float32)
    test_data_np = np.array(test_data_pd)
    test_size = len(test_data_np)
    test_x = test_data_np[:, 0:3].T
    test_labels = test_data_np[:, 3:].astype(int).reshape((-1,))
    test_y = np.zeros((test_size, N_cl))
    test_y[np.arange(test_size), test_labels] = 1
    test_y_T = test_y.T

    test_y_hat = model.forward(test_x)
    test_acc, test_loss = calculate_loss_accuracy(test_y_hat, test_y_T, test_labels)
    print("Test accuracy:", test_acc * 100, "%", "Test loss:", test_loss)

    # Plot loss and accuracy history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss History")
    plt.subplot(1, 2, 2)
    plt.plot(acc_history)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy History")
    plt.tight_layout()

    # Plot actual 3D prediction
    prediction = np.argmax(test_y_hat, axis=0)
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(test_x[0, :], test_x[1, :], test_x[2, :], c=test_labels)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Ground Truth')

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(test_x[0, :], test_x[1, :], test_x[2, :], c=prediction)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('Actual Prediction')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
