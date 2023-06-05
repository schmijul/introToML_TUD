import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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


class NeuralNetwork:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.linear1 = LinearLayer(input_dim, hidden_dim)
        self.sigmoid = SigmoidLayer()
        self.linear2 = LinearLayer(hidden_dim, output_dim)
        self.softmax = SoftmaxLayer()

    def forward_pass(self, inputs):
        hidden = self.linear1.forward(inputs)
        activated = self.sigmoid.forward(hidden)
        outputs = self.linear2.forward(activated)
        y_hat = self.softmax.forward(outputs)
        return activated, y_hat

    def calculate_loss_accuracy(self, y_hat, y_T, labels, size_data_set):
        labels_hat = np.argmax(y_hat, axis=0)
        emp_loss = (-y_T * np.log(y_hat)).sum() / size_data_set
        acc = (labels_hat == labels).sum() / size_data_set
        return acc, emp_loss


class Backpropagation:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def update_weights_biases(self, nn, activated, y_hat, x_T, y_T, size_data_set):
        delta2 = y_hat - y_T
        grad_w2 = np.matmul(delta2, activated.T) / size_data_set
        grad_b2 = np.mean(delta2, axis=1, keepdims=True)
        delta1 = np.matmul(nn.linear2.weights.T, delta2) * (activated * (1 - activated))
        grad_w1 = np.matmul(delta1, x_T.T) / size_data_set
        grad_b1 = np.mean(delta1, axis=1, keepdims=True)
        nn.linear2.weights -= self.learning_rate * grad_w2
        nn.linear2.biases -= self.learning_rate * grad_b2
        nn.linear1.weights -= self.learning_rate * grad_w1
        nn.linear1.biases -= self.learning_rate * grad_b1


def train(nn, backpropagation, x_T, y_T, labels, size_data_set, epochs, print_loss_every):
    for t in range(epochs):
        activated, y_hat = nn.forward_pass(x_T)
        backpropagation.update_weights_biases(nn, activated, y_hat, x_T, y_T, size_data_set)
        if (t + 1) % print_loss_every == 0:
            acc, emp_loss = nn.calculate_loss_accuracy(y_hat, y_T, labels, size_data_set)
            print("Epoch:", t + 1, "Model accuracy:", acc * 100, "%", "Emp. loss:", emp_loss)
    final_acc, final_loss = nn.calculate_loss_accuracy(y_hat, y_T, labels, size_data_set)
    print("Final model accuracy:", final_acc * 100, "%", "Final emp. loss:", final_loss)


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


    data_colors = ["tab:green", "tab:orange", "tab:blue", "tab:red", "tab:purple", "tab:cyan", "tab:gray"]
    for cl in range(N_cl):
        print("Number of training data points in class", cl, ":", len(x[:, labels == cl].T))
        xcl_np = x[:, labels == cl]
      
    input_dim = x.shape[0]
    hidden_dim = 12
    output_dim = N_cl
    epochs = 1000
    print_loss_every = 100
    learning_rate = 1

    nn = NeuralNetwork(input_dim, hidden_dim, output_dim)
    backpropagation = Backpropagation(learning_rate)

    train(nn, backpropagation, x, y_T, labels, size_data_set, epochs, print_loss_every)


if __name__ == "__main__":
    main()
