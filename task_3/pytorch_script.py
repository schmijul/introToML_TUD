import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class TwoLayerPerceptron(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TwoLayerPerceptron, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.softmax(x)
        return x

def load_data(file):
    data = pd.read_csv(file, delimiter = ',', dtype = np.float32)
    data = torch.tensor(data.values)
    labels = nn.functional.one_hot(data[:, 2].long(), num_classes=3)
    data = data[:, :2]
    return data, labels

def find_best_learning_rate(data_train, labels_train, data_val, labels_val, model, learning_rates, epochs):
    criterion = nn.CrossEntropyLoss()
    best_learning_rate = None
    best_accuracy = 0

    for learning_rate in learning_rates:
        model_tmp = TwoLayerPerceptron(NUM_INPUT_NODES, NUM_HIDDEN_NODES, NUM_OUTPUT_NODES)
        model_tmp.load_state_dict(model.state_dict())
        optimizer = optim.SGD(model_tmp.parameters(), lr=learning_rate)
        train_loader = DataLoader(TensorDataset(data_train, labels_train.argmax(dim=1)), batch_size=BATCH_SIZE, shuffle=True)
        
        for epoch in range(epochs):
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model_tmp(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        outputs_val = model_tmp(data_val)
        _, predicted = torch.max(outputs_val, 1)
        accuracy = (predicted == labels_val.argmax(dim=1)).float().mean().item()
        print(f"Learning Rate: {learning_rate}, Accuracy: {accuracy}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_learning_rate = learning_rate

    return best_learning_rate

def train_model(data, labels, model, learning_rate, epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    train_loader = DataLoader(TensorDataset(data, labels.argmax(dim=1)), batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

def plot_classification(data, labels, model):
    

    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros(X.shape)
    
    prediction_data = torch.tensor(np.dstack((X, Y)).reshape(-1, 2))
    prediction_loader = DataLoader(TensorDataset(prediction_data), batch_size=64, shuffle=False)
    
    Z = np.zeros(X.shape)
    with torch.no_grad():
        for inputs in prediction_loader:
            inputs = inputs[0].type(torch.FloatTensor)
            outputs = model(inputs)


            _, predicted = torch.max(outputs, 1)
            predicted = predicted.numpy()
            Z[np.unravel_index(range(len(predicted)), X.shape)] = predicted
    
    plt.contourf(X, Y, Z, cmap=plt.cm.Spectral, alpha=0.8)
    plt.scatter(data[:, 0], data[:, 1], c=labels.argmax(dim=1), s=40, cmap=plt.cm.Spectral)
    plt.show()


def main():
    # Load Data
    data, labels = load_data(DATA_FILE)
    
    # split Data into Train and Validation Data 
    data_train = data[:int(0.8 * len(data))]
    labels_train = labels[:int(0.8 * len(labels))]

    data_val = data[int(0.8 * len(data)):]
    labels_val = labels[int(0.8 * len(labels)):]
    
    # Define Model
    model = TwoLayerPerceptron(NUM_INPUT_NODES, NUM_HIDDEN_NODES, NUM_OUTPUT_NODES)

    # Find best learning rate
    learning_rates_to_test = [10**(-x) for x in range(1, 6)]
    best_learning_rate = find_best_learning_rate(data_train, labels_train, data_val, labels_val, model=model, learning_rates=learning_rates_to_test, epochs =EPOCHS_FOR_LEARNING_RATE_TEST)

    # Train with best learning rate
    train_model(data, labels, model, best_learning_rate, NUM_EPOCHS)

    # save model
    torch.save(model.state_dict(), "model.pt")
    # Visualize Results
    
    plot_classification(data, labels, model)

if __name__ == "__main__":
    # data_file
    DATA_FILE = "labeled-dataset-tud-logo.txt"
    # Define Constants
    NUM_INPUT_NODES = 2
    NUM_HIDDEN_NODES = 25
    NUM_OUTPUT_NODES = 3
    NUM_EPOCHS = 10
    BATCH_SIZE = 64
    EPOCHS_FOR_LEARNING_RATE_TEST = 10
    main()
