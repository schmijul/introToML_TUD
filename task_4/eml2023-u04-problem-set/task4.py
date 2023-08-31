import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

from networks.mlp import *
from networks.cnn import *
from networks.lstmnet import *
from networks.gru import *


def get_data():
    # read MNIST training data
    train_data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    # read MNIST test data
    test_data = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )

    return train_data, test_data


# Set device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")








def plot_accuracy(model, test_loader):
    model.eval()
    correct_per_label = [0] * 10  # List to store the number of correct predictions per label
    total_per_label = [0] * 10  # List to store the total number of examples per label

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.view(images.size(0), -1).to(device)
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                correct_per_label[label] += correct[i].item()
                total_per_label[label] += 1

    accuracy_per_label = [correct_per_label[i] / total_per_label[i] for i in range(10)]

    # Plot accuracy per label
    labels = [str(i) for i in range(10)]
    plt.bar(labels, accuracy_per_label)
    plt.xlabel('Label')
    plt.ylabel('Accuracy')
    plt.title('Accuracy per Predicted Label')
    plt.show()

def plot_predictions(model, test_loader,num_examples = 5):
    figsize=(6*num_examples, 3)
    figure, axes = plt.subplots(1, num_examples,figsize=figsize)
    model.eval()
    with torch.no_grad():dataiter = iter(test_loader)
    images, labels = dataiter.__next__()
    images = images.to(device)
    labels = labels.tolist()
    outputs = model(images.view(images.size(0), -1))
    _, predicted = torch.max(outputs.data, 1)
    predictions = predicted.tolist()

    for i in range(num_examples):
        axes[i].imshow(images[i].squeeze().cpu(), cmap='gray')
        axes[i].set_title(f"Prediction: {predictions[i]} | True Label: {labels[i]}", fontsize=10)
        axes[i].axis('off')

    plt.show()


if __name__ == "__main__":
    # Set hyperparameters
    input_size = 28 * 28  # MNIST image size
    hidden_sizes = [32*7*7,128]  # hidden layer sizes
    output_size = 10  # Number of classes (digits 0-9)
    activation = nn.ReLU()  # Example activation function

    learning_rate = 0.01  # Learning rate for optimizer
    momentum = 0.9  
    batch_size = 128
    num_epochs = 2

    train_data, test_data = get_data()

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # Initialize model, optimizer, and loss function
    mlp = MLP(input_size, hidden_sizes, output_size, activation).to(device)
    optimizer = optim.SGD(mlp.parameters(), lr=learning_rate, momentum=momentum)
    criterion = nn.CrossEntropyLoss()
 
    # Train and test the model
    train_mlp(mlp, optimizer, criterion, train_loader, num_epochs)
    test_mlp(mlp, test_loader)

    plot_accuracy(mlp,test_loader)
    
    """
 
    convnet = CNN(output_size).to(device)
    optimizer = optim.SGD(convnet.parameters(), lr=learning_rate, momentum=momentum)
    criterion = nn.CrossEntropyLoss()

    train_cnn(convnet, optimizer, criterion, train_loader, num_epochs)
    test_cnn(convnet, test_loader)
    


    lstmnet = LSTM(output_size).to(device)
    optimizer = optim.SGD(lstmnet.parameters(), lr=learning_rate, momentum=momentum)
    criterion = nn.CrossEntropyLoss()

    train_lstm(lstmnet, optimizer, criterion, train_loader, num_epochs)
    test_lstm(lstmnet, test_loader)

    grunet = GRUNet(output_size).to(device)
    optimizer = optim.SGD(grunet.parameters(), lr=learning_rate, momentum=momentum)
    criterion = nn.CrossEntropyLoss()

    train_gru(grunet, optimizer, criterion, train_loader, num_epochs)
    test_gru(grunet, test_loader)

    """