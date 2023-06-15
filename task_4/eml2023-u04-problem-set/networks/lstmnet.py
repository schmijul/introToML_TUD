
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size_1=32*7*7, hidden_size2=128, output_size=10):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size_1
        self.conv = nn.Conv2d(28, hidden_size_1, kernel_size=3, stride=1, padding=1)  # Update input channels
        self.lstm = nn.LSTM(hidden_size_1, hidden_size_1, batch_first=True)
        self.hiddenlayer = nn.Linear(hidden_size_1, hidden_size2)
        self.outputlayer = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)  # Reshape for LSTM input [batch_size, height, width, channels]
        x = x.reshape(x.size(0), x.size(1), -1)  # Flatten height and width dimensions
        _, (h_n, _) = self.lstm(x)
        x = self.hiddenlayer(h_n.squeeze(0))
        x = torch.relu(x)
        x = self.outputlayer(x)
        x = torch.softmax(x, dim=1)
        return x


def train_lstm(model, optimizer, criterion, train_loader, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.permute(0, 2, 3, 1)  # Permute dimensions for LSTM input
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}")

def test_lstm(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    predictions = []  # List to store predicted labels

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.permute(0, 2, 3, 1)  # Permute dimensions for LSTM input
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Append predicted labels to the list
            predictions.extend(predicted.tolist())

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")
    return accuracy