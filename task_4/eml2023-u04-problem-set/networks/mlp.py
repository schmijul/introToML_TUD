import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Define MLP architecture
class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, activation):
        super(MLP, self).__init__()
        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.append(nn.Linear(input_size, hidden_sizes[0]))
        for i in range(1, len(hidden_sizes)):
            self.hidden_layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)
        self.activation = activation

    def forward(self, x):
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        x = self.output_layer(x)
        x = torch.softmax(x, dim=1)
        return x

# Define training function
def train_mlp(model, optimizer, criterion, train_loader, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.view(images.size(0), -1).to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}")



def test_mlp(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    predictions = []  # List to store predicted labels

    with torch.no_grad():
        dataiter = iter(test_loader)  # Create a new iterator for the test_loader
        for images, labels in dataiter:
            images = images.view(images.size(0), -1).to(device)
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

