import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms

from cnn import CNN

# Create dataloaders
def create_data_loaders(batch_size):
    transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                    transforms.RandomRotation(90),
                                    transforms.ToTensor()])
    emnist_train_dataset = torchvision.datasets.EMNIST(root='data', split='letters', train=True, transform=transform, download=True)
    emnist_test_dataset = torchvision.datasets.EMNIST(root='data', split='letters', train=False, transform=transform, download=True)
    
    train_loader = torch.utils.data.DataLoader(emnist_train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(emnist_test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

# Training function
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}')

# Testing function
def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    accuracy = correct / total * 100
    print(f'Test Accuracy: {accuracy:.2f}%')

def main():
    batch_size = 64
    train_loader, test_loader = create_data_loaders(batch_size)
    
    cnn_model = CNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)
    
    num_epochs = 10
    train_model(cnn_model, train_loader, criterion, optimizer, num_epochs)
    test_model(cnn_model, test_loader)

if __name__ == "__main__":
    main()
