import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from autoencoder import Autoencoder

def train_autoencoder(model, train_loader, optimizer, criterion, num_epochs, loss_threshold):
    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch_data in train_loader:
            inputs, _ = batch_data
            optimizer.zero_grad()
            outputs, _ = model(inputs)
            
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        average_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {average_loss}")

        if average_loss < loss_threshold:
            print("Validation loss threshold achieved. Training stopped.")
            break

def save_model(model, filename):
    torch.save(model.state_dict(), filename)

def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = torchvision.datasets.FashionMNIST(
        root='./data', train=True, transform=transform, download=True
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

    autoencoder = Autoencoder()
    optimizer = optim.Adam(autoencoder.parameters())
    criterion = nn.MSELoss()

    num_epochs = 100
    validation_loss_threshold = 3e-2

    train_autoencoder(autoencoder, train_loader, optimizer, criterion, num_epochs, validation_loss_threshold)
    save_model(autoencoder, 'trained_autoencoder_model.pth')



if __name__ == "__main__":
    main()
