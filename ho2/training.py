
import matplotlib.pyplot as plt
import torchvision

from torch.utils.data import DataLoader

import torch
import torch.nn as nn
import torch.optim as optim


from models.autoencoders import CNNAutoencoder


def load_fashion_mnist(batch_size, show_dataset=False):

    """"
    Load Fashion MNIST dataset and return train and test loaders.
    
    params
    ------  
    batch_size: int
        Batch size for train and test loaders.
    show_dataset: bool
        If True, show a sample image from the dataset.

    returns
    -------
    trainloader: torch.utils.data.DataLoader
        Train loader.   
    testloader: torch.utils.data.DataLoader
        Test loader. 
    """
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize((0.5,), (0.5,))])
    
    trainset = torchvision.datasets.FashionMNIST('./data', download=True, train=True, transform=transform)
    testset = torchvision.datasets.FashionMNIST('./data', download=True, train=False, transform=transform)
    
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    
    if show_dataset:
        dataiter = iter(trainloader)
        images, labels = next(dataiter)

        plt.imshow(images[0].numpy().squeeze(), cmap='gray_r')
        plt.title(f'Label: {labels[0]}')
        plt.show()
        
    return trainloader, testloader




def initialize_weights(model):
    
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

def train_autoencoder(model, lr, trainloader, testloader, epochs,target_loss=0.03):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        for images, _ in trainloader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()
            
        # Calculate validation loss
        validation_loss = 0.0
        for images, _ in testloader:
            outputs = model(images)
            validation_loss += criterion(outputs, images).item()
        
        validation_loss /= len(testloader)
        print(f"Epoch: {epoch+1}/{epochs}, Validation Loss: {validation_loss:.4f}")
        
        # Stop training if validation loss is below 0.03
        if validation_loss < target_loss:
            print("Stopping training. Validation loss below 0.03 achieved.")
            break

    return model

if __name__ == "__main__":
    BATCH_SIZE = 64
    NUM_EPOCHS = 50
    LR = 0.03
    LATENT_DIM = 3

    trainloader, testloader = load_fashion_mnist(batch_size=BATCH_SIZE)
    
    # Initialize the model
    cnnautoencoder = CNNAutoencoder(LATENT_DIM)
    #initialize_weights(cnnautoencoder)
    
    # Train the model
    train_autoencoder(model=cnnautoencoder,
                        lr=LR,
                        trainloader=trainloader, 
                        testloader=testloader,
                        epochs=NUM_EPOCHS)

    # save model
    torch.save(cnnautoencoder.state_dict(), 'models/cnnautoencoder.pth')
    print("Model Saved Successfully")