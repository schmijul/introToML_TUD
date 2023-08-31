
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

def train_autoencoder(model, trainloader, epochs, criterion=nn.MSELoss()):
    optimizer = optim.Adam(model.parameters())
    
    for epoch in range(epochs):
        for images, _ in trainloader:  # We don't need labels
            # Print shapes of images and outputs
            
            optimizer.zero_grad()
            
            outputs = model(images)
            print(f"Input shape: {images.shape}, Output shape: {outputs.shape}")
            loss = criterion(outputs, images)
            
            loss.backward()
            optimizer.step()
            # Print shapes of images and outputs
            print(f"Input shape: {images.shape}, Output shape: {outputs.shape}")
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

            


if __name__ == "__main__":



    BATCH_SIZE = 64
    NUM_EPOCHS = 2

    NUM_INPUT_CHANNELS = 1
    ENCODER_CHANNELS = [64, 32, 16, 2]
    DECODER_CHANNELS = [16, 32, 64, 1]
    LATENT_DIM = 3
    KERNEL_SIZE = 3
    PADDING = 1
    ENCODER_STRIDES = [2, 2, 1, 1]
    DECODER_STRIDES = [1, 1, 2, 2]


    trainloader, _ = load_fashion_mnist(batch_size=BATCH_SIZE)
    
    # Initialize the model
    cnnautoencoder = CNNAutoencoder(LATENT_DIM)
    
    # Train the model
    train_autoencoder(model=cnnautoencoder, trainloader= trainloader, epochs=NUM_EPOCHS)