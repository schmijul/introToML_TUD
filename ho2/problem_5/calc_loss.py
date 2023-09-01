import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from autoencoder import Autoencoder  

def calculate_test_loss(model, data_loader, criterion):
    total_loss = 0.0
    with torch.no_grad():
        for batch_data in data_loader:
            inputs, _ = batch_data
            outputs, _ = model(inputs)
            loss = criterion(outputs, inputs)
            total_loss += loss.item()
    return total_loss / len(data_loader)

def calculate_class_mse(model, data_loader, criterion):
    class_mse = torch.zeros(10)
    class_counts = torch.zeros(10)

    with torch.no_grad():
        for batch_data in data_loader:
            inputs, labels = batch_data
            outputs, _ = model(inputs)
            loss = criterion(outputs, inputs)

            class_mse[labels] += loss.sum()
            class_counts[labels] += inputs.size(0)

    class_mse /= class_counts
    return class_mse

def visualize_reconstructions(model, data_loader):
    num_samples = 1
    fig, axes = plt.subplots(10, num_samples, figsize=(num_samples, 10))

    with torch.no_grad():
        for i in range(10):
            class_mask = data_loader.dataset.targets == i
            class_samples = data_loader.dataset.data[class_mask]
            class_samples = class_samples[:num_samples].unsqueeze(1).float() / 255.0

            reconstructed_samples, _ = model(class_samples)
            
            for j in range(num_samples):
                axes[i, j].imshow(class_samples[j, 0], cmap='gray')
                axes[i, j].axis('off')
                axes[i + 1, j].imshow(reconstructed_samples[j, 0], cmap='gray')
                axes[i + 1, j].axis('off')

    plt.suptitle('Original and Reconstructed Images for Each Class')
    plt.show()

def main():
    autoencoder = Autoencoder()
    autoencoder.load_state_dict(torch.load('trained_autoencoder_model.pth'))
    autoencoder.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    test_dataset = torchvision.datasets.FashionMNIST(
        root='./data', train=False, transform=transform, download=True
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

    criterion = nn.MSELoss()

    test_loss = calculate_test_loss(autoencoder, test_loader, criterion)
    class_mse = calculate_class_mse(autoencoder, test_loader, criterion)

    print(f"Test Loss: {test_loss:.4f}")
    print("Class-wise MSE:")
    for i in range(10):
        print(f"Class {i}: {class_mse[i]:.4f}")
    # total average mse:
    print(f"Total Average MSE: {class_mse.mean():.4f}")
    # write into txt file:
    with open('test_loss.txt', 'w') as f:
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write("Class-wise MSE:\n")
        for i in range(10):
            f.write(f"Class {i}: {class_mse[i]:.4f}\n")
        f.write(f"Total Average MSE: {class_mse.mean():.4f}\n")

    visualize_reconstructions(autoencoder, test_loader)

if __name__ == "__main__":
    main()
