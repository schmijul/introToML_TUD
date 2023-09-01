import torch
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
from autoencoder import Autoencoder  # Import your Autoencoder class here

def visualize_image_and_reconstruction(autoencoder, data_loader, image_index):
    image, _ = next(iter(data_loader))
    original_image = image[image_index].squeeze().numpy()

    with torch.no_grad():
        reconstructed_image, _ = autoencoder(image[image_index].unsqueeze(0))

    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(original_image, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Reconstructed Image")
    plt.imshow(reconstructed_image.squeeze().numpy(), cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_dataset = torchvision.datasets.FashionMNIST(
        root='./data', train=True, transform=transform, download=True
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    autoencoder = Autoencoder()
    autoencoder.load_state_dict(torch.load('trained_autoencoder_model.pth'))
    autoencoder.eval()

    visualize_image_and_reconstruction(autoencoder, train_loader, image_index=1)
