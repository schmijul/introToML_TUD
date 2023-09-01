import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torchvision
import torchvision.transforms as transforms
from autoencoder import Autoencoder 

def get_latent_representations(model, data_loader):
    latent_representations = []
    class_labels = []

    with torch.no_grad():
        for batch_data in data_loader:
            inputs, labels = batch_data
            encoded = model.encoder(inputs)
            latent_representations.append(encoded)
            class_labels.append(labels)

    latent_representations = torch.cat(latent_representations, dim=0)
    class_labels = torch.cat(class_labels, dim=0)
    return latent_representations, class_labels

def plot_latent_space(latent_representations, class_labels):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'purple', 'orange', 'pink']
    for class_idx in range(10):
        class_mask = class_labels == class_idx
        class_latents = latent_representations[class_mask]
        ax.scatter(class_latents[:, 0], class_latents[:, 1], class_latents[:, 2], c=colors[class_idx], label=str(class_idx))

    ax.set_xlabel('Latent Dimension 1')
    ax.set_ylabel('Latent Dimension 2')
    ax.set_zlabel('Latent Dimension 3')
    ax.legend()
    plt.title('Latent Space Visualization')
    # save as jpg
    plt.savefig('latent_space.jpg')
    plt.show()

def main():
    autoencoder = Autoencoder()
    autoencoder.load_state_dict(torch.load('trained_autoencoder_model.pth'))
    autoencoder.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = torchvision.datasets.FashionMNIST(
        root='./data', train=True, transform=transform, download=True
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=False)

    latent_representations, class_labels = get_latent_representations(autoencoder, train_loader)
    plot_latent_space(latent_representations, class_labels)



if __name__ == "__main__":
    main()
