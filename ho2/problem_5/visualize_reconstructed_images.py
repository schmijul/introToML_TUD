# Import required libraries
import matplotlib.pyplot as plt
import numpy as np
import torch
from autoencoder import Autoencoder
# Function to generate a 3D grid
def generate_3d_grid(min_vals, max_vals, steps=12):
    x_vals = np.linspace(min_vals[0], max_vals[0], steps)
    y_vals = np.linspace(min_vals[1], max_vals[1], steps)
    z_vals = np.linspace(min_vals[2], max_vals[2], steps)
    grid = np.array(np.meshgrid(x_vals, y_vals, z_vals)).T.reshape(-1, 3)
    return grid

# Function to decode grid points and get reconstructed images
def decode_grid_points(model, grid_points):
    grid_points_tensor = torch.FloatTensor(grid_points)
    with torch.no_grad():
        decoded_images = model.decoder(grid_points_tensor)
    return decoded_images

# Function to visualize and save decoded images in a grid
def visualize_and_save_images(decoded_images, num_figs=12, fig_size=(15, 15), save_path='./'):
    decoded_images_np = decoded_images.numpy().reshape(-1, 28, 28)
    num_images_per_fig = decoded_images_np.shape[0] // num_figs

    for fig_num in range(num_figs):
        fig, axes = plt.subplots(12, 12, figsize=fig_size)
        axes = axes.ravel()
        
        start_idx = fig_num * num_images_per_fig
        end_idx = (fig_num + 1) * num_images_per_fig

        for i, img in enumerate(decoded_images_np[start_idx:end_idx]):
            axes[i].imshow(img, cmap='gray')
            axes[i].axis('off')
        
        plt.subplots_adjust(wspace=0.5, hspace=0.5)
        plt.savefig(f"{save_path}/decoded_images_fig_{fig_num + 1}.jpg")
        plt.close()

# Main function to perform all steps
def main():
    # Load the trained model
    model_path = 'trained_autoencoder_model.pth'  # Replace this with the path to your model
    autoencoder = Autoencoder()
    autoencoder.load_state_dict(torch.load(model_path))
    autoencoder.eval()

    # Generate 12x12x12 3D grid points
    min_vals = np.array([-54.102005, -4.751048, -14.630363])
    max_vals = np.array([-6.7525697, 51.515377, 31.410765])
    grid_points = generate_3d_grid(min_vals, max_vals, steps=12)

    # Decode these grid points to get the images
    decoded_images = decode_grid_points(autoencoder, grid_points)

    # Visualize and save the decoded images in 12 figures
    visualize_and_save_images(decoded_images, num_figs=12, fig_size=(15, 15), save_path='./')

if __name__ == "__main__":
    main()
