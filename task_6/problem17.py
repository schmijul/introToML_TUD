import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets
from torchvision.transforms import ToTensor


def get_data():
    """
    Fetches and returns the MNIST training data.

    Returns
    -------
    train_data : torchvision.datasets.MNIST
        The MNIST training data.
    """
    train_data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )
    return train_data


def normalize_center(data):
    """
    Normalize and center the dataset.

    Parameters
    ----------
    data : numpy.ndarray
        The dataset to be normalized and centered.

    Returns
    -------
    normalized_centered_data : numpy.ndarray
        The normalized and centered dataset.
    """
    mean = np.mean(data, axis=0)
    centered_data = data - mean
    std = np.std(centered_data, axis=0)
    epsilon = 1e-8  # Small epsilon value to avoid division by zero
    normalized_centered_data = centered_data / (std + epsilon)
    return normalized_centered_data


def compute_principal_components(data, num_components):
    """
    Compute the principal components using SVD.

    Parameters
    ----------
    data : numpy.ndarray
        The dataset.

    num_components : int
        The number of principal components to compute.

    Returns
    -------
    principal_components : numpy.ndarray
        The principal components.

    coefficients : numpy.ndarray
        The coefficients of the data points on the principal components.
    """
    _, _, Vt = np.linalg.svd(data, full_matrices=False)
    principal_components = Vt[:num_components, :]
    coefficients = np.dot(data, principal_components.T)
    return principal_components, coefficients


def calculate_explained_variance(principal_components):
    """
    Calculate the percentage of explained variance over the number of principal components.

    Parameters
    ----------
    principal_components : numpy.ndarray
        The principal components.

    Returns
    -------
    explained_variance : numpy.ndarray
        The percentage of explained variance for each principal component.
    """
    eigenvalues = np.linalg.eigvals(np.cov(principal_components.T))
    sorted_eigenvalues = np.sort(eigenvalues)[::-1]
    explained_variance = sorted_eigenvalues / np.sum(sorted_eigenvalues)
    return explained_variance


def reconstruct_images(data, principal_components, num_components):
    """
    Reconstruct the data using the specified number of principal components.

    Parameters
    ----------
    data : numpy.ndarray
        The dataset.

    principal_components : numpy.ndarray
        The principal components.

    num_components : int
        The number of principal components to use for reconstruction.

    Returns
    -------
    reconstructed_data : numpy.ndarray
        The reconstructed data.
    """
    coefficients = np.dot(data, principal_components.T)
    reconstructed_data = np.dot(coefficients[:, :num_components], principal_components[:num_components, :])
    return reconstructed_data


def calculate_mse(original_data, reconstructed_data):
    """
    Calculate the mean squared error (MSE) between the original data and the reconstructed data.

    Parameters
    ----------
    original_data : numpy.ndarray
        The original data.

    reconstructed_data : numpy.ndarray
        The reconstructed data.

    Returns
    -------
    mse : float
        The mean squared error.
    """
    mse = np.mean((original_data - reconstructed_data) ** 2)
    return mse


# Load the MNIST training dataset
train_data = get_data()
images = train_data.data.numpy()
num_samples = images.shape[0]
num_features = images.shape[1] * images.shape[2]
data = images.reshape(num_samples, num_features)

# Normalize and center the dataset
normalized_centered_data = normalize_center(data)

# Set the number of principal components
num_components = 2

# Compute the principal components and coefficients
principal_components, coefficients = compute_principal_components(normalized_centered_data, num_components)

# Plot the mean of the dataset
plt.imshow(principal_components.mean(axis=0).reshape(28, 28), cmap='gray')
plt.title('Mean of the Dataset')
plt.axis('off')
plt.show()

# Plot the first 2 principal components
fig, ax = plt.subplots(nrows=1, ncols=num_components, figsize=(8, 4))
for i in range(num_components):
    ax[i].imshow(principal_components[i].reshape(28, 28), cmap='gray')
    ax[i].set_title(f'Principal Component {i+1}')
    ax[i].axis('off')
plt.show()

# Calculate the explained variance
explained_variance = calculate_explained_variance(principal_components)
cumulative_variance = np.cumsum(explained_variance)

# Plot the percentage of explained variance over the number of principal components
plt.plot(np.arange(1, num_components+1), cumulative_variance, marker='o')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Percentage of Explained Variance')
plt.grid(True)
plt.show()

# Determine the number of components needed to explain certain percentages of variance
variance_thresholds = [0.5, 0.9, 0.95, 0.99]
num_components_needed = []

for threshold in variance_thresholds:
    num_components_needed.append(np.argmax(cumulative_variance >= threshold) + 1)
    print(f"Number of components needed to explain {threshold*100}% variance: {num_components_needed[-1]}")

# Load the MNIST test dataset
test_data = get_data()
test_images = test_data.data.numpy()
test_data = test_images.reshape(test_images.shape[0], num_features)

# Reconstruct test sample images for different values of Q and calculate MSE
Q_values = [2, 5, 10, 20]
reconstructed_images = {}

for Q in Q_values:
    reconstructed_data = reconstruct_images(test_data, principal_components, Q)
    reconstructed_images[Q] = reconstructed_data

# Calculate MSE for the reconstructed images
mse_values = []
for Q, reconstructed_data in reconstructed_images.items():
    mse = calculate_mse(test_data, reconstructed_data)
    mse_values.append(mse)
    print(f"MSE for Q={Q}: {mse}")

# Plot the reconstructions of the first test sample images
num_samples_to_plot = 5

fig, axs = plt.subplots(nrows=len(Q_values), ncols=num_samples_to_plot + 1, figsize=(12, 2 * len(Q_values)))

for i, Q in enumerate(Q_values):
    reconstructed_data = reconstructed_images[Q]
    mse = mse_values[i]
    axs[i, 0].imshow(test_images[0], cmap='gray')
    axs[i, 0].set_title('Original Image')
    axs[i, 0].axis('off')

    for j in range(num_samples_to_plot):
        axs[i, j + 1].imshow(reconstructed_data[j].reshape(28, 28), cmap='gray')
        axs[i, j + 1].set_title(f'Reconstructed Image (MSE={mse:.4f})')
        axs[i, j + 1].axis('off')

plt.tight_layout()
plt.show()
