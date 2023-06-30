import numpy as np
from PIL import Image
import cv2

def convolution_plain_python(matrix, kernel):
    # Dimensions of the input matrix and the kernel
    m_rows, m_cols = len(matrix), len(matrix[0])
    k_rows, k_cols = len(kernel), len(kernel[0])

    # Prepare an output matrix
    output = [[0 for _ in range(m_cols - k_cols + 1)] for _ in range(m_rows - k_rows + 1)]

    # Perform the convolution operation
    for i in range(m_rows - k_rows + 1):
        for j in range(m_cols - k_cols + 1):
            # Apply the kernel to the patch
            for ki in range(k_rows):
                for kj in range(k_cols):
                    output[i][j] += matrix[i + ki][j + kj] * kernel[ki][kj]

    return output  



def convoltuion_scipy(matrix, kernel):
    from scipy.signal import convolve2d
    return convolve2d(matrix, kernel, mode="valid")


def maxpooling(matrix, dim):
    output = np.zeros((int(matrix.shape[0]/dim), int(matrix.shape[1]/dim)))
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            output[i][j] = np.max(matrix[i*dim:(i+1)*dim, j*dim:(j+1)*dim])

    return output
if __name__ == "__main__":


    X = [
        [0,0,0,0,0,0],
        [0,1,1,1,1,0],
        [0,1,1,1,1,0],
        [0,1,1,1,1,0],
        [0,1,1,1,1,0],
        [0,0,0,0,0,0]
         ]


    K1 = [
        [-1, -2, -1,],
        [0, 0, 0,],
        [1, 2, 1,],
         ]


    K2 = [
        [-1, 0, 1,],
        [-2, 0, 2,],
        [-1, 0, 1,],
        ]
    

    X_conv_K1 =convolution_plain_python(X, K1)

    Y1= maxpooling(np.array(X_conv_K1), 2)


    print("X_conv_K1:\n ", np.array(X_conv_K1))
    print("\n")
    print("Y1: \n ", Y1)




    # Load the image

    img = np.array(Image.open("circle.png"))
    import matplotlib.pyplot as plt
    
    img_conv_k1 = convolution_plain_python(img, K1)

    Y1 = maxpooling(np.array(img_conv_k1), 2)
    

    # show img, img_conv_k1, Y1 in one figure
    fig, ax = plt.subplots(1, 3, figsize=(15, 15))
    ax[0].imshow(img, cmap="gray")

    ax[1].imshow(img_conv_k1, cmap="gray")

    ax[2].imshow(Y1, cmap="gray")

    plt.show()