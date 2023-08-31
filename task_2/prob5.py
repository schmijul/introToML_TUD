import numpy as np
import math
import matplotlib.pyplot as plt
from network import Network

if __name__ == "__main__":
    
    
    EPOCHS = 10000
    LEARNING_RATE = 1e-3
    TRAIN_DATA_SET_LEN = 41
    TEST_DATA_SET_LEN = 20  # Number of data points in the test set

    # Create training set
    x_train = np.linspace(-2, 2, TRAIN_DATA_SET_LEN).reshape(1, -1)
    y_train = (7 / 2 * np.cos(np.pi / 4 * x_train) ** 2).reshape(1, -1)

    # Create test set
    x_test = np.linspace(-2.5, 2.5, TEST_DATA_SET_LEN).reshape(1, -1)
    y_test = (7 / 2 * np.cos(math.pi / 4 * x_test) ** 2).reshape(1, -1)

    # Initialize and train the network
    network = Network(input_size=1, hidden_size=2, output_size=1)
    loss_history = network.train(x_train, y_train, EPOCHS, LEARNING_RATE)

    # Plot the loss per epoch
    plt.plot(loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()

    # Test the network on the test set
    yhat_test = network.predict(x_test)
    emp_loss_test = np.sum((y_test - yhat_test) ** 2) / TEST_DATA_SET_LEN
    print(f'Empirical loss for test set: {emp_loss_test}')

    # Get learned weights and biases
    learned_weights = network.get_weights()
    learned_biases = network.get_biases()
    print("Learned weights :")
    print("w-1 :", learned_weights['w1'])
    print("w-2 :", learned_weights['w2'])
    print(" ")
    print("Learned biases :")
    print("b-1 :", learned_biases['b1'])
    print("b-2 :", learned_biases['b2'])    
    
