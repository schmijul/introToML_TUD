import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class LinearLayer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2. / input_size)
        self.bias = np.zeros(output_size)
        self.output = None
        self.input = None

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

def forward_pass(x, hidden_layer, output_layer):
    hidden_layer.input = x
    hidden_layer.output =np.tanh(np.dot(x,hidden_layer.weights) + hidden_layer.bias)# np.maximum(0, np.dot(x, hidden_layer.weights) + hidden_layer.bias)
    
    output_layer.input = hidden_layer.output
    output_layer.output = softmax(np.dot(hidden_layer.output, output_layer.weights) + output_layer.bias)

def backpropagation(hidden_layer, output_layer, output, target, learning_rate):
    output_error = output - target
    

    hidden_error = output_error.dot(output_layer.weights.T)
    
    output_layer.weights -= learning_rate * hidden_layer.output.T.dot(output_error)
    output_layer.bias -= learning_rate * np.sum(output_error, axis=0)
    hidden_layer.weights -= learning_rate * hidden_layer.input.T.dot(hidden_error)

    hidden_layer.bias -= learning_rate * np.sum(hidden_error, axis=0)

def train(data, labels, hidden_layer, output_layer, learning_rate, epochs, batch_size):
    for epoch in range(epochs):
        permutation = np.random.permutation(data.shape[0])
        data_shuffled = data[permutation]
        labels_shuffled = labels[permutation]

        for i in range(0, data.shape[0], batch_size):
            batch_data = data_shuffled[i:i+batch_size]
            batch_labels = labels_shuffled[i:i+batch_size]

            forward_pass(batch_data, hidden_layer, output_layer)
            backpropagation(hidden_layer, output_layer, output_layer.output, batch_labels, learning_rate)
            
def evaluate_model(data, labels, hidden_layer, output_layer):
    forward_pass(data, hidden_layer, output_layer)
    predictions = np.argmax(output_layer.output, axis=1)
    labels = np.argmax(labels, axis=1)
    return np.sum(predictions == labels) / len(labels)

def predict(data, labels, hidden_layer,output_layer):
    forward_pass(data, hidden_layer, output_layer)
    predictions = np.argmax(output_layer.output, axis=1)
    return predictions

def find_best_learning_rate(data_train, labels_train, data_val, labels_val, hidden_nodes, output_nodes, learning_rates, epochs, batch_size):
    best_learning_rate = None
    best_accuracy = 0

    for learning_rate in learning_rates:
        hidden_layer = LinearLayer(2, hidden_nodes)
        output_layer = LinearLayer(hidden_nodes, output_nodes)
        
        train(data_train, labels_train, hidden_layer, output_layer, learning_rate, epochs, batch_size)
        accuracy = evaluate_model(data_val, labels_val, hidden_layer, output_layer)
        
        print(f'Learning rate: {learning_rate}, Validation accuracy: {accuracy}')
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_learning_rate = learning_rate
            
    return best_learning_rate

def plot_classification(data, labels, hidden_layer, output_layer):
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros(X.shape)
    
    for i in range(100):
        for j in range(100):
            data_point = np.array([[X[i, j], Y[i, j]]])
            forward_pass(data_point, hidden_layer, output_layer)
            Z[i, j] = np.argmax(output_layer.output)

    plt.contourf(X, Y, Z, cmap=plt.cm.Spectral, alpha=0.8)
    plt.scatter(data[:, 0], data[:, 1], c=labels, s=40, cmap=plt.cm.Spectral)
    plt.show()

def load_data(file):
    data = pd.read_csv(file, delimiter = ',', dtype = np.float32)
    data = np.array(data)
    labels = np.eye(3)[data[:, 2].astype(int)]
    data = data[:, :2]
    return data, labels

def main():
    
    
    
    
    # Load Data
    data, labels = load_data(DATA_FILE)
    
    # split Data into Train and Validation Data 
    data_train = data[:int(0.8 * data.shape[0])]
    labels_train = labels[:int(0.8 * labels.shape[0])]

    data_val = data[int(0.8 * data.shape[0]):]
    labels_val = labels[int(0.8 * labels.shape[0]):]
    
    
    input_shape = data.shape[1]
    output_shape = labels.shape[1] 
    
    # Define Layers
    hidden_layer = LinearLayer(input_shape, NUM_HIDDEN_NODES)
    output_layer = LinearLayer(NUM_HIDDEN_NODES, output_shape)
    # Find best learning rate
    learning_rates_to_test = [10**(-x) for x in range(1, 6)]

    best_learning_rate = find_best_learning_rate(data_train, labels_train, data_val, labels_val, hidden_nodes=NUM_HIDDEN_NODES, output_nodes=output_shape,learning_rates=learning_rates_to_test, epochs=EPOCHS_FOR_LEARNING_RATE_TEST, batch_size=BATCH_SIZE)
    
    
    
    # Train with best learning rate
    train(data, labels, hidden_layer, output_layer, learning_rate=best_learning_rate, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE)
    
    
    # Save Weights and Biases
    trained_mlp_parameters ={"hidden-layer-weights":hidden_layer.weights,
                            "hidden-layer-bias":hidden_layer.bias,
                            "output-layer-weights": output_layer.weights,
                            "output-layer-bias":output_layer.bias}

    np.save('mlp_numpy_parameters.npy', trained_mlp_parameters)
    # Save predictions 
    predictions = predict(data, labels, hidden_layer, output_layer)
    print(predictions)

    np.save('mlp_numpy_pred.npy', predictions)
    # Visualize Results
    #plot_classification(data, labels, hidden_layer, output_layer)

if __name__ == "__main__":
    # data_file
    DATA_FILE = "labeled-dataset-tud-logo.txt"
    # Define Constants
    NUM_HIDDEN_NODES = 25
    NUM_EPOCHS = 10000
    BATCH_SIZE = 64
    EPOCHS_FOR_LEARNING_RATE_TEST = 100
    main()
