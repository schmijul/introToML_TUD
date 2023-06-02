import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

class LinearLayer:
    def __init__(self, input_dim, output_dim):
        self.weights = np.random.randn(input_dim, output_dim)
        self.biases = np.random.randn(output_dim)

    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = np.matmul(inputs, self.weights) + self.biases
        return self.outputs

class SigmoidLayer:
    def __init__(self):
        pass
    
    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = 1/(1 + np.exp(-inputs))
        return self.outputs

        
class TwoLayerNet:
    def __init__(self, input_dim=2, hidden_dim=4, output_dim=1):
        self.linear1 = LinearLayer(input_dim, hidden_dim)
        self.sigmoid = SigmoidLayer()
        self.linear2 = LinearLayer(hidden_dim, output_dim)

    def forward(self, inputs):
        output_layer_1 = self.linear1.forward(inputs)
        sigmoid_output_layer_1 = self.sigmoid.forward(output_layer_1)
        y_hat = self.linear2.forward(sigmoid_output_layer_1)
        return y_hat
    
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def target_fct(x1, x2):
    return ((x1**2 + x2**2)**(7/12)).reshape(-1, 1)

def lossfct(y_hat, y_true):
    return np.sum((y_hat - y_true)**2) / y_true.shape[0]

def plot_training_progress(training_progress):
    #plt.plot(training_progress)
    plt.semilogy(training_progress)
    plt.xlabel("Epoch [linear scale]")
    plt.ylabel("Loss [log scale]")
    plt.title("Training Progress")
    plt.show()

def backpropagation(model, x, y, current_lr, gradient_clip=None):
    # Forward Pass
    y_hat = model.forward(x)
    loss = lossfct(y_hat, y)

    # Backward Pass
    d_loss = 2 * (y_hat - y) / y.shape[0]
    grad_w2 = np.matmul(model.linear1.outputs.T, d_loss)
    grad_b2 = np.sum(d_loss, axis=0)
    d_sigmoid = sigmoid(model.linear1.outputs) * (1 - sigmoid(model.linear1.outputs))
    grad_w1 = np.matmul(x.T, np.matmul(d_loss, model.linear2.weights.T) * d_sigmoid)
    grad_b1 = np.sum(np.matmul(d_loss, model.linear2.weights.T) * d_sigmoid, axis=0)

    # Gradient Clipping
    gradient_clip = None  # Set gradient_clip if desired
    if gradient_clip is not None:
        grad_w2 = np.clip(grad_w2, -gradient_clip, gradient_clip)
        grad_b2 = np.clip(grad_b2, -gradient_clip, gradient_clip)
        grad_w1 = np.clip(grad_w1, -gradient_clip, gradient_clip)
        grad_b1 = np.clip(grad_b1, -gradient_clip, gradient_clip)

    # Update Parameters
    model.linear2.weights -= current_lr * grad_w2
    model.linear2.biases -= current_lr * grad_b2
    model.linear1.weights -= current_lr * grad_w1
    model.linear1.biases -= current_lr * grad_b1

    return loss

def train(model, x_train, y_train, num_epochs, init_lr=1, target_loss=None, scale_data=False, decay_lr=None, decay_interval=None, gradient_clip=None, verbose=False):
    current_lr = init_lr
    best_loss = np.inf

    training_progress = []
    start_time = time.time()

    best_weights_and_biases = None

    if (verbose):
        progress_bar = tqdm(total=num_epochs, ncols=80, desc="Training:")

    if (scale_data):
        y_min = np.min(y_train)
        y_max = np.max(y_train)
        y_train = (y_train - y_min) / (y_max - y_min)

        x_min = np.min(x_train)
        x_max = np.max(x_train)
        x_train = (x_train - x_min) / (x_max - x_min)

    for current_epoch in range(num_epochs):

        idx = np.random.permutation(x_train.shape[0])
        x_train = x_train[idx]
        y_train = y_train[idx]

        loss = backpropagation(model, x_train, y_train, current_lr,gradient_clip=gradient_clip)
        if (scale_data):
            loss = loss *(y_max - y_min) + y_min

        training_progress.append(loss)

        elapsed_time = time.time() - start_time
        
        if (loss < best_loss):
            best_loss = loss
            no_improvement_counter = 0
            best_weights_and_biases = {"w1":model.linear1.weights.copy(), 
                                       "b1":model.linear1.biases.copy(), 
                                       "w2":model.linear2.weights.copy(), 
                                       "b2":model.linear2.biases.copy()}
        else:
            no_improvement_counter += 1

        

        if ((decay_lr is not None) and (decay_interval is not None)):
            if ((no_improvement_counter >= decay_interval) and ((current_epoch + 1) % decay_interval == 0)):
                current_lr *= decay_lr
                no_improvement_counter = 0        

        if ((target_loss) and (loss <= target_loss)):
            print(f"Reached target loss after {current_epoch} epochs and {elapsed_time} seconds")
            if (verbose):
                progress_bar.close()
                return training_progress
        
        if (verbose):
            progress_bar.update(1)
            if (current_epoch % 10000 == 0):  
                print(f"Epoch: {current_epoch} Loss: {loss} Best Loss :{best_loss} Elapsed Time: {elapsed_time:.2f}s")
    if (verbose):
        progress_bar.close()
    

    return best_weights_and_biases,training_progress


def plot_3d(x1, x2, y):
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x1, x2, y)
    plt.show()


def main():

    x_train = np.array(np.meshgrid(x1, x2)).T.reshape(-1, 2)
    y_train = target_fct(x_train[:, 0], x_train[:, 1])
    
    model = TwoLayerNet()

    best_weights_and_biases, train_loss = train(model, x_train, y_train, num_epochs=num_epochs, init_lr=init_lr, target_loss=target_loss,gradient_clip=gradient_clip, scale_data=scale_data, verbose=verbose)

    plot_training_progress(train_loss)

    # save model state
    weights_and_biases = best_weights_and_biases

    np.save("weights_and_biases.npy", weights_and_biases)

if (__name__ == "__main__"):

    # Script Parameters
    verbose = True
    num_epochs = int(1e6)


    # Hyper Parameters
    init_lr = 0.1
    N1 = 5

    # Optimization Parameters
    decay_lr = 0.9
    decay_interval = 1000
    scale_data = True
    gradient_clip = False

    # Target
    target_loss = 0.02
    
    
    # Data
    x1 = np.arange(-5, 5, 0.1)
    x2 = np.arange(-5, 5, 0.1)
 
    main()