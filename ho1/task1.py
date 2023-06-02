import numpy as np
import matplotlib.pyplot as plt

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
        z1 = self.linear1.forward(inputs)
        g_z1 = self.sigmoid.forward(z1)
        y_hat = self.linear2.forward(g_z1)
        return y_hat
    
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def target_fct(x1, x2):
    return ((x1**2 + x2**2)**(7/12)).reshape(-1, 1)

def lossfct(y_hat, y_true):
    return np.sum((y_hat - y_true)**2) / y_true.shape[0]

def plot_training_progress(training_progress):
    plt.plot(training_progress)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Progress")
    plt.show()

def train(model, x_train, y_train, num_epochs, batch_size, init_lr=1,target_loss=None, scale_data=False, decay_lr=None, decay_interval=None, gradient_clip=None, verbose=False):
    current_lr = init_lr
    best_loss = np.inf

    training_progress = []
    if scale_data:
        y_min = np.min(y_train)
        y_max = np.max(y_train)
        y_train = (y_train - y_min) / (y_max - y_min)

        x_min = np.min(x_train)
        x_max = np.max(x_train)
        x_train = (x_train - x_min) / (x_max - x_min)
    for current_epoch in range(num_epochs):

        # Shuffle the data
        idx = np.random.permutation(x_train.shape[0])
        x_train = x_train[idx]
        y_train = y_train[idx]

        # Batch training
        for batch_start in range(0, x_train.shape[0], batch_size):
            batch_end = batch_start + batch_size
            x_batch = x_train[batch_start:batch_end]
            y_batch = y_train[batch_start:batch_end]

            # Forward Pass
            y_hat = model.forward(x_batch)
            batch_loss = lossfct(y_hat, y_batch)
            training_progress.append(batch_loss)

            # Check if loss has improved
            if batch_loss < best_loss:
                best_loss = batch_loss
                no_improvement_counter = 0
            else:
                no_improvement_counter += 1

            # Decay learning rate if necessary
            if decay_lr is not None and decay_interval is not None:
                if no_improvement_counter >= decay_interval and (current_epoch + 1) % decay_interval == 0:
                    current_lr *= decay_lr
                    no_improvement_counter = 0

            # Backward Pass
            d_loss = 2 * (y_hat - y_batch) / y_batch.shape[0]
            grad_w2 = np.matmul(model.linear1.outputs.T, d_loss)
            grad_b2 = np.sum(d_loss, axis=0)
            d_sigmoid = sigmoid(model.linear1.outputs) * (1 - sigmoid(model.linear1.outputs))
            grad_w1 = np.matmul(x_batch.T, np.matmul(d_loss, model.linear2.weights.T) * d_sigmoid)
            grad_b1 = np.sum(np.matmul(d_loss, model.linear2.weights.T) * d_sigmoid, axis=0)

            # Gradient Clipping
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

        y_hat = model.forward(x_train)
        loss = lossfct(y_hat, y_train)
        if verbose:
            if current_epoch % 10000 == 0:
                if scale_data:
                    y_hat_rescaled = y_hat * (y_max - y_min) + y_min
                    y_train_rescaled = y_train * (y_max - y_min) + y_min
                    loss = lossfct(y_hat_rescaled, y_train_rescaled)
                print(f"Epoch: {current_epoch} Loss: {loss}")



        # if Loss <=0.02 end training here

        if (target_loss) and (loss <= target_loss):
            print(f"reached target loss")
            return training_progress
    return training_progress

def plot_3d(x1, x2, y):
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x1, x2, y)
    plt.show()

if __name__ == "__main__":
    verbose = True
    num_epochs = 100000
    batch_size = 32
    decay_lr = 0.9
    decay_interval = 1000
    scale_data = True
    init_lr = 0.1
    target_loss = 0.02

    N1 = 10

    x1 = np.arange(-5, 5, 0.1)
    x2 = np.arange(-5, 5, 0.1)
    x_train = np.array(np.meshgrid(x1, x2)).T.reshape(-1, 2)
    y_train = target_fct(x_train[:, 0], x_train[:, 1])
    
    model = TwoLayerNet()

    train_loss = train(model, x_train, y_train, num_epochs=num_epochs, batch_size=batch_size, init_lr=init_lr, scale_data=scale_data, verbose=verbose)

    plot_training_progress(train_loss)
    # save model state
    weights_and_biases = {"w1": model.linear1.weights, "b1": model.linear1.biases, "w2": model.linear2.weights, "b2": model.linear2.biases}
    np.save("weights_and_biases.npy", weights_and_biases)
