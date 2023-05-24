# eml2023-u03

## Task 9 Numpy MLP

I implemented a Linear Layer Class:

```python

class LinearLayer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2. / input_size)
        self.bias = np.zeros(output_size)

```

That is initialized with random weights and a bias of 0. The forward pass is implemented as follows:

```python

def forward_pass(x, hidden_layer, output_layer):
    hidden_layer.input = x
    hidden_layer.output = np.maximum(0, np.dot(x, hidden_layer.weights) + hidden_layer.bias)
    
    output_layer.input = hidden_layer.output
    output_layer.output = softmax(np.dot(hidden_layer.output, output_layer.weights) + output_layer.bias)

```

And the backpropagation is implemented as follows:

```python
def backpropagation(hidden_layer, output_layer, output, target, learning_rate):
    output_error = output - target
    hidden_error = output_error.dot(output_layer.weights.T)
    
    output_layer.weights -= learning_rate * hidden_layer.output.T.dot(output_error)
    output_layer.bias -= learning_rate * np.sum(output_error, axis=0)
    
    hidden_layer.weights -= learning_rate * hidden_layer.input.T.dot(hidden_error)
    hidden_layer.bias -= learning_rate * np.sum(hidden_error, axis=0)
```


The training is implemented as follows:

```python
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

```

## Task 11 Pytorch MLP

I implemented a MLP with Pytorch:

```python
class TwoLayerPerceptron(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TwoLayerPerceptron, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.softmax(x)
        return x
```

and optimized it with SGD:

```python
def train_model(data, labels, model, learning_rate, epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    train_loader = DataLoader(TensorDataset(data, labels.argmax(dim=1)), batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

```


## Comparison

To compare the train and runtime I created the script comparisson.py:

To run :
```bash
python comparisson.py
```

This Script runs the Numpy and Pytorch Script multiple times and measures the runtime.

The models parameters for the numpy and pytorch model are saved, where the numpy models saves the weights and biases as an array and the pytorch model saves the model as a .pt file.