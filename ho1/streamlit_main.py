import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
import networkx as nx
import pandas as pd
import task1
import task2

st.set_option('deprecation.showPyplotGlobalUse', False)
# Streamlit App
st.title("Neural Network Training Visualization")

# Parameter einstellen

# Choose Task
task_choice = st.selectbox("Choose Task", ("Task 1", "Task 2"))

num_epochs = st.number_input("Number of Epochs", value=1000000, step=1000)
batch_size = st.number_input("Batch Size", value=100)
init_lr = st.number_input("Initial Learning Rate", value=0.1)
decay_lr = st.number_input("Decay Learning Rate", value=0.9)
decay_interval = st.number_input("Decay Interval", value=1000)
scale_data = st.checkbox("Scale Data [0-1]")
target_loss = st.number_input("Target Loss", value=0.02)
N1 = st.number_input("Hidden Layer Dimension (N1)",value=5)

# Data
x1 = np.arange(-5, 5, 0.1)
x2 = np.arange(-5, 5, 0.1)



def visualize_model_architecture(model):
    G = nx.DiGraph()

    # Add nodes for input
    G.add_node("Input 1")
    G.add_node("Input 2")

    # Get the output dimension of the first layer
    if (task_choice == "Task 1"):
        N1 = model.linear1.weights.shape[1]
        input_dim =2

    elif (task_choice == "Task 2"):
        N1 = model.linear1.weights.shape[0]
        input_dim = 3
    # Add nodes for middle layer
    for i in range(N1):
        G.add_node(f"Middle Node {i + 1}")

    # Add node for output
    G.add_node("Output Node")

   

    # Define the layout positions for nodes
    pos = {}
    for i in range(input_dim):
        pos[f"Input {i + 1}"] = (0, -3 + 2 * i)

    for i in range(N1):
        pos[f"Middle Node {i + 1}"] = (1, -3 + 2 * i)



    pos["Output Node"] = (2, 0)

    # Add edges
    for i in range(input_dim):
        for j in range(N1):
            G.add_edge(f"Input {i + 1}", f"Middle Node {j + 1}")

    for i in range(N1):
        G.add_edge(f"Middle Node {i + 1}", "Output Node")
    
        

    # Draw the graph
    plt.figure(figsize=(8, 6))
    nx.draw_networkx(G, pos, with_labels=True, node_color='lightblue', node_size=1000, font_size=10, edge_color='gray')
    plt.title("Model Architecture")
    plt.axis('off')

# Trainingsfunktion
@st.cache_data()
def train_and_plot_performance():
    if task_choice == "Task 1":
       
        x_train = np.array(np.meshgrid(x1, x2)).T.reshape(-1, 2)
        y_train = task1.target_fct(x_train[:, 0], x_train[:, 1])

        model = task1.TwoLayerNet(hidden_dim=N1)
    elif task_choice == "Task 2":
        labeled_data_pd = pd.read_csv("labeled-dataset-3d-rings.txt", delimiter=',', dtype=np.float32)
        labeled_data_np = np.array(labeled_data_pd)
        size_data_set = len(labeled_data_np)
        np.random.shuffle(labeled_data_np)
        x = labeled_data_np[:, 0:3].T
        labels = labeled_data_np[:, 3:].astype(int).reshape((-1,))
        N_cl = np.max(labels).astype(int) + 1
        y = np.zeros((size_data_set, N_cl))
        y[np.arange(size_data_set), labels] = 1
        y_T = y.T
        model = task2.TwoLayerNet(input_dim=3, hidden_dim=N1, output_dim=N_cl)
    visualize_model_architecture(model)
    st.pyplot()
    training_progress = []

    plot_placeholder = st.empty()  # Create a placeholder for the plot

    for epoch in range(num_epochs):
       
        if task_choice == "Task 1":
            loss = task1.backpropagation(model, x_train, y_train, init_lr)
        elif task_choice == "Task 2":
            task2.backpropagation(model, x, y_T, init_lr)
            y_hat = model.forward(x)

            loss = task2.calculate_loss_accuracy(y_hat, y_T, labels)[1]
        training_progress.append(loss)

        if (epoch + 1) % 1000 == 0:
            print(f"Epoch {epoch + 1}: Loss = {loss:.4f}")
            if task_choice == "Task 1":
                plot_3d_predictions(model, x1, x2, epoch+1, plot_placeholder, y_train)
            
            elif task_choice == "Task 2":
                plot_3d_predictions(model, x[:,0].T, x[:,1].T, epoch+1, plot_placeholder, labels)

    return training_progress


def plot_3d_predictions(model, x1, x2, epoch, placeholder, y_train):
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(121, projection='3d')
    x1_grid, x2_grid = np.meshgrid(x1, x2)
    x_grid = np.column_stack((x1_grid.ravel(), x2_grid.ravel()))
    y_pred = model.forward(x_grid)
    y_pred_grid = y_pred.reshape(x1_grid.shape)
    ax.scatter(x1_grid, x2_grid, y_pred_grid)
    ax.set_title(f"Prediction at Epoch {epoch}")

    ax = fig.add_subplot(122, projection='3d')
    ax.scatter(x_grid[:, 0], x_grid[:, 1], y_train)
    ax.set_title("Ground Truth")

    # Clear previous plot
    placeholder.pyplot(fig)

    # Close the figure to release memory
    plt.close(fig)


# Trainingsdurchlauf und Performance plotten
if st.button("Start Training"):
    progress_bar = st.progress(0)
    training_progress = train_and_plot_performance()
    progress_bar.progress(1.0)

    # Plot Performance
    plt.plot(training_progress)
    plt.xlabel("Epoch [linear scale]")
    plt.ylabel("Loss")
    plt.title("Training Progress")
    st.pyplot()
