
from autoencoder import Autoencoder
def calculate_layer_dimensions_and_parameters(model):
    total_params = 0
    layers = {}
    for name, param in model.named_parameters():
        layer_dim = param.size()
        num_params = param.numel()
        total_params += num_params
        layers[name] = (layer_dim, num_params)
        print(f"{name} - Dimensions: {layer_dim}, Parameters: {num_params}")
        
    print(f"Total Trainable Parameters: {total_params}")
    # write layers dict and total params into txt tile
    with open('layer_dimensions_and_parameters.txt', 'w') as f:
        f.write("Layers:\n")
        for name, (layer_dim, num_params) in layers.items():
            f.write(f"{name} - Dimensions: {layer_dim}, Parameters: {num_params}\n")
        f.write(f"Total Trainable Parameters: {total_params}\n")
    
   
if __name__ == "__main__":
    autoencoder = Autoencoder()
    calculate_layer_dimensions_and_parameters(autoencoder)

    