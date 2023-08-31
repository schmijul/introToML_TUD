from models.autoencoders import CNNAutoencoder

def calculate_layer_dimensions_and_parameters(model):
    total_params = 0
    for name, param in model.named_parameters():
        layer_dim = param.size()
        num_params = param.numel()
        total_params += num_params
        print(f"{name} - Dimensions: {layer_dim}, Parameters: {num_params}")
        
    print(f"Total Trainable Parameters: {total_params}")

if __name__ == "__main__":


    NUM_INPUT_CHANNELS = 1
    ENCODER_CHANNELS = [64, 32, 16, 2]
    DECODER_CHANNELS = [16, 32, 64, 1]
    LATENT_DIM = 3
    KERNEL_SIZE = 3
    PADDING = 1
    ENCODER_STRIDES = [2, 2, 1, 1]
    DECODER_STRIDES = [1, 1, 2, 2]

    model = CNNAutoencoder(
                            input_channels=NUM_INPUT_CHANNELS,
                            encoder_channels=ENCODER_CHANNELS,
                            decoder_channels=DECODER_CHANNELS,
                            latent_dim=LATENT_DIM,
                            kernel_size=KERNEL_SIZE,
                            padding=PADDING,
                            encoder_strides=ENCODER_STRIDES,
                            decoder_strides=DECODER_STRIDES
    )
    
    calculate_layer_dimensions_and_parameters(model)