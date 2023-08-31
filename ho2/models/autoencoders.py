import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNAutoencoder(nn.Module):
    def __init__(self, input_channels, encoder_channels, decoder_channels, latent_dim,
                 kernel_size, padding, encoder_strides, decoder_strides):
        super(CNNAutoencoder, self).__init__()
        
        # Encoder
        self.enc_layers = nn.ModuleList()
        in_channels = input_channels
        for out_channels, stride in zip(encoder_channels, encoder_strides):
            self.enc_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding))
            in_channels = out_channels
            
        self.enc_fc = nn.Linear(encoder_channels[-1] * 2 * 2, latent_dim)
        
        # Decoder
        self.dec_fc = nn.Linear(latent_dim, encoder_channels[-1] * 2 * 2)
        
        self.dec_layers = nn.ModuleList()
        in_channels = encoder_channels[-1]
        for out_channels, stride in zip(decoder_channels, decoder_strides):
            self.dec_layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding))
            in_channels = out_channels

    def forward(self, x):
        for layer in self.enc_layers:
            x = F.relu(layer(x))
            
        x = x.view(-1, encoder_channels[-1] * 2 * 2)
        x = self.enc_fc(x)
        
        x = self.dec_fc(x)
        x = x.view(-1, encoder_channels[-1], 2, 2)
        
        for layer in self.dec_layers[:-1]:
            x = F.relu(layer(x))
        
        x = torch.sigmoid(self.dec_layers[-1](x))
        
        return x
