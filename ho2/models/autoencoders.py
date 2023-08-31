import torch
import torch.nn as nn
import torch.nn.functional as F
class CNNAutoencoder(nn.Module):
    def __init__(self, latent_dim):
        super(CNNAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(16, 2, kernel_size=3, padding=1),
            nn.ReLU(True)
        )
        
        self.enc_fc = nn.Linear(2 * 7 * 7, latent_dim)
        
        # Decoder
        self.dec_fc = nn.Linear(latent_dim, 2 * 7 * 7)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2, 16, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size = x.size(0)
        
        # Encoder
        x = self.encoder(x)
        x = x.view(batch_size, -1)
        x = self.enc_fc(x)
        
        # Decoder
        x = self.dec_fc(x)
        x = x.view(batch_size, 2, 7, 7)
        x = self.decoder(x)
        
        return x




