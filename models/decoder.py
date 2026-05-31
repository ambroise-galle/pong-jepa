import torch
import torch.nn as nn

class SpatialDecoder(nn.Module):
    """
    Spatial Decoder that reconstructs a single (1, 64, 64) frame 
    from the 256-dimensional JEPA latent representation.
    """
    def __init__(self, latent_dim=256):
        super().__init__()
        # Linear layer to project latent vector to a feature map shape
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 64 * 4 * 4),
            nn.ReLU()
        )
        
        # Deconvolutional layers to upscale 4x4 -> 8x8 -> 16x16 -> 32x32 -> 64x64
        self.deconvs = nn.Sequential(
            # Input: (64, 4, 4)
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),  # Output: (64, 8, 8)
            nn.ReLU(),
            
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # Output: (32, 16, 16)
            nn.ReLU(),
            
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # Output: (16, 32, 32)
            nn.ReLU(),
            
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),   # Output: (1, 64, 64)
            nn.Sigmoid()  # Restrict reconstructed pixels to [0.0, 1.0]
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 64, 4, 4)
        x = self.deconvs(x)
        return x
