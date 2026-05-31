import torch
import torch.nn as nn

class QNetwork(nn.Module):
    """Simple MLP Q-network that operates on top of frozen JEPA representations."""
    def __init__(self, latent_dim=256, action_dim=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, z):
        return self.net(z)
