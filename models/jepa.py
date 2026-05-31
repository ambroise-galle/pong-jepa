import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class Encoder(nn.Module):
    def __init__(self, in_channels=4, latent_dim=256):
        super().__init__()
        # Standard Nature CNN architecture adapted for 64x64
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),  # Output: (32, 15, 15)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),           # Output: (64, 6, 6)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),           # Output: (64, 4, 4)
            nn.ReLU()
        )
        
        self.flat = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, latent_dim),
            nn.LayerNorm(latent_dim)  # Normalization helps stabilize representation space
        )

    def forward(self, x):
        features = self.convs(x)
        features = self.flat(features)
        z = self.fc(features)
        return z

class Predictor(nn.Module):
    def __init__(self, latent_dim=256, action_dim=6, action_emb_dim=32):
        super().__init__()
        # action embedding allows the network to learn similarities between actions
        self.action_embed = nn.Embedding(action_dim, action_emb_dim)
        
        # Dynamics MLP to transition the latent representation forward in time
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim + action_emb_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, latent_dim)
        )

    def forward(self, z, action):
        a_emb = self.action_embed(action)
        x = torch.cat([z, a_emb], dim=-1)
        z_next = self.mlp(x)
        return z_next

class RewardDiscountPredictor(nn.Module):
    """Predicts reward and terminal status from latent state and action."""
    def __init__(self, latent_dim=256, action_dim=6, action_emb_dim=32):
        super().__init__()
        self.action_embed = nn.Embedding(action_dim, action_emb_dim)
        
        # Reward head (continuous value or classification, Pong rewards are in {-1, 0, 1})
        self.reward_mlp = nn.Sequential(
            nn.Linear(latent_dim + action_emb_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)  # continuous scalar prediction
        )
        
        # Terminal/Done head (binary logit)
        self.done_mlp = nn.Sequential(
            nn.Linear(latent_dim + action_emb_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)  # logit for probability of game over
        )

    def forward(self, z, action):
        a_emb = self.action_embed(action)
        x = torch.cat([z, a_emb], dim=-1)
        
        reward = self.reward_mlp(x).squeeze(-1)
        done_logit = self.done_mlp(x).squeeze(-1)
        return reward, done_logit

class JEPAWorldModel(nn.Module):
    def __init__(self, in_channels=4, latent_dim=256, action_dim=6, ema_decay=0.99):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.ema_decay = ema_decay

        # Online networks
        self.online_encoder = Encoder(in_channels=in_channels, latent_dim=latent_dim)
        self.predictor = Predictor(latent_dim=latent_dim, action_dim=action_dim)
        self.reward_done_predictor = RewardDiscountPredictor(latent_dim=latent_dim, action_dim=action_dim)

        # Target encoder (EMA of online encoder)
        self.target_encoder = copy.deepcopy(self.online_encoder)
        
        # Freeze target encoder weights (they are updated manually via EMA, not SGD)
        for param in self.target_encoder.parameters():
            param.requires_grad = False

    def update_target_encoder(self):
        """EMA update of target encoder parameters."""
        with torch.no_grad():
            for online_p, target_p in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
                target_p.data.mul_(self.ema_decay).add_(online_p.data, alpha=1.0 - self.ema_decay)

    def get_representation(self, x):
        """Get context representation from online encoder."""
        return self.online_encoder(x)

    def get_target_representation(self, x):
        """Get target representation (stop-grad, target encoder)."""
        with torch.no_grad():
            return self.target_encoder(x)

    def compute_jepa_loss(self, states, actions, next_states, rewards, dones):
        # 1. Get representations
        z_t = self.get_representation(states)
        
        # 2. Get target representations (using Target Encoder, no gradients)
        with torch.no_grad():
            z_target_next = self.get_target_representation(next_states)
            
        # 3. Predict next representation
        z_pred_next = self.predictor(z_t, actions)
        
        # 4. Normalize embeddings (BYOL style - prevents representation collapse)
        z_pred_norm = F.normalize(z_pred_next, dim=-1)
        z_target_norm = F.normalize(z_target_next, dim=-1)
        
        # Prediction Loss: L2 distance between normalized embeddings
        pred_loss = F.mse_loss(z_pred_norm, z_target_norm)
        
        # 5. Predict reward and terminal conditions
        pred_rewards, pred_done_logits = self.reward_done_predictor(z_t, actions)
        
        # Reward loss (MSE)
        reward_loss = F.mse_loss(pred_rewards, rewards)
        
        # Done/Terminal loss (Binary Cross Entropy)
        done_loss = F.binary_cross_entropy_with_logits(pred_done_logits, dones)
        
        total_loss = pred_loss + 0.5 * reward_loss + 0.1 * done_loss
        
        return {
            "loss": total_loss,
            "pred_loss": pred_loss,
            "reward_loss": reward_loss,
            "done_loss": done_loss,
            "z_t_norm": z_t.norm(dim=-1).mean(),
            "z_pred_norm": z_pred_next.norm(dim=-1).mean(),
            "z_target_norm": z_target_next.norm(dim=-1).mean()
        }
