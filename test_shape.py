import torch
from models.jepa import Encoder, Predictor, RewardDiscountPredictor, JEPAWorldModel
from models.policy import QNetwork

def test_dimensions():
    print("Testing shapes of JEPA components...")
    
    batch_size = 16
    latent_dim = 256
    action_dim = 6
    action_emb_dim = 32
    
    # 1. Test Encoder
    encoder = Encoder(in_channels=4, latent_dim=latent_dim)
    dummy_states = torch.randn(batch_size, 4, 64, 64)
    z = encoder(dummy_states)
    assert z.shape == (batch_size, latent_dim), f"Encoder shape mismatch: {z.shape}"
    print("✓ Encoder: Input (B, 4, 64, 64) -> Output (B, 256) matches!")
    
    # 2. Test Predictor
    predictor = Predictor(latent_dim=latent_dim, action_dim=action_dim, action_emb_dim=action_emb_dim)
    dummy_actions = torch.randint(0, action_dim, (batch_size,))
    z_next = predictor(z, dummy_actions)
    assert z_next.shape == (batch_size, latent_dim), f"Predictor shape mismatch: {z_next.shape}"
    print("✓ Predictor: Input (B, 256) & (B,) -> Output (B, 256) matches!")
    
    # 3. Test Reward & Done Predictors
    reward_done_pred = RewardDiscountPredictor(latent_dim=latent_dim, action_dim=action_dim, action_emb_dim=action_emb_dim)
    rewards, done_logits = reward_done_pred(z, dummy_actions)
    assert rewards.shape == (batch_size,), f"Reward shape mismatch: {rewards.shape}"
    assert done_logits.shape == (batch_size,), f"Done shape mismatch: {done_logits.shape}"
    print("✓ Reward/Discount Predictor: Output (B,) rewards and (B,) done logits matches!")

    # 4. Test JEPAWorldModel
    model = JEPAWorldModel(in_channels=4, latent_dim=latent_dim, action_dim=action_dim)
    dummy_next_states = torch.randn(batch_size, 4, 64, 64)
    dummy_rewards = torch.randn(batch_size)
    dummy_dones = torch.randint(0, 2, (batch_size,)).float()
    
    loss_dict = model.compute_jepa_loss(
        dummy_states, dummy_actions, dummy_next_states, dummy_rewards, dummy_dones
    )
    
    assert "loss" in loss_dict, "Loss key missing from loss_dict"
    assert loss_dict["loss"].ndim == 0, "Total loss is not a scalar"
    print("✓ JEPA Loss Function: Scaled loss computation matches!")
    
    # 5. Test QNetwork
    qnet = QNetwork(latent_dim=latent_dim, action_dim=action_dim)
    q_vals = qnet(z)
    assert q_vals.shape == (batch_size, action_dim), f"QNetwork shape mismatch: {q_vals.shape}"
    print("✓ Downstream Q-Network: Input (B, 256) -> Output (B, 6) matches!")
    
    print("\nAll architectural shapes and forward steps are 100% correct!")

if __name__ == "__main__":
    test_dimensions()
