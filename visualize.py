import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from utils.env import make_pong
from models.jepa import JEPAWorldModel
from models.policy import QNetwork

def get_rollout_data(env, encoder, num_steps=500, device="cpu"):
    """Runs the environment with a random policy and returns transitions and representations."""
    obs, info = env.reset()
    
    states_list = []
    z_list = []
    actions_list = []
    rewards_list = []
    
    with torch.no_grad():
        for _ in range(num_steps):
            action = env.action_space.sample()
            
            # Save original frame stack and compute representation
            states_list.append(obs.copy())
            
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0) / 255.0
            z = encoder(obs_t).squeeze(0).cpu().numpy()
            z_list.append(z)
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            actions_list.append(action)
            rewards_list.append(reward)
            
            if terminated or truncated:
                obs, info = env.reset()
                
    return np.array(states_list), np.array(z_list), np.array(actions_list), np.array(rewards_list)

def plot_latent_space(z_list, states_list, save_path="data/latent_space_tsne.png"):
    """Projects representations to 2D using t-SNE or PCA and colors by ball positions."""
    print("Projecting latent representations to 2D...")
    
    # 1. Compute projection (handling small sample sizes for robustness)
    n_samples = len(z_list)
    perplexity = min(30, n_samples - 1)
    if n_samples < 5:
        print("Too few points for t-SNE. Using PCA instead...")
        pca = PCA(n_components=2, random_state=42)
        z_2d = pca.fit_transform(z_list)
    else:
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        z_2d = tsne.fit_transform(z_list)
    
    # 2. Extract heuristic ball y-positions to color-code the plot
    # This helps verify if the latent space actually encodes the physics of the game!
    ball_y_positions = []
    for state in states_list:
        frame = state[-1]  # Get most recent frame in the stack
        ball_y_indices, _ = np.where(frame[10:60, 8:55] > 140)
        if len(ball_y_indices) > 0:
            ball_y_positions.append(ball_y_indices.mean())
        else:
            ball_y_positions.append(-1.0)  # Unknown/Not visible
            
    ball_y_positions = np.array(ball_y_positions)
    visible_mask = ball_y_positions >= 0
    
    plt.figure(figsize=(10, 8), dpi=150)
    
    # Plot non-visible ball states as gray
    plt.scatter(z_2d[~visible_mask, 0], z_2d[~visible_mask, 1], c="gray", alpha=0.3, label="Ball not visible/centered")
    
    # Plot visible ball states colored by Y-position
    sc = plt.scatter(
        z_2d[visible_mask, 0], 
        z_2d[visible_mask, 1], 
        c=ball_y_positions[visible_mask], 
        cmap="coolwarm", 
        alpha=0.8, 
        label="Ball visible (colored by Y-position)"
    )
    
    plt.colorbar(sc, label="Ball Y-coordinate in Frame")
    plt.title("JEPA Latent Space t-SNE Projection (Pong)", fontsize=14)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    print(f"Saved latent space visualization to {save_path}")
    plt.close()

def evaluate_dynamics_drift(model, z_list, actions, device="cpu", save_path="data/dynamics_drift.png"):
    """Computes and plots prediction drift over a multi-step open-loop rollout in latent space."""
    print("Evaluating dynamics model prediction drift...")
    model.eval()
    
    z_t = torch.as_tensor(z_list[0], device=device).unsqueeze(0)
    z_pred_history = [z_t.squeeze(0).cpu().numpy()]
    
    # Open-loop rollout: feed actions to predictor without correcting with the actual encoder
    with torch.no_grad():
        for i in range(len(actions) - 1):
            action_t = torch.as_tensor([actions[i]], device=device, dtype=torch.long)
            # Predict next latent state from previous predicted state
            z_t = model.predictor(z_t, action_t)
            z_pred_history.append(z_t.squeeze(0).cpu().numpy())
            
    z_pred_history = np.array(z_pred_history)
    
    # Calculate MSE drift at each time step
    mse_drift = np.mean((z_list - z_pred_history)**2, axis=1)
    
    plt.figure(figsize=(8, 5), dpi=150)
    plt.plot(mse_drift, color="crimson", linewidth=2.0, label="Open-loop prediction drift (MSE)")
    plt.title("JEPA Predictor Multi-Step Drift in Latent Space", fontsize=12)
    plt.xlabel("Time Step (Prediction Horizon)")
    plt.ylabel("Mean Squared Error (MSE)")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    print(f"Saved prediction drift plot to {save_path}")
    plt.close()

def play_agent(env_id, jepa_path, dqn_path, device="cpu", episodes=5):
    """Play Pong using pre-trained JEPA encoder and DQN QNetwork."""
    print(f"Running visual evaluation in environment for {episodes} episodes...")
    env = make_pong(env_id=env_id, render_mode="human")
    
    # Load model checkpoints
    jepa_checkpoint = torch.load(jepa_path, map_location=device, weights_only=False)

    jepa_model = JEPAWorldModel(in_channels=4, latent_dim=256, action_dim=6).to(device)
    jepa_model.load_state_dict(jepa_checkpoint["model_state_dict"])
    encoder = jepa_model.online_encoder
    encoder.eval()
    
    dqn = QNetwork(latent_dim=256, action_dim=6).to(device)
    dqn.load_state_dict(torch.load(dqn_path, map_location=device, weights_only=False))

    dqn.eval()
    
    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        ep_reward = 0
        
        while not done:
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0) / 255.0
                z = encoder(obs_t)
                q_values = dqn(z)
                action = q_values.argmax(dim=-1).item()
                
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            
        print(f"Visual Episode {ep+1} | Score: {ep_reward:.1f}")
        
    env.close()

def main():
    parser = argparse.ArgumentParser(description="Evaluate and visualize JEPA latent space and dynamics model")
    parser.add_argument("--jepa_path", type=str, default="checkpoints/jepa_best.pt", help="Path to pre-trained JEPA model")
    parser.add_argument("--dqn_path", type=str, default="checkpoints/dqn_policy_best.pt", help="Path to trained DQN policy model")
    parser.add_argument("--env_id", type=str, default="PongNoFrameskip-v4", help="Environment ID")
    parser.add_argument("--device", type=str, default="auto", help="Execution device")
    parser.add_argument("--num_steps", type=int, default=1000, help="Steps to collect for offline visualization")
    parser.add_argument("--play", action="store_true", help="Launch interactive play window with trained agent")
    parser.add_argument("--episodes", type=int, default=5, help="Number of visualization play episodes")
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    if args.play:
        play_agent(args.env_id, args.jepa_path, args.dqn_path, device=device, episodes=args.episodes)
        return

    # Offline analysis
    print(f"Loading pre-trained JEPA model: {args.jepa_path}")
    if not os.path.exists(args.jepa_path):
        raise FileNotFoundError(f"Checkpoint {args.jepa_path} not found.")
        
    checkpoint = torch.load(args.jepa_path, map_location=device, weights_only=False)

    model = JEPAWorldModel(in_channels=4, latent_dim=256, action_dim=6).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    print("Collecting environment rollout transitions...")
    env = make_pong(env_id=args.env_id)
    states_list, z_list, actions, rewards = get_rollout_data(env, model.online_encoder, num_steps=args.num_steps, device=device)
    env.close()
    
    # 1. Plot Latent Space
    plot_latent_space(z_list, states_list, save_path="data/latent_space_tsne.png")
    
    # 2. Evaluate Prediction Drift
    evaluate_dynamics_drift(model, z_list, actions, device=device, save_path="data/dynamics_drift.png")
    
    print("Evaluation and visualization completed successfully.")

if __name__ == "__main__":
    main()
