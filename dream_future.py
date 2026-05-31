import os
import argparse
import numpy as np
import torch
import cv2
from PIL import Image

from utils.env import make_pong
from models.jepa import JEPAWorldModel
from models.decoder import SpatialDecoder
from models.policy import QNetwork

def make_panel(true_img, recon_img, dream_img, step_idx, action_name, true_reward, pred_reward, pred_done_prob):
    # Scale up each image from 64x64 to 256x256 using nearest-neighbor
    true_up = cv2.resize(true_img, (256, 256), interpolation=cv2.INTER_NEAREST)
    recon_up = cv2.resize(recon_img, (256, 256), interpolation=cv2.INTER_NEAREST)
    dream_up = cv2.resize(dream_img, (256, 256), interpolation=cv2.INTER_NEAREST)
    
    # Convert grayscale to RGB for colorful annotations
    if len(true_up.shape) == 2:
        true_up = cv2.cvtColor(true_up, cv2.COLOR_GRAY2RGB)
    if len(recon_up.shape) == 2:
        recon_up = cv2.cvtColor(recon_up, cv2.COLOR_GRAY2RGB)
    if len(dream_up.shape) == 2:
        dream_up = cv2.cvtColor(dream_up, cv2.COLOR_GRAY2RGB)
        
    # Draw label on true_up
    cv2.putText(true_up, "True Env", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(true_up, f"Act: {action_name}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(true_up, f"Rew: {true_reward:+.1f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    
    # Draw label on recon_up
    cv2.putText(recon_up, "Decoded Real", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(recon_up, "From Latent z_t", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Draw label on dream_up
    cv2.putText(dream_up, "Dream", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
    cv2.putText(dream_up, f"Imagined t+{step_idx}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 1)
    cv2.putText(dream_up, f"Pred Rew: {pred_reward:+.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
    cv2.putText(dream_up, f"Done Prob: {pred_done_prob:.2%}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
    
    # Concatenate horizontally
    canvas = np.concatenate([true_up, recon_up, dream_up], axis=1)
    
    # Add borders between panels to look professional
    cv2.line(canvas, (256, 0), (256, 256), (80, 80, 80), 3)
    cv2.line(canvas, (512, 0), (512, 256), (80, 80, 80), 3)
    
    return canvas

def main():
    parser = argparse.ArgumentParser(description="Visualize World Model Imagination Rollout ('Dreaming')")
    parser.add_argument("--jepa_path", type=str, default="checkpoints/jepa_best.pt", help="Path to pre-trained JEPA")
    parser.add_argument("--decoder_path", type=str, default="checkpoints/decoder_best.pt", help="Path to trained Spatial Decoder")
    parser.add_argument("--policy_path", type=str, default="checkpoints/dqn_policy_best.pt", help="Path to trained Q-network")
    parser.add_argument("--env_id", type=str, default="PongNoFrameskip-v4", help="Atari environment")
    parser.add_argument("--warmup_steps", type=int, default=80, help="Steps to run in real env before starting dream")
    parser.add_argument("--dream_steps", type=int, default=40, help="Steps to imagine forward")
    parser.add_argument("--device", type=str, default="auto", help="Execution device")
    parser.add_argument("--save_path", type=str, default="checkpoints/dream_rollout.gif", help="Path to save output GIF")
    parser.add_argument("--use_wandb", action="store_true", help="Log output GIF to W&B")
    parser.add_argument("--project_name", type=str, default="pong-jepa", help="W&B project name")
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Running dreaming rollout on: {device}")

    # 1. Load models
    print("Loading pre-trained models...")
    if not os.path.exists(args.jepa_path):
        raise FileNotFoundError(f"JEPA model not found at {args.jepa_path}")
    if not os.path.exists(args.decoder_path):
        raise FileNotFoundError(f"Spatial Decoder model not found at {args.decoder_path}")
    if not os.path.exists(args.policy_path):
        raise FileNotFoundError(f"Policy model not found at {args.policy_path}")

    # Load JEPA World Model
    jepa_checkpoint = torch.load(args.jepa_path, map_location=device, weights_only=False)
    jepa_action_dim = jepa_checkpoint["model_state_dict"]["predictor.action_embed.weight"].shape[0]
    jepa_model = JEPAWorldModel(in_channels=4, latent_dim=256, action_dim=jepa_action_dim).to(device)
    jepa_model.load_state_dict(jepa_checkpoint["model_state_dict"])
    encoder = jepa_model.online_encoder
    predictor = jepa_model.predictor
    reward_done_predictor = jepa_model.reward_done_predictor

    encoder.eval()
    predictor.eval()
    reward_done_predictor.eval()

    # Load Spatial Decoder
    decoder = SpatialDecoder(latent_dim=256).to(device)
    decoder.load_state_dict(torch.load(args.decoder_path, map_location=device, weights_only=False))
    decoder.eval()

    # Load DQN Policy Q-Network
    policy_state_dict = torch.load(args.policy_path, map_location=device, weights_only=False)
    policy_action_dim = policy_state_dict["net.4.weight"].shape[0] if "net.4.weight" in policy_state_dict else 6
    q_net = QNetwork(latent_dim=256, action_dim=policy_action_dim).to(device)
    q_net.load_state_dict(policy_state_dict)
    q_net.eval()

    print("All models successfully loaded and set to eval.")

    # 2. Setup Environment
    print(f"Creating environment: {args.env_id}")
    env = make_pong(env_id=args.env_id)
    
    # Resolve action names based on action space dimension
    if env.action_space.n == 3:
        action_names = {0: "NOOP", 1: "UP", 2: "DOWN"}
    else:
        try:
            raw_meanings = env.unwrapped.get_action_meanings()
            action_names = {i: raw_meanings[i] for i in range(len(raw_meanings))}
        except Exception:
            action_names = {i: f"ACTION_{i}" for i in range(env.action_space.n)}

    obs, _ = env.reset()
    done = False
    
    # 3. Warmup Phase (run the DQN policy in real environment to get active gameplay)
    print(f"Warming up environment for {args.warmup_steps} steps...")
    for step in range(args.warmup_steps):
        with torch.no_grad():
            state_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0) / 255.0
            z = encoder(state_tensor)
            q_values = q_net(z)
            action = q_values.argmax(dim=-1).item()
            
        obs, reward, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            print("Warning: Environment terminated during warmup! Resetting...")
            obs, _ = env.reset()

    # 4. Dreaming Phase
    print(f"Warmup completed. Starting {args.dream_steps}-step dreaming imagination rollout...")
    
    panels = []
    
    # Initial state encoding
    with torch.no_grad():
        state_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0) / 255.0
        z_t = encoder(state_tensor)
        
    # We initialize the dream latent code z_dream as the starting true latent code z_t
    z_dream = z_t.clone()

    for k in range(1, args.dream_steps + 1):
        # Decide action based on the true state representation using the policy
        with torch.no_grad():
            q_values = q_net(z_t)
            action = q_values.argmax(dim=-1).item()
            action_name = action_names.get(action, f"ACT_{action}")
            
        # Step environment (Real Ground Truth path)
        next_obs, true_reward, terminated, truncated, _ = env.step(action)
        
        # Encode next real state
        with torch.no_grad():
            next_state_tensor = torch.as_tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0) / 255.0
            z_next = encoder(next_state_tensor)
            
        # Decode the real state representation (Reconstruction path)
        with torch.no_grad():
            recon_frame_tensor = decoder(z_next)
            
        # Transition the dream representation (Imagination path)
        action_tensor = torch.tensor([action], device=device, dtype=torch.long)
        with torch.no_grad():
            z_dream_next = predictor(z_dream, action_tensor)
            dream_frame_tensor = decoder(z_dream_next)
            
            # Predict reward and done logit for the dreamed state
            pred_reward_tensor, pred_done_logit = reward_done_predictor(z_dream, action_tensor)
            pred_reward = pred_reward_tensor.item()
            pred_done_prob = torch.sigmoid(pred_done_logit).item()
            
        # Convert tensors to numpy arrays for visualization
        true_frame = next_obs[-1].astype(np.uint8)  # actual raw frame (most recent)
        recon_frame = (recon_frame_tensor[0, 0].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        dream_frame = (dream_frame_tensor[0, 0].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        
        # Build side-by-side comparison canvas
        canvas = make_panel(
            true_frame, 
            recon_frame, 
            dream_frame, 
            k, 
            action_name, 
            true_reward, 
            pred_reward, 
            pred_done_prob
        )
        panels.append(canvas)
        
        # Advance representations
        z_t = z_next
        z_dream = z_dream_next
        obs = next_obs
        
        if terminated or truncated:
            print(f"Environment terminated at step {k} of dream rollout.")
            break

    env.close()

    # 5. Compile and Save GIF
    print(f"Saving compiled dreaming rollout GIF to {args.save_path}...")
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    gif_frames = [Image.fromarray(canvas) for canvas in panels]
    gif_frames[0].save(
        args.save_path,
        save_all=True,
        append_images=gif_frames[1:],
        duration=120,
        loop=0
    )
    print("Dreaming GIF compiled successfully!")

    # 6. Log to W&B if requested
    if args.use_wandb:
        try:
            import wandb
            # Check if there is an active run, otherwise init a new one
            if wandb.run is None:
                wandb.init(project=args.project_name, name="latent-dream-rollout", config=vars(args))
            
            wandb.log({
                "imagination_rollout": wandb.Video(args.save_path, format="gif", fps=8, caption="JEPA Latent dreaming rollout")
            })
            print("Successfully uploaded dreaming GIF to Wandb.")
            wandb.finish()
        except ImportError:
            print("Wandb not installed. Skipping upload.")

if __name__ == "__main__":
    main()
