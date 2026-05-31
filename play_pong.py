import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from utils.env import make_pong
from models.jepa import JEPAWorldModel
from models.policy import QNetwork

class SimpleRLBuffer:
    """A lightweight buffer for storing latent transitions (z_t, a_t, r_t, z_{t+1}, done_t)."""
    def __init__(self, capacity, latent_dim=256, device="cpu"):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0
        
        self.states = torch.zeros((capacity, latent_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros(capacity, dtype=torch.long, device=device)
        self.rewards = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.next_states = torch.zeros((capacity, latent_dim), dtype=torch.float32, device=device)
        self.dones = torch.zeros(capacity, dtype=torch.float32, device=device)

    def add(self, z, action, reward, z_next, done):
        self.states[self.ptr] = z.detach()
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = z_next.detach()
        self.dones[self.ptr] = float(done)
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idxs = torch.randint(0, self.size, (batch_size,), device=self.device)
        return (
            self.states[idxs],
            self.actions[idxs],
            self.rewards[idxs],
            self.next_states[idxs],
            self.dones[idxs]
        )

def main():
    parser = argparse.ArgumentParser(description="Train DQN Policy on top of frozen JEPA representations")
    parser.add_argument("--jepa_path", type=str, default="checkpoints/jepa_best.pt", help="Path to pre-trained JEPA checkpoint")
    parser.add_argument("--env_id", type=str, default="PongNoFrameskip-v4", help="Atari environment")
    parser.add_argument("--episodes", type=int, default=500, help="Number of training episodes")
    parser.add_argument("--buffer_size", type=int, default=50000, help="Replay buffer size")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--target_update_freq", type=int, default=1000, help="Target network update frequency (steps)")
    parser.add_argument("--epsilon_start", type=float, default=1.0, help="Starting epsilon")
    parser.add_argument("--epsilon_end", type=float, default=0.05, help="Ending epsilon")
    parser.add_argument("--epsilon_decay", type=int, default=100000, help="Number of steps for linear decay of epsilon")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--device", type=str, default="auto", help="Device to run training on")
    parser.add_argument("--render", action="store_true", help="Render environment during training")
    parser.add_argument("--use_wandb", action="store_true", help="Log metrics to Weights & Biases")
    parser.add_argument("--project_name", type=str, default="pong-jepa", help="Wandb project name")
    parser.add_argument("--reward_shaping", action="store_true", help="Use dense reward shaping (+0.2 on paddle ball hit)")


    args = parser.parse_args()

    # Initialize wandb
    if args.use_wandb:
        try:
            import wandb
            wandb.init(project=args.project_name, config=vars(args))
            print("Successfully initialized Wandb RL logging.")
        except ImportError:
            print("Wandb not installed. Defaulting to standard training only.")
            args.use_wandb = False


    # Device configuration
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device
    print(f"Running RL training on: {device}")

    # 1. Load pre-trained JEPA model and extract the online encoder
    print(f"Loading pre-trained JEPA from {args.jepa_path}...")
    if not os.path.exists(args.jepa_path):
        raise FileNotFoundError(f"Checkpoint not found at {args.jepa_path}. Please train the JEPA model first!")
    
    checkpoint = torch.load(args.jepa_path, map_location=device, weights_only=False)

    
    # We instantiate a model, load weights, and extract encoder
    jepa_model = JEPAWorldModel(in_channels=4, latent_dim=256, action_dim=6).to(device)
    jepa_model.load_state_dict(checkpoint["model_state_dict"])
    
    encoder = jepa_model.online_encoder
    encoder.eval()
    
    # Freeze encoder parameters
    for param in encoder.parameters():
        param.requires_grad = False
    print("Pre-trained Encoder successfully loaded and frozen.")

    # 2. Setup Gymnasium environment
    print(f"Creating environment: {args.env_id}")
    render_mode = "human" if args.render else None
    env = make_pong(env_id=args.env_id, render_mode=render_mode, reward_shaping=args.reward_shaping)
    action_dim = env.action_space.n

    # 3. Initialize DQN policy networks
    q_net = QNetwork(latent_dim=256, action_dim=action_dim).to(device)
    target_q_net = QNetwork(latent_dim=256, action_dim=action_dim).to(device)
    target_q_net.load_state_dict(q_net.state_dict())
    
    optimizer = optim.AdamW(q_net.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    buffer = SimpleRLBuffer(capacity=args.buffer_size, latent_dim=256, device=device)
    
    global_step = 0
    best_return = -21.0
    episode_returns = []

    print("Beginning RL training loop...")
    for ep in range(1, args.episodes + 1):
        obs, info = env.reset()
        done = False
        ep_return = 0
        
        # Get initial state representation
        with torch.no_grad():
            state_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0) / 255.0
            z = encoder(state_tensor).squeeze(0)  # Shape (latent_dim,)
            
        while not done:
            # 1. Epsilon-greedy action selection
            epsilon = max(
                args.epsilon_end,
                args.epsilon_start - (args.epsilon_start - args.epsilon_end) * (global_step / args.epsilon_decay)
            )
            
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q_values = q_net(z.unsqueeze(0))
                    action = q_values.argmax(dim=-1).item()
            
            # 2. Step environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_return += reward
            
            # 3. Get next state representation
            with torch.no_grad():
                next_state_tensor = torch.as_tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0) / 255.0
                z_next = encoder(next_state_tensor).squeeze(0)
            
            # 4. Store transition
            buffer.add(z, action, reward, z_next, done)
            
            # Move forward
            obs = next_obs
            z = z_next
            global_step += 1

            # 5. Optimization step
            if buffer.size > args.batch_size:
                bz, ba, br, bz_next, bd = buffer.sample(args.batch_size)
                
                # Current Q-values
                current_q = q_net(bz).gather(1, ba.unsqueeze(-1)).squeeze(-1)
                
                # Target Q-values
                with torch.no_grad():
                    next_q = target_q_net(bz_next).max(dim=-1)[0]
                    target_q = br + args.gamma * next_q * (1.0 - bd)
                
                loss = loss_fn(current_q, target_q)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            # 6. Update Target network
            if global_step % args.target_update_freq == 0:
                target_q_net.load_state_dict(q_net.state_dict())

        # Track history
        episode_returns.append(ep_return)
        mean_return_10 = np.mean(episode_returns[-10:])
        
        print(f"Episode {ep:03d} | Length: {global_step} total steps | Return: {ep_return:.1f} | Epsilon: {epsilon:.3f} | Mean 10-Ep Return: {mean_return_10:.2f}")

        # Log episode metrics to Wandb
        if args.use_wandb:
            wandb.log({
                "episode": ep,
                "return": ep_return,
                "mean_return_10": mean_return_10,
                "epsilon": epsilon,
                "steps": global_step
            }, step=ep)

        # Periodic gameplay evaluation & GIF generation
        if args.use_wandb and ep % 50 == 0:
            print(f"Episode {ep:03d} | Generating and logging evaluation gameplay GIF...")
            try:
                eval_env = make_pong(env_id=args.env_id)
                eval_obs, _ = eval_env.reset()
                eval_done = False
                eval_frames = []
                eval_return = 0
                eval_step_cnt = 0
                
                # Limit to 1500 steps to keep memory and CPU usage low
                while not eval_done and eval_step_cnt < 1500:
                    # Capture the current frame from frame stack (last channel is most recent frame)
                    frame = eval_obs[-1]
                    frame_rgb = np.stack([frame]*3, axis=-1)
                    eval_frames.append(frame_rgb)
                    
                    with torch.no_grad():
                        state_tensor = torch.as_tensor(eval_obs, dtype=torch.float32, device=device).unsqueeze(0) / 255.0
                        z_eval = encoder(state_tensor)
                        q_values = q_net(z_eval)
                        eval_action = q_values.argmax(dim=-1).item()
                        
                    eval_obs, eval_reward, eval_terminated, eval_truncated, _ = eval_env.step(eval_action)
                    eval_done = eval_terminated or eval_truncated
                    eval_return += eval_reward
                    eval_step_cnt += 1
                
                eval_env.close()
                
                if len(eval_frames) > 0:
                    from PIL import Image
                    # Subsample frames to reduce final file size and increase play speed
                    gif_frames = [Image.fromarray(f) for f in eval_frames[::2]]
                    gif_path = os.path.join(args.save_dir, f"eval_play_episode_{ep}.gif")
                    os.makedirs(args.save_dir, exist_ok=True)
                    gif_frames[0].save(
                        gif_path, 
                        save_all=True, 
                        append_images=gif_frames[1:], 
                        duration=80, 
                        loop=0
                    )
                    
                    wandb.log({
                        "eval_gameplay_gif": wandb.Image(gif_path, caption=f"Evaluation Gameplay GIF at Episode {ep}"),
                        "eval_return": eval_return
                    }, step=ep)
                    print(f"Successfully uploaded gameplay GIF to Wandb with evaluation return {eval_return:.1f}")
            except Exception as gif_err:
                print(f"Warning: Failed to generate or upload gameplay GIF: {gif_err}")

        # Checkpoint saving
        os.makedirs(args.save_dir, exist_ok=True)
        if ep_return > best_return:
            best_return = ep_return
            torch.save(q_net.state_dict(), os.path.join(args.save_dir, "dqn_policy_best.pt"))
            print(f"--> Saved new best policy checkpoint with return {best_return:.1f}")

        # Save regular checkpoint
        if ep % 50 == 0:
            torch.save(q_net.state_dict(), os.path.join(args.save_dir, f"dqn_policy_{ep}ep.pt"))

    print("RL training completed!")
    env.close()
    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
