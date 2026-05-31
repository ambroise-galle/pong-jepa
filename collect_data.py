import os
import argparse
import numpy as np
import gymnasium as gym
from utils.env import make_pong
from utils.buffer import ReplayBuffer

def get_heuristic_action(state, epsilon=0.15):
    """
    Heuristic Pong policy: Tracks the ball Y-position and moves the paddle towards it.
    Input state: numpy array of shape (4, 64, 64) with uint8 values in [0, 255].
    """
    if np.random.rand() < epsilon:
        return np.random.choice([0, 1, 2])  # NOOP, UP, DOWN in the restricted 3-action space

    # Use the most recent frame in the stack (index -1)
    frame = state[-1]

    # Crop out scores (top of the screen)
    # The resolution is 64x64. Let's look at y from 10 to 64.
    # Right paddle (controlled agent) is at columns 56-59
    # Left paddle (opponent) is at columns 4-7
    # Ball can be anywhere in columns 8-55
    
    # 1. Find ball position
    # The ball is bright white in grayscale (value usually > 140)
    ball_y_indices, ball_x_indices = np.where(frame[10:60, 8:55] > 140)
    if len(ball_y_indices) > 0:
        # Get mean position and adjust for crop offset (10)
        ball_y = ball_y_indices.mean() + 10
    else:
        ball_y = None

    # 2. Find paddle position
    paddle_y_indices, paddle_x_indices = np.where(frame[10:60, 56:59] > 100)
    if len(paddle_y_indices) > 0:
        paddle_y = paddle_y_indices.mean() + 10
    else:
        paddle_y = None

    # Heuristic control rule
    if ball_y is None or paddle_y is None:
        return np.random.choice([0, 1, 2])  # NOOP/UP/DOWN randomly if objects are not found
    
    # Restricted controls: 1 is UP, 2 is DOWN, 0 is NOOP
    if ball_y < paddle_y - 2:
        return 1  # Move paddle UP (raw 2)
    elif ball_y > paddle_y + 2:
        return 2  # Move paddle DOWN (raw 3)
    else:
        return 0  # Stay (NOOP)


def main():
    parser = argparse.ArgumentParser(description="Collect Pong transitions for JEPA pre-training")
    parser.add_argument("--env_id", type=str, default="PongNoFrameskip-v4", help="Gym environment ID")
    parser.add_argument("--num_steps", type=int, default=100000, help="Number of transitions to collect")
    parser.add_argument("--save_path", type=str, default="data/pong_transitions_100k.npz", help="Path to save dataset")
    parser.add_argument("--epsilon", type=float, default=0.2, help="Exploration epsilon for heuristic policy")
    parser.add_argument("--render", action="store_true", help="Render the environment during collection")
    args = parser.parse_args()

    print(f"Initializing {args.env_id} environment...")
    render_mode = "human" if args.render else None
    env = make_pong(env_id=args.env_id, render_mode=render_mode, width=64, height=64, k=4)
    
    buffer = ReplayBuffer(capacity=args.num_steps, state_shape=(4, 64, 64))
    
    obs, info = env.reset()
    episodes = 0
    ep_reward = 0
    ep_steps = 0
    total_rewards = []
    
    print(f"Starting data collection for {args.num_steps} steps using epsilon-heuristic policy...")
    
    for step in range(args.num_steps):
        action = get_heuristic_action(obs, epsilon=args.epsilon)
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        done = terminated or truncated
        
        # Save to replay buffer
        buffer.add(obs, action, next_obs, reward, done)
        
        obs = next_obs
        ep_reward += reward
        ep_steps += 1
        
        if done:
            episodes += 1
            total_rewards.append(ep_reward)
            if episodes % 5 == 0:
                print(f"Step {step+1}/{args.num_steps} | Episode {episodes} ended | Length: {ep_steps} | Reward: {ep_reward:.1f} | Mean Reward: {np.mean(total_rewards[-10:]):.2f}")
            
            obs, info = env.reset()
            ep_reward = 0
            ep_steps = 0

    print("Data collection complete.")
    env.close()
    
    # Save collected dataset
    buffer.save(args.save_path)

if __name__ == "__main__":
    main()
