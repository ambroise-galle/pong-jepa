import os
import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, capacity, state_shape=(4, 64, 64), device="cpu"):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0

        # Pre-allocate memory using numpy arrays (highly efficient)
        self.states = np.zeros((capacity, *state_shape), dtype=np.uint8)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.next_states = np.zeros((capacity, *state_shape), dtype=np.uint8)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)

    def add(self, state, action, next_state, reward, done):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.next_states[self.ptr] = next_state
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)

        # Convert to torch tensors and normalize states to [0.0, 1.0]
        states_t = torch.as_tensor(self.states[idxs], device=self.device, dtype=torch.float32) / 255.0
        actions_t = torch.as_tensor(self.actions[idxs], device=self.device, dtype=torch.int64)
        next_states_t = torch.as_tensor(self.next_states[idxs], device=self.device, dtype=torch.float32) / 255.0
        rewards_t = torch.as_tensor(self.rewards[idxs], device=self.device, dtype=torch.float32)
        dones_t = torch.as_tensor(self.dones[idxs], device=self.device, dtype=torch.float32)

        return states_t, actions_t, next_states_t, rewards_t, dones_t

    def save(self, filepath):
        """Save replay buffer to a compressed numpy archive."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        np.savez_compressed(
            filepath,
            states=self.states[:self.size],
            actions=self.actions[:self.size],
            next_states=self.next_states[:self.size],
            rewards=self.rewards[:self.size],
            dones=self.dones[:self.size],
            ptr=self.ptr,
            size=self.size
        )
        print(f"Saved {self.size} transitions to {filepath}")

    def load(self, filepath):
        """Load replay buffer from a compressed numpy archive."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No file found at {filepath}")
        
        data = np.load(filepath)
        loaded_size = data['size']
        
        if loaded_size > self.capacity:
            raise ValueError(f"Loaded dataset size ({loaded_size}) exceeds buffer capacity ({self.capacity})")
        
        self.states[:loaded_size] = data['states']
        self.actions[:loaded_size] = data['actions']
        self.next_states[:loaded_size] = data['next_states']
        self.rewards[:loaded_size] = data['rewards']
        self.dones[:loaded_size] = data['dones']
        
        self.size = int(loaded_size)
        self.ptr = int(data['ptr'])
        print(f"Successfully loaded {self.size} transitions from {filepath}")
