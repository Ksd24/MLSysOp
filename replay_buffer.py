from collections import deque
import random
import numpy as np
import torch
import multiprocessing as mp

class ReplayBuffer:
    """
    A replay buffer to store experience tuples and sample mini-batches.
    Designed for use within a single process (e.g., the learner).
    """
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        # No lock needed here as it's intended for single-process use,
        # experiences are fed into it via a Queue from other processes.

    def add(self, state, action, reward, next_state, done):
        """Adds an experience tuple to the buffer."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        Samples a batch of experiences from the buffer.
        Args:
            batch_size (int): The number of experiences to sample.
        Returns:
            tuple: Tensors for (states, actions, rewards, next_states, dones)
        """
        if len(self.buffer) < batch_size:
            return None # Not enough experiences to sample a full batch

        batch = random.sample(self.buffer, batch_size)
        
        # Unpack experiences and convert to NumPy arrays first
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert NumPy arrays to PyTorch tensors
        states_t = torch.tensor(np.array(states), dtype=torch.float32)
        actions_t = torch.tensor(np.array(actions), dtype=torch.long)
        rewards_t = torch.tensor(np.array(rewards), dtype=torch.float32)
        next_states_t = torch.tensor(np.array(next_states), dtype=torch.float32)
        dones_t = torch.tensor(np.array(dones), dtype=torch.float32) # Convert bool to float for calculations

        return states_t, actions_t, rewards_t, next_states_t, dones_t

    def __len__(self):
        """Returns the current size of the buffer."""
        return len(self.buffer)

if __name__ == '__main__':
    # Simple test for the ReplayBuffer
    buffer = ReplayBuffer(capacity=10)
    
    # Add some dummy experiences
    for i in range(15):
        state = np.array([i, i+1], dtype=np.float32)
        action = i % 4
        reward = float(i)
        next_state = np.array([i+1, i+2], dtype=np.float32)
        done = (i % 5 == 0)
        buffer.add(state, action, reward, next_state, done)
    
    print(f"Buffer size: {len(buffer)}") # Should be 10 due to maxlen
    
    # Sample a batch
    batch_size = 5
    if len(buffer) >= batch_size:
        states, actions, rewards, next_states, dones = buffer.sample(batch_size)
        print(f"\nSampled Batch (size {batch_size}):")
        print(f"States shape: {states.shape}, dtype: {states.dtype}")
        print(f"Actions shape: {actions.shape}, dtype: {actions.dtype}")
        print(f"Rewards shape: {rewards.shape}, dtype: {rewards.dtype}")
        print(f"Next States shape: {next_states.shape}, dtype: {next_states.dtype}")
        print(f"Dones shape: {dones.shape}, dtype: {dones.dtype}")
        
        print("\nFirst sampled experience:")
        print(f"  State: {states[0].numpy()}")
        print(f"  Action: {actions[0].item()}")
        print(f"  Reward: {rewards[0].item()}")
        print(f"  Next State: {next_states[0].numpy()}")
        print(f"  Done: {dones[0].item()}")
    else:
        print("Not enough experiences in buffer to sample.")