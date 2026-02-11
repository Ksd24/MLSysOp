import torch
import torch.nn as nn

class QNetwork(nn.Module):
    """
    Defines the Deep Q-Network architecture.
    """
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        # Input layer: state_dim (e.g., 2 for x,y coordinates)
        # Hidden layers: two fully connected layers with ReLU activation
        # Output layer: action_dim (e.g., 4 for Up, Down, Left, Right)
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, state):
        """
        Forward pass through the network.
        Args:
            state (torch.Tensor): The current state observation.
        Returns:
            torch.Tensor: Q-values for each action in the given state.
        """
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

if __name__ == '__main__':
    # Simple test for the QNetwork
    state_dim = 2  # (x, y)
    action_dim = 4 # Up, Down, Left, Right

    model = QNetwork(state_dim, action_dim)
    print("QNetwork Architecture:")
    print(model)

    # Create a dummy state tensor
    dummy_state = torch.randn(1, state_dim) # Batch size of 1
    print(f"\nDummy State Input Shape: {dummy_state.shape}")

    # Get Q-values
    q_values = model(dummy_state)
    print(f"Output Q-values Shape: {q_values.shape}")
    print(f"Output Q-values: {q_values}")

    dummy_batch_states = torch.randn(5, state_dim) # Batch size of 5
    batch_q_values = model(dummy_batch_states)
    print(f"\nBatch States Input Shape: {dummy_batch_states.shape}")
    print(f"Batch Output Q-values Shape: {batch_q_values.shape}")
    print(f"Batch Output Q-values (first 2): {batch_q_values[:2]}")