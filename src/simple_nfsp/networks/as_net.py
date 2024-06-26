import torch
import torch.nn as nn
import torch.nn.functional as F

class ASNet(nn.Module):
    """Defines the neural network architecture for the average-strategy network."""

    def __init__(self, state_dim: int, action_dim: int) -> None:
        """Initializes the AS-Net with a simple feedforward architecture.

        Args:
            state_dim (int): Dimensionality of the state space.
            action_dim (int): Dimensionality of the action space.
        """
        super(ASNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)  # First fully connected layer
        self.head = nn.Linear(128, action_dim)  # Output layer

    def forward(self, state: torch.Tensor, legal_actions_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            state (torch.Tensor): Input state.
            legal_actions_mask (torch.Tensor): Mask of legal actions.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = F.relu(self.fc1(state))
        logits = self.head(x)

        # Apply legal actions mask
        masked_logits = logits + (legal_actions_mask - 1) * 1e9  # Use a large negative number to mask
        probabilities = F.softmax(masked_logits, dim=-1)
        return probabilities