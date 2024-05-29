import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Union


class BRNet(nn.Module):
    """Defines the neural network architecture for the best-response network."""

    def __init__(self, state_dim: int, action_dim: int) -> None:
        """Initializes the BR-Net with a simple feedforward architecture.

        Args:
            state_dim (int): Dimensionality of the state space.
            action_dim (int): Dimensionality of the action space.
        """
        super(BRNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)  # First fully connected layer
        self.fc2 = nn.Linear(128, 128)  # Second fully connected layer
        self.head = nn.Linear(128, action_dim) # Output layer

    def forward(self, state: torch.Tensor, legal_actions_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            state (torch.Tensor): Input state.
            legal_actions_mask (torch.Tensor): Mask of legal actions.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        logits = self.head(x)

        masked_logits = logits + (legal_actions_mask - 1) * 1e9  # Use a large negative number to mask
        probabilities = F.softmax(masked_logits, dim=-1)
        return probabilities
