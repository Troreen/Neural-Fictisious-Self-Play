import torch
import torch.nn as nn
import torch.nn.functional as F

class BRNet(nn.Module):
    """Defines the neural network architecture for the best-response network."""

    def __init__(self, state_dim: int, action_dim: int) -> None:
        """Initializes the BR-Net with a simple feedforward architecture.

        Args:
            state_dim (int): Dimensionality of the state space.
            action_dim (int): Dimensionality of the action space.
        """
        super(BRNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.head = nn.Linear(128, action_dim)

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
        return masked_logits