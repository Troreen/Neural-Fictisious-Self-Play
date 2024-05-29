from typing import List, Union

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from simple_nfsp.agents.memory import RLMemory, SLMemory
from simple_nfsp.networks.as_net import ASNet
from simple_nfsp.networks.br_net import BRNet


class NFSPAgent:
    """Implements the NFSP agent, combining both RL and SL strategies."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        anticipatory_param: float,
        rl_learning_rate: float = 1e-4,
        sl_learning_rate: float = 1e-4,
        gamma: float = 0.99,
        device: str = "cpu",
    ) -> None:
        """Initializes the NFSP agent with both a best-response (BR-Net) and average-strategy (AS-Net) network.

        Args:
            state_dim (int): Dimensionality of the state space.
            action_dims (int): Dimensionality of the action space.
            anticipatory_param (float): Anticipatory parameter to balance between BR and AS
                strategies.
            rl_learning_rate (float, optional): Learning rate for the best-response network.
                Defaults to 1e-4.
            sl_learning_rate (float, optional): Learning rate for the average-strategy network.
                Defaults to 1e-4.
            gamma (float, optional): Discount factor for future rewards. Defaults to 0.99.
            device (str, optional): Device to run the agent on. Defaults to "cpu".
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.anticipatory_param = anticipatory_param
        self.device = device
        self.gamma = gamma

        # Whether the last action was best-response based
        self.last_action_best_response = False

        self.rl_memory = RLMemory(capacity=20000)  # RL experiences memory
        self.sl_memory = SLMemory(capacity=20000)  # SL experiences memory

        self.br_net = BRNet(state_dim, action_dim).to(device)  # Best-response network
        self.as_net = ASNet(state_dim, action_dim).to(device)  # Average-strategy network
        self.target_br_net = BRNet(state_dim, action_dim).to(device)  # Target best-response network
        self.update_target_network()

        self.br_optimizer = optim.Adam(self.br_net.parameters(), lr=rl_learning_rate)
        self.as_optimizer = optim.Adam(self.as_net.parameters(), lr=sl_learning_rate)

    def select_action(
        self, state: List[int], legal_actions: List[int], epsilon: float = 0.1
    ) -> Union[int, List[int]]:
        """Selects an action using an epsilon-greedy strategy for the RL part and a deterministic
        strategy for the SL part based on the anticipatory parameter.

        Args:
            state (List[int]): The current state.
            legal_actions (List[int]): The list of legal actions.
            epsilon (float): The exploration rate for epsilon-greedy action selection.

        Returns:
            Union[int, List[int]]: The selected action(s).
        """
        # Convert state to tensor for network input
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        legal_actions_mask = torch.zeros(self.action_dim).to(self.device)
        legal_actions_mask[legal_actions] = 1

        # Decide between using the average strategy network or best-response network
        use_br = np.random.rand() < self.anticipatory_param
        self.last_action_best_response = use_br

        # With probability (1 - anticipatory_param), use the average strategy network
        with torch.no_grad():
            if use_br:
                # Epsilon-greedy action selection for exploring RL network's strategy
                if np.random.rand() < epsilon:
                    # Explore: randomly choose among legal actions
                    action = np.random.choice(legal_actions)
                else:
                    # Exploit: choose the best action based on the network's output
                    q_values = self.br_net(state_tensor, legal_actions_mask)
                    action = q_values.argmax().item()
            else:
                probabilities  = self.as_net(state_tensor, legal_actions_mask)

                action = torch.multinomial(probabilities, 1).item()

        return action

    def store_experience_rl(
        self,
        state: List[int],
        action: int,
        reward: float,
        next_state: List[int],
        done: bool,
        legal_actions_mask: List[int],
    ) -> None:
        """Stores an experience tuple in the RL memory.

        Args:
            state (List[int]): The current state.
            action (int): The action taken.
            reward (float): The reward received.
            next_state (List[int]): The next state after the action.
            done (bool): Whether the episode has ended.
            legal_actions_mask (List[int]): The mask of legal actions.
        """
        self.rl_memory.add_experience(state, action, reward, next_state, done, legal_actions_mask)

    def store_experience_sl(
        self,
        state: List[int],
        action: int,
        legal_actions_mask: List[int],
    ) -> None:
        """Stores an experience tuple in the SL memory.

        Args:
            state (List[int]): The current state.
            action (int): The action taken.
            legal_actions_mask (List[int]): The mask of legal actions.
        """
        self.sl_memory.add_experience(state, action, legal_actions_mask)

    def update_target_network(self) -> None:
        """Updates the target best-response network with the weights of the current best-response
        network.
        """
        self.target_br_net.load_state_dict(self.br_net.state_dict())

    def train(self, batch_size: int) -> None:
        """Trains both the Best-Response Network (BR-Net) and the Average-Strategy Network (AS-Net)

        Trains both the Best-Response Network (BR-Net) and the Average-Strategy Network (AS-Net)
        using sampled experiences from their respective memories.

        Args:
            batch_size (int): Batch size for training.
        """
        if len(self.rl_memory) >= batch_size:
            # Sample from RL memory for BR-Net training
            rl_samples = self.rl_memory.sample(batch_size)
            rl_states, rl_actions, rl_rewards, rl_next_states, rl_dones, rl_legal_actions = zip(*rl_samples)

            # Convert lists of numpy arrays to numpy arrays
            rl_states = np.stack(rl_states)
            rl_next_states = np.stack(rl_next_states)

            # Convert to tensors
            rl_states = torch.tensor(rl_states, dtype=torch.float32).to(self.device)
            rl_next_states = torch.tensor(rl_next_states, dtype=torch.float32).to(self.device)
            rl_actions = torch.tensor(rl_actions, dtype=torch.int64).view(-1, 1).to(self.device)
            rl_rewards = torch.tensor(rl_rewards, dtype=torch.float32).to(self.device)
            rl_dones = torch.tensor(rl_dones, dtype=torch.float32).to(self.device)

            # Convert legal actions to masks
            rl_legal_actions_masks = torch.zeros((batch_size, self.action_dim), device=self.device)
            for idx, actions in enumerate(rl_legal_actions):
                rl_legal_actions_masks[idx, actions] = 1

            # BR-Net Q-learning update
            # Using the formula: Q(s, a) = Q(s, a) + alpha * (r + gamma * max(Q(s', a')) - Q(s, a))
            q_values = self.br_net(rl_states, rl_legal_actions_masks).gather(1, rl_actions)  # Q(s, a)
            next_q_values = (
                self.target_br_net(rl_next_states, rl_legal_actions_masks).max(1)[0].detach()
            )  # max(Q(s', a'))
            expected_q_values = rl_rewards + self.gamma * next_q_values * (
                1 - rl_dones
            )  # r + gamma * max(Q(s', a')) * (1 - done)
            # Note: 1 - done is used to zero out the Q-value if the next state is terminal
            # Note: detach() is used to prevent backpropagation through next_q_values (target network)
            # Note: gather() is used to select the Q-values for the actions taken
            # Note: There is a subtle difference between the formula and the code. Can you spot it?
            #       Yes, the formula is using - Q(s, a) while the code does not. This is because
            #       the loss function already includes a minus sign, so we are effectively minimizing
            #       the negative Q-value, which is equivalent to maximizing the Q-value.

            # Compute loss and backpropagate for BR-Net
            br_loss = F.mse_loss(q_values.squeeze(-1), expected_q_values)
            self.br_optimizer.zero_grad()
            br_loss.backward()
            self.br_optimizer.step()

        if len(self.sl_memory) >= batch_size:
            # Sample from SL memory for AS-Net training
            sl_samples = self.sl_memory.sample(batch_size)
            sl_states, sl_actions, sl_legal_actions = zip(*sl_samples)

            sl_states = np.stack(sl_states)

            # Convert to tensors
            sl_states = torch.tensor(sl_states, dtype=torch.float32).to(self.device)
            sl_actions = torch.tensor(sl_actions, dtype=torch.int64).to(self.device)

            # Efficient tensor construction for legal actions mask
            # Convert legal actions to masks
            sl_legal_actions_masks = torch.zeros((batch_size, self.action_dim), device=self.device)
            for idx, actions in enumerate(sl_legal_actions):
                sl_legal_actions_masks[idx, actions] = 1

            # AS-Net supervised learning update
            action_probs = self.as_net(sl_states, sl_legal_actions_masks)
            as_loss = F.cross_entropy(action_probs, sl_actions)

            # AS-Net supervised learning update
            action_probs = self.as_net(sl_states, sl_legal_actions_masks)
            as_loss = F.cross_entropy(action_probs, sl_actions)

            # Compute loss and backpropagate for AS-Net
            self.as_optimizer.zero_grad()
            as_loss.backward()
            self.as_optimizer.step()