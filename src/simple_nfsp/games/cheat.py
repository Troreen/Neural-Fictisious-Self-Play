import random
from typing import Dict, List, Tuple
import numpy as np
from simple_nfsp.games.components.deck import Deck, NUM_UNIQUE_CARDS, NUM_REPLICAS
from simple_nfsp.games.components.player import Player
import logging
import torch

logger = logging.getLogger(__name__)

"""
We have a complicated action representation in Cheat since there is a large action space.

Here are the action representation details:

Since we have sequential actions with different legalities at each step, we are describing the
details here:

- The action representation is always in the form of one-hot encoding in the size of the action
    space.
- Here is the action sequence for a player turn:
    - Declare if the player is challenging the previous player's claim (0 or 1).
    - Declare the number of cards to play. [1, 0, 0, 0] -> 1 card, [0, 1, 0, 0] -> 2 cards, etc.
    - Declare the rank of the cards to play. This is a 13-dimensional vector where each element
        corresponds to a different rank. [0, 1, 0, ..., 0] -> 3, [0, 0, 1, ..., 0] -> 4, etc. The
        ranks go from 2 to 14 where 2 is 2, 3 is 3, ..., 10 is 10, Jack is 11, Queen is 12, King is
        13, and Ace is 14.
    - For each card to play (defined in the first step);
        - Declare the card to play. Again this is a 13-dimensional vector where each element
            corresponds to a different rank. [0, 0, ..., 0, 1] -> Ace, [1, 0, ..., 0, 0] -> 2, etc.
- All the actions will be padded to the maximum length of the action space (13).
- The legality of the actions will be determined based on the action history and the player's hand.
"""


class CheatGame:
    def __init__(self, num_players: int = 2, num_rounds: int = 50):
        self.num_players = num_players
        self.players = [Player(f"Player {i}") for i in range(self.num_players)]

        self.max_number_of_rounds = num_rounds

        self.invalid_penalty = -99

        self.player_hand_size = NUM_UNIQUE_CARDS
        self.max_unique_pile_len = NUM_UNIQUE_CARDS

        # (players action space + opponents turn exposure) * max number of rounds
        # players action space = challenge (1) + num_cards (1) + rank (1) + cards (4) = 7
        # opponents turn exposure = num_cards (1) + rank (1) + challenge (1) = 3
        self.max_history_len = 10 * self.max_number_of_rounds * 2

        # Maximum info_state size
        self.state_space = len(self.players[0].hand) + self.max_history_len

        # Action space for neural network - Head size for output layers
        self.action_space = NUM_UNIQUE_CARDS + 1 # 1 shifted since 0 represents "none"

        self.action_types = ['Challenge', 'Num Cards to Declare', 'Rank to Declare', 'Card to Play', 'Card to Play', 'Card to Play', 'Card to Play']

        self.reset()

    def reset(self) -> List[int]:
        """Reset the game state and return the initial information state.

        Returns:
            List[int]: The initial information state.
        """
        self.deck = Deck()
        self.num_rounds = 0
        cards_p_p = len(self.deck) // self.num_players
        for player in self.players:
            player.reset()
            player.hand = self.deck.deal(cards_p_p)

        self.next_action = 1  # first action is always num_cards
        self.reset_turn()

        self.state = np.array(
            [], dtype=int
        )  # empty state - we are using padding for stability
        self.player_history = [[] for _ in range(self.num_players)]

        self.current_player = 0
        self.central_pile = np.zeros(NUM_UNIQUE_CARDS, dtype=int)
        self.if_last_claim_bluff = False

        self.done = False

        return self.get_info_state()

    def legal_actions(self) -> List[int]:
        """Get the legal actions for the current player.

        Returns:
            List[int]: The legal actions for the current player.
        """
        if self.next_action == 0: # challenge
            return [0, 1]
        elif self.next_action == 1: # num_cards
            return list(range(1, min(len(self.players[self.current_player].hand) + 1, 5)))
        elif self.next_action == 2: # rank
            return list(range(1, 14))
        else: # cards
            return self.players[self.current_player].get_unique_cards()

    def process_action(self, action: int):
        """Process the action taken by the player.

        The action can be an integer from 1 to 13 indicating the action taken by the player. The
        action is processed based on the current state of the game.

        Args:
            action (int): The action taken by the player.
        """
        if self.next_action == 0: # challenge
            self.challenge_declared = action
            self.next_action = 1 # no matter the challenge, next action is num_cards
            if self.challenge_declared:
                self.turn_over = True
        elif self.next_action == 1: # num_cards
            self.num_cards_declared = action
            self.next_action = 2 # rank
            self.cards_played = [0] * 4
        elif self.next_action == 2:
            self.rank_declared = action
            self.next_action = 3 # card
        else: # cards
            self.cards_played[self.next_action - 3] = action
            if self.num_cards_declared == self.next_action - 3 + 1:
                self.next_action = 0 # challenge
                self.turn_over = True
            else:
                self.next_action += 1

    def reset_turn(self) -> None:
        """Reset the turn after the round ends.

        The a players turn can end in two ways:
            - The player successfully plays the cards.
            - The player challenges the previous player's claim.
        """
        self.num_cards_declared = None
        self.rank_declared = None
        self.cards_played = [None] * 4
        self.challenge_declared = None
        self.turn_over = False

    def get_info_state(self) -> np.array:
        """Get the current information state.

        An information state consists of the following components:
            - The current player's hand. This is vector of length 13, where each element represents
                the number of cards of a particular rank in the player's hand.
            - The action history. This is the sequence of actions that have been played in the game
                so far that the player has knowledge of. This is a vector of length max_history_len
                for stability (Neural Networks expect fixed input sizes).

        Returns:
            np.array: The current information state.
        """
        pad_len = self.max_history_len - len(self.player_history[self.current_player])
        return np.concatenate(
            [
                self.players[self.current_player].hand,
                np.pad(
                    self.player_history[self.current_player],
                    (0, pad_len),
                    mode="constant",
                ),
            ]
        )

    def step(self, action: int) -> Tuple[List[int], int, bool, Dict]:
        """Take a step in the environment given an action.

        Args:
            action (int): The action taken by the player.
        Returns:
            Tuple[List[int], int, bool, Dict]: A tuple containing the following elements:
                - The information state for the next player.
                - The reward for the current player.
                - Whether the game is over.
                - Additional information.
        """
        assert not self.done, "Game is already over. Please reset the game."

        # Check if the action is valid / if not return PENALTY reward
        if action not in self.legal_actions():
            # logger.warning(f"Invalid action: {action }, penalizing player.")
            return (
                self.get_info_state(),
                self.invalid_penalty,
                self.done,
                {},
            )

        # process the action
        self.process_action(action)

        # update history
        self.state = np.append(self.state, action)
        self.player_history[self.current_player].append(action)

        # opponent also gets exposed to partial information
        # if the action is a challenge +
        if self.next_action - 1 < 3:
            self.player_history[1 - self.current_player].append(action)

        # Check if the turn is over first
        if not self.turn_over:
            return (
                self.get_info_state(),
                0,
                self.done,
                {},
            )

        # action handling
        if self.challenge_declared:
            self.resolve_challenge()
        else:
            # update players hand
            self.players[self.current_player].remove_cards(self.cards_played)

            # update central pile
            self.add_to_pile(self.cards_played)

            # Check if bluff
            self.if_last_claim_bluff = self.is_bluff()
        self.reset_turn()

        # check for terminal state
        if self.is_terminal():
            self.done = True  # For stopping recalling step()
        else:
            # update current player
            self.next_player()

        # Update the number of rounds
        self.num_rounds += 1

        return (
            self.get_info_state(),
            self.get_reward(1 - self.current_player),
            self.done,
            {},
        )

    def next_player(self) -> None:
        """Update the current player to the next player."""
        self.current_player = 1 - self.current_player

    def add_to_pile(self, cards: List[int]) -> None:
        """Add cards to the central pile.

        Args:
            cards (List[int]): The cards to add to the central pile.
        """
        assert len(cards) == 4, "Invalid cards representation."
        cards_to_pile = np.zeros(NUM_UNIQUE_CARDS, dtype=int)
        for card in cards:
            cards_to_pile[card - 1] += 1
        self.central_pile += cards_to_pile

    def is_bluff(self) -> bool:
        """Checks if the action was a bluff.

        Returns:
            bool: True if the last claim was a bluff, False otherwise.
        """
        # check if all the cards played are of the declared rank
        return not all(
            self.rank_declared == card
            for card in self.cards_played[: self.num_cards_declared]
        )

    def resolve_challenge(self) -> None:
        """Resolves if the previous player's claim was a bluff or not.

        If the previous player's claim was a bluff, the previous player receives the central pile.
        Otherwise, the challenging player receives the central pile.
        """
        if self.if_last_claim_bluff:
            self.players[1 - self.current_player].add_cards(self.central_pile)
        else:
            self.players[self.current_player].add_cards(self.central_pile)

        # reset central pile
        self.central_pile = np.zeros(NUM_UNIQUE_CARDS, dtype=int)

    def is_terminal(self) -> bool:
        """Check if the game is in a terminal state.

        The game ends if either player has no cards left in their hand, or if the number of rounds
        exceeds the maximum number of rounds.

        Returns:
            bool: True if the game is in a terminal state, False otherwise.
        """
        return self.num_rounds >= self.max_number_of_rounds or any(
            len(player.hand) == 0 for player in self.players
        )

    def get_reward(self, player_idx: int) -> int:
        """Get the reward for the player.

        The reward is +1 if the player wins, -1 if the player loses, and 0 if the game is a draw.
        ! Check discussion on top of the file for more reward options.

        Args:
            player_idx (int): The index of the player.

        Returns:
            int: The reward for the player.
        """
        if (
            self.done
        ):  # game is over - this doesn't necessarily have to be a reward condition
            return (
                -1
                if len(self.players[player_idx].hand)
                > len(self.players[1 - player_idx].hand)
                else 1
            )
        return 0
