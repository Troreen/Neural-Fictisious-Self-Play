from simple_nfsp.games.components.deck import NUM_UNIQUE_CARDS
from typing import List
import numpy as np


class Player:
    def __init__(self, name: str) -> None:
        self.name = name
        self.hand = np.zeros(NUM_UNIQUE_CARDS + 1, dtype=int)

    def add_cards(self, cards: np.array) -> None:
        """Update the player's hand with new cards.

        Args:
            cards (List[Card]): The cards to add to the player's hand.
        """
        self.hand += cards

    def remove_card(self, card: int) -> None:
        """Remove card from the player's hand.

        Args:
            cards (int): The card to remove from the player's hand.
        """
        self.hand[card] -= 1

    def get_unique_cards(self) -> List[int]:
        """Get the unique cards in the player's hand."""
        cards = np.where(self.hand > 0)[0]
        return [card for card in cards]

    def get_cards_as_list(self) -> List[int]:
        """Get the player's hand as a list."""
        cards = []
        for card, count in enumerate(self.hand):
            cards.extend([card] * count)
        return cards

    def reset(self) -> None:
        """Reset the player's hand."""
        self.hand = np.zeros(NUM_UNIQUE_CARDS+1, dtype=int)