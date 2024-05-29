from simple_nfsp.games.components.deck import NUM_UNIQUE_CARDS
from typing import List
import numpy as np


class Player:
    def __init__(self, name: str) -> None:
        self.name = name
        self.hand = np.zeros(NUM_UNIQUE_CARDS, dtype=int)

    def add_cards(self, cards: np.array) -> None:
        """Update the player's hand with new cards.

        Args:
            cards (List[Card]): The cards to add to the player's hand.
        """
        self.hand += cards

    def remove_cards(self, cards: np.array) -> None:
        """Remove cards from the player's hand.

        Args:
            cards (List[Card]): The cards to remove from the player's hand.
        """
        card_to_remove = np.zeros(NUM_UNIQUE_CARDS, dtype=int)
        for card in cards:
            if card != 0:
                card_to_remove[card-1] += 1
        self.hand -= card_to_remove

    def get_unique_cards(self) -> List[int]:
        """Get the unique cards in the player's hand."""
        cards = np.where(self.hand > 0)[0]
        return [card + 1 for card in cards]

    def reset(self) -> None:
        """Reset the player's hand."""
        self.hand = np.zeros(NUM_UNIQUE_CARDS, dtype=int)
