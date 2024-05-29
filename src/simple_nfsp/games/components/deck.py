import numpy as np

NUM_UNIQUE_CARDS = 13
NUM_REPLICAS = 4


class Deck:
    def __init__(self) -> None:
        self.cards: np.array = np.array([NUM_REPLICAS] * NUM_UNIQUE_CARDS, dtype=int)

    def __len__(self) -> int:
        """Return the number of cards in the deck.

        Returns:
            int: The number of cards in the deck.
        """
        return np.sum(self.cards)

    # Deal cards from the deck
    def deal(self, num_cards: int = 1) -> np.array:
        """Deal a specified number of cards from the deck.

        Args:
            num_cards (int, optional): The number of cards to deal. Defaults to 1.

        Returns:
            np.array: The dealt cards.
        """
        dealt_cards = np.zeros(NUM_UNIQUE_CARDS, dtype=int)
        for _ in range(num_cards):
            valid_cards = np.where(self.cards > 0)[0]
            if len(valid_cards) == 0:
                raise ValueError("No cards left in the deck.")
            card = np.random.choice(valid_cards)
            self.cards[card] -= 1
            dealt_cards[card] += 1
        return dealt_cards

    # Check if deck is empty
    def is_empty(self):
        return len(self.cards) == 0
