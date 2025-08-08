import random
from constants import SUITS, RANKS, CARD_VALUES, NUM_DECKS, RESHUFFLE_THRESHOLD

class Card:
    def __init__(self, rank, suit):
        self.rank = rank
        self.suit = suit
        self.value = CARD_VALUES[rank]

    def __str__(self):
        return f"{self.rank}{self.suit}"

    def is_ace(self):
        return self.rank == 'A'
