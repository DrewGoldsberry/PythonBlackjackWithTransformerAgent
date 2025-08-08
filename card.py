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


class Deck:
    def __init__(self):
        self.cards = []
        self.discarded = []
        self.build()
        self.shuffle()

    def build(self):
        self.cards = [
            Card(rank, suit)
            for _ in range(NUM_DECKS)
            for suit in SUITS
            for rank in RANKS
        ]

    def shuffle(self):
        random.shuffle(self.cards)

    def draw(self):
        if len(self.cards) <= RESHUFFLE_THRESHOLD:
            self.discard_to_deck()
            self.shuffle()
        return self.cards.pop()

    def discard_to_deck(self):
        self.cards += self.discarded
        self.discarded = []

    def discard(self, card):
        self.discarded.append(card)