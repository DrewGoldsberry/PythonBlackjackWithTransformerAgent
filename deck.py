# deck.py

import random
from constants import SUITS, RANKS, CARD_VALUES, NUM_DECKS, RESHUFFLE_THRESHOLD
from card import Card

class Deck:
    def __init__(self):
        self.cards = []
        self.discarded = []
        self.build()
        self.shuffle()

    def build(self):
        """Builds a fresh multi-deck stack"""
        self.cards = [
            Card(rank, suit)
            for _ in range(NUM_DECKS)
            for suit in SUITS
            for rank in RANKS
        ]
        print(f"Built new deck with {len(self.cards)} cards.")

    def shuffle(self):
        random.shuffle(self.cards)
        print("Shuffled deck.")

    def draw(self):
        """Draws a card and reshuffles if needed"""
        if len(self.cards) <= RESHUFFLE_THRESHOLD:
            print(f"Deck low ({len(self.cards)} cards left). Reshuffling...")
        card = self.cards.pop()
        self.discard(card)    
        return card

    def discard(self, card):
        self.discarded.append(card)

    def discard_to_deck(self):
        print(f"Rebuilding deck from {len(self.discarded)} discarded cards.")
        self.cards += self.discarded
        self.discarded = []