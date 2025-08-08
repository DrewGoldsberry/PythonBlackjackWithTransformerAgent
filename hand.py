# hand.py

class Hand:
    def __init__(self):
        self.cards = []
        self.is_split = False  # Track if this hand was split
        self.has_doubled = False
        self.bet = 0
        self.can_double = True  # Track if doubling down is allowed
        self.hit_above_17 = False  # Track if hit above 17
        self.stood_below_17 = False  # Track if stood below 17

    def add_card(self, card):
        self.cards.append(card)

    def get_values(self):
        total = 0
        num_aces = 0
        for card in self.cards:
            if card.is_ace():
                num_aces += 1
            total += card.value

        # Convert Aces from 11 to 1 if needed
        while total > 21 and num_aces > 0:
            total -= 10
            num_aces -= 1

        return total
    def get_original_delt_values(self):
        """Returns the original values of the cards without adjusting for Aces."""
        total = 0
        num_aces = 0
        for card in self.cards[:2]: # Only consider the first two cards
            if card.is_ace():
                num_aces += 1
            total += card.value

        # Convert Aces from 11 to 1 if needed
        while total > 21 and num_aces > 0:
            total -= 10
            num_aces -= 1

        return total
    def get_first_card_value(self):
        """Returns the value of the first card in the hand."""
        total = 0
        num_aces = 0
        for card in self.cards[:1]: # Only consider the first two cards
            if card.is_ace():
                num_aces += 1
            total += card.value

        # Convert Aces from 11 to 1 if needed
        while total > 21 and num_aces > 0:
            total -= 10
            num_aces -= 1

        return total

    def is_blackjack(self):
        return len(self.cards) == 2 and self.get_values() == 21

    def is_busted(self):
        return self.get_values() > 21

    def can_split(self):
        return len(self.cards) == 2 and self.cards[0].rank == self.cards[1].rank

    def reset(self):
        self.cards = []
        self.is_split = False
        self.has_doubled = False
        self.bet = 0