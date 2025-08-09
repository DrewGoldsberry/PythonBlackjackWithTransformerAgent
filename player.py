# player.py

from hand import Hand

class Player:
    def __init__(self, name, bankroll=500, is_human=False):
        self.name = name
        self.bankroll = bankroll
        self.hands = [Hand()]
        self.active_hand_index = 0
        self.is_human = is_human
        self.is_finished = False
    def current_hand(self):
        return self.hands[self.active_hand_index]

    def reset_for_round(self):
        self.hands = [Hand()]
        self.active_hand_index = 0

    def place_bet(self, amount):
        if self.bankroll >= amount:
            self.current_hand().bet = amount
            self.bankroll -= amount
        else:
            self.current_hand().bet = self.bankroll
            self.bankroll -= self.bankroll
            amount = self.bankroll

        if self.bankroll == amount:
            print(f"{self.name} has run out of money!")
            self.is_finished = True

    def win_bet(self, hand):
        self.bankroll += 2 * hand.bet

    def draw_bet(self, hand):
        self.bankroll += hand.bet

    def lose_bet(self, hand):
        pass  # Bet already deducted on placement

    def split_hand(self):
        hand = self.current_hand()
        if hand.can_split() and self.bankroll >= hand.bet:
            new_card = hand.cards.pop()
            new_hand = Hand()
            new_hand.add_card(new_card)
            new_hand.bet = hand.bet
            self.bankroll -= hand.bet
            hand.is_split = True
            new_hand.is_split = True
            self.hands.insert(self.active_hand_index + 1, new_hand)

    def double_down(self):
        hand = self.current_hand()
        if self.bankroll >= hand.bet:
            self.bankroll -= hand.bet
            hand.bet *= 2
            hand.has_doubled = True