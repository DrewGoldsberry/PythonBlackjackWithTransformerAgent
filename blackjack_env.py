# blackjack_env.py

from deck import Deck
from player import Player
from hand import Hand
from constants import RESHUFFLE_THRESHOLD, AGENT_BANKROLL_TARGET, AGENT_STARTING_BANKROLL
from agent_player import AgentPlayer
import math
from helpers import is_lambda
from rewards import REWARDS_BINDINGS

class BlackjackEnv:
    def __init__(self, player=None):
        self.deck = Deck()
        self.dealer = Player("Dealer")
        self.shuffle_cards_next_reset = False
        
        if player is None:
            self.players = [Player("Human")]
        else:
            self.players = [player]

    def reset(self, bet_amount=10):
        print("\n===== New Round =====")
        if len(self.deck.cards) <= RESHUFFLE_THRESHOLD:
            self.deck.discard_to_deck()
            self.deck.shuffle()

        for player in self.players:
            player.reset_for_round()
            bet=0
            if isinstance(player, AgentPlayer):
                bet = player.decide_bet()
            else:
                bet = bet_amount
            player.place_bet(bet)  # TODO: Make this configurable
        if self.players[0].is_finished:
            self.players[0].bankroll = AGENT_STARTING_BANKROLL
            self.players[0].is_finished = False
        self.dealer.reset_for_round()

        # Initial deal: two cards to each player and dealer
        for _ in range(2):
            for player in self.players:
                card = self.deck.draw()
                player.current_hand().add_card(card)
                print(f"{player.name} draws {card}")
            dealer_card = self.deck.draw()
            self.dealer.current_hand().add_card(dealer_card)
            if _ == 0:
                print(f"Dealer shows {dealer_card}")
            elif _ == 1:
                print("Dealer's second card is hidden")

    def play_round(self, skip_dealer=False):
        for player in self.players:
            print(f"\n--- {player.name}'s turn ---")
            self.play_player(player)

        if not skip_dealer:
            print(f"\n--- Dealer's turn ---")
            self.play_dealer()

        for player in self.players:
            for hand in player.hands:
                self.evaluate_hand(player, hand)

    def play_player(self, player):
        if isinstance(player, AgentPlayer):
            for i in range(len(player.hands)):
                player.active_hand_index = i
                hand = player.current_hand()
                if hand.has_stood or hand.is_blackjack() or hand.is_busted() or hand.has_doubled:
                    continue  # Skip if player has already stood

                while True:
                    action = player.decide_action(self.dealer.current_hand().cards[0])
                    if action == "hit":
                        if hand.get_values() >= 17:
                            hand.hit_above_17 = True
                        card = self.deck.draw()
                        print(f'User hit for another Card {card}')
                        hand.add_card(card)
                        hand.can_double = False  # Can't double after hitting

                    elif action == "stand":
                        print(f'User stands')
                        hand.stood_below_17 = True if hand.get_values() < 17 else False
                        hand.has_stood = True
                        break
                    elif action == "double" and hand.can_double:
                        if player.bankroll >= hand.bet:
                            player.double_down()
                            card = self.deck.draw()
                            print(f'User double downed for another Card {card}')
                            hand.add_card(card)
                            break
                    elif action == "split":
                        if hand.can_split():
                            print("user split")
                            player.split_hand()
                            player.current_hand().cards.append(self.deck.draw())
                            
        else:
        # Human players play manually through the UI
            return
    def play_dealer(self):
        hand = self.dealer.current_hand()
        print(f"Dealer cards: {', '.join(str(c) for c in hand.cards)}")
        
        while hand.get_values() < 17:
            card = self.deck.draw()
            hand.add_card(card)
            print(f"Dealer hits and draws {card}")

        dealer_val = hand.get_values()
        print(f"Dealer stands with {dealer_val}")

    def evaluate_hand(self, player, hand):
        dealer_val = self.dealer.current_hand().get_values()
        player_val = hand.get_values()

        
        print (f"{player.name} has {player_val}, Dealer has {dealer_val}")
        round_over=False        
        if hand.is_busted():
            print(f"{player.name} busted and loses bet of {hand.bet}")
            player.lose_bet(hand)
            round_over= True
        elif dealer_val > 21 or player_val > dealer_val:
            print(f"{player.name} wins and gains {2 * hand.bet}")
            round_over= True
            player.win_bet(hand)
            player.current_hand().is_winner = True
        elif player_val == dealer_val:
            print(f"{player.name} pushes and gets back {hand.bet}")
            round_over= True
            player.draw_bet(hand)
            player.current_hand().is_winner = True
        else:
            print(f"{player.name} loses bet of {hand.bet}")
            round_over= True
            player.lose_bet(hand)

        
        
        if round_over and isinstance(player, AgentPlayer):
            reward = 0
            rules = []
            for reward_binding in REWARDS_BINDINGS:
                if reward_binding.bool_function(self, player):
                    temp_reward = 0
                    if is_lambda(reward_binding.reward):
                        temp_reward+=reward_binding.reward(self, player)
                    else:
                        temp_reward+=reward_binding.reward
                    
                    reward+=temp_reward
                    rules.append(reward_binding.label + f" (reward: {temp_reward})")

            print("")
            for rule in rules:
                print(rule)
            print("")    
            print(f"Reward: {reward} Balance: {player.bankroll} Bet: {hand.bet}")
            
            if player.trajectories:
                    # We only need to update the last trajectory because in training we apply the rewards to the past trajectories
                    traj = player.trajectories[-1]
                    # Already has reward — optionally update or skip
                    token_seq = None
                    action_idx = None
                    if len(traj)==2:
                        token_seq, action_idx = traj
                    if len(traj) == 3:
                        token_seq, action_idx, _ = traj

                    player.trajectories[-1] = token_seq, action_idx, reward
