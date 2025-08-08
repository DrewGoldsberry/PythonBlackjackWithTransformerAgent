# blackjack_env.py

from deck import Deck
from player import Player
from hand import Hand
from constants import RESHUFFLE_THRESHOLD
from agent_player import AgentPlayer
class BlackjackEnv:
    def __init__(self, num_players=1, agent=None, use_agent=False):
        self.deck = Deck()
        self.players = [Player(f"Player {i+1}") for i in range(num_players)]
        self.dealer = Player("Dealer")
        self.num_players = num_players
        self.shuffle_cards_next_reset = False
        self.agent = agent
        if use_agent:
            self.players = [agent]
        else:
            self.players = [Player("Human")]

    def reset(self, bet_amount=10):
        print("\n===== New Round =====")
        if len(self.deck.cards) <= RESHUFFLE_THRESHOLD:
            self.deck.discard_to_deck()
            self.deck.shuffle()

        for player in self.players:
            player.reset_for_round()
            player.place_bet(bet_amount)  # TODO: Make this configurable
        if self.players[0].is_finished:
            self.players[0].bankroll == 500
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

                while True:
                    if hand.is_blackjack() or hand.is_busted():
                        break

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

        reward = 0
        if isinstance(player, AgentPlayer):
            if hand.is_busted():
                reward = -1
            elif dealer_val > 21 or player_val > dealer_val:
                reward = 6
            elif player_val == dealer_val:
                reward = 0
            else:
                reward = -1
            if hand.has_doubled and hand.is_busted():
                reward += -5
            if (hand.get_original_delt_values()>= 17 and hand.is_busted()):
                reward += -5
            
            if (hand.get_original_delt_values() > 11 and hand.has_doubled):
                reward += -5
            elif hand.get_original_delt_values() > 8 and hand.has_doubled and self.dealer.current_hand().get_first_card_value() < 8:
                reward += 5
            
            if hand.get_original_delt_values() >=17 and len(hand.cards) == 2:
                reward += 5
            if self.dealer.current_hand().get_first_card_value()< 7 and hand.is_busted():
                reward = -5
            if hand.get_original_delt_values() < 16 and len(hand.cards) > 2 and self.dealer.current_hand().get_first_card_value() >= 7:
                reward += 5
            if hand.get_original_delt_values() < 16 and len(hand.cards) == 2 and self.dealer.current_hand().get_first_card_value() >= 7: 
                reward += -5   
            if hand.get_original_delt_values() < 16 and len(hand.cards)== 2 and self.dealer.current_hand().get_first_card_value()<= 7:
                reward += -10
            if hand.hit_above_17:
                reward += -5
            
            if hand.get_values() <= 11:
                reward += -5

            if hand.get_original_delt_values() >11 and hand.has_doubled:
                reward -= 5
            if hand.get_original_delt_values() <16 and not hand.has_doubled and len(hand.cards) > 2  and self.dealer.current_hand().get_first_card_value() >= 7:
                reward +=  5
            if hand.stood_below_17 and self.dealer.current_hand().get_first_card_value() < 7:
                reward += 5
            print(f"Reward for {player.name} with hand {hand.get_values()}: {reward}")
        
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
        elif player_val == dealer_val:
            print(f"{player.name} pushes and gets back {hand.bet}")
            round_over= True
            player.draw_bet(hand)
        else:
            print(f"{player.name} loses bet of {hand.bet}")
            round_over= True
            player.lose_bet(hand)
        
        if round_over and isinstance(player, AgentPlayer):
            if player.trajectories:
                    traj = player.trajectories[-1]
                    # Already has reward — optionally update or skip
                    token_seq = None
                    action_idx = None
                    if len(traj)==2:
                        token_seq, action_idx = traj
                    if len(traj) == 3:
                        token_seq, action_idx, _ = traj

                    player.trajectories[-1] = token_seq, action_idx, reward