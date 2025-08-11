# blackjack_env.py

from deck import Deck
from player import Player
from hand import Hand
from constants import RESHUFFLE_THRESHOLD, AGENT_BANKROLL_TARGET, AGENT_STARTING_BANKROLL
from agent_player import AgentPlayer
import math
from action_rewards import calculate_action_reward, calculate_bet_reward

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

        # Mark the start of a new hand for agent players
        for player in self.players:
            if isinstance(player, AgentPlayer):
                player.start_new_hand()

        for player in self.players:
            player.reset_for_round()
            bet=0
            if isinstance(player, AgentPlayer):
                # Record bankroll before bet decision
                bet = player.decide_bet()
                
                # Apply per-action reward for betting decision
                bet_reward, bet_reasons = calculate_bet_reward(
                    bet, player.bankroll, player.episode_starting_bankroll
                )
                self.apply_per_action_reward(player, bet_reward, bet_reasons)
                
                # Record bankroll after bet placement
                player.bankroll_history.append(player.bankroll)
            else:
                bet = bet_amount
            player.place_bet(bet)  # TODO: Make this configurable
            
        # Check if agent player needs to be reset or episode is complete
        agent_player = None
        for player in self.players:
            if isinstance(player, AgentPlayer):
                agent_player = player
                break       
                  
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
                if hand.has_stood or hand.is_blackjack() or hand.is_busted() or hand.has_doubled or hand.has_invalid_split:
                    continue  # Skip if player has already stood
                looping_without_ending=0       
                while True:
                    looping_without_ending+=1
                    action = player.decide_action(self.dealer.current_hand().cards[0])
                    
                    # Calculate per-action reward before processing the action
                    dealer_up_card = self.dealer.current_hand().get_first_card_value()
                    action_reward, action_reasons = calculate_action_reward(
                        action, hand, dealer_up_card, player.bankroll
                    )
                    
                    if action == "hit":
                        if hand.get_values() >= 17:
                            hand.hit_above_17 = True
                        card = self.deck.draw()
                        print(f'User hit for another Card {card}')
                        hand.add_card(card)
                        hand.can_double = False  # Can't double after hitting
                        looping_without_ending = 0
                        
                        # Apply per-action reward immediately
                        self.apply_per_action_reward(player, action_reward, action_reasons)
                        
                    elif action == "stand":
                        print(f'User stands')
                        hand.stood_below_17 = True if hand.get_values() < 17 else False
                        hand.has_stood = True
                        
                        # Apply per-action reward immediately
                        self.apply_per_action_reward(player, action_reward, action_reasons)
                        break
                        
                    elif action == "double" and hand.can_double:
                        if player.bankroll >= hand.bet:
                            player.double_down()
                            card = self.deck.draw()
                            print(f'User double downed for another Card {card}')
                            hand.add_card(card)
                            looping_without_ending=0
                            
                            # Apply per-action reward immediately
                            self.apply_per_action_reward(player, action_reward, action_reasons)
                            break
                            
                    elif action == "split":
                        if hand.can_split():
                            print("user split")
                            player.split_hand()
                            player.current_hand().cards.append(self.deck.draw())
                            looping_without_ending=0
                            
                            # Apply per-action reward immediately
                            self.apply_per_action_reward(player, action_reward, action_reasons)
                        else:
                            print("Invalid split.")
                            hand.has_invalid_split = True
                            # Apply penalty for invalid split
                            self.apply_per_action_reward(player, action_reward, action_reasons)
                            break  # Mark as invalids split if conditions not met
                    
                            
        else:
        # Human players play manually through the UI
            return

    def apply_per_action_reward(self, player, reward, reasons):
        """Apply immediate reward for an action to the most recent trajectory."""
        if not isinstance(player, AgentPlayer) or len(player.trajectories) == 0:
            return
        
        # Get the most recent trajectory (the action we just took)
        most_recent_idx = len(player.trajectories) - 1
        traj = player.trajectories[most_recent_idx]
        
        # Apply reward to the most recent action or bet
        if len(traj) == 2:  # Action trajectory: (token_seq, action_idx)
            token_seq, action_idx = traj
            player.trajectories[most_recent_idx] = (token_seq, action_idx, reward)
        elif len(traj) == 4:  # Bet trajectory: (token_seq, None, "bet", log_prob_bet)
            token_seq, none_placeholder, bet_marker, log_prob_bet = traj
            player.trajectories[most_recent_idx] = (token_seq, none_placeholder, bet_marker, log_prob_bet, reward)
        elif len(traj) >= 3:  # Already has a reward, add to it
            if len(traj) == 3:  # Action with reward
                token_seq, action_idx, existing_reward = traj
                new_reward = existing_reward + reward
                player.trajectories[most_recent_idx] = (token_seq, action_idx, new_reward)
            elif len(traj) == 5:  # Bet with reward
                token_seq, none_placeholder, bet_marker, log_prob_bet, existing_reward = traj
                new_reward = existing_reward + reward
                player.trajectories[most_recent_idx] = (token_seq, none_placeholder, bet_marker, log_prob_bet, new_reward)
        
        # Print action reward feedback
        if reward != 0:
            print(f"Action Reward: {reward:.3f} ({', '.join(reasons)})")
        
        # Track for logging
        player.accumulated_reward += reward
            
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
            # Increment hands played counter
            player.increment_hands_played()
            
            # Complete the hand tracking for bankroll history
            player.complete_hand()
            
            print(f"Hand completed. Balance: {player.bankroll} Bet: {hand.bet}")
            print(f"Trajectories from current hand: {len(player.trajectories) - player.current_hand_start_idx}")

    def check_and_complete_episode(self, player):
        """Check if episode is complete."""
        if not isinstance(player, AgentPlayer):
            return False, None
            
        episode_complete, reason = player.is_episode_complete()
        
        if episode_complete:
            bankroll_change = player.bankroll - player.episode_starting_bankroll
            
            print(f"Episode Summary:")
            print(f"  Hands played: {player.episode_hands_played}")
            print(f"  High-risk bets: {player.episode_high_risk_bets}")
            print(f"  Bankroll change: {bankroll_change}")
            print(f"  Final accumulated reward: {player.accumulated_reward:.2f}")
            print(f"  Epsilon: {player.epsilon:.4f}")
            
            # NOTE: Only per-action rewards are used - no hand or episode bonuses
            
            return True, reason
        
        return False, None
