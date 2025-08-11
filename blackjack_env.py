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
                    if action == "hit":
                        if hand.get_values() >= 17:
                            hand.hit_above_17 = True
                        card = self.deck.draw()
                        print(f'User hit for another Card {card}')
                        hand.add_card(card)
                        hand.can_double = False  # Can't double after hitting
                        looping_without_ending = 0
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
                            looping_without_ending=0
                            break
                    elif action == "split":
                        if hand.can_split():
                            print("user split")
                            player.split_hand()
                            player.current_hand().cards.append(self.deck.draw())
                            looping_without_ending=0
                        else:
                            hand.has_invalid_split = True
                            break  # Mark as invalids split if conditions not met
                    
                    if looping_without_ending>5:
                        hand.has_stood = True
                        break
                            
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
            # Increment hands played counter
            player.increment_hands_played()
            
            # Calculate hand-level rewards (smaller magnitude for accumulation)
            hand_reward = 0
            rules = []
            for reward_binding in REWARDS_BINDINGS:
                if reward_binding.bool_function(self, player):
                    temp_reward = 0
                    if is_lambda(reward_binding.reward):
                        temp_reward+=reward_binding.reward(self, player)
                    else:
                        temp_reward+=reward_binding.reward
                    
                    hand_reward+=temp_reward
                    rules.append(reward_binding.label + f" (reward: {temp_reward})")

            print("")
            for rule in rules:
                print(rule)
            print("")    
            
            # Apply hand reward only to trajectories from the current hand
            recent_trajectories = []
            hand_start_idx = player.current_hand_start_idx
            
            for i in range(hand_start_idx, len(player.trajectories)):
                traj = player.trajectories[i]
                if len(traj) == 2:  # Action trajectory
                    token_seq, action_idx = traj
                    player.trajectories[i] = (token_seq, action_idx, hand_reward)
                    recent_trajectories.append(i)
                elif len(traj) == 4:  # Bet trajectory
                    token_seq, none_placeholder, bet_marker, log_prob_bet = traj
                    player.trajectories[i] = (token_seq, none_placeholder, bet_marker, log_prob_bet, hand_reward)
                    recent_trajectories.append(i)
            
            # Track accumulated reward for logging only
            player.accumulated_reward += hand_reward
            
            # Complete the hand tracking for bankroll history
            player.complete_hand()
            
            # Apply bankroll growth rewards if we have enough history
            growth_applied = 0
            if len(player.hand_trajectory_ranges) >= 2:
                growth_applied = player.apply_bankroll_growth_rewards(lookback_hands=3)
                if growth_applied > 0:
                    growth_reward = player.calculate_bankroll_growth_reward(lookback_hands=3)
                    print(f"Bankroll Growth Reward: {growth_reward:.3f} applied to {growth_applied} trajectories")
            
            print(f"Hand Reward: {hand_reward} Total Accumulated: {player.accumulated_reward:.2f} Balance: {player.bankroll} Bet: {hand.bet}")
            print(f"Applied reward {hand_reward} to {len(recent_trajectories)} trajectories from current hand (idx {hand_start_idx}+)")

    def check_and_complete_episode(self, player):
        """Check if episode is complete and apply accumulated rewards if so."""
        if not isinstance(player, AgentPlayer):
            return False, None
            
        episode_complete, reason = player.is_episode_complete()
        
        if episode_complete:
            bankroll_change = player.bankroll - player.episode_starting_bankroll
            
            # Calculate episode outcome bonus/penalty
            episode_bonus = 0
            if reason == "target_reached":
                episode_bonus = 10  # Moderate bonus for reaching target
                print(f"🎉 TARGET REACHED! Episode bonus: +{episode_bonus}")
            elif reason == "bankrupt":
                episode_bonus = -5  # Moderate penalty for bankruptcy
                print(f"💸 BANKRUPTCY! Episode penalty: {episode_bonus}")
            elif reason == "high_risk_limit":
                episode_bonus = -3  # Smaller penalty for high-risk limit
                print(f"⚠️  HIGH RISK LIMIT! Episode penalty: {episode_bonus}")
            
            # Add episode bonus to all trajectories that already have hand rewards
            bonus_applied_count = 0
            if episode_bonus != 0:
                for i in range(len(player.trajectories)):
                    traj = player.trajectories[i]
                    if len(traj) == 3:  # Action with hand reward
                        token_seq, action_idx, hand_reward = traj
                        new_reward = hand_reward + episode_bonus
                        player.trajectories[i] = (token_seq, action_idx, new_reward)
                        bonus_applied_count += 1
                    elif len(traj) == 5:  # Bet with hand reward
                        token_seq, none_placeholder, bet_marker, log_prob_bet, hand_reward = traj
                        new_reward = hand_reward + episode_bonus
                        player.trajectories[i] = (token_seq, none_placeholder, bet_marker, log_prob_bet, new_reward)
                        bonus_applied_count += 1
            
            print(f"Episode Summary:")
            print(f"  Hands played: {player.episode_hands_played}")
            print(f"  High-risk bets: {player.episode_high_risk_bets}")
            print(f"  Bankroll change: {bankroll_change}")
            print(f"  Accumulated hand rewards: {player.accumulated_reward:.2f}")
            print(f"  Episode bonus applied to {bonus_applied_count} actions: {episode_bonus}")
            print(f"  Epsilon: {player.epsilon:.4f}")
            
            # NOTE: Each action now has individual hand reward + episode outcome bonus
            # This provides both immediate feedback and long-term consequence learning
            
            return True, reason
        
        return False, None
