# agent_player.py

from player import Player
from tokenizer import tokenize_state
from collections import deque
from  transformer_agent import TransformerAgent
from constants import AGENT_BANKROLL_TARGET, AGENT_STARTING_BANKROLL
import torch
ACTIONS = ["hit", "stand", "double", "split"]

class AgentPlayer(Player):
    def __init__(self, name, agent:TransformerAgent, epsilon=0.0,is_training=True):
        super().__init__(name)
        self.agent = agent
        self.epsilon = epsilon
        self.trajectories = deque([], 256)  # a bit larger for safety
        self.is_training = is_training
        
        # Episode tracking for bankroll-based rewards
        self.episode_hands_played = 0
        self.episode_high_risk_bets = 0
        self.episode_starting_bankroll = self.bankroll
        self.accumulated_reward = 0.0
        
        # Track bankroll and bet history for better credit assignment
        self.bankroll_history = []
        self.bet_history = []
        self.last_bet_amount = 0
        
        # Track trajectory boundaries for each hand
        self.current_hand_start_idx = 0
        
        # Track hand-specific data for bankroll history rewards
        self.hand_trajectory_ranges = []  # [(start_idx, end_idx, pre_hand_bankroll, post_hand_bankroll)]
        self.pre_hand_bankroll = self.bankroll

    def decide_action(self, dealer_card):
        hand = self.current_hand()
        if hand.is_blackjack() or hand.is_busted():
            return "stand"

        token_seq = tokenize_state(
            player_hand=hand.cards,
            dealer_card=dealer_card,
            bankroll=self.bankroll,
            bet=hand.bet
        )
        action_idx, probs, _ = self.agent.act(token_seq, epsilon=self.epsilon)
        action = ACTIONS[action_idx]

        # Store action with bankroll tracking
        self.bankroll_history.append(self.bankroll)
        self.trajectories.append((token_seq, action_idx))
        return action

    def decide_bet(self):
        """Sample a continuous bet fraction from Beta policy and convert to amount."""
        token_seq = tokenize_state(
            player_hand=[],
            dealer_card=None,
            bankroll=self.bankroll,
            bet=0,
        )

        # Sample fraction in [0,1], plus log_prob for REINFORCE
        bet_frac, log_prob_bet = self.agent.sample_bet(token_seq, training=self.is_training)

        # Turn fraction into amount, clamp minimally
        bet_amount = max(1.0, float(bet_frac.item() * max(0.0, self.bankroll)))

        # Mark this as a bet step so trainer can use the log_prob:
        # (token_seq, None, "bet", log_prob_bet)
        self.bankroll_history.append(self.bankroll)
        self.trajectories.append((token_seq, None, "bet", log_prob_bet))
        
        # Track bet amount and high-risk betting
        self.last_bet_amount = bet_amount
        self.bet_history.append(bet_amount)
        if bet_amount > self.bankroll * 0.5:
            self.episode_high_risk_bets += 1

        return bet_amount

    def start_new_episode(self):
        """Reset episode tracking when starting a new training episode."""
        self.episode_hands_played = 0
        self.episode_high_risk_bets = 0
        self.bankroll = AGENT_STARTING_BANKROLL
        self.episode_starting_bankroll = self.bankroll
        self.accumulated_reward = 0.0
        self.bankroll_history = []
        self.bet_history = []
        self.last_bet_amount = 0
        self.current_hand_start_idx = 0
        self.hand_trajectory_ranges = []
        self.pre_hand_bankroll = self.bankroll
        # Don't clear trajectories here - they need to be processed first!

    def start_new_hand(self):
        """Mark the start of a new hand in trajectories."""
        self.current_hand_start_idx = len(self.trajectories)
        self.pre_hand_bankroll = self.bankroll

    def complete_hand(self):
        """Mark the completion of the current hand and store trajectory range."""
        end_idx = len(self.trajectories) - 1
        if end_idx >= self.current_hand_start_idx:
            hand_range = (self.current_hand_start_idx, end_idx, self.pre_hand_bankroll, self.bankroll)
            self.hand_trajectory_ranges.append(hand_range)
        
    def calculate_bankroll_growth_reward(self, lookback_hands=3):
        """Calculate reward based on bankroll growth over recent hands."""
        if len(self.hand_trajectory_ranges) < 2:
            return 0.0
        
        # Look at the last few hands for trend analysis
        recent_hands = self.hand_trajectory_ranges[-lookback_hands:]
        
        # Calculate total growth over the period
        start_bankroll = recent_hands[0][2]  # pre_hand_bankroll of first hand
        end_bankroll = recent_hands[-1][3]   # post_hand_bankroll of last hand
        growth_rate = (end_bankroll - start_bankroll) / start_bankroll if start_bankroll > 0 else 0
        
        # Reward positive growth, penalize losses
        if growth_rate > 0.1:  # > 10% growth
            return min(0.5, growth_rate * 2)  # Cap at 0.5, scale by growth
        elif growth_rate < -0.1:  # > 10% loss
            return max(-0.5, growth_rate * 2)  # Cap at -0.5, scale by loss
        else:
            return growth_rate  # Small reward/penalty for small changes
    
    def apply_bankroll_growth_rewards(self, lookback_hands=3):
        """Apply bankroll growth rewards to recent hand trajectories."""
        if len(self.hand_trajectory_ranges) < 2:
            return 0
        
        growth_reward = self.calculate_bankroll_growth_reward(lookback_hands)
        if abs(growth_reward) < 0.01:  # Skip very small rewards
            return 0
        
        # Apply to trajectories from recent hands
        recent_hands = self.hand_trajectory_ranges[-lookback_hands:]
        applied_count = 0
        
        for start_idx, end_idx, _, _ in recent_hands:
            for i in range(start_idx, end_idx + 1):
                if i < len(self.trajectories):
                    traj = self.trajectories[i]
                    if len(traj) == 3:  # Action with current reward
                        token_seq, action_idx, current_reward = traj
                        new_reward = current_reward + growth_reward
                        self.trajectories[i] = (token_seq, action_idx, new_reward)
                        applied_count += 1
                    elif len(traj) == 5:  # Bet with current reward
                        token_seq, none_placeholder, bet_marker, log_prob_bet, current_reward = traj
                        new_reward = current_reward + growth_reward
                        self.trajectories[i] = (token_seq, none_placeholder, bet_marker, log_prob_bet, new_reward)
                        applied_count += 1
        
        return applied_count

    def increment_hands_played(self):
        """Call this when a hand is completed."""
        self.episode_hands_played += 1

    def get_current_hand_trajectories(self):
        """Get trajectories for the current hand only."""
        return list(self.trajectories)[self.current_hand_start_idx:]

    def is_episode_complete(self):
        """Check if episode should end based on bankroll goals or hand limits."""
        # Episode ends if: bankrupt, reached target, or played 100 hands with high-risk betting
        if self.bankroll <= 5:
            return True, "bankrupt"
        if self.bankroll >= AGENT_BANKROLL_TARGET:
            return True, "target_reached"
        if self.episode_hands_played >= 10 and self.episode_high_risk_bets >= 10:
            return True, "high_risk_limit"
        return False, "continue"
    
    def get_bankroll_history(self):
        """Return the bankroll at each decision point in the episode."""
        return self.bankroll_history.copy()
    
    def get_bet_amount_for_trajectory(self, trajectory_index):
        """Get the bet amount corresponding to a specific trajectory index."""
        bet_index = 0
        for i, traj in enumerate(self.trajectories):
            if i == trajectory_index:
                if len(traj) == 4 and traj[2] == "bet":  # Bet trajectory
                    return self.bet_history[bet_index] if bet_index < len(self.bet_history) else 0
                else:
                    # For action trajectories, return the most recent bet
                    return self.bet_history[bet_index - 1] if bet_index > 0 else 0
            if len(traj) == 4 and traj[2] == "bet":
                bet_index += 1
        return 0
