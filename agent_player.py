# agent_player.py

from player import Player
from tokenizer import tokenize_state
from collections import deque
from  transformer_agent import TransformerAgent
import torch
ACTIONS = ["hit", "stand", "double", "split"]

class AgentPlayer(Player):
    def __init__(self, name, agent:TransformerAgent, epsilon=0.1):
        super().__init__(name)
        self.agent = agent
        self.epsilon = epsilon
        self.trajectories = deque([], 256)  # a bit larger for safety

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

        # Store action (legacy format): (token_seq, action_idx)
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
        bet_frac, log_prob_bet = self.agent.sample_bet(token_seq, training=True)

        # Turn fraction into amount, clamp minimally
        bet_amount = max(1.0, float(bet_frac.item() * max(0.0, self.bankroll)))

        # Mark this as a bet step so trainer can use the log_prob:
        # (token_seq, None, "bet", log_prob_bet)
        self.trajectories.append((token_seq, None, "bet", log_prob_bet))

        return bet_amount
