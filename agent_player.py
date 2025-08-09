
from player import Player
from tokenizer import tokenize_state
from collections import deque
from  transformer_agent import TransformerAgent
ACTIONS = ["hit", "stand", "double", "split"]

class AgentPlayer(Player):
    def __init__(self, name, agent:TransformerAgent, epsilon=0.1):
        super().__init__(name)
        self.agent = agent
        self.epsilon = epsilon
        self.trajectories = deque([],64)

    def decide_action(self, dealer_card):
        hand = self.current_hand()

        if hand.is_blackjack() or hand.is_busted():
            return "stand"  # force stop if already done

        # Tokenize state and act
        token_seq = tokenize_state(
            player_hand=hand.cards,
            dealer_card=dealer_card,
            bankroll=self.bankroll,
            bet=hand.bet
        )
        action_idx, probs, _ = self.agent.act(token_seq, epsilon=self.epsilon)
        action = ACTIONS[action_idx]

        # Save for training
        self.trajectories.append((token_seq, action_idx))

        return action

    def decide_bet(self):
        """Use the agent to decide a bet amount based on current bankroll."""
        token_seq = tokenize_state(
            player_hand=[],
            dealer_card=None,
            bankroll=self.bankroll,
            bet=0,
        )
        bet_fraction = self.agent.predict_bet(token_seq)
        # Track this bet decision so the bet head can receive feedback
        self.trajectories.append((token_seq, None))
        bet_amount = max(1, float(bet_fraction * self.bankroll))
        return bet_amount
