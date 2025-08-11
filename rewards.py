from constants import AGENT_STARTING_BANKROLL
import math

class Reward:
    def __init__(self, label, reward, bool_function):
        self.label = label
        self.bool_function = bool_function
        self.reward = reward

# Updated rewards system focused on bankroll management and strategy
# Reduced magnitudes since rewards accumulate over episodes
REWARDS_BINDINGS = [
    
   
    # Strategy penalties - more severe for bad basic strategy
   
    # Reward("Good Bankroll Growth", lambda env, player: 
    #        min(0.3, max(0, (player.bankroll - getattr(player, 'episode_starting_bankroll', AGENT_STARTING_BANKROLL)) / AGENT_STARTING_BANKROLL * 0.5)), 
    #        lambda env, player: hasattr(player, 'episode_starting_bankroll')),

    # # Win/Loss rewards - balanced to encourage proper play over just avoiding busts
    # Reward("Hand Win", 0.3, lambda env, player: 
    #        hasattr(player.current_hand(), 'is_winner') and player.current_hand().is_winner),
    
    # Reward("Blackjack Bonus", 0.5, lambda env, player: player.current_hand().is_blackjack()),
   
]
