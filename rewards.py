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
    
    # Core strategy rewards - stronger incentives for basic strategy
    Reward("Good Hit (< 17 vs High Card)", 0.8, lambda env, player:
           not player.current_hand().is_busted() and
           len(player.current_hand().cards) > 2 and
           player.current_hand().get_original_delt_values() <= 16 and
           env.dealer.current_hand().get_first_card_value() >= 7),
    
    Reward("Good Hit (Low Hand vs Low Dealer)", 0.6, lambda env, player:
           not player.current_hand().is_busted() and
           len(player.current_hand().cards) > 2 and
           player.current_hand().get_original_delt_values() <= 11 and
           env.dealer.current_hand().get_first_card_value() <= 6),
    
    Reward("Good Stand (>= 17)", 0.1, lambda env, player:
            not player.current_hand().is_busted() and
            player.current_hand().get_values() >= 17 and
            player.current_hand().has_stood),
    
    Reward("Good Stand (12-16 vs Low Dealer)", 0.3, lambda env, player:
           not player.current_hand().is_busted() and
           12 <= player.current_hand().get_values() <= 16 and
           player.current_hand().has_stood and
           env.dealer.current_hand().get_first_card_value() <= 6),

    Reward("Good Double (9-11 vs Low Card)", 0.4, lambda env, player:
           player.current_hand().has_doubled and
           9 <= player.current_hand().get_original_delt_values() <= 11 and
           env.dealer.current_hand().get_first_card_value() <= 6 and
           not player.current_hand().is_busted()),
    
    # Strategy penalties - more severe for bad basic strategy
    Reward("Bad Hit (>= 17)", -0.2, lambda env, player:  # Reduced penalty - sometimes correct vs dealer A
           hasattr(player.current_hand(), 'hit_above_17') and 
           player.current_hand().hit_above_17 and
           env.dealer.current_hand().get_first_card_value() < 10),  # Only penalize vs non-high cards
    
    Reward("Bad Stand (< 17 vs High Card)", -1.0, lambda env, player:  # Increased penalty
           hasattr(player.current_hand(), 'stood_below_17') and
           player.current_hand().stood_below_17 and
           player.current_hand().get_values() < 17 and
           env.dealer.current_hand().get_first_card_value() >= 7),
    
    Reward("Bad Hit (12-16 vs Low Dealer)", -0.6, lambda env, player:  # New penalty for hitting when should stand
           len(player.current_hand().cards) > 2 and
           12 <= player.current_hand().get_original_delt_values() <= 16 and
           env.dealer.current_hand().get_first_card_value() <= 6 and
           not player.current_hand().has_stood),
    Reward("Invalid Split", -0.5, lambda env, player:  # Penalty for invalid split
            player.current_hand().has_invalid_split),
    # Bankroll management rewards (key focus)
    Reward("Conservative Betting (Low Bankroll)", 0.2, lambda env, player:
           player.bankroll < AGENT_STARTING_BANKROLL * 0.5 and 
           player.current_hand().bet <= player.bankroll * 0.2),
           
    Reward("Moderate Risk Betting", 0.1, lambda env, player:
           player.bankroll >= AGENT_STARTING_BANKROLL * 0.5 and
           0.1 <= (player.current_hand().bet / max(1, player.bankroll)) <= 0.3),
    
    Reward("Dangerous High-Risk Betting", -0.8, lambda env, player:
           player.current_hand().bet > player.bankroll * 0.5),
    
    Reward("Near Bankruptcy Warning", -1.0, lambda env, player: 
           player.bankroll < AGENT_STARTING_BANKROLL * 0.2),
           
    Reward("Good Bankroll Growth", lambda env, player: 
           min(0.3, max(0, (player.bankroll - getattr(player, 'episode_starting_bankroll', AGENT_STARTING_BANKROLL)) / AGENT_STARTING_BANKROLL * 0.5)), 
           lambda env, player: hasattr(player, 'episode_starting_bankroll')),

    # Win/Loss rewards - balanced to encourage proper play over just avoiding busts
    Reward("Hand Win", 0.3, lambda env, player: 
           hasattr(player.current_hand(), 'is_winner') and player.current_hand().is_winner),
    
    Reward("Hand Bust", -0.25, lambda env, player: player.current_hand().is_busted()),  # Reduced from -0.4
    
    Reward("Blackjack Bonus", 0.5, lambda env, player: player.current_hand().is_blackjack()),
    
    # Additional strategic rewards to encourage proper basic strategy
    Reward("Strategic Hit Reward", 0.4, lambda env, player:
           not player.current_hand().is_busted() and
           len(player.current_hand().cards) > 2 and
           ((player.current_hand().get_original_delt_values() <= 11) or  # Always hit 11 or less
            (12 <= player.current_hand().get_original_delt_values() <= 16 and env.dealer.current_hand().get_first_card_value() >= 7))),  # Hit 12-16 vs high cards,
]
