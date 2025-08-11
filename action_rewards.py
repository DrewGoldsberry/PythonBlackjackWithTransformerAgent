"""
Per-action reward system for immediate feedback on blackjack decisions.
Provides rewards immediately after each action based on basic strategy principles.
"""

def calculate_action_reward(action, player_hand, dealer_up_card, player_bankroll):
    """
    Calculate immediate reward for an action based on basic strategy.
    
    Args:
        action: The action taken ('hit', 'stand', 'double', 'split')
        player_hand: The player's hand object
        dealer_up_card: The dealer's up card value (1-11)
        player_bankroll: Current player bankroll
    
    Returns:
        float: Reward value for the action
    """
    reward = 0.0
    reasons = []
    
    player_total = player_hand.get_values()
    
    # Check if hand is soft (has Ace counting as 11)
    has_ace_as_11 = False
    total_raw = sum(card.value for card in player_hand.cards)
    num_aces = sum(1 for card in player_hand.cards if card.is_ace())
    if num_aces > 0 and total_raw - (num_aces * 10) <= 11:
        # If we can count at least one ace as 11 without busting
        has_ace_as_11 = True
    is_soft = has_ace_as_11
    
    can_double = player_hand.can_double
    
    # Basic strategy rewards for HIT action
    if action == "hit":
        reward, reasons = _evaluate_hit_action(player_total, dealer_up_card, is_soft, reasons)
    
    # Basic strategy rewards for STAND action
    elif action == "stand":
        reward, reasons = _evaluate_stand_action(player_total, dealer_up_card, is_soft, reasons)
    
    # Basic strategy rewards for DOUBLE action
    elif action == "double":
        reward, reasons = _evaluate_double_action(player_total, dealer_up_card, is_soft, can_double, player_bankroll, reasons)
    
    # Basic strategy rewards for SPLIT action
    elif action == "split":
        reward, reasons = _evaluate_split_action(player_hand, dealer_up_card, player_bankroll, reasons)
    
    return reward, reasons

def _evaluate_hit_action(player_total, dealer_up_card, is_soft, reasons):
    """Evaluate hit action based on basic strategy."""
    reward = 0.0
    
    # Always hit on low totals
    if player_total <= 11:
        reward += 0.8
        reasons.append("Correct Hit (≤11)")
    
    # Hit 12-16 vs dealer high cards (7-A)
    elif 12 <= player_total <= 16 and dealer_up_card >= 7:
        reward += 0.6
        reasons.append("Correct Hit (12-16 vs High)")
    
    # Hit soft 17 vs dealer 2-6, 8-A (NOT vs 7)
    elif is_soft and player_total <= 16 and dealer_up_card >= 7:
        if dealer_up_card <= 6:
            reward += 0.4
            reasons.append("Correct Hit (Soft 17 vs 2-6,8-A)")
        else:  # vs dealer 7
            reward -= 0.6
            reasons.append("Bad Hit (Soft 17 vs >=7)")
    
    # Penalize hitting when should stand
    elif player_total >= 17 and not is_soft:
        reward -= 0.8
        reasons.append("Bad Hit (≥17 Hard)")
    
    # Penalize hitting 12-16 vs dealer low cards (2-6)
    elif 12 <= player_total <= 16 and dealer_up_card <= 6:
        reward -= 0.6
        reasons.append("Bad Hit (12-16 vs Low)")
    
    return reward, reasons

def _evaluate_stand_action(player_total, dealer_up_card, is_soft, reasons):
    """Evaluate stand action based on basic strategy."""
    reward = 0.0
    
    # Always stand on 18+ 
    if player_total >= 18 and player_total <= 21:
        reward += 0.3
        reasons.append("Correct Stand (≥18)")
    
    # Soft 17 handling - should stand vs dealer 7, hit vs others
    elif is_soft and player_total >= 17:
        if dealer_up_card >= 7:
            reward += 0.3
            reasons.append("Correct Stand (Soft 17 vs 7)")
        else:  # vs dealer 2-6
            reward -= 0.4
            reasons.append("Bad Stand (Soft 17 vs 2-6,8-A)")
    
    # Hard 17 - always stand
    elif not is_soft and player_total == 17:
        reward += 0.3
        reasons.append("Correct Stand (≥17)")
    
    # Stand 12-16 vs dealer low cards (2-6)
    elif 12 <= player_total <= 16 and dealer_up_card <= 6:
        reward += 0.5
        reasons.append("Correct Stand (12-16 vs Low)")
    
    # Penalize standing when should hit
    elif player_total < 12:
        reward -= 0.8
        reasons.append("Bad Stand (<12)")
    
    # Penalize standing 12-16 vs dealer high cards (7-A)
    elif 12 <= player_total <= 16 and dealer_up_card >= 7:
        reward -= 1.0
        reasons.append("Bad Stand (12-16 vs High)")
    
    return reward, reasons

def _evaluate_double_action(player_total, dealer_up_card, is_soft, can_double, player_bankroll, reasons):
    """Evaluate double down action based on basic strategy."""
    reward = 0.0
    
    if not can_double:
        reward -= 0.5
        reasons.append("Invalid Double (Not Allowed)")
        return reward, reasons
    
    # Good double situations
    if 9 <= player_total <= 11:
        if dealer_up_card <= 6:
            reward += 0.8
            reasons.append("Excellent Double (9-11 vs Low)")
        elif dealer_up_card <= 9:
            reward += 0.4
            reasons.append("Good Double (9-11 vs Mid)")
        else:
            reward -= 0.2
            reasons.append("Risky Double (9-11 vs High)")
    
    # Soft doubles (A,2 through A,7)
    elif is_soft and 13 <= player_total <= 18:
        if dealer_up_card >= 4 and dealer_up_card <= 6:
            reward += 0.6
            reasons.append("Good Soft Double")
        else:
            reward -= 0.3
            reasons.append("Poor Soft Double")
    
    # Bad double situations
    else:
        reward -= 0.6
        reasons.append("Bad Double Situation")
    
    # Bankroll consideration for doubling
    bet_ratio = player_bankroll / max(1, player_bankroll + 100)  # Rough bet estimation
    if bet_ratio < 0.1:  # Low bankroll
        reward -= 0.3
        reasons.append("Risky Double (Low Bankroll)")
    
    return reward, reasons

def _evaluate_split_action(player_hand, dealer_up_card, player_bankroll, reasons):
    """Evaluate split action based on basic strategy."""
    reward = 0.0
    
    if not player_hand.can_split():
        reward -= 1.0
        reasons.append("Invalid Split")
        return reward, reasons
    
    # Get the paired card value
    pair_value = player_hand.cards[0].value if player_hand.cards[0].value <= 10 else 10
    
    # Always split Aces and 8s
    if pair_value == 1 or pair_value == 8:
        reward += 1.0
        reasons.append("Excellent Split (A-A or 8-8)")
    
    # Never split 5s or 10s
    elif pair_value == 5 or pair_value == 10:
        reward -= 1.0
        reasons.append("Bad Split (5-5 or 10-10)")
    
    # Context-dependent splits
    elif pair_value == 2 or pair_value == 3:
        if dealer_up_card <= 7:
            reward += 0.4
            reasons.append("Good Split (2-2/3-3 vs Low)")
        else:
            reward -= 0.4
            reasons.append("Bad Split (2-2/3-3 vs High)")
    
    elif pair_value == 6:
        if dealer_up_card <= 6:
            reward += 0.3
            reasons.append("Good Split (6-6 vs Low)")
        else:
            reward -= 0.5
            reasons.append("Bad Split (6-6 vs High)")
    
    elif pair_value == 7:
        if dealer_up_card <= 7:
            reward += 0.5
            reasons.append("Good Split (7-7 vs Low-Mid)")
        else:
            reward -= 0.5
            reasons.append("Bad Split (7-7 vs High)")
    
    elif pair_value == 9:
        if dealer_up_card in [7, 10, 1]:  # 7, 10, or Ace
            reward -= 0.3
            reasons.append("Bad Split (9-9 vs 7/10/A)")
        else:
            reward += 0.6
            reasons.append("Good Split (9-9)")
    
    # Bankroll consideration for splitting
    bet_ratio = player_bankroll / max(1, player_bankroll + 200)  # Rough estimation for split bet
    if bet_ratio < 0.2:  # Low bankroll
        reward -= 0.2
        reasons.append("Risky Split (Low Bankroll)")
    
    return reward, reasons

def calculate_bet_reward(bet_amount, bankroll, episode_starting_bankroll):
    """
    Calculate reward for betting decision based on bankroll management.
    
    Args:
        bet_amount: The amount being bet
        bankroll: Current bankroll
        episode_starting_bankroll: Starting bankroll for the episode
    
    Returns:
        tuple: (reward, reasons)
    """
    reward = 0.0
    reasons = []
    
    if bankroll <= 0:
        return 0.0, ["No Bankroll"]
    
    bet_ratio = bet_amount / bankroll
    bankroll_ratio = bankroll / episode_starting_bankroll
    
    # Bankroll management rewards
    if bankroll_ratio < 0.2:  # Low bankroll situation
        if bet_ratio <= 0.1:
            reward += 0.3
            reasons.append("Conservative Betting (Low Bankroll)")
        elif bet_ratio <= 0.3:
            reward -= 0.2
            reasons.append("Moderate Risk (Low Bankroll)")
        else:
            reward -= 0.8
            reasons.append("High Risk (Low Bankroll)")
    
    elif bankroll_ratio >= 1.5:  # High bankroll situation
        if bet_ratio <= 0.2:
            reward -= 0.1
            reasons.append("Too Conservative (High Bankroll)")
        elif bet_ratio <= 0.4:
            reward += 0.2
            reasons.append("Good Aggression (High Bankroll)")
        else:
            reward -= 0.3
            reasons.append("Excessive Risk (High Bankroll)")
    
    else:  # Normal bankroll situation
        if bet_ratio <= 0.3:
            reward += 0.1
            reasons.append("Reasonable Betting")
        else:
            reward -= 0.5
            reasons.append("High Risk Betting")
    
    return reward, reasons
