# test_trained_agent.py

import torch
import os
from transformer_agent import TransformerAgent
from agent_player import AgentPlayer
from blackjack_env import BlackjackEnv
from constants import AGENT_STARTING_BANKROLL

def test_trained_agent(num_hands=1000, model_path="./models/blackjack_agent_ep.pt"):
    """
    Test a trained agent over multiple hands and collect detailed statistics.
    """
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return None
    
    # Load the trained agent
    print(f"Loading trained model from {model_path}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = TransformerAgent.load(model_path).to(device)
    
    # Create player with NO exploration (epsilon=0 for pure exploitation)
    player = AgentPlayer("TestBot", agent=agent, epsilon=0.0, is_training=False)
    player.bankroll = AGENT_STARTING_BANKROLL  # Reset to starting bankroll
    
    # Create environment
    env = BlackjackEnv(player=player)
    
    # Initialize statistics
    stats = {
        'hands_played': 0,
        'wins': 0,
        'losses': 0,
        'pushes': 0,
        'busts': 0,
        'blackjacks': 0,
        'doubles': 0,
        'splits': 0,
        'money_won': 0.0,
        'money_lost': 0.0,
        'total_bet': 0.0,
        'final_bankroll': 0.0,
        'starting_bankroll': player.bankroll,
        'max_bankroll': player.bankroll,
        'min_bankroll': player.bankroll,
        'bankroll_history': []
    }
    
    print(f"Starting test with {num_hands} hands...")
    print(f"Starting bankroll: ${player.bankroll:.2f}")
    print(f"Using device: {device}")
    print("=" * 60)
    
    for hand_num in range(1, num_hands + 1):
        # Store bankroll before hand
        bankroll_before = player.bankroll
        
        # Reset environment and play
        env.reset()
        
        # Get bet amount for statistics
        current_bet = player.current_hand().bet
        stats['total_bet'] += current_bet
        
        # Track hand types
        if player.current_hand().has_doubled:
            stats['doubles'] += 1
        if len(player.hands) > 1:  # Split occurred
            stats['splits'] += 1
        if player.current_hand().is_blackjack():
            stats['blackjacks'] += 1
        
        # Play the round
        env.play_round()
        
        # Calculate bankroll change
        bankroll_after = player.bankroll
        bankroll_change = bankroll_after - bankroll_before
        
        # Update statistics based on outcome
        if player.current_hand().is_busted():
            stats['busts'] += 1
            stats['losses'] += 1
            stats['money_lost'] += abs(bankroll_change)
        elif player.current_hand().is_winner:
            if env.dealer.current_hand().get_values() == player.current_hand().get_values():
                stats['pushes'] += 1  # Tie
            else:
                stats['wins'] += 1
                if bankroll_change > 0:
                    stats['money_won'] += bankroll_change
        else:
            stats['losses'] += 1
            stats['money_lost'] += abs(bankroll_change)
        
        # Update bankroll tracking
        stats['max_bankroll'] = max(stats['max_bankroll'], bankroll_after)
        stats['min_bankroll'] = min(stats['min_bankroll'], bankroll_after)
        stats['bankroll_history'].append(bankroll_after)
        
        stats['hands_played'] += 1
        
        # Clear trajectories to prevent memory issues
        player.trajectories.clear()
        
        # Progress update every 100 hands
        if hand_num % 100 == 0:
            win_rate = stats['wins'] / stats['hands_played'] * 100
            current_profit = bankroll_after - stats['starting_bankroll']
            print(f"Hand {hand_num:4d} | Win Rate: {win_rate:5.1f}% | "
                  f"Bankroll: ${bankroll_after:7.2f} | "
                  f"Profit: ${current_profit:+7.2f}")
        
        # Stop if bankroll is exhausted
        if player.bankroll <= 0:
            print(f"\nBankroll exhausted at hand {hand_num}!")
            break
    
    # Final calculations
    stats['final_bankroll'] = player.bankroll
    net_profit = stats['final_bankroll'] - stats['starting_bankroll']
    win_rate = stats['wins'] / stats['hands_played'] * 100 if stats['hands_played'] > 0 else 0
    
    # Print detailed results
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    
    print(f"Hands Played:      {stats['hands_played']:,}")
    print(f"Wins:              {stats['wins']:,} ({stats['wins']/stats['hands_played']*100:.1f}%)")
    print(f"Losses:            {stats['losses']:,} ({stats['losses']/stats['hands_played']*100:.1f}%)")
    print(f"Pushes:            {stats['pushes']:,} ({stats['pushes']/stats['hands_played']*100:.1f}%)")
    print(f"Busts:             {stats['busts']:,} ({stats['busts']/stats['hands_played']*100:.1f}%)")
    print(f"Blackjacks:        {stats['blackjacks']:,} ({stats['blackjacks']/stats['hands_played']*100:.1f}%)")
    print(f"Doubles:           {stats['doubles']:,} ({stats['doubles']/stats['hands_played']*100:.1f}%)")
    print(f"Splits:            {stats['splits']:,} ({stats['splits']/stats['hands_played']*100:.1f}%)")
    
    print(f"\nFINANCIAL SUMMARY:")
    print(f"Starting Bankroll: ${stats['starting_bankroll']:,.2f}")
    print(f"Final Bankroll:    ${stats['final_bankroll']:,.2f}")
    print(f"Net Profit/Loss:   ${net_profit:+,.2f}")
    print(f"Total Money Won:   ${stats['money_won']:,.2f}")
    print(f"Total Money Lost:  ${stats['money_lost']:,.2f}")
    print(f"Total Bet:         ${stats['total_bet']:,.2f}")
    print(f"Max Bankroll:      ${stats['max_bankroll']:,.2f}")
    print(f"Min Bankroll:      ${stats['min_bankroll']:,.2f}")
    
    # Calculate ROI and other metrics
    if stats['total_bet'] > 0:
        house_edge = (stats['money_lost'] - stats['money_won']) / stats['total_bet'] * 100
        print(f"House Edge:        {house_edge:.2f}%")
    
    roi = net_profit / stats['starting_bankroll'] * 100
    print(f"ROI:               {roi:+.2f}%")
    
    # Risk metrics
    max_drawdown = stats['starting_bankroll'] - stats['min_bankroll']
    print(f"Max Drawdown:      ${max_drawdown:.2f} ({max_drawdown/stats['starting_bankroll']*100:.1f}%)")
    
    # Average bet size
    avg_bet = stats['total_bet'] / stats['hands_played'] if stats['hands_played'] > 0 else 0
    print(f"Average Bet:       ${avg_bet:.2f}")
    
    print("=" * 60)
    
    return stats

def plot_bankroll_history(stats):
    """
    Optional function to plot bankroll over time (requires matplotlib).
    """
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 6))
        plt.plot(stats['bankroll_history'])
        plt.title('Bankroll Over Time')
        plt.xlabel('Hand Number')
        plt.ylabel('Bankroll ($)')
        plt.grid(True, alpha=0.3)
        plt.axhline(y=stats['starting_bankroll'], color='r', linestyle='--', label='Starting Bankroll')
        plt.legend()
        plt.tight_layout()
        
        # Save plot
        plot_path = 'bankroll_history.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Bankroll history plot saved as: {plot_path}")
        
    except ImportError:
        print("matplotlib not available - skipping bankroll plot")

if __name__ == "__main__":
    # Run the test
    results = test_trained_agent(num_hands=1000)
    
    if results:
        # Optionally plot results
        plot_bankroll_history(results)
        
        # Save detailed results to file
        with open("test_results.txt", "w") as f:
            f.write(f"Test Results for Trained Blackjack Agent\n")
            f.write(f"{'='*50}\n\n")
            f.write(f"Hands Played: {results['hands_played']:,}\n")
            f.write(f"Wins: {results['wins']:,} ({results['wins']/results['hands_played']*100:.1f}%)\n")
            f.write(f"Losses: {results['losses']:,} ({results['losses']/results['hands_played']*100:.1f}%)\n")
            f.write(f"Pushes: {results['pushes']:,} ({results['pushes']/results['hands_played']*100:.1f}%)\n")
            f.write(f"Net Profit/Loss: ${results['final_bankroll'] - results['starting_bankroll']:+,.2f}\n")
            f.write(f"Final Bankroll: ${results['final_bankroll']:,.2f}\n")
            f.write(f"Total Money Won: ${results['money_won']:,.2f}\n")
            f.write(f"Total Money Lost: ${results['money_lost']:,.2f}\n")
            f.write(f"Total Bet: ${results['total_bet']:,.2f}\n")
        
        print("\nDetailed results saved to: test_results.txt")
