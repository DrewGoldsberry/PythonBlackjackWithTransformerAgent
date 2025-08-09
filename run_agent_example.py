from transformer_agent import TransformerAgent
from hand import Hand
from card import Card
from constants import ACTIONS
from tokenizer import tokenize_state
print("")

print("Loading agent...")
# Load the pre-trained agent model
agent = TransformerAgent.load("./models/blackjack_agent_ep.pt")
print("Agent loaded successfully.")
# Example usage
# You can now use the agent to make decisions in your game environment.
player_hand = Hand()
player_hand.add_card(card=Card(rank="6", suit="Hearts"))
player_hand.add_card(card=Card(rank="4", suit="Hearts"))
bankroll = 1000  # Example bankroll
bet=10  # Example bet amount
dealer_card = Card(rank="7", suit="Diamonds")
token_seq = tokenize_state(player_hand.cards, dealer_card, bankroll, bet)
action, probs, bet_fraction = agent.act(token_seq)

print("")
print(f"Action taken: {ACTIONS[action]}, bet fraction: {bet_fraction}, with probabilities: {probs}")
print("")

# This code snippet demonstrates how to use the TransformerAgent to make decisions based on the current game state.
# You can integrate this into your game loop or environment to allow the agent to play Blackjack.
# You must stop the round when it busts or has a blackjack otherwise it might keep hitting.
