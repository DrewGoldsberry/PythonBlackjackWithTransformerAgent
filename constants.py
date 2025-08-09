# constants.py

# Pygame window
SCREEN_WIDTH = 1024
SCREEN_HEIGHT = 768
FPS = 60

# Colors
GREEN = (34, 139, 34)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (220, 20, 60)
YELLOW = (255, 215, 0)
BLUE = (30, 144, 255)

# Cards
SUITS = ['♠', '♥', '♦', '♣']
RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
CARD_VALUES = {
    '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
    '9': 9, '10': 10, 'J': 10, 'Q': 10, 'K': 10, 'A': 11  # Aces start as 11
}

AGENT_BANKROLL_TARGET = 4000
AGENT_STARTING_BANKROLL = 500
# Gameplay
NUM_DECKS = 6
RESHUFFLE_THRESHOLD = 52 * NUM_DECKS // 3  # Shuffle when 1/3 through
ACTIONS = ['hit', 'stand', 'double', 'split']