
CARD_VALUE_TOKENS = {
    '2': 2,  '3': 3,  '4': 4,  '5': 5,  '6': 6,
    '7': 7,  '8': 8,  '9': 9,  '10': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14,
    'UNKNOWN': 1,  # For padding
}

MAX_PLAYER_CARDS = 5
MAX_TOKEN_SEQ_LEN = 10  # Dealer + Player + Bankroll + Bet
MAX_BANKROLL = 1000
NUM_BUCKETS = 20

def tokenize_state(player_hand, dealer_card, bankroll, bet):
    """
    Convert Blackjack game state to a sequence of integer tokens for the agent.
    Returns: torch.LongTensor of shape (1, seq_len)
    """
    import torch

    tokens = []

    # Dealer card (only the visible one)
    dealer_value = dealer_card.rank if dealer_card else 'UNKNOWN'
    tokens.append(CARD_VALUE_TOKENS.get(dealer_value, 1))

    # Player hand (rank only, pad to MAX_PLAYER_CARDS)
    for card in player_hand:
        tokens.append(CARD_VALUE_TOKENS.get(card.rank, 1))

    while len(tokens) < MAX_PLAYER_CARDS + 1:
        tokens.append(0)  # padding

    # Bucketed bankroll and bet
    bankroll_bucket = min(int((bankroll / MAX_BANKROLL) * NUM_BUCKETS), NUM_BUCKETS - 1)
    bet_bucket = min(int((bet / MAX_BANKROLL) * NUM_BUCKETS), NUM_BUCKETS - 1)

    tokens.append(100 + bankroll_bucket)  # Token range: 100–119
    tokens.append(120 + bet_bucket)       # Token range: 120–139

    token_tensor = torch.LongTensor(tokens).unsqueeze(0)  # shape: (1, seq_len)
    return token_tensor