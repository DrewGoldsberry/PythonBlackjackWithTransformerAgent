# transformer_agent.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from tokenizer import tokenize_state

class TransformerAgent(nn.Module):
    def __init__(self, 
                 vocab_size=256,
                 d_model=128,
                 nhead=4,
                 num_layers=2,
                 max_seq_len=20,
                 num_actions=4):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.num_actions = num_actions

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.policy_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, num_actions)
        )

        # Head for predicting bet size as a fraction between 0 and 1
        self.bet_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )

    def forward(self, token_seq):
        seq_len = token_seq.size(1)
        positions = torch.arange(0, seq_len, device=token_seq.device).unsqueeze(0)
        x = self.token_embedding(token_seq) + self.pos_embedding(positions)
        x = self.transformer(x)
        x = x[:, -1]  # Use the last token
        action_logits = self.policy_head(x)
        bet = self.bet_head(x).squeeze(-1)
        return action_logits, bet

    def act(self, token_seq, epsilon=0.0):
        """
        Select an action based on the output logits. Epsilon controls exploration.
        """
        logits, bet = self.forward(token_seq)
        probs = F.softmax(logits, dim=-1)

        if torch.rand(1).item() < epsilon:
            action = torch.randint(0, self.num_actions, (1,))
        else:
            action = torch.multinomial(probs, num_samples=1)

        return action.item(), probs[0], bet.item()

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': {
                'vocab_size': self.vocab_size,
                'd_model': self.token_embedding.embedding_dim,
                'nhead': self.transformer.layers[0].self_attn.num_heads,
                'num_layers': len(self.transformer.layers),
                'max_seq_len': self.max_seq_len,
                'num_actions': self.num_actions
            }
        }, path)
        print(f"Agent saved to {path}")

    @classmethod
    def load(cls, path, device='cpu'):
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint['config']
        model = cls(**config)
        # Allow loading models that might not have the bet head yet
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        model.to(device)
        print(f"Agent loaded from {path}")
        return model
    
    def act_from_game_state(self, player_hand, dealer_card, bankroll, bet, epsilon=0.0):
        """
        Full game-state-based act() method that uses the tokenizer.
        """
        token_seq = tokenize_state(player_hand, dealer_card, bankroll, bet)
        logits, bet_fraction = self.forward(token_seq)
        probs = F.softmax(logits, dim=-1)

        if torch.rand(1).item() < epsilon:
            action = torch.randint(0, self.num_actions, (1,))
        else:
            action = torch.multinomial(probs, num_samples=1)

        return action.item(), probs[0].detach(), bet_fraction.item()

    def predict_bet(self, token_seq):
        """Return a bet fraction (0-1) given a tokenized state."""
        with torch.no_grad():
            action, prob, bet = self.act(token_seq)
            return bet
