# transformer_agent.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from tokenizer import tokenize_state
from torch.distributions import Beta   # NEW

class TransformerAgent(nn.Module):
    def __init__(self, 
                 vocab_size=256,
                 d_model=128,
                 nhead=4,
                 num_layers=5,
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

        # ==== NEW: Beta policy for continuous bet in [0, 1] ====
        self.bet_trunk = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
        )
        self.bet_alpha = nn.Linear(d_model, 1)
        self.bet_beta  = nn.Linear(d_model, 1)
        # =======================================================

    def forward(self, token_seq):
        seq_len = token_seq.size(1)
        positions = torch.arange(0, seq_len, device=token_seq.device).unsqueeze(0)
        x = self.token_embedding(token_seq) + self.pos_embedding(positions)
        x = self.transformer(x)
        x = x[:, -1]  # Use the last token as state summary

        action_logits = self.policy_head(x)

        # ---- NEW: produce Beta(α, β) parameters (>1 to avoid extreme spikes) ----
        h = self.bet_trunk(x)
        alpha = F.softplus(self.bet_alpha(h)) + 1.0
        beta  = F.softplus(self.bet_beta(h))  + 1.0
        # -------------------------------------------------------------------------

        return action_logits, (alpha.squeeze(-1), beta.squeeze(-1))

    def act(self, token_seq, epsilon=0.0):
        logits, (alpha, beta) = self.forward(token_seq)
        probs = F.softmax(logits, dim=-1)

        if torch.rand(1).item() < epsilon:
            action = torch.randint(0, self.num_actions, (1,))
        else:
            action = torch.multinomial(probs, num_samples=1)

        # For legacy compatibility, return a bet *mean* here
        bet_mean = (alpha / (alpha + beta)).detach().item()
        return action.item(), probs[0], bet_mean

    # === NEW: sampling API used by AgentPlayer for training ===
    def sample_bet(self, token_seq, training=True):
        with torch.set_grad_enabled(training):
            _, (alpha, beta) = self.forward(token_seq)
            dist = Beta(alpha, beta)
            if training:
                # Sample with reparameterization-ish gradient support
                b = dist.rsample() if hasattr(dist, "rsample") else dist.sample()
            else:
                # Greedy = mean of Beta
                b = alpha / (alpha + beta)

            # Clamp to [0,1] just in case of numeric weirdness
            b = b.clamp(0.0, 1.0)
            log_prob = dist.log_prob(b)
            return b.squeeze(0), log_prob.squeeze(0)   # scalars

    def predict_bet(self, token_seq):
        """Return the mean bet fraction (0-1) given a tokenized state; no grad."""
        with torch.no_grad():
            _, (alpha, beta) = self.forward(token_seq)
            b = alpha / (alpha + beta)
            return b.squeeze(0).item()

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
        # strict=False allows loading older checkpoints without the Beta head
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        model.to(device)
        print(f"Agent loaded from {path}")
        return model

    def act_from_game_state(self, player_hand, dealer_card, bankroll, bet, epsilon=0.0):
        token_seq = tokenize_state(player_hand, dealer_card, bankroll, bet)
        logits, (alpha, beta) = self.forward(token_seq)
        probs = F.softmax(logits, dim=-1)
        if torch.rand(1).item() < epsilon:
            action = torch.randint(0, self.num_actions, (1,))
        else:
            action = torch.multinomial(probs, num_samples=1)
        bet_mean = (alpha / (alpha + beta)).item()
        return action.item(), probs[0].detach(), bet_mean
