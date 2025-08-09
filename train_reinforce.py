# train_reinforce.py

import torch
import torch.nn.functional as F
from transformer_agent import TransformerAgent
from agent_player import AgentPlayer
from blackjack_env import BlackjackEnv
from torch.optim import Adam
import os
NUM_EPISODES = 100000
SAVE_EVERY = 500
EPSILON = 0.2
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128
agent = None
MODEL_PATH = "./models/blackjack_agent_ep.pt"

#agent = TransformerAgent().to(DEVICE)
if (os.path.exists(MODEL_PATH) and agent is None):
    #using existing model if available
    agent = TransformerAgent.load(MODEL_PATH).to(DEVICE)
elif agent is None:
    agent = TransformerAgent().to(DEVICE)


optimizer = Adam(agent.parameters(), lr=LR)

win_count = 0
player = AgentPlayer("Bot", agent=agent)

env = BlackjackEnv(player=player)

for episode in range(1, NUM_EPISODES + 1):
    env.reset()
    env.play_round()

    # Pull and reverse for reward backfill
    traj = list(player.trajectories)
    traj.reverse()

    # Find the terminal reward (if any) and propagate backward to earlier steps
    reward = 0.0
    if episode % BATCH_SIZE == 0 or episode==NUM_EPISODES:
        for i in range(len(traj)):
            item = traj[i]

            # Item shapes:
            #  - action step: (token_seq, action_idx)  OR (token_seq, action_idx, reward)
            #  - bet step:    (token_seq, None, "bet", log_prob_bet) OR with +reward later
            if len(item) == 2:
                # (token_seq, action_idx) => attach current running reward
                token_seq, action_idx = item
                traj[i] = (token_seq, action_idx, reward)

            elif len(item) == 3:
                # Could be an action with reward (token_seq, action_idx, reward)
                token_seq, action_idx_or_none, third = item
                if action_idx_or_none is None:
                    # This is ambiguous; in our design, a bet step always has len==4
                    # so len==3 means action with explicit reward. Keep as is:
                    reward = float(third)
                    traj[i] = (token_seq, action_idx_or_none, reward)
                else:
                    # action with reward
                    reward = float(third)
                    traj[i] = (token_seq, action_idx_or_none, reward)

            elif len(item) == 4:
                # Bet step: (token_seq, None, "bet", log_prob_bet) possibly without reward yet
                token_seq, none_placeholder, bet_marker, log_prob_bet = item
                assert bet_marker == "bet"
                # Attach current reward for consistency:
                traj[i] = (token_seq, none_placeholder, bet_marker, log_prob_bet, reward)

        traj.reverse()

        # === Compute loss ===
        loss = torch.tensor(0.0, device=DEVICE)
        total_reward = 0.0

        loss_terms = []          # collect scalar tensors
        total_reward = 0.0

        for item in traj:
            if len(item) == 3:
                # action step: (token_seq, action_idx, r)
                token_seq, action_idx, r = item
                r_t = torch.tensor(r, dtype=torch.float32, device=DEVICE)
                if token_seq is not None:
                    token_seq = token_seq.to(DEVICE)
                    logits, _ = agent(token_seq)

                if action_idx is not None and token_seq is not None:
                    log_probs = F.log_softmax(logits, dim=-1)
                    log_prob = log_probs[0, action_idx]                  # scalar
                    loss_terms.append(-log_prob * r_t)                   # DO NOT do +=
                    total_reward += float(r)

            elif len(item) == 5:
                # bet step: (token_seq, None, "bet", log_prob_bet, r)
                token_seq, _, bet_marker, log_prob_bet, r = item
                assert bet_marker == "bet"
                r_t = torch.tensor(r, dtype=torch.float32, device=log_prob_bet.device)
                loss_terms.append(-log_prob_bet * r_t)                   # DO NOT do +=
                total_reward += float(r)

        # After the loop: sum once (no retain_graph needed)
        if loss_terms:
            loss = torch.stack(loss_terms).sum()
        else:
            loss = torch.zeros((), device=DEVICE)
        if len(traj) >1:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()                  
            optimizer.step()
        
        player.trajectories.clear()      # clear trajectories for next episode

        if total_reward > 0:
            win_count += 1

        
        print(f"Episode {episode} | Win Rate: {win_count:.2f} | Loss: {loss.item():.4f}")
        win_count = 0
        agent.save(f"models/blackjack_agent_ep.pt")
        print(f"Model saved at episode {episode}")
