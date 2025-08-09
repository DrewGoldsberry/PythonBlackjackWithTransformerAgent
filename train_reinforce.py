# train_reinforce.py

import torch
import torch.nn.functional as F
from transformer_agent import TransformerAgent
from agent_player import AgentPlayer
from blackjack_env import BlackjackEnv
from torch.optim import Adam

# === Config ===
NUM_EPISODES = 10000
SAVE_EVERY = 50
EPSILON = 0.2  # exploration
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
# === Init ===
#uncomment to train from scratch
agent = TransformerAgent().to(DEVICE)
#using existing model
#agent = TransformerAgent.load("./models/blackjack_agent_ep.pt").to(DEVICE)

optimizer = Adam(agent.parameters(), lr=LR)

win_count = 0
player = AgentPlayer("Bot", agent=agent)
env = BlackjackEnv(agent=player, use_agent=True)
for episode in range(1, NUM_EPISODES + 1):
    # 1. Create environment with an agent player
    env.reset()

    # 2. Run full game loop
    env.play_round()

    # 3. Collect trajectory from agent player
    player = env.players[0]
    if not isinstance(player, AgentPlayer):
        continue

    loss = torch.tensor(0.0, device=DEVICE)
    total_reward = 0
    trajectories = None
    trajectories = list(player.trajectories)
    trajectories.reverse()
    reward = 0
    for i in range(len(trajectories)):
        if (len(trajectories[i]) == 2):
            token_seq, action_idx = trajectories[i]
            trajectories[i] = token_seq, action_idx, reward
        else:
            token_seq, action_idx, _ = trajectories[i]
            reward = _

    trajectories.reverse()
    for token_seq, action_idx, reward in trajectories:
        token_seq = token_seq.to(DEVICE)
        logits, bet_pred = agent(token_seq)
        if action_idx is not None:
            # Update policy head using selected action
            log_probs = F.log_softmax(logits, dim=-1)
            log_prob = log_probs[0, action_idx]
            loss += -log_prob * reward
        else:
            # Update bet head: encourage larger bets for positive rewards and smaller for negative
            loss += -bet_pred[0] * reward
        total_reward += reward

    if total_reward > 0:
        win_count += 1

    # 4. Backpropagate
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 5. Print progress
    if episode % 50 == 0:
        print(f"Episode {episode} | Win Rate: {win_count/50:.2f} | Loss: {loss.item():.4f}")
        win_count = 0

    # 6. Save model
    if episode % SAVE_EVERY == 0:
        agent.save(f"models/blackjack_agent_ep.pt")
