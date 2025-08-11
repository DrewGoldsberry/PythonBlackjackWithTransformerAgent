# train_reinforce.py

import torch
import torch.nn.functional as F
from transformer_agent import TransformerAgent
from agent_player import AgentPlayer
from blackjack_env import BlackjackEnv
from torch.optim import Adam
import os

NUM_EPISODES = 1000  # Reduced since episodes are now longer
CHANGE_EPSILON_EVERY = 5
EPSILON_START = .3  # Reduced from 0.5 for more stable learning
EPSILON_END = 0.001
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32  # Smaller batch size for episode-based training
agent = None
MODEL_PATH = "./models/blackjack_agent_ep.pt"

agent = TransformerAgent().to(DEVICE)
optimizer = Adam(agent.parameters(), lr=LR)

episode_wins = 0
total_episodes = 0
player = AgentPlayer("Bot", agent=agent, epsilon=EPSILON_START, is_training=True)
env = BlackjackEnv(player=player)

# Episode-based training loop
conservative_episodes = 0  # Track overly conservative play

for episode in range(1, NUM_EPISODES + 1):
    # Standard epsilon decay
    if episode % CHANGE_EPSILON_EVERY == 0:
        player.epsilon = max(EPSILON_END, player.epsilon * 0.98)  # Slower decay
    
    # Anti-conservative mechanism: if agent is too conservative, increase exploration
    if episode > 100 and episode % 50 == 0:  # Check every 50 episodes after episode 100
        # If epsilon is very low, occasionally boost it to encourage exploration
        if player.epsilon < 0.05 and conservative_episodes > 5:
            player.epsilon = min(0.15, player.epsilon * 10)  # Temporary exploration boost
            conservative_episodes = 0
            print(f"🔄 Exploration boost! Epsilon increased to {player.epsilon:.4f}")
    
    # Start new episode - clear trajectories from previous episode if needed
    if len(player.trajectories) > 0:
        print(f"Warning: Clearing {len(player.trajectories)} unprocessed trajectories from previous episode")
        player.trajectories.clear()
    
    player.start_new_episode()
    episode_complete = False
    hands_in_episode = 0
    
    print(f"\n=== EPISODE {episode} START ===")
    print(f"Starting bankroll: {player.bankroll} | Epsilon: {player.epsilon:.4f}")
    
    # Play hands until episode completion condition is met
    while not episode_complete:
        env.reset()
        env.play_round()
        hands_in_episode += 1
        
        # Check if episode is complete after each hand
        episode_complete, reason = env.check_and_complete_episode(player)
        
        if episode_complete:
            print(f"\n=== EPISODE {episode} COMPLETE ===")
            print(f"Reason: {reason} | Hands played: {hands_in_episode}")
            print(f"Final bankroll: {player.bankroll}")
            
            # Detect overly conservative play (too many stands on low hands)
            stand_count = sum(1 for traj in player.trajectories if len(traj) == 3 and traj[1] == 1)  # action_idx 1 = stand
            hit_count = sum(1 for traj in player.trajectories if len(traj) == 3 and traj[1] == 0)   # action_idx 0 = hit
            if stand_count > 0 and hit_count / max(1, stand_count) < 0.3:  # Less than 30% hits
                conservative_episodes += 1
                print(f"⚠️  Conservative play detected (Hit ratio: {hit_count/(stand_count+hit_count):.2f})")
            else:
                conservative_episodes = max(0, conservative_episodes - 1)  # Reduce counter for good play
            
            if reason == "target_reached":
                episode_wins += 1
                
            total_episodes += 1
            
            # Train on trajectories with individual hand rewards + episode bonuses
            if len(player.trajectories) > 0:
                loss_terms = []
                total_reward = 0.0
                
                for traj_item in player.trajectories:
                    if len(traj_item) == 3:
                        # Action step: (token_seq, action_idx, reward)
                        token_seq, action_idx, reward = traj_item
                        if token_seq is not None and action_idx is not None:
                            r_t = torch.tensor(reward, dtype=torch.float32, device=DEVICE)
                            token_seq = token_seq.to(DEVICE)
                            logits, _ = agent(token_seq)
                            log_probs = F.log_softmax(logits, dim=-1)
                            log_prob = log_probs[0, action_idx]
                            loss_terms.append(-log_prob * r_t)
                            total_reward += float(reward)
                    
                    elif len(traj_item) == 5:
                        # Bet step: (token_seq, None, "bet", log_prob_bet, reward)
                        token_seq, _, bet_marker, log_prob_bet, reward = traj_item
                        if bet_marker == "bet":
                            r_t = torch.tensor(reward, dtype=torch.float32, device=log_prob_bet.device)
                            loss_terms.append(-log_prob_bet * r_t)
                            total_reward += float(reward)
                
                # Compute and apply gradients
                if loss_terms:
                    loss = torch.stack(loss_terms).sum()
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=1.0)  # Gradient clipping
                    optimizer.step()
                    
                    print(f"Training Loss: {loss.item():.4f} | Total Reward: {total_reward:.2f}")
                else:
                    print("No valid trajectories for training")
                
                # Clear trajectories for next episode
                player.trajectories.clear()
            else:
                print("No trajectories found for training")
        
        # Safety check to prevent infinite loops
        if hands_in_episode > 200:  # Maximum hands per episode
            print(f"Episode {episode} exceeded maximum hands limit")
            episode_complete = True
            
    # Periodic logging and model saving
    if episode % 10 == 0:
        win_rate = episode_wins / max(1, total_episodes) * 100
        print(f"\nEpisode {episode} Summary:")
        print(f"  Win Rate: {win_rate:.1f}% ({episode_wins}/{total_episodes})")
        print(f"  Epsilon: {player.epsilon:.4f}")
        
        # Save model periodically
        agent.save(MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")
        
        # Reset counters
        episode_wins = 0
        total_episodes = 0

print("\nTraining completed!")
agent.save(MODEL_PATH)
print(f"Final model saved to {MODEL_PATH}")
