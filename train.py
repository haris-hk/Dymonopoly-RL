from rl import train_dqn
import os
import torch

# ================================================================
#  Dynamic Pricing Configuration
# ================================================================
# Set this to your trained market bot path to enable dynamic pricing
# Example: "models/best_model/best_model.zip"
MARKET_BOT_PATH = "models/best_model/best_model.zip"  # Change to None for static prices
MARKET_UPDATE_FREQUENCY = 5  # Update prices every 5 turns

# ================================================================
#  Train DQN Agent
# ================================================================
# Train with 2 players to match the human vs AI game setup
model, rewards = train_dqn(
    episodes=500,
    max_steps_per_episode=200,
    batch_size=64,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_end=0.1,
    epsilon_decay=0.995,
    learning_rate=1e-3,
    num_players=2,  # Match the human vs AI game (2 players)
    market_bot_path=MARKET_BOT_PATH,  # Enable dynamic pricing with market bot
    market_update_frequency=MARKET_UPDATE_FREQUENCY,
)

# Save trained policy for gameplay
save_dir = os.path.join("models", "best_model")
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "dqn_policy.pt")
try:
    torch.save(model.state_dict(), save_path)
    print(f"Saved trained policy to: {save_path}")
except Exception as e:
    print(f"Warning: failed to save trained policy: {e}")