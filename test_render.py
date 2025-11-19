import numpy as np
from rl import DymonopolyEnv

# Create environment with human rendering
env = DymonopolyEnv(num_properties=40, num_players=4)
env.render_mode = "human"

print("\n🎮 Starting Dymonopoly Market Simulation...")
print("⏸️  Press Ctrl+C to stop early\n")

obs, info = env.reset()

try:
    for step in range(50):  # Simulate 50 turns (10 price updates)
        # Random action: price multipliers for all properties
        action = env.action_space.sample()
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render every turn
        env.render()
        
        if terminated:
            print("🏁 Game ended - Only one player remaining!")
            break
        
        # Optional: pause every 5 turns for readability
        if (step + 1) % 5 == 0:
            input("Press Enter to continue...")

except KeyboardInterrupt:
    print("\n\n⏹️  Simulation stopped by user")

print("\n✅ Simulation complete!")