# Playing Human vs AI in Monopoly

This guide will help you play against your trained DQN model in a 2-player Monopoly game.

## Prerequisites

Make sure you have all required packages installed:
```bash
pip install torch numpy gymnasium
```

## Step 1: Train the Model (if not already trained)

If you haven't trained a model yet, run:
```bash
python train.py
```

This will:
- Train a DQN agent for 500 episodes
- Save the trained model to `models/best_model/dqn_policy.pt`
- The training will take some time depending on your hardware

You should see output like:
```
Episode 10/500 | Avg Reward (last 10): -2.45 | Epsilon: 0.951
Episode 20/500 | Avg Reward (last 10): -1.82 | Epsilon: 0.905
...
```

## Step 2: Play Against the AI

Once training is complete, run:
```bash
python play_human_vs_ai.py
```

## Game Controls

### During Your Turn:

1. **View Game State**: The terminal will display:
   - Both players' cash and net worth
   - Current positions on the board
   - Owned properties and developments
   - Jail status

2. **Make Decisions**: You'll be prompted to enter action numbers:
   - `[0]` End Turn - Skip to next player
   - `[1]` Buy Property - Purchase the property you landed on
   - `[2]` Pay Jail Fine - Pay $50 to get out of jail
   - `[3]` Use Get Out of Jail Card - Use your jail card
   - `[4]` Accept Jail - Stay in jail for up to 3 turns
   - `[5+]` Build/Sell actions - Build houses/hotels or sell properties

3. **Building Houses/Hotels**:
   - You can only build on properties where you own the complete color set
   - Houses must be built evenly across the color group
   - Cost is shown next to each build option

4. **Selling Assets**:
   - Sell buildings for 50% refund of house cost
   - Mortgage properties for 50% of purchase price
   - Useful when running low on cash

### Game Rules:

- **Starting Cash**: $1,500 per player
- **Passing GO**: Collect $200
- **Jail**: 
  - Pay $50 fine immediately
  - Use a "Get Out of Jail Free" card
  - Or stay in jail (automatically released after 3 turns)
- **Bankruptcy**: Game ends when a player runs out of money
- **Max Turns**: Default 100 turns (winner by net worth if no bankruptcy)

## Customization

You can modify `play_human_vs_ai.py` to change:

```python
MODEL_PATH = "models/best_model/dqn_policy.pt"  # Path to your model
MAX_TURNS = 100  # Maximum turns before game ends
AI_EPSILON = 0.0  # AI exploration (0.0 = fully trained, 0.1 = 10% random)
```

## Troubleshooting

### Model Not Found Error
```
❌ ERROR: Model not found at models/best_model/dqn_policy.pt
```
**Solution**: Run `python train.py` first to train the model.

### Game Freezes
**Solution**: The game waits for your input. Type an action number and press ENTER.

### Invalid Action Error
```
❌ Invalid action X. Please choose from the available actions.
```
**Solution**: Only enter action numbers shown in the available actions list.

## Game Flow Example

```
╔═══════════════════════ TURN 5 ════════════════════════╗
======================================================================

    👤 Player 1 (YOU (Human))
    💰 Cash: $1350 | 📊 Net Worth: $1450 | Status: ACTIVE
    📍 Position (6): Oriental Avenue
    🎫 Jail Cards: 0
    🏠 Owned Properties (2):
       • Mediterranean Avenue [brown] - No development
       • Baltic Avenue [brown] - No development

    🤖 Player 2 (AI Agent)
    💰 Cash: $1280 | 📊 Net Worth: $1480 | Status: ACTIVE
    📍 Position (12): St. Charles Place
    🎫 Jail Cards: 0
    🏠 Owned Properties (3):
       • Reading Railroad [N/A] - No development
       • Vermont Avenue [light_blue] - No development
       • Connecticut Avenue [light_blue] - No development

----------------------------------------------------------------------

🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹
                    👤 YOUR TURN                    
🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹🔹

🎲 DECISION TYPE: DECISION
----------------------------------------------------------------------
🏢 Landed on: Oriental Avenue ($100)
   Type: property | Color: light_blue

📋 AVAILABLE ACTIONS:
----------------------------------------------------------------------
   [0] end_turn
   [1] buy_property (Cost: $100.0)
----------------------------------------------------------------------

🎯 Player 1, enter action number: 1
```

## Strategy Tips

### Against the AI:
- **Early Game**: Buy properties aggressively to deny the AI monopolies
- **Mid Game**: Focus on completing color sets for building opportunities
- **Late Game**: Build houses/hotels on high-rent properties
- **Cash Management**: Keep enough cash for rent payments
- **Jail Strategy**: Early game, pay to get out. Late game, stay in jail to avoid landing on expensive properties

### The AI's Behavior:
- The AI uses a Deep Q-Network trained through reinforcement learning
- It has learned strategies like property acquisition, building timing, and cash management
- The AI plays deterministically (no randomness) by default
- It evaluates actions based on expected long-term rewards

## Have Fun!

Enjoy playing against your AI! See if you can beat the agent you trained. 🎮🎲

