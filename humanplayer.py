

from rl import DymonopolyDecisionEnv
import copy
import random
import sys
from collections import deque
from typing import Dict, List, Optional, Tuple
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# --- Game Running Functions ---
def print_game_state(env: DymonopolyDecisionEnv, current_player_id: int):
    """Prints the essential game state for the given player."""
    player_id = current_player_id
    print("\n" + "=" * 50)
    print(f"MONOPOLY GAME TURN {env.turn_count + 1}")
    print("=" * 50)
    
    # Current Player Status
    cash = env.player_cash[player_id]
    pos = env.player_positions[player_id]
    tile_name = env.properties[pos]['name']
    net_worth = env._player_net_worth(player_id)
    
    print(f"👤 Player {player_id + 1} (YOU) | Cash: ${cash:.2f} | Net Worth: ${net_worth:.2f}")
    print(f"📍 Current Position ({pos}): {tile_name}")
    print(f"🗃️ Jail Cards: {env.player_jail_cards[player_id]}")

    # Property List
    owned_props = env._select_properties_owned(player_id)
    if owned_props:
        print("\n🏠 Your Owned Properties:")
        for idx in owned_props:
            prop = env.properties[idx]
            houses = env.property_houses[idx]
            cost = prop.get('house_cost', 0)
            
            # Determine development status display
            dev_level = "None"
            if houses == 5:
                dev_level = "Hotel"
            elif houses > 0:
                dev_level = f"{houses} Houses"
            
            print(f"- {prop['name']} (Color: {prop.get('color', 'N/A')}) [Dev: {dev_level}] [Price: ${prop['price']:.0f}]")
    
    # All Player Status
    print("\n👥 All Player Status:")
    for i in range(env.num_players):
        if i == player_id: continue
        
        opp_cash = env.player_cash[i]
        opp_worth = env._player_net_worth(i)
        status = "JAIL" if env.player_in_jail[i] else "ACTIVE"
        status = "BANKRUPT" if env.player_bankrupt[i] else status
        print(f"- Player {i + 1} | Cash: ${opp_cash:.0f} | Worth: ${opp_worth:.0f} | Status: {status}")
    print("-" * 50)


def human_policy(env: DymonopolyDecisionEnv, player_id: int, info: Dict) -> int:
    """Prompts the human player for an action based on the current context."""
    ctx = info["context"]
    mask = info["action_mask"]
    valid_actions = [i for i, m in enumerate(mask) if m == 1]
    
    action_map = {idx: env.action_meanings[idx] for idx in valid_actions}
    
    if not valid_actions:
        print("No valid actions available. Skipping turn (Action 1).")
        return 1

    action_prompt = f"🎲 Decision Type: {ctx.get('type', 'idle').upper()}\n"
    
    if ctx.get("type") == "buy":
        prop = env.properties[ctx["property_id"]]
        action_prompt += f"    Landed on: {prop['name']} (Price: ${ctx['price']:.0f})\n"
    elif ctx.get("type") == "sell":
        prop = env.properties[ctx["property_id"]]
        refund = prop.get('house_cost', 0) * 0.5
        action_prompt += f"    FORCED ACTION: Low cash! Must sell development on {prop['name']} (Refund: ${refund:.0f}) or risk bankruptcy.\n"
    elif ctx.get("type") == "jail":
        action_prompt += f"    You are in Jail. Fine is $50.\n"
    
    action_prompt += "\nSelect Action:\n"
    for idx, meaning in action_map.items():
        action_prompt += f"  [{idx}] {meaning}\n"
    
    print(action_prompt)

    while True:
        try:
            choice = input(f"Player {player_id + 1} - Enter action index: ").strip()
            action = int(choice)
            if action in valid_actions:
                return action
            print("Invalid action index or action not available in current context. Try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")
        except EOFError:
            return 1 # Default to skip

def bot_policy(env: DymonopolyDecisionEnv, player_id: int, ctx: Dict) -> int:
    """A simple heuristic policy for non-human players."""
    ctx_type = ctx.get("type")
    
    # 1. Jail Decision (Prioritize escape)
    if ctx_type == "jail":
        if 5 in ctx.get("valid_actions", []): # Use card
            return 5
        if 4 in ctx.get("valid_actions", []): # Pay fine
            return 4
    
    # 2. Buying Decision (Buy if affordable and maintain a cash buffer)
    if ctx_type == "buy":
        price = ctx.get("price", 0)
        # Buy if we can afford it and still have at least 50% of starting cash left
        if env.player_cash[player_id] > price and env.player_cash[player_id] >= env.starting_cash * 0.5:
            return 0 # buy_property
    
    # 3. Selling Decision (Only sell if absolutely forced to by low cash context)
    if ctx_type == "sell":
        # The 'sell' context is usually triggered by low cash, so the bot should sell to survive.
        return 3 # sell_house
    
    # 4. Spontaneous Management (Bot only acts if a full turn has been completed)
    if ctx_type == "idle":
        # Check if the bot can build
        buildable = env._get_buildable_monopolies(player_id)
        if buildable:
            # Build on the first affordable property
            return 2 # build_house_or_hotel (Note: this action is handled differently in the flexible loop)
            
    # Default: skip_decision
    return 1 

def get_player_choice(prompt: str, choices: Dict[int, str], player_id: int) -> Optional[int]:
    """Helper for complex multi-step input."""
    print("\n" + "=" * 50)
    print(prompt)
    for idx, name in choices.items():
        print(f"  [{idx}] {name}")
    print("=" * 50)

    while True:
        try:
            choice = input(f"Player {player_id + 1} - Enter index: ").strip()
            if choice == "": return None # Allow empty choice to cancel/skip
            idx = int(choice)
            if idx in choices:
                return idx
            print("Invalid index. Try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")
        except EOFError:
            return None

def _asset_management_menu(env: DymonopolyDecisionEnv, player_id: int):
    """Interactive management loop for human player."""
    management_active = True
    
    while management_active:
        print("\n--- ASSET MANAGEMENT MENU ---")
        cash = env.player_cash[player_id]
        print(f"Current Cash: ${cash:.2f}")

        # Determine available high-level actions
        has_monopoly = bool(env._get_buildable_monopolies(player_id))
        has_buildings = bool(env._get_sellable_buildings(player_id))
        has_property = bool(env._get_all_owned_properties(player_id))
        
        menu_choices = {
            1: "Build House/Hotel (If Monopoly Owned)",
            2: "Sell House/Hotel (Development)",
            3: "Sell Property (Mortgage)",
            4: "Done Management (End Turn Phase)"
        }
        
        # --- High-Level Choice ---
        choice = get_player_choice("Select Management Action:", menu_choices, player_id)
        if choice is None or choice == 4:
            management_active = False
            continue

        # --- Sub-Menus ---
        
        # 1. Build House/Hotel
        if choice == 1:
            if not has_monopoly:
                print("Cannot build: You do not own a full color set or cannot build evenly.")
                continue
            
            buildable_list = env._get_buildable_monopolies(player_id)
            if not buildable_list:
                print("Cannot build: No affordable or valid building spots currently exist (check affordability and even development rule).")
                continue

            build_options = {}
            for prop_id in buildable_list:
                prop = env.properties[prop_id]
                houses = env.property_houses[prop_id]
                cost = prop.get('house_cost', 0)
                next_level = 'Hotel' if houses == 4 else f'{houses + 1} House'
                build_options[prop_id] = f"{prop['name']} (Current: {houses} houses) -> {next_level} [Cost: ${cost:.0f}]"
                
            prop_id_to_build = get_player_choice("Select Property to Build On:", build_options, player_id)
            
            if prop_id_to_build is not None:
                prop_id_to_build = int(prop_id_to_build)
                # Apply the action directly
                result = env._handle_build_for_player({}, player_id, prop_id_to_build)
                if result > 0:
                    print(f"SUCCESS: Built on {env.properties[prop_id_to_build]['name']}.")
                else:
                    print("FAILURE: Could not build (check cash/rules).")

        # 2. Sell House/Hotel
        elif choice == 2:
            if not has_buildings:
                print("Cannot sell building: You have no houses or hotels to liquidate.")
                continue

            sellable_buildings = env._get_sellable_buildings(player_id)
            sell_options = {}
            for prop_id in sellable_buildings:
                prop = env.properties[prop_id]
                houses = env.property_houses[prop_id]
                refund = prop.get('house_cost', 0) * 0.5
                sell_options[prop_id] = f"{prop['name']} (Current: {houses} development) [Refund: ${refund:.0f}]"
            
            prop_id_to_sell_building = get_player_choice("Select Property to Sell Development From:", sell_options, player_id)

            if prop_id_to_sell_building is not None:
                prop_id_to_sell_building = int(prop_id_to_sell_building)
                result = env._handle_sell_building_for_player({}, player_id, prop_id_to_sell_building)
                if result > 0:
                    print(f"SUCCESS: Sold development from {env.properties[prop_id_to_sell_building]['name']}.")
                else:
                    print("FAILURE: Could not sell development.")
        
        # 3. Sell Property
        elif choice == 3:
            if not has_property:
                print("Cannot sell property: You own no properties.")
                continue
                
            owned_props = env._get_all_owned_properties(player_id)
            sell_prop_options = {}
            for prop_id in owned_props:
                prop = env.properties[prop_id]
                sell_price = prop['price'] * 0.5
                if env.property_houses[prop_id] > 0:
                    status = "Blocked (Must sell buildings first)"
                else:
                    status = f"Mortgage Price: ${sell_price:.0f}"
                sell_prop_options[prop_id] = f"{prop['name']} ({status})"

            prop_id_to_sell = get_player_choice("Select Property to Sell (Mortgage):", sell_prop_options, player_id)

            if prop_id_to_sell is not None:
                prop_id_to_sell = int(prop_id_to_sell)
                result = env._handle_sell_property_for_player({}, player_id, prop_id_to_sell)
                if result > 0:
                    print(f"SUCCESS: Sold (mortgaged) property {env.properties[prop_id_to_sell]['name']}.")
                else:
                    print("FAILURE: Could not sell property.")
        
        print(f"Current Cash after management: ${env.player_cash[player_id]:.2f}")


def run_flexible_game(num_players: int = 4, num_human_players: int = 1, max_turns: int = 50):
    """Runs a game with a customizable mix of human and bot players."""
    
    if num_human_players > num_players or num_players < 2:
        print("Invalid configuration: num_human_players cannot exceed num_players, and must have at least 2 players total.")
        return
        
    env = DymonopolyDecisionEnv(num_players=num_players, max_turns=max_turns)
    env._reset_game_state()
    
    human_ids = set(range(min(num_human_players, num_players)))
    current_player_idx = 0
    terminated = False
    truncated = False
    
    print(f"Starting Dymonopoly Game (Players 1-{num_human_players} are Human, Players {num_human_players + 1}-{num_players} are Bots)")
    
    while not terminated and not truncated:
        player_id = current_player_idx
        
        # Skip bankrupt players
        num_active_players = num_players - np.sum(env.player_bankrupt)
        if num_active_players <= 1:
            terminated = True
            break
            
        while env.player_bankrupt[player_id]:
            current_player_idx = (current_player_idx + 1) % env.num_players
            player_id = current_player_idx
            if current_player_idx == player_id:
                break
        
        # 1. Prepare turn context (rolls dice, lands, determines next available action)
        env._prepare_turn_context(player_id)
        
        info = {"action_mask": env.pending_mask.copy(), "context": env.pending_context}
        action = 1 # Default action is skip

        if player_id in human_ids:
            # Human Player Decision
            print(f"\n{'< ' * 5} PLAYER {player_id + 1}'s TURN (HUMAN) {' >' * 5}")
            print_game_state(env, player_id)
            
            # Get initial move/action (Buy/Jail/Skip/Manage)
            action = human_policy(env, player_id, info)
            
            # If the human chooses Management, open the interactive menu
            if action == 6: # manage_assets
                _asset_management_menu(env, player_id)
                action = 1 # Force action to skip so the main loop moves to the next player
            elif action == 3 and info['context']['type'] == 'sell':
                 # If forced to sell, execute it
                prop_id = info['context'].get("property_id")
                env._handle_sell_building_for_player({}, player_id, prop_id)
                action = 1 # After forced sale, skip the rest of the phase

        else:
            # Bot Player Decision
            print(f"\n{'< ' * 5} PLAYER {player_id + 1}'s TURN (BOT) {' >' * 5}")
            action = bot_policy(env, player_id, info['context'])
            print(f"Bot chooses action: [{action}] {env.action_meanings[action]}")
            
            # Bot executes spontaneous management only if bot_policy returns build action (2)
            if action == 2:
                # Bot decides to build. We find the best place to build now.
                buildable_list = env._get_buildable_monopolies(player_id)
                if buildable_list:
                    prop_id_to_build = buildable_list[0] # Bot just takes the first valid option
                    env._handle_build_for_player({}, player_id, prop_id_to_build)
                action = 1 # Always follow up management actions with a skip

        # 2. Apply action (Buy, Jail, Skip)
        pre_worth = env._player_net_worth(player_id)
        # Execute the chosen action
        env._apply_action_for_player(action, player_id)
        env.pending_context = None
        
        # 3. Check for bankruptcy
        env._check_and_handle_bankruptcy(player_id)
        post_worth = env._player_net_worth(player_id)
        net_worth_change = post_worth - pre_worth

        print(f"\n[Turn Result for P{player_id + 1}] Action taken: {env.action_meanings[action]} | Net Worth Change: {net_worth_change:+.2f}")
        
        # Check end conditions
        num_active_players = num_players - np.sum(env.player_bankrupt)
        terminated = num_active_players <= 1
        
        env.turn_count += 1 
        truncated = env.turn_count >= max_turns
        
        current_player_idx = (current_player_idx + 1) % env.num_players
            
    # Print game end summary
    print("\n" + "#" * 50)
    if terminated:
        winner_id = np.argmax(~env.player_bankrupt)
        print(f"GAME OVER: Player {winner_id + 1} wins!")
    elif truncated:
        # Determine current highest net worth player
        net_worths = [env._player_net_worth(i) for i in range(num_players)]
        winner_id = np.argmax(net_worths)
        print(f"GAME OVER: Max turns ({max_turns}) reached. Player {winner_id + 1} leads with ${net_worths[winner_id]:.2f}.")
    print("#" * 50)

# Renamed for clarity, as it runs a human vs. bots hybrid.
def run_hybrid_game(max_turns: int = 50):
    """Runs a single game of Dymonopoly with a human player (Player 0) vs 3 heuristic opponents."""
    
    env = DymonopolyDecisionEnv(num_players=4, max_turns=max_turns)
    
    obs, info = env.reset()
    terminated = False
    truncated = False
    
    print("Starting Dymonopoly Game (Human Player 1 vs 3 Heuristic Opponents)")
    
    while not terminated and not truncated:
        print_game_state(env, 0)
        
        # Player 0 (Human) turn and decision
        action = human_policy(env, 0, info)
        
        # Note: env.step handles Player 0's decision AND the subsequent 3 bot turns.
        obs, reward, terminated, truncated, info = env.step(action) 
        
        print(f"\n[Turn Result] Action taken: {env.action_meanings[action]} | Net Worth Change Reward: {reward:.4f}")
        
        if terminated:
            print("\n" + "#" * 50)
            print(f"GAME OVER: Player 1 is BANKRUPT.")
            print("#" * 50)
        elif truncated:
            print("\n" + "#" * 50)
            print(f"GAME OVER: Max turns ({max_turns}) reached.")
            print("#" * 50)

# Renamed for clarity, as it runs all human players.
def run_multiplayer_game(num_players: int = 4, max_turns: int = 50):
    """Runs a single game of Dymonopoly with multiple human players."""
    # This now just calls the flexible game function with all humans.
    return run_flexible_game(num_players=num_players, num_human_players=num_players, max_turns=max_turns)


# --- Example Execution Block (Optional, for demonstration) ---
# To run the game, uncomment the following lines and run the script:

if __name__ == "__main__":
    # Example: Run a game with 2 Human Players (P1, P2) and 2 Bot Players (P3, P4)
    try:
        run_flexible_game(num_players=4, num_human_players=2, max_turns=50)
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        print("Note: This script requires an interactive console for input() to work.")