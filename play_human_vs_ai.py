"""
Human vs AI Monopoly Game
Play against your trained DQN model in a 2-player Monopoly game.
"""

import os
import sys
import torch
import numpy as np
from typing import Dict, Optional
from rl import DymonopolyDecisionEnv, QNetwork, select_action


def print_game_state(env: DymonopolyDecisionEnv, current_player_id: int, show_full_board: bool = False):
    """Prints the game state for both players."""
    print("\n" + "=" * 70)
    print(f"╔═══════════════════════ TURN {env.turn_count + 1} ════════════════════════╗")
    print("=" * 70)
    
    # Show both players' status
    for pid in range(env.num_players):
        cash = env.player_cash[pid]
        pos = env.player_positions[pid]
        tile_name = env.properties[pos]['name']
        net_worth = env._player_net_worth(pid)
        jail_status = "🔒 IN JAIL" if env.player_in_jail[pid] else ""
        bankrupt_status = "💀 BANKRUPT" if env.player_bankrupt[pid] else ""
        
        player_type = "YOU (Human)" if pid == 0 else "AI Agent"
        marker = "👤" if pid == 0 else "🤖"
        is_current = ">>> " if pid == current_player_id else "    "
        
        status = bankrupt_status or jail_status or "ACTIVE"
        
        print(f"{is_current}{marker} Player {pid + 1} ({player_type})")
        print(f"    💰 Cash: ${cash:.0f} | 📊 Net Worth: ${net_worth:.0f} | Status: {status}")
        print(f"    📍 Position ({pos}): {tile_name}")
        print(f"    🎫 Jail Cards: {env.player_jail_cards[pid]}")
        
        # Show owned properties
        owned_props = env._select_properties_owned(pid)
        if owned_props:
            print(f"    🏠 Owned Properties ({len(owned_props)}):")
            for idx in owned_props:
                prop = env.properties[idx]
                houses = env.property_houses[idx]
                
                # Determine development status
                if houses == 5:
                    dev = "🏨 Hotel"
                elif houses > 0:
                    dev = "🏠" * houses
                else:
                    dev = "No development"
                
                color = prop.get('color', 'N/A')
                print(f"       • {prop['name']} [{color}] - {dev}")
        print()
    
    print("-" * 70)


def asset_management_menu(env: DymonopolyDecisionEnv, player_id: int) -> bool:
    """Interactive asset management menu for human player. Returns True if actions were taken."""
    management_active = True
    actions_taken = False
    
    while management_active:
        print("\n" + "=" * 70)
        print("🏠 ASSET MANAGEMENT MENU")
        print("=" * 70)
        cash = env.player_cash[player_id]
        print(f"💰 Current Cash: ${cash:.0f}")
        
        # Determine available options
        buildable = env._get_buildable_monopolies(player_id)
        sellable_buildings = env._get_sellable_buildings(player_id)
        sellable_properties = env._get_all_owned_properties(player_id)
        
        print("\n📋 Management Options:")
        print("   [1] Build House/Hotel")
        print("   [2] Sell House/Hotel")
        print("   [3] Sell Property")
        print("   [4] Done (Return to Main Turn)")
        print("-" * 70)
        
        try:
            choice = input("\n🎯 Select option: ").strip()
            
            if choice == "":
                continue
            
            option = int(choice)
            
            # Build House/Hotel
            if option == 1:
                if not buildable:
                    print("❌ Cannot build: No complete color sets or insufficient funds.")
                    continue
                
                print("\n🏗️  BUILDABLE PROPERTIES:")
                print("-" * 70)
                for i, prop_id in enumerate(buildable, 1):
                    prop = env.properties[prop_id]
                    houses = env.property_houses[prop_id]
                    cost = prop.get('house_cost', 0)
                    next_level = 'Hotel' if houses == 4 else f'{houses + 1} House(s)'
                    print(f"   [{i}] {prop['name']} [{prop.get('color')}]")
                    print(f"       Current: {houses} house(s) → Next: {next_level} | Cost: ${cost:.0f}")
                print("-" * 70)
                
                try:
                    build_choice = input("\n🎯 Select property to build on (or Enter to cancel): ").strip()
                    if build_choice == "":
                        continue
                    
                    build_idx = int(build_choice) - 1
                    if 0 <= build_idx < len(buildable):
                        prop_id = buildable[build_idx]
                        result = env._handle_build_for_player({}, player_id, prop_id)
                        if result > 0:
                            print(f"✅ Successfully built on {env.properties[prop_id]['name']}!")
                            actions_taken = True
                        else:
                            print(f"❌ Failed to build on {env.properties[prop_id]['name']}.")
                    else:
                        print("❌ Invalid selection.")
                except ValueError:
                    print("❌ Invalid input.")
            
            # Sell Building
            elif option == 2:
                if not sellable_buildings:
                    print("❌ No houses or hotels to sell.")
                    continue
                
                print("\n💵 SELLABLE BUILDINGS:")
                print("-" * 70)
                for i, prop_id in enumerate(sellable_buildings, 1):
                    prop = env.properties[prop_id]
                    houses = env.property_houses[prop_id]
                    refund = prop.get('house_cost', 0) * 0.5
                    dev_level = 'Hotel' if houses == 5 else f'{houses} House(s)'
                    print(f"   [{i}] {prop['name']} [{prop.get('color')}]")
                    print(f"       Current: {dev_level} | Refund: ${refund:.0f}")
                print("-" * 70)
                
                try:
                    sell_choice = input("\n🎯 Select property to sell building from (or Enter to cancel): ").strip()
                    if sell_choice == "":
                        continue
                    
                    sell_idx = int(sell_choice) - 1
                    if 0 <= sell_idx < len(sellable_buildings):
                        prop_id = sellable_buildings[sell_idx]
                        result = env._handle_sell_building_for_player({}, player_id, prop_id)
                        if result > 0:
                            print(f"✅ Successfully sold building from {env.properties[prop_id]['name']}!")
                            actions_taken = True
                        else:
                            print(f"❌ Failed to sell building from {env.properties[prop_id]['name']}.")
                    else:
                        print("❌ Invalid selection.")
                except ValueError:
                    print("❌ Invalid input.")
            
            # Sell Property
            elif option == 3:
                if not sellable_properties:
                    print("❌ No properties to sell.")
                    continue
                
                print("\n💵 SELLABLE PROPERTIES:")
                print("-" * 70)
                for i, prop_id in enumerate(sellable_properties, 1):
                    prop = env.properties[prop_id]
                    houses = env.property_houses[prop_id]
                    refund = env._get_property_price(prop_id) * 0.5
                    
                    if houses > 0:
                        # Add building refund
                        building_refund = houses * prop.get('house_cost', 0) * 0.5
                        total_refund = refund + building_refund
                        status = f"Has {houses} building(s) - Total Refund: ${total_refund:.0f}"
                    else:
                        status = f"Refund: ${refund:.0f}"
                    
                    print(f"   [{i}] {prop['name']} [{prop.get('color', 'N/A')}] - {status}")
                print("-" * 70)
                
                try:
                    sell_choice = input("\n🎯 Select property to sell (or Enter to cancel): ").strip()
                    if sell_choice == "":
                        continue
                    
                    sell_idx = int(sell_choice) - 1
                    if 0 <= sell_idx < len(sellable_properties):
                        prop_id = sellable_properties[sell_idx]
                        result = env._handle_sell_property_for_player({}, player_id, prop_id)
                        if result > 0:
                            print(f"✅ Successfully sold {env.properties[prop_id]['name']}!")
                            actions_taken = True
                        else:
                            print(f"❌ Failed to sell {env.properties[prop_id]['name']}.")
                    else:
                        print("❌ Invalid selection.")
                except ValueError:
                    print("❌ Invalid input.")
            
            # Done
            elif option == 4:
                management_active = False
            
            else:
                print("❌ Invalid option. Please select 1-4.")
        
        except ValueError:
            print("❌ Invalid input. Please enter a number.")
        except (EOFError, KeyboardInterrupt):
            print("\n\n⚠️  Returning to main turn...")
            management_active = False
    
    return actions_taken


def get_human_action(env: DymonopolyDecisionEnv, player_id: int, info: Dict) -> int:
    """Get action from human player via terminal input."""
    ctx = info["context"]
    mask = info["action_mask"]
    valid_actions = [i for i, m in enumerate(mask) if m == 1]
    
    if not valid_actions:
        print("⚠️  No valid actions available. Ending turn.")
        return env.END_TURN
    
    # Display context information
    ctx_type = ctx.get("type", "decision")
    print(f"\n🎲 DECISION TYPE: {ctx_type.upper()}")
    print("-" * 70)
    
    if ctx_type == "decision" and ctx.get("buy_target") is not None:
        prop_id = ctx["buy_target"]
        prop = env.properties[prop_id]
        price = ctx.get("buy_price", 0)
        print(f"🏢 Landed on: {prop['name']} (${price:.0f})")
        print(f"   Type: {prop.get('type', 'property')} | Color: {prop.get('color', 'N/A')}")
    
    if ctx_type == "jail_entry":
        print("🚔 You've been sent to JAIL!")
        print(f"   Fine: $50 | Jail Cards Available: {env.player_jail_cards[player_id]}")
    
    # For decision context, offer asset management option
    if ctx_type == "decision":
        print(f"\n📋 QUICK ACTIONS:")
        print("-" * 70)
        print(f"   [0] End Turn")
        if ctx.get("buy_target") is not None:
            print(f"   [1] Buy Property (${ctx.get('buy_price', 0):.0f})")
        print(f"   [M] Manage Assets (Build/Sell Houses & Properties)")
        print("-" * 70)
        
        while True:
            try:
                choice = input(f"\n🎯 Player {player_id + 1}, enter choice: ").strip().upper()
                
                if choice == "":
                    print("⚠️  No input provided.")
                    continue
                
                # Manage assets
                if choice == "M":
                    asset_management_menu(env, player_id)
                    # After management, ask again
                    continue
                
                # Regular action
                action = int(choice)
                
                if action == 0:
                    return env.END_TURN
                elif action == 1 and ctx.get("buy_target") is not None:
                    return env.BUY_PROPERTY
                else:
                    print(f"❌ Invalid action.")
            
            except ValueError:
                print("❌ Invalid input. Enter a number or 'M' for management.")
            except (EOFError, KeyboardInterrupt):
                print("\n\n⚠️  Game interrupted by user.")
                sys.exit(0)
    
    # For non-decision contexts (jail, etc), show all valid actions
    else:
        print(f"\n📋 AVAILABLE ACTIONS:")
        print("-" * 70)
        
        action_meanings = env.action_meanings
        for action_idx in valid_actions:
            meaning = action_meanings[action_idx] if action_idx < len(action_meanings) else f"action_{action_idx}"
            
            # Add helpful context for specific actions
            if action_idx == env.PAY_JAIL_FINE:
                meaning += " (Cost: $50)"
            
            print(f"   [{action_idx}] {meaning}")
        
        print("-" * 70)
        
        # Get user input
        while True:
            try:
                choice = input(f"\n🎯 Player {player_id + 1}, enter action number: ").strip()
                
                if choice == "":
                    print("⚠️  No input provided. Please enter an action number.")
                    continue
                
                action = int(choice)
                
                if action in valid_actions:
                    return action
                else:
                    print(f"❌ Invalid action {action}. Please choose from the available actions.")
            
            except ValueError:
                print("❌ Invalid input. Please enter a number.")
            except (EOFError, KeyboardInterrupt):
                print("\n\n⚠️  Game interrupted by user.")
                sys.exit(0)


def ai_take_management_actions(env: DymonopolyDecisionEnv, player_id: int) -> bool:
    """AI makes intelligent building/selling decisions. Returns True if actions were taken."""
    actions_taken = False
    
    # Strategy 1: Build on monopolies if affordable and maintain cash buffer
    buildable = env._get_buildable_monopolies(player_id)
    cash_buffer = env.starting_cash * 0.3  # Keep 30% of starting cash as buffer
    
    for prop_id in buildable:
        cost = env.properties[prop_id].get('house_cost', 0)
        if env.player_cash[player_id] > cost + cash_buffer:
            result = env._handle_build_for_player({}, player_id, prop_id)
            if result > 0:
                prop_name = env.properties[prop_id]['name']
                houses = env.property_houses[prop_id]
                print(f"   🏗️  AI built on {prop_name} (now has {houses} house(s))")
                actions_taken = True
                # Build one at a time to be strategic
                break
    
    # Strategy 2: Sell buildings if cash is critically low
    if env.player_cash[player_id] < 100:  # Critical cash level
        sellable = env._get_sellable_buildings(player_id)
        if sellable:
            prop_id = sellable[0]  # Sell from first available
            result = env._handle_sell_building_for_player({}, player_id, prop_id)
            if result > 0:
                prop_name = env.properties[prop_id]['name']
                print(f"   💵 AI sold building from {prop_name} (low on cash)")
                actions_taken = True
    
    return actions_taken


def get_ai_action(
    policy_net: QNetwork,
    env: DymonopolyDecisionEnv,
    obs: Dict,
    mask: np.ndarray,
    device: torch.device,
    epsilon: float = 0.0,
    player_id: int = 1
) -> int:
    """Get action from trained AI model."""
    # First, check if AI should manage assets (build/sell)
    # Only do this during decision context
    ctx = env.pending_context
    if ctx and ctx.get("type") == "decision":
        managed = ai_take_management_actions(env, player_id)
        if managed:
            # After management, prepare context again for next decision
            env._prepare_turn_context(player_id)
            obs = env._get_obs()
            mask = env.pending_mask
    
    # Then make main decision (buy, end turn, etc)
    state = env.flatten_observation(obs)
    action = select_action(policy_net, state, mask, epsilon, device)
    return action


def play_human_vs_ai(
    model_path: str = "models/best_model/dqn_policy.pt",
    max_turns: int = 100,
    ai_epsilon: float = 0.0  # Set to 0 for fully trained behavior, >0 for exploration
):
    """
    Run a 2-player game: Human (Player 1) vs AI (Player 2).
    
    Args:
        model_path: Path to trained DQN model weights
        max_turns: Maximum number of turns before game ends
        ai_epsilon: Exploration rate for AI (0.0 = fully exploit learned policy)
    """
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ ERROR: Model not found at {model_path}")
        print("Please train a model first using train.py")
        return
    
    # Initialize environment with 2 players
    print("🎮 Initializing Monopoly: Human vs AI")
    print("=" * 70)
    env = DymonopolyDecisionEnv(num_players=2, max_turns=max_turns)
    
    # Load trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"💻 Using device: {device}")
    
    policy_net = QNetwork(env.flat_observation_size, env.action_space.n).to(device)
    
    try:
        policy_net.load_state_dict(torch.load(model_path, map_location=device))
        policy_net.eval()
        print(f"✅ Loaded trained model from: {model_path}")
    except Exception as e:
        print(f"❌ ERROR loading model: {e}")
        return
    
    print(f"🎲 Starting game with max {max_turns} turns")
    print(f"👤 Player 1: HUMAN")
    print(f"🤖 Player 2: AI Agent (epsilon={ai_epsilon})")
    print("=" * 70)
    input("\n⏸️  Press ENTER to start the game...")
    
    # Initialize game
    obs, info = env.reset()
    terminated = False
    truncated = False
    current_player = 0
    
    # Game loop
    while not terminated and not truncated:
        # Check if current player is bankrupt, skip their turn
        if env.player_bankrupt[current_player]:
            current_player = (current_player + 1) % env.num_players
            continue
        
        # Prepare turn context for current player
        env.current_player = current_player
        env._prepare_turn_context(current_player)
        
        # Display game state
        print_game_state(env, current_player)
        
        # Get action based on player type
        if current_player == 0:
            # Human player
            print(f"\n{'🔹' * 35}")
            print(f"{'  ' * 10}👤 YOUR TURN {'  ' * 10}")
            print(f"{'🔹' * 35}\n")
            action = get_human_action(env, current_player, {
                "context": env.pending_context,
                "action_mask": env.pending_mask
            })
            
        else:
            # AI player
            print(f"\n{'🔸' * 35}")
            print(f"{'  ' * 10}🤖 AI's TURN {'  ' * 10}")
            print(f"{'🔸' * 35}\n")
            
            action = get_ai_action(
                policy_net,
                env,
                obs,
                env.pending_mask,
                device,
                ai_epsilon,
                current_player
            )
            
            action_name = env.action_meanings[action] if action < len(env.action_meanings) else f"action_{action}"
            print(f"🤖 AI chose: [{action}] {action_name}")
            input("\n⏸️  Press ENTER to continue...")
        
        # Apply action
        pre_worth = env._player_net_worth(current_player)
        reward = env._apply_action_for_player(action, current_player)
        env.pending_context = None
        
        # Check bankruptcy
        env._check_and_handle_bankruptcy(current_player)
        post_worth = env._player_net_worth(current_player)
        
        print(f"\n📊 Turn Result:")
        print(f"   Action: {env.action_meanings[action] if action < len(env.action_meanings) else f'action_{action}'}")
        print(f"   Net Worth Change: {post_worth - pre_worth:+.2f}")
        
        # Check if only one player remains
        active_players = env.num_players - np.sum(env.player_bankrupt)
        if active_players <= 1:
            terminated = True
        
        # Move to next player
        if not terminated:
            current_player = (current_player + 1) % env.num_players
            # Update observation for AI's next turn
            obs = env._get_obs()
        
        env.turn_count += 1
        truncated = env.turn_count >= max_turns
    
    # Game over - display results
    print("\n" + "🏁" * 35)
    print(f"{'  ' * 12}GAME OVER{'  ' * 12}")
    print("🏁" * 35)
    
    if terminated:
        # Find winner (non-bankrupt player)
        for pid in range(env.num_players):
            if not env.player_bankrupt[pid]:
                player_type = "👤 HUMAN" if pid == 0 else "🤖 AI"
                net_worth = env._player_net_worth(pid)
                print(f"\n🎉 WINNER: Player {pid + 1} ({player_type})")
                print(f"   Final Net Worth: ${net_worth:.2f}")
                break
    elif truncated:
        # Find player with highest net worth
        net_worths = [env._player_net_worth(i) for i in range(env.num_players)]
        winner_id = np.argmax(net_worths)
        player_type = "👤 HUMAN" if winner_id == 0 else "🤖 AI"
        print(f"\n⏱️  Game reached max turns ({max_turns})")
        print(f"🎉 WINNER by Net Worth: Player {winner_id + 1} ({player_type})")
        print(f"   Final Net Worth: ${net_worths[winner_id]:.2f}")
        
        print(f"\n📊 Final Standings:")
        for pid in range(env.num_players):
            player_type = "👤 HUMAN" if pid == 0 else "🤖 AI"
            status = "💀 BANKRUPT" if env.player_bankrupt[pid] else "ACTIVE"
            print(f"   Player {pid + 1} ({player_type}): ${net_worths[pid]:.2f} - {status}")
    
    print("\n" + "=" * 70)
    print("Thank you for playing!")
    print("=" * 70)


if __name__ == "__main__":
    """
    Run the game directly from terminal:
    python play_human_vs_ai.py
    """
    
    # You can customize these parameters
    MODEL_PATH = os.path.join("models", "best_model", "dqn_policy.pt")
    MAX_TURNS = 100
    AI_EPSILON = 0.0  # 0.0 = no exploration, fully use learned policy
    
    try:
        play_human_vs_ai(
            model_path=MODEL_PATH,
            max_turns=MAX_TURNS,
            ai_epsilon=AI_EPSILON
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  Game interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        import traceback
        traceback.print_exc()
