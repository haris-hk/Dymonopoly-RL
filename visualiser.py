import os
import pygame
import numpy as np
from rl import DymonopolyEnv, DymonopolyDecisionEnv
import random
from gymnasium import spaces

# Try to import SAC model components (optional - for market pricing)
try:
    from stable_baselines3 import SAC
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from gymnasium.wrappers import FlattenObservation
    SAC_AVAILABLE = True
except ImportError as e:
    print(f"[Market Model] SAC not available: {e}")
    SAC_AVAILABLE = False
except OSError as e:
    print(f"[Market Model] SAC loading error (possibly PyTorch DLL issue): {e}")
    SAC_AVAILABLE = False

# Try to import DQN model components (for AI player)
try:
    import torch
    import torch.nn as nn
    DQN_AVAILABLE = True
except ImportError as e:
    print(f"[AI Model] PyTorch not available: {e}")
    DQN_AVAILABLE = False
except OSError as e:
    print(f"[AI Model] PyTorch loading error: {e}")
    DQN_AVAILABLE = False


# QNetwork class for DQN (mirrors rl.py definition)
class QNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
        )

    def forward(self, x):
        return self.model(x)

# Game states
class GameState:
    AWAITING_ROLL = "awaiting_roll"
    AWAITING_DECISION = "awaiting_decision"  # After landing, can buy/build/sell/skip
    AWAITING_TILE_SELECT = "awaiting_tile_select"  # Selecting tile for build/sell
    GAME_OVER = "game_over"

chance_cards = [
    {"type": "money", "amount": 200, "text": "Bank error in your favour. Collect £200."},
    {"type": "money", "amount": -100, "text": "Pay speeding fine of £100."},
    {"type": "move", "target": 0, "text": "Advance to Go. Collect £200."},
    {"type": "go_to_jail", "text": "Go directly to Jail. Do not pass Go, do not collect £200."},
    {"type": "jail_card", "text": "Get Out of Jail Free. This card may be kept until needed or sold."},
]

chest_cards = [
    {"type": "money", "amount": 150, "text": "Your building loan matures. Collect £150."},
    {"type": "money", "amount": -50, "text": "Doctor's fees. Pay £50."},
    {"type": "money", "amount": 100, "text": "Life insurance matures. Collect £100."},
    {"type": "go_to_jail", "text": "Go directly to Jail. Do not pass Go, do not collect £200."},
    {"type": "jail_card", "text": "Get Out of Jail Free. This card may be kept until needed or sold."},
]

class ActionButton:
    """Simple rectangular button with enable/disable support."""

    def __init__(self, rect, label, callback, base_color, enabled_getter=lambda: True):
        self.rect = rect
        self.label = label
        self.callback = callback
        self.base_color = base_color
        self.enabled_getter = enabled_getter
        self.dynamic_color = None

    def is_enabled(self):
        return self.enabled_getter()

    def handle_click(self, pos):
        if self.rect.collidepoint(pos) and self.is_enabled():
            self.callback()
            return True
        return False

    def draw(self, surface, font):
        color = self.dynamic_color or self.base_color
        color = color if self.is_enabled() else (160, 160, 160)
        pygame.draw.rect(surface, color, self.rect, border_radius=6)
        pygame.draw.rect(surface, (30, 30, 30), self.rect, 2, border_radius=6)
        text = font.render(self.label, True, (0, 0, 0))
        surface.blit(text, text.get_rect(center=self.rect.center))

class MonopolyVisualizer:
    def __init__(self, env=None):
        pygame.init()
        
        # Support both env types - prefer DymonopolyDecisionEnv
        if env is None:
            self.env = DymonopolyDecisionEnv(num_players=2)
            self.env.reset()
            self.using_decision_env = True
        elif isinstance(env, DymonopolyDecisionEnv):
            self.env = env
            self.using_decision_env = True
        else:
            # Legacy DymonopolyEnv support
            self.env = env
            self.using_decision_env = False
        
        self.width = 1100
        self.height = 780
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Dymonopoly - Human vs AI")

        self.corner_size = 100
        self.edge_size = 58
        self.board_size = self.corner_size * 2 + self.edge_size * 9
        self.margin_left = 20
        self.margin_top = (self.height - self.board_size) // 2
        self.board_rect = pygame.Rect(
            self.margin_left,
            self.margin_top,
            self.board_size,
            self.board_size,
        )
        self.info_rect = pygame.Rect(
            self.board_rect.right + 20,
            self.board_rect.top,
            self.width - (self.board_rect.right + 40),
            self.board_size,
        )


        # Palette
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.BOARD_EDGE = (0, 0, 0)
        self.BOARD_CENTER = (213, 235, 247)
        self.BANNER_RED = (199, 48, 35)
        self.CHANCE_ORANGE = (247, 161, 70)
        self.CHEST_BLUE = (153, 204, 255)
        self.PROPERTY_COLORS = {
            "brown": (94, 63, 39),
            "light_blue": (173, 216, 230),
            "pink": (255, 182, 193),
            "orange": (255, 165, 0),
            "red": (237, 28, 36),
            "yellow": (255, 242, 0),
            "green": (34, 177, 76),
            "dark_blue": (0, 91, 150),
        }

        # Fonts
        self.font_small = pygame.font.SysFont("arial", 16)
        self.font_medium = pygame.font.SysFont("arial", 20, bold=True)
        self.font_large = pygame.font.SysFont("arial", 28, bold=True)
        self.font_corner = pygame.font.SysFont("arial", 22, bold=True)
        self.font_price = pygame.font.SysFont("arial", 14)
        self.font_banner = pygame.font.SysFont("arial", 80, bold=True)
        self.font_stats = pygame.font.SysFont("arial", 18)
        self.font_house_num = pygame.font.SysFont("arial", 12, bold=True)

        self.tiles = self._build_tiles()
        self.layout = self._calculate_layout()
        self.max_players = 2  # Human vs AI
        
        # For DecisionEnv compatibility - ensure all players start at GO (position 0)
        if self.using_decision_env:
            # Force reset positions to 0 (GO) for a fresh game
            self.env.player_positions = np.zeros(self.max_players, dtype=np.int32)
            self.player_positions = [0] * self.max_players
        else:
            self.env.num_players = self.max_players
            self.player_positions = [0] * self.max_players
            self.env.current_player = 0
            self.env.turn_counter = 0

        self.collision_offsets = {}
        for i in range(21, 30):   # top row tiles 21..29
            self.collision_offsets[i] = (20, 0)
        for i in range(31, 40):   # right column tiles 31..39
            self.collision_offsets[i] = (0, 20)

        # Game state machine
        self.game_state = GameState.AWAITING_ROLL
        self.pending_action_type = None  # "build" or "sell_building" or "sell_property"
        
        # Legacy state vars (for compatibility)
        self.awaiting_roll = True
        self.awaiting_decision = False
        self.last_roll_total = None
        self.last_dice = None
        self.player_last_rolls = [None] * self.max_players
        self.selected_tile = 0
        self.message_log = ["Player 1 (Human) - Click 'Roll Dice' to begin"]
        self.show_property_card = False
        self.last_clicked_property = None
        self.show_event_card = False
        self.current_event_card = None
        self.current_event_type = None
        
        # Buildable/Sellable lists for current player
        self.buildable_tiles = []
        self.sellable_building_tiles = []
        self.sellable_property_tiles = []

        self.images_dir = os.path.join(os.path.dirname(__file__), "images")
        self.board_surface = self._load_board_surface()
        self.dice_images = self._load_dice_images()

        self.clock = pygame.time.Clock()
        self.action_buttons = self._create_action_buttons()
        self.sell_count = 0  # Counter for sell building
        
        # Plus/Minus button rects for sell building counter
        self.plus_rect = pygame.Rect(0, 0, 26, 30)
        self.minus_rect = pygame.Rect(0, 0, 26, 30)
        
        # ========== MARKET PRICE OVERLAY ==========
        self.show_market_overlay = False
        self.market_overlay_start_time = 0
        self.market_overlay_duration = 15000  # 15 seconds
        self.market_price_changes = []  # Store price changes for overlay
        
        # ========== MARKET PRICE MODEL (SAC) ==========
        # Load trained SAC model for dynamic market pricing
        self.market_model = None
        self.market_vec_normalize = None
        self.market_update_interval = 5  # Update prices every 5 turns
        self.base_prices = self._get_base_prices()  # Store original prices
        
        if SAC_AVAILABLE:
            model_path = os.path.join(os.path.dirname(__file__), "models", "last_model.zip")
            normalize_path = os.path.join(os.path.dirname(__file__), "models", "vec_normalize.pkl")
            
            if os.path.exists(model_path):
                try:
                    # Create dummy env for VecNormalize loading
                    self.market_model = SAC.load(model_path, device="cpu")
                    if os.path.exists(normalize_path):
                        # Load the VecNormalize stats
                        self.market_vec_normalize = VecNormalize.load(normalize_path, DummyVecEnv([self._make_dummy_market_env]))
                        self.market_vec_normalize.training = False
                        self.market_vec_normalize.norm_reward = False
                    print("[Market Model] SAC model loaded successfully!")
                except Exception as e:
                    print(f"[Market Model] Failed to load: {e}")
                    self.market_model = None
            else:
                print(f"[Market Model] Model not found at {model_path}")
        else:
            print("[Market Model] SAC not available - market prices will remain static")
        
        # ========== AI PLAYER MODEL (DQN) ==========
        # Load trained DQN model for AI player decisions
        self.ai_policy = None
        self.ai_device = None
        self.ai_action_delay = 1000  # ms delay between AI actions for visibility
        self.ai_last_action_time = 0
        self.ai_thinking = False  # Flag to show AI is "thinking"
        self.ai_actions_this_turn = 0  # Counter to limit actions per turn
        self.ai_max_actions_per_turn = 2  # Max actions before auto end turn (buy + maybe build)
        
        if DQN_AVAILABLE:
            dqn_model_path = os.path.join(os.path.dirname(__file__), "models", "best_model", "dqn_policy.pt")
            
            if os.path.exists(dqn_model_path):
                try:
                    self.ai_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    # Calculate network dimensions
                    input_dim = self.env.flat_observation_size
                    output_dim = self.env.action_space.n
                    
                    self.ai_policy = QNetwork(input_dim, output_dim).to(self.ai_device)
                    self.ai_policy.load_state_dict(torch.load(dqn_model_path, map_location=self.ai_device))
                    self.ai_policy.eval()
                    print(f"[AI Model] DQN policy loaded successfully! (input={input_dim}, output={output_dim})")
                except Exception as e:
                    print(f"[AI Model] Failed to load DQN: {e}")
                    import traceback
                    traceback.print_exc()
                    self.ai_policy = None
            else:
                print(f"[AI Model] Model not found at {dqn_model_path}")
        else:
            print("[AI Model] PyTorch not available - AI will use random actions")
    
    def _make_dummy_market_env(self):
        """Create a dummy market env for VecNormalize compatibility."""
        if not SAC_AVAILABLE:
            return None
        env = DymonopolyEnv(num_properties=40, num_players=4)
        env = FlattenObservation(env)
        return env
    
    def _get_base_prices(self):
        """Extract base prices from tiles (for price multiplier reference)."""
        prices = np.zeros(40, dtype=np.float32)
        for i, tile in enumerate(self.tiles):
            if "price" in tile:
                prices[i] = tile["price"]
        return prices
    
    def _get_market_observation(self):
        """Build observation dict matching DymonopolyEnv format for SAC model."""
        num_properties = 40
        num_players = self.max_players
        
        # Property prices (current)
        property_prices = np.zeros(num_properties, dtype=np.float32)
        for i, tile in enumerate(self.tiles):
            if "price" in tile:
                property_prices[i] = tile["price"]
        
        # Property owners
        property_owners = np.full(num_properties, -1, dtype=np.float32)
        if self.using_decision_env:
            for i in range(num_properties):
                property_owners[i] = self.env.property_owner[i]
        else:
            for i, tile in enumerate(self.tiles):
                property_owners[i] = tile.get("owned_by", -1)
        
        # Visiting frequency
        visiting_freq = np.zeros(num_properties, dtype=np.float32)
        for i, tile in enumerate(self.tiles):
            visiting_freq[i] = tile.get("visiting_freq", 0)
        
        # Trading frequency
        trading_freq = np.zeros(num_properties, dtype=np.float32)
        for i, tile in enumerate(self.tiles):
            trading_freq[i] = tile.get("trading_frequency", 0)
        
        # Player cash
        player_cash = np.zeros(num_players, dtype=np.float32)
        if self.using_decision_env:
            player_cash = self.env.player_cash.astype(np.float32)
        else:
            for i in range(num_players):
                player_cash[i] = 1500  # default
        
        # Current player
        current_player = self.env.current_player if hasattr(self.env, 'current_player') else 0
        
        return {
            "property_prices": property_prices,
            "property_owners": property_owners,
            "visiting_frequency": visiting_freq,
            "trading_frequency": trading_freq,
            "player_cash": player_cash,
            "current_player": current_player
        }
    
    def _flatten_observation(self, obs_dict):
        """Flatten observation dict to match FlattenObservation wrapper."""
        # Order must match spaces.Dict ordering (alphabetical by key)
        # current_player, player_cash, property_owners, property_prices, trading_frequency, visiting_frequency
        parts = []
        
        # current_player as one-hot (4 players in training)
        current_player_onehot = np.zeros(4, dtype=np.float32)
        current_player_onehot[min(obs_dict["current_player"], 3)] = 1.0
        parts.append(current_player_onehot)
        
        # player_cash (pad to 4 players)
        player_cash = np.zeros(4, dtype=np.float32)
        player_cash[:len(obs_dict["player_cash"])] = obs_dict["player_cash"]
        parts.append(player_cash)
        
        parts.append(obs_dict["property_owners"])
        parts.append(obs_dict["property_prices"])
        parts.append(obs_dict["trading_frequency"])
        parts.append(obs_dict["visiting_frequency"])
        
        return np.concatenate(parts).astype(np.float32)
    
    def _update_market_prices(self):
        """Use SAC model to update property prices based on market conditions."""
        if self.market_model is None:
            return
        
        # Get observation
        obs_dict = self._get_market_observation()
        obs_flat = self._flatten_observation(obs_dict)
        
        # Normalize if VecNormalize is available
        if self.market_vec_normalize is not None:
            obs_flat = self.market_vec_normalize.normalize_obs(obs_flat)
        
        # Get action from model (deterministic)
        action, _ = self.market_model.predict(obs_flat, deterministic=True)
        
        # Convert action [-1, 1] to price multipliers [0.4, 2.0]
        # Formula from rl.py: multiplier = 1.2 + 0.8 * action
        multipliers = 1.2 + 0.8 * action
        
        # Apply multipliers to base prices
        price_changes = []
        for i, tile in enumerate(self.tiles):
            if "price" in tile and self.base_prices[i] > 0:
                old_price = tile["price"]
                new_price = int(self.base_prices[i] * multipliers[i])
                # Clamp to reasonable range
                new_price = max(10, min(new_price, int(self.base_prices[i] * 3)))
                
                if new_price != old_price:
                    change = new_price - old_price
                    change_pct = ((new_price / old_price) - 1) * 100
                    price_changes.append((tile["name"], old_price, new_price, change_pct))
                    tile["price"] = new_price
        
        # Show market overlay with price changes
        if price_changes:
            # Sort by absolute percentage change and take top changes
            self.market_price_changes = sorted(price_changes, key=lambda x: abs(x[3]), reverse=True)[:8]
            self.show_market_overlay = True
            self.market_overlay_start_time = pygame.time.get_ticks()

    # ========== AI PLAYER METHODS ==========
    
    def _get_ai_action_mask(self):
        """Build action mask for AI player based on current game state."""
        if not self.using_decision_env:
            return np.ones(125, dtype=np.float32)
        
        mask = np.zeros(self.env.action_space.n, dtype=np.float32)
        current_player = self._get_current_player()
        
        # End turn is always valid
        mask[0] = 1.0  # END_TURN
        
        # Buy property - only if on purchasable, unowned tile with enough cash
        pos = self.player_positions[current_player]
        tile = self.tiles[pos]
        if tile["type"] in ["property", "railroad", "utility"]:
            owner = self._get_property_owner(pos)
            price = tile.get("price", 0)
            cash = self._get_player_cash(current_player)
            if owner == -1 and cash >= price:
                mask[1] = 1.0  # BUY_PROPERTY
        
        # Jail actions
        if self.using_decision_env and self.env.player_in_jail[current_player]:
            mask[2] = 1.0  # PAY_JAIL_FINE (if has $50)
            if self.env.player_jail_cards[current_player] > 0:
                mask[3] = 1.0  # USE_JAIL_CARD
            mask[4] = 1.0  # ACCEPT_JAIL (skip turn)
        
        # Build actions (indices 5-44 for tiles 0-39)
        build_offset = 5
        for tile_idx in range(40):
            if self._can_build_on_tile(current_player, tile_idx):
                mask[build_offset + tile_idx] = 1.0
        
        # Sell building actions (indices 45-84 for tiles 0-39)
        sell_building_offset = 45
        for tile_idx in range(40):
            if self._can_sell_building_on_tile(current_player, tile_idx):
                mask[sell_building_offset + tile_idx] = 1.0
        
        # Sell property actions (indices 85-124 for tiles 0-39)
        sell_property_offset = 85
        for tile_idx in range(40):
            if self._can_sell_property(current_player, tile_idx):
                mask[sell_property_offset + tile_idx] = 1.0
        
        return mask
    
    def _get_ai_observation(self):
        """Build flattened observation for AI model."""
        if not self.using_decision_env:
            return np.zeros(self.env.flat_observation_size, dtype=np.float32)
        
        obs = self.env._get_obs()
        return self.env.flatten_observation(obs)
    
    def _can_build_on_tile(self, player_id, tile_idx):
        """Check if player can build on a tile."""
        if not self.using_decision_env:
            return False
        
        tile = self.tiles[tile_idx]
        if tile.get("type") != "property":
            return False
        
        owner = self._get_property_owner(tile_idx)
        if owner != player_id:
            return False
        
        houses = self._get_property_houses(tile_idx)
        if houses >= 5:  # Max houses (hotel)
            return False
        
        # Check if player owns all properties in color group
        color = tile.get("color")
        if not self._owns_color_group(player_id, color):
            return False
        
        # Check if player has enough cash
        house_cost = self._get_house_cost(tile_idx)
        cash = self._get_player_cash(player_id)
        return cash >= house_cost
    
    def _can_sell_building_on_tile(self, player_id, tile_idx):
        """Check if player can sell a building on a tile."""
        tile = self.tiles[tile_idx]
        if tile.get("type") != "property":
            return False
        
        owner = self._get_property_owner(tile_idx)
        if owner != player_id:
            return False
        
        houses = self._get_property_houses(tile_idx)
        return houses > 0
    
    def _can_sell_property(self, player_id, tile_idx):
        """Check if player can sell a property."""
        tile = self.tiles[tile_idx]
        if tile.get("type") not in ["property", "railroad", "utility"]:
            return False
        
        owner = self._get_property_owner(tile_idx)
        if owner != player_id:
            return False
        
        # Can only sell if no houses
        houses = self._get_property_houses(tile_idx)
        return houses == 0
    
    def _select_ai_action(self):
        """Select action for AI using DQN policy or random fallback."""
        if self.ai_policy is None or not DQN_AVAILABLE:
            # Random fallback - pick random valid action
            mask = self._get_ai_action_mask()
            valid_actions = np.where(mask > 0)[0]
            if len(valid_actions) == 0:
                return 0  # End turn
            return int(random.choice(valid_actions))
        
        # Use trained DQN policy
        state = self._get_ai_observation()
        mask = self._get_ai_action_mask()
        
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.ai_device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.ai_policy(state_tensor).cpu().numpy()[0]
        
        # Mask invalid actions
        q_values[mask == 0] = -np.inf
        return int(np.argmax(q_values))
    
    def _execute_ai_action(self, action):
        """Execute an AI action and update game state."""
        current_player = self._get_current_player()
        
        # Action constants (matching rl.py)
        END_TURN = 0
        BUY_PROPERTY = 1
        PAY_JAIL_FINE = 2
        USE_JAIL_CARD = 3
        ACCEPT_JAIL = 4
        BUILD_OFFSET = 5
        SELL_BUILDING_OFFSET = 45
        SELL_PROPERTY_OFFSET = 85
        
        if action == END_TURN:
            self._push_message(f"AI ends turn")
            self.ai_actions_this_turn = 0
            self._end_turn()
            return
        
        elif action == BUY_PROPERTY:
            pos = self.player_positions[current_player]
            tile = self.tiles[pos]
            self._push_message(f"AI buys {tile['name']}")
            self._ai_execute_buy_property(current_player, pos)
            self.ai_actions_this_turn += 1
            # After buying, end turn to prevent excessive transactions
            self._push_message(f"AI ends turn")
            self.ai_actions_this_turn = 0
            self._end_turn()
            return
        
        elif action == PAY_JAIL_FINE:
            if self.using_decision_env:
                self.env.player_cash[current_player] -= 50
                self.env.player_in_jail[current_player] = False
                self.env.jail_turns_remaining[current_player] = 0
            self._push_message(f"AI pays $50 to leave jail")
            return
        
        elif action == USE_JAIL_CARD:
            if self.using_decision_env:
                self.env.player_jail_cards[current_player] -= 1
                self.env.player_in_jail[current_player] = False
                self.env.jail_turns_remaining[current_player] = 0
            self._push_message(f"AI uses Get Out of Jail Free card")
            return
        
        elif action == ACCEPT_JAIL:
            self._push_message(f"AI stays in jail")
            self.ai_actions_this_turn = 0
            self._end_turn()
            return
        
        elif BUILD_OFFSET <= action < SELL_BUILDING_OFFSET:
            tile_idx = action - BUILD_OFFSET
            tile = self.tiles[tile_idx]
            self._push_message(f"AI builds on {tile['name']}")
            self._ai_execute_build_house(current_player, tile_idx)
            self.ai_actions_this_turn += 1
            # End turn after building
            self._push_message(f"AI ends turn")
            self.ai_actions_this_turn = 0
            self._end_turn()
            return
        
        elif SELL_BUILDING_OFFSET <= action < SELL_PROPERTY_OFFSET:
            tile_idx = action - SELL_BUILDING_OFFSET
            tile = self.tiles[tile_idx]
            self._push_message(f"AI sells building on {tile['name']}")
            self._ai_execute_sell_building(current_player, tile_idx, 1)
            self.ai_actions_this_turn += 1
            # End turn after selling building
            self._push_message(f"AI ends turn")
            self.ai_actions_this_turn = 0
            self._end_turn()
            return
        
        elif action >= SELL_PROPERTY_OFFSET:
            tile_idx = action - SELL_PROPERTY_OFFSET
            tile = self.tiles[tile_idx]
            self._push_message(f"AI sells {tile['name']}")
            self._ai_execute_sell_property(current_player, tile_idx)
            self.ai_actions_this_turn += 1
            # End turn after selling property
            self._push_message(f"AI ends turn")
            self.ai_actions_this_turn = 0
            self._end_turn()
            return
        
        # Unknown action - end turn
        self._push_message(f"AI: unknown action {action}")
        self._end_turn()
    
    def _ai_execute_buy_property(self, player_id, tile_idx):
        """Execute buying a property (AI version)."""
        tile = self.tiles[tile_idx]
        price = tile.get("price", 0)
        
        if self.using_decision_env:
            self.env.player_cash[player_id] -= price
            self.env.property_owner[tile_idx] = player_id
        tile["owned_by"] = player_id
    
    def _ai_execute_build_house(self, player_id, tile_idx):
        """Execute building a house on a property (AI version)."""
        house_cost = self._get_house_cost(tile_idx)
        
        if self.using_decision_env:
            self.env.player_cash[player_id] -= house_cost
            self.env.property_houses[tile_idx] += 1
    
    def _ai_execute_sell_building(self, player_id, tile_idx, count=1):
        """Execute selling buildings on a property (AI version)."""
        house_cost = self._get_house_cost(tile_idx)
        refund = (house_cost // 2) * count
        
        if self.using_decision_env:
            self.env.player_cash[player_id] += refund
            self.env.property_houses[tile_idx] = max(0, self.env.property_houses[tile_idx] - count)
    
    def _ai_execute_sell_property(self, player_id, tile_idx):
        """Execute selling a property (AI version)."""
        tile = self.tiles[tile_idx]
        price = tile.get("price", 0)
        refund = price // 2
        
        if self.using_decision_env:
            self.env.player_cash[player_id] += refund
            self.env.property_owner[tile_idx] = -1
        tile["owned_by"] = -1
    
    def _process_ai_turn(self):
        """Process AI player's turn - called from game loop."""
        current_player = self._get_current_player()
        
        # Only process if it's AI's turn (player 1)
        if current_player != 1:
            return False
        
        current_time = pygame.time.get_ticks()
        
        # Check if we need to wait for action delay
        if current_time - self.ai_last_action_time < self.ai_action_delay:
            return True  # Still processing
        
        # Handle different game states
        if self.game_state == GameState.AWAITING_ROLL:
            self._push_message("AI is rolling dice...")
            self.ai_actions_this_turn = 0  # Reset action counter at start of turn
            self._handle_roll()
            self.ai_last_action_time = current_time
            return True
        
        elif self.game_state == GameState.AWAITING_DECISION:
            # Check if AI has exceeded max actions this turn
            if self.ai_actions_this_turn >= self.ai_max_actions_per_turn:
                self._push_message("AI ends turn")
                self._end_turn()
                self.ai_actions_this_turn = 0
                self.ai_last_action_time = current_time
                return True
            
            # Get AI's decision
            action = self._select_ai_action()
            self._execute_ai_action(action)
            self.ai_last_action_time = current_time
            return True
        
        return False

    def _build_tiles(self):
        return [
    {"name": "GO", "type": "corner", "corner": "go"},
    {"name": "Old Kent Road", "type": "property", "color": "brown", "price": 60, "owned_by": -1, "rent": {"base": 2, "1h": 10, "2h": 30, "3h": 90, "4h": 160, "hotel": 250}, "visiting_freq": 0, "trading_frequency": 0},
    {"name": "Community Chest", "type": "chest"},
    {"name": "Whitechapel Road", "type": "property", "color": "brown", "price": 60, "owned_by": -1, "rent": {"base": 4, "1h": 20, "2h": 60, "3h": 180, "4h": 320, "hotel": 450}, "visiting_freq": 0, "trading_frequency": 0},
    {"name": "Income Tax", "type": "tax", "price": 200},
    {"name": "Kings Cross Station", "type": "railroad", "price": 200, "owned_by": -1, "rent": {"1_rr": 25, "2_rr": 50, "3_rr": 100, "4_rr": 200}, "visiting_freq": 0, "trading_frequency": 0},
    {"name": "The Angel, Islington", "type": "property", "color": "light_blue", "price": 100, "owned_by": -1, "rent": {"base": 6, "1h": 30, "2h": 90, "3h": 270, "4h": 400, "hotel": 550}, "visiting_freq": 0, "trading_frequency": 0},
    {"name": "Chance", "type": "chance"},
    {"name": "Euston Road", "type": "property", "color": "light_blue", "price": 100, "owned_by": -1, "rent": {"base": 6, "1h": 30, "2h": 90, "3h": 270, "4h": 400, "hotel": 550}, "visiting_freq": 0, "trading_frequency": 0},
    {"name": "Pentonville Road", "type": "property", "color": "light_blue", "price": 120, "owned_by": -1, "rent": {"base": 8, "1h": 40, "2h": 100, "3h": 300, "4h": 450, "hotel": 600}, "visiting_freq": 0, "trading_frequency": 0},
    {"name": "In Jail / Just Visiting", "type": "corner", "corner": "jail"},
    {"name": "Pall Mall", "type": "property", "color": "pink", "price": 140, "owned_by": -1, "rent": {"base": 10, "1h": 50, "2h": 150, "3h": 450, "4h": 625, "hotel": 750}, "visiting_freq": 0, "trading_frequency": 0},
    {"name": "Electric Company", "type": "utility", "price": 150, "owned_by": -1, "rent": {"one_util": "4x dice", "both_utils": "10x dice"}, "visiting_freq": 0, "trading_frequency": 0},
    {"name": "Whitehall", "type": "property", "color": "pink", "price": 140, "owned_by": -1, "rent": {"base": 10, "1h": 50, "2h": 150, "3h": 450, "4h": 625, "hotel": 750}, "visiting_freq": 0, "trading_frequency": 0},
    {"name": "Northumberland Avenue", "type": "property", "color": "pink", "price": 160, "owned_by": -1, "rent": {"base": 12, "1h": 60, "2h": 180, "3h": 500, "4h": 700, "hotel": 900}, "visiting_freq": 0, "trading_frequency": 0},
    {"name": "Marylebone Station", "type": "railroad", "price": 200, "owned_by": -1, "rent": {"1_rr": 25, "2_rr": 50, "3_rr": 100, "4_rr": 200}, "visiting_freq": 0, "trading_frequency": 0},
    {"name": "Bow Street", "type": "property", "color": "orange", "price": 180, "owned_by": -1, "rent": {"base": 14, "1h": 70, "2h": 200, "3h": 550, "4h": 750, "hotel": 950}, "visiting_freq": 0, "trading_frequency": 0},
    {"name": "Community Chest", "type": "chest"},
    {"name": "Marlborough Street", "type": "property", "color": "orange", "price": 180, "owned_by": -1, "rent": {"base": 14, "1h": 70, "2h": 200, "3h": 550, "4h": 750, "hotel": 950}, "visiting_freq": 0, "trading_frequency": 0},
    {"name": "Vine Street", "type": "property", "color": "orange", "price": 200, "owned_by": -1, "rent": {"base": 16, "1h": 80, "2h": 220, "3h": 600, "4h": 800, "hotel": 1000}, "visiting_freq": 0, "trading_frequency": 0},
    {"name": "Free Parking", "type": "corner", "corner": "free"},
    {"name": "Strand", "type": "property", "color": "red", "price": 220, "owned_by": -1, "rent": {"base": 18, "1h": 90, "2h": 250, "3h": 700, "4h": 875, "hotel": 1050}, "visiting_freq": 0, "trading_frequency": 0},
    {"name": "Chance", "type": "chance"},
    {"name": "Fleet Street", "type": "property", "color": "red", "price": 220, "owned_by": -1, "rent": {"base": 18, "1h": 90, "2h": 250, "3h": 700, "4h": 875, "hotel": 1050}, "visiting_freq": 0, "trading_frequency": 0},
    {"name": "Trafalgar Square", "type": "property", "color": "red", "price": 240, "owned_by": -1, "rent": {"base": 20, "1h": 100, "2h": 300, "3h": 750, "4h": 925, "hotel": 1100}, "visiting_freq": 0, "trading_frequency": 0},
    {"name": "Fenchurch St. Station", "type": "railroad", "price": 200, "owned_by": -1, "rent": {"1_rr": 25, "2_rr": 50, "3_rr": 100, "4_rr": 200}, "visiting_freq": 0, "trading_frequency": 0},
    {"name": "Leicester Square", "type": "property", "color": "yellow", "price": 260, "owned_by": -1, "rent": {"base": 22, "1h": 110, "2h": 330, "3h": 800, "4h": 975, "hotel": 1150}, "visiting_freq": 0, "trading_frequency": 0},
    {"name": "Coventry Street", "type": "property", "color": "yellow", "price": 260, "owned_by": -1, "rent": {"base": 22, "1h": 110, "2h": 330, "3h": 800, "4h": 975, "hotel": 1150}, "visiting_freq": 0, "trading_frequency": 0},
    {"name": "Water Works", "type": "utility", "price": 150, "owned_by": -1, "rent": {"one_util": "4x dice", "both_utils": "10x dice"}, "visiting_freq": 0, "trading_frequency": 0},
    {"name": "Piccadilly", "type": "property", "color": "yellow", "price": 280, "owned_by": -1, "rent": {"base": 24, "1h": 120, "2h": 360, "3h": 850, "4h": 1025, "hotel": 1200}, "visiting_freq": 0, "trading_frequency": 0},
    {"name": "Go To Jail", "type": "corner", "corner": "goto"},
    {"name": "Regent Street", "type": "property", "color": "green", "price": 300, "owned_by": -1, "rent": {"base": 26, "1h": 130, "2h": 390, "3h": 900, "4h": 1100, "hotel": 1275}, "visiting_freq": 0, "trading_frequency": 0},
    {"name": "Oxford Street", "type": "property", "color": "green", "price": 300, "owned_by": -1, "rent": {"base": 26, "1h": 130, "2h": 390, "3h": 900, "4h": 1100, "hotel": 1275}, "visiting_freq": 0, "trading_frequency": 0},
    {"name": "Community Chest", "type": "chest"},
    {"name": "Bond Street", "type": "property", "color": "green", "price": 320, "owned_by": -1, "rent": {"base": 28, "1h": 150, "2h": 450, "3h": 1000, "4h": 1200, "hotel": 1400}, "visiting_freq": 0, "trading_frequency": 0},
    {"name": "Liverpool St. Station", "type": "railroad", "price": 200, "owned_by": -1, "rent": {"1_rr": 25, "2_rr": 50, "3_rr": 100, "4_rr": 200}, "visiting_freq": 0, "trading_frequency": 0},
    {"name": "Chance", "type": "chance"},
    {"name": "Park Lane", "type": "property", "color": "dark_blue", "price": 350, "owned_by": -1, "rent": {"base": 35, "1h": 175, "2h": 500, "3h": 1100, "4h": 1300, "hotel": 1500}, "visiting_freq": 0, "trading_frequency": 0},
    {"name": "Super Tax", "type": "tax", "price": 100},
    {"name": "Mayfair", "type": "property", "color": "dark_blue", "price": 400, "owned_by": -1, "rent": {"base": 50, "1h": 200, "2h": 600, "3h": 1400, "4h": 1700, "hotel": 2000}, "visiting_freq": 0, "trading_frequency": 0}
]
        
    def _calculate_layout(self):
        layout = [None] * 40

        left = self.board_rect.left
        top = self.board_rect.top
        right = self.board_rect.right
        bottom = self.board_rect.bottom

        layout[0] = {
            "rect": pygame.Rect(right - self.corner_size, bottom - self.corner_size, self.corner_size, self.corner_size),
            "orientation": "corner-br",
        }
        layout[10] = {
            "rect": pygame.Rect(left, bottom - self.corner_size, self.corner_size, self.corner_size),
            "orientation": "corner-bl",
        }
        layout[20] = {
            "rect": pygame.Rect(left, top, self.corner_size, self.corner_size),
            "orientation": "corner-tl",
        }
        layout[30] = {
            "rect": pygame.Rect(right - self.corner_size, top, self.corner_size, self.corner_size),
            "orientation": "corner-tr",
        }

        for offset in range(1, 10):
            idx = offset
            layout[idx] = {
                "rect": pygame.Rect(
                    right - self.corner_size - offset * self.edge_size,
                    bottom - self.corner_size,
                    self.edge_size,
                    self.corner_size,
                ),
                "orientation": "bottom",
            }

        for offset in range(1, 10):
            idx = 10 + offset
            layout[idx] = {
                "rect": pygame.Rect(
                    left,
                    bottom - self.corner_size - offset * self.edge_size,
                    self.corner_size,
                    self.edge_size,
                ),
                "orientation": "left",
            }

        for offset in range(1, 10):
            idx = 20 + offset
            layout[idx] = {
                "rect": pygame.Rect(
                    left + self.corner_size + (offset - 1) * self.edge_size,
                    top,
                    self.edge_size,
                    self.corner_size,
                ),
                "orientation": "top",
            }

        for offset in range(1, 10):
            idx = 30 + offset
            layout[idx] = {
                "rect": pygame.Rect(
                    right - self.corner_size,
                    top + self.corner_size + (offset - 1) * self.edge_size,
                    self.corner_size,
                    self.edge_size,
                ),
                "orientation": "right",
            }

        return layout

    def _load_board_surface(self):
        board_path = os.path.join(self.images_dir, "dymonopoly_board.png")
        if os.path.exists(board_path):
            board_img = pygame.image.load(board_path).convert()
            return pygame.transform.smoothscale(board_img, (self.board_size, self.board_size))

        fallback = pygame.Surface((self.board_size, self.board_size))
        fallback.fill((240, 240, 240))
        pygame.draw.rect(fallback, (50, 50, 50), fallback.get_rect(), 6)
        return fallback

    def _load_dice_images(self):
        value_map = {
            1: "one",
            2: "two",
            3: "three",
            4: "four",
            5: "five",
            6: "six",
        }
        images = {}
        for value, name in value_map.items():
            file_path = os.path.join(self.images_dir, f"dice-six-faces-{name}.png")
            if os.path.exists(file_path):
                img = pygame.image.load(file_path).convert_alpha()
                images[value] = pygame.transform.smoothscale(img, (44, 44))
            else:
                surface = pygame.Surface((44, 44))
                surface.fill((255, 255, 255))
                pygame.draw.rect(surface, (0, 0, 0), surface.get_rect(), 2)
                label = self.font_medium.render(str(value), True, (0, 0, 0))
                surface.blit(label, label.get_rect(center=surface.get_rect().center))
                images[value] = surface
        return images

    def _create_action_buttons(self):
        buttons = []
        button_width = self.info_rect.width - 32
        button_height = 36
        start_x = self.info_rect.left + 16
        # Position buttons after Dice Rolls and Current Tile sections
        start_y = self.info_rect.top + 420
        
        # Narrower width for Sell Building to leave room for +/- counter
        sell_building_width = button_width - 90  # Leave 90px for counter controls

        # Button enabled conditions
        def can_roll():
            return self.game_state == GameState.AWAITING_ROLL
        
        def can_buy():
            return self.game_state == GameState.AWAITING_DECISION and self._can_buy_current_tile()
        
        def can_skip():
            return self.game_state == GameState.AWAITING_DECISION
        
        def can_build():
            return self.game_state == GameState.AWAITING_DECISION and len(self._get_buildable_tiles()) > 0
        
        def can_sell_building():
            return self.game_state == GameState.AWAITING_DECISION and len(self._get_sellable_building_tiles()) > 0
        
        def can_sell_property():
            return self.game_state == GameState.AWAITING_DECISION and len(self._get_sellable_property_tiles()) > 0

        specs = [
            ("Roll Dice", self._handle_roll, can_roll, (76, 175, 80), button_width),  # Green
            ("Buy", self._handle_buy, can_buy, (100, 149, 237), button_width),  # Blue
            ("End Turn", self._handle_skip, can_skip, (220, 60, 60), button_width),  # Red when enabled
            ("Build House", self._handle_build, can_build, (76, 175, 80), button_width),  # Green
            ("Sell Building", self._handle_sell_building, can_sell_building, (210, 180, 140), sell_building_width),  # Tan - narrower
            ("Sell Property", self._handle_sell_property_btn, can_sell_property, (255, 165, 0), button_width),  # Orange
        ]

        for idx, (label, callback, enabled_fn, color, width) in enumerate(specs):
            rect = pygame.Rect(
                start_x,
                start_y + idx * (button_height + 5),
                width,
                button_height,
            )
            buttons.append(ActionButton(rect, label, callback, color, enabled_fn))
        
        return buttons
    
    # Helper methods for game logic
    def _get_current_player(self):
        """Get current player index."""
        if self.using_decision_env:
            return self.env.current_player
        else:
            return self.env.current_player % self.max_players
    
    def _get_player_cash(self, player_id):
        """Get player's cash."""
        if self.using_decision_env:
            return self.env.player_cash[player_id]
        else:
            return self.env.player_cash[player_id] if hasattr(self.env, 'player_cash') else 1500
    
    def _get_property_owner(self, tile_idx):
        """Get property owner."""
        if self.using_decision_env:
            return self.env.property_owner[tile_idx]
        else:
            return self.env.property_owners[tile_idx] if hasattr(self.env, 'property_owners') else -1
    
    def _get_property_houses(self, tile_idx):
        """Get number of houses on property."""
        if self.using_decision_env:
            return int(self.env.property_houses[tile_idx])
        else:
            return int(self.env.property_houses[tile_idx]) if hasattr(self.env, 'property_houses') else 0
    
    def _get_house_cost(self, tile_idx):
        """Get house cost for a property."""
        tile = self.tiles[tile_idx]
        # Default house costs by color group
        color_costs = {
            "brown": 50, "light_blue": 50,
            "pink": 100, "orange": 100,
            "red": 150, "yellow": 150,
            "green": 200, "dark_blue": 200
        }
        color = tile.get("color", "")
        return tile.get("house_cost", color_costs.get(color, 100))
    
    def _owns_color_group(self, player_id, color):
        """Check if player owns all properties in a color group."""
        if not color:
            return False
        
        color_tiles = [i for i, t in enumerate(self.tiles) if t.get("color") == color]
        for tile_idx in color_tiles:
            owner = self._get_property_owner(tile_idx)
            if owner != player_id:
                return False
        return True
    
    def _can_buy_current_tile(self):
        """Check if current tile (where player is standing) can be bought."""
        current_player = self._get_current_player()
        current_position = self.player_positions[current_player]
        tile = self.tiles[current_position]
        if tile["type"] not in {"property", "railroad", "utility"}:
            return False
        owner = self._get_property_owner(current_position)
        if owner >= 0:
            return False
        price = tile.get("price", 0)
        return self._get_player_cash(current_player) >= price
    
    def _get_buildable_tiles(self):
        """Get list of tiles where current player can build."""
        if self.using_decision_env:
            current_player = self._get_current_player()
            return self.env._list_buildable_properties(current_player)
        return []
    
    def _get_sellable_building_tiles(self):
        """Get list of tiles where current player can sell buildings."""
        if self.using_decision_env:
            current_player = self._get_current_player()
            return self.env._list_sell_building_candidates(current_player)
        return []
    
    def _get_sellable_property_tiles(self):
        """Get list of tiles where current player can sell property."""
        if self.using_decision_env:
            current_player = self._get_current_player()
            return self.env._list_sell_property_candidates(current_player)
        return []
    
    def _wrap_text(self, text, font, max_width):
        words = text.split()
        lines = []
        current = []
        for word in words:
            candidate = " ".join(current + [word])
            if font.size(candidate)[0] <= max_width:
                current.append(word)
            else:
                if current:
                    lines.append(" ".join(current))
                current = [word]
        if current:
            lines.append(" ".join(current))
        return lines if lines else [""]

    def _push_message(self, text):
        self.message_log.append(text)
        self.message_log = self.message_log[-5:]

    def _price_tint(self, index):
        base = self.env.base_prices[index]
        current = self.env.current_prices[index]
        ratio = current / base if base else 1.0
        delta = ratio - 1.0
        if delta > 0.15:
            return (0, 120, 0, 70)
        if delta > 0.05:
            return (0, 200, 0, 55)
        if delta < -0.15:
            return (200, 0, 0, 70)
        if delta < -0.05:
            return (255, 60, 60, 55)
        return (255, 255, 255, 0)

    def _angle_for_orientation(self, orientation):
        mapping = {
            "bottom": 0,
            "left": 90,
            "top": 180,
            "right": 270,
            "corner-br": 0,
            "corner-bl": 90,
            "corner-tl": 180,
            "corner-tr": 270,
        }
        return mapping[orientation]

    def _draw_property_card(self, tile_index):
        """
        Draws a Title Deed card for the property at the given tile index.
        Shows the card as an overlay on the board.
        """
        tile = self.tiles[tile_index]
        
        # Only draw for properties, railroads, and utilities
        if tile["type"] not in ["property", "railroad", "utility"]:
            return
        
        # --- Configuration & Setup ---
        CARD_WIDTH = 220
        CARD_HEIGHT = 320
        
        # Position card in center of the board
        x = self.board_rect.centerx - CARD_WIDTH // 2
        y = self.board_rect.centery - CARD_HEIGHT // 2
        
        # RGB Colors for the Property Groups
        COLORS = {
            "brown": (139, 69, 19),
            "light_blue": (173, 216, 230),
            "pink": (255, 105, 180),
            "orange": (255, 165, 0),
            "red": (255, 0, 0),
            "yellow": (255, 255, 0),
            "green": (0, 128, 0),
            "dark_blue": (0, 0, 139),
            "white": (255, 255, 255),
            "black": (0, 0, 0),
            "mortgage_red": (200, 0, 0),
            "railroad_gray": (180, 180, 180),
            "utility_cream": (245, 245, 220),
        }
        
        # Helper to get house cost based on color
        def get_house_cost(color):
            if color in ["brown", "light_blue"]: return 50
            if color in ["pink", "orange"]: return 100
            if color in ["red", "yellow"]: return 150
            if color in ["green", "dark_blue"]: return 200
            return 0
        
        # Fonts
        font_header = pygame.font.SysFont("Arial", 18, bold=True)
        font_sub = pygame.font.SysFont("Arial", 14)
        font_bold = pygame.font.SysFont("Arial", 14, bold=True)
        font_small = pygame.font.SysFont("Arial", 12)
        
        # --- Draw Card Background ---
        card_rect = pygame.Rect(x, y, CARD_WIDTH, CARD_HEIGHT)
        pygame.draw.rect(self.screen, COLORS["white"], card_rect)
        pygame.draw.rect(self.screen, COLORS["black"], card_rect, 3)
        
        property_name = tile["name"]
        p_price = tile.get("price", 0)
        
        if tile["type"] == "property":
            p_color = tile["color"]
            p_rent = tile["rent"]
            
            # --- Draw Header (Color Box) ---
            header_rect = pygame.Rect(x + 5, y + 5, CARD_WIDTH - 10, 50)
            pygame.draw.rect(self.screen, COLORS.get(p_color, (100, 100, 100)), header_rect)
            pygame.draw.rect(self.screen, COLORS["black"], header_rect, 1)
            
            # Draw Name inside Header
            text_col = COLORS["white"] if p_color in ["brown", "red", "green", "dark_blue"] else COLORS["black"]
            name_surf = font_header.render(property_name.upper(), True, text_col)
            name_rect = name_surf.get_rect(center=header_rect.center)
            self.screen.blit(name_surf, name_rect)
            
            # --- Draw Rent Details ---
            curr_y = y + 65
            margin_l = x + 15
            margin_r = x + CARD_WIDTH - 15
            
            # Title Deed Text
            title_surf = font_bold.render("TITLE DEED", True, COLORS["black"])
            self.screen.blit(title_surf, title_surf.get_rect(center=(x + CARD_WIDTH // 2, curr_y)))
            curr_y += 22
            
            # Helper function to draw a row of text
            def draw_row(label, value, is_bold=False):
                nonlocal curr_y
                f = font_bold if is_bold else font_sub
                lbl_s = f.render(label, True, COLORS["black"])
                val_s = f.render(str(value), True, COLORS["black"])
                self.screen.blit(lbl_s, (margin_l, curr_y))
                val_rect = val_s.get_rect(topright=(margin_r, curr_y))
                self.screen.blit(val_s, val_rect)
                curr_y += 18
            
            # Rent Rows
            draw_row("RENT", f"£{p_rent['base']}")
            draw_row("With 1 house", f"£{p_rent['1h']}")
            draw_row("With 2 houses", f"£{p_rent['2h']}")
            draw_row("With 3 houses", f"£{p_rent['3h']}")
            draw_row("With 4 houses", f"£{p_rent['4h']}")
            curr_y += 2
            draw_row("WITH HOTEL", f"£{p_rent['hotel']}", is_bold=True)
            curr_y += 8
            
            # Separator Line
            pygame.draw.line(self.screen, COLORS["black"], (margin_l, curr_y), (margin_r, curr_y), 1)
            curr_y += 8
            
            # House/Hotel Costs
            house_cost = get_house_cost(p_color)
            cost_text = font_small.render(f"Houses cost £{house_cost} each", True, COLORS["black"])
            self.screen.blit(cost_text, (margin_l, curr_y))
            curr_y += 16
            
            hotel_text = font_small.render(f"Hotel £{house_cost} + 4 houses", True, COLORS["black"])
            self.screen.blit(hotel_text, (margin_l, curr_y))
            curr_y += 20
            
            # Mortgage Value
            mortgage_val = p_price // 2
            mort_text = font_bold.render(f"Mortgage: £{mortgage_val}", True, COLORS["mortgage_red"])
            mort_rect = mort_text.get_rect(center=(x + CARD_WIDTH // 2, curr_y))
            self.screen.blit(mort_text, mort_rect)
            
        elif tile["type"] == "railroad":
            # Railroad card
            header_rect = pygame.Rect(x + 5, y + 5, CARD_WIDTH - 10, 50)
            pygame.draw.rect(self.screen, COLORS["railroad_gray"], header_rect)
            pygame.draw.rect(self.screen, COLORS["black"], header_rect, 1)
            
            name_surf = font_header.render(property_name.upper(), True, COLORS["black"])
            name_rect = name_surf.get_rect(center=header_rect.center)
            self.screen.blit(name_surf, name_rect)
            
            curr_y = y + 70
            margin_l = x + 15
            
            rent_info = tile.get("rent", {})
            rents = [
                ("1 Railroad owned", f"£{rent_info.get('1_rr', 25)}"),
                ("2 Railroads owned", f"£{rent_info.get('2_rr', 50)}"),
                ("3 Railroads owned", f"£{rent_info.get('3_rr', 100)}"),
                ("4 Railroads owned", f"£{rent_info.get('4_rr', 200)}"),
            ]
            for label, val in rents:
                lbl = font_sub.render(label, True, COLORS["black"])
                self.screen.blit(lbl, (margin_l, curr_y))
                curr_y += 16
                v = font_bold.render(val, True, COLORS["black"])
                self.screen.blit(v, (margin_l + 20, curr_y))
                curr_y += 22
            
            # Price
            price_text = font_bold.render(f"Price: £{p_price}", True, COLORS["black"])
            self.screen.blit(price_text, price_text.get_rect(center=(x + CARD_WIDTH // 2, curr_y + 20)))
            
        elif tile["type"] == "utility":
            # Utility card
            header_rect = pygame.Rect(x + 5, y + 5, CARD_WIDTH - 10, 50)
            pygame.draw.rect(self.screen, COLORS["utility_cream"], header_rect)
            pygame.draw.rect(self.screen, COLORS["black"], header_rect, 1)
            
            name_surf = font_header.render(property_name.upper(), True, COLORS["black"])
            name_rect = name_surf.get_rect(center=header_rect.center)
            self.screen.blit(name_surf, name_rect)
            
            curr_y = y + 70
            margin_l = x + 15
            
            info_lines = [
                "If one Utility is owned,",
                "rent is 4x dice roll.",
                "",
                "If both Utilities are owned,",
                "rent is 10x dice roll.",
            ]
            for line in info_lines:
                text = font_sub.render(line, True, COLORS["black"])
                self.screen.blit(text, (margin_l, curr_y))
                curr_y += 18
            
            # Price
            price_text = font_bold.render(f"Price: £{p_price}", True, COLORS["black"])
            self.screen.blit(price_text, price_text.get_rect(center=(x + CARD_WIDTH // 2, curr_y + 30)))
        
        # Draw "Click to close" text at bottom
        close_text = font_small.render("Click anywhere to close", True, (100, 100, 100))
        self.screen.blit(close_text, close_text.get_rect(center=(x + CARD_WIDTH // 2, y + CARD_HEIGHT - 15)))

    def _draw_event_card(self):
        """
        Draws a Chance or Community Chest card overlay on the board.
        Uses self.current_event_card and self.current_event_type.
        """
        if not self.current_event_card or not self.current_event_type:
            return
        
        card_data = self.current_event_card
        card_type = self.current_event_type
        
        # Card Styling
        CARD_WIDTH = 400
        CARD_HEIGHT = 250
        
        # Center the card on the board
        x = self.board_rect.centerx - CARD_WIDTH // 2
        y = self.board_rect.centery - CARD_HEIGHT // 2
        
        # Colors
        CARD_BG_COLOR = (255, 255, 255)
        CARD_BORDER_COLOR = (0, 0, 0)
        TEXT_COLOR = (0, 0, 0)
        
        # Fonts
        font_title = pygame.font.SysFont("Arial", 28, bold=True)
        font_body = pygame.font.SysFont("Arial", 20)
        font_small = pygame.font.SysFont("Arial", 12)
        
        # 1. Draw Card Background & Border
        card_rect = pygame.Rect(x, y, CARD_WIDTH, CARD_HEIGHT)
        pygame.draw.rect(self.screen, CARD_BG_COLOR, card_rect)
        pygame.draw.rect(self.screen, CARD_BORDER_COLOR, card_rect, 3)
        
        # 2. Draw colored header bar
        is_chance = "CHANCE" in card_type.upper()
        header_color = self.CHANCE_ORANGE if is_chance else self.CHEST_BLUE
        header_rect = pygame.Rect(x + 5, y + 5, CARD_WIDTH - 10, 50)
        pygame.draw.rect(self.screen, header_color, header_rect)
        pygame.draw.rect(self.screen, CARD_BORDER_COLOR, header_rect, 1)
        
        # 3. Draw Title
        title_surf = font_title.render(card_type.upper(), True, CARD_BORDER_COLOR)
        title_rect = title_surf.get_rect(center=header_rect.center)
        self.screen.blit(title_surf, title_rect)
        
        # 4. Draw Body Text (Wrapped)
        margin = 30
        max_text_width = CARD_WIDTH - (margin * 2)
        wrapped_lines = self._wrap_text(card_data.get("text", ""), font_body, max_text_width)
        
        # Calculate starting Y to vertically center the text block below header
        line_height = font_body.get_linesize()
        total_text_height = len(wrapped_lines) * line_height
        text_area_top = y + 70  # Below header
        text_area_height = CARD_HEIGHT - 100  # Leave room for header and close text
        start_text_y = text_area_top + (text_area_height - total_text_height) // 2
        
        for i, line in enumerate(wrapped_lines):
            line_surf = font_body.render(line, True, TEXT_COLOR)
            line_rect = line_surf.get_rect(center=(x + CARD_WIDTH // 2, start_text_y + i * line_height))
            self.screen.blit(line_surf, line_rect)
        
        # 5. Draw icon (? for Chance, treasure chest symbol for Community Chest)
        icon_text = "?" if is_chance else "💰"
        icon_font = pygame.font.SysFont("Arial", 48, bold=True)
        icon_color = (230, 230, 230)  # Faint background
        icon_surf = icon_font.render(icon_text, True, icon_color)
        icon_rect = icon_surf.get_rect(center=(x + CARD_WIDTH // 2, y + CARD_HEIGHT // 2 + 20))
        # Draw icon behind text (faint watermark effect)
        self.screen.blit(icon_surf, icon_rect)
        
        # Re-draw text on top of icon
        for i, line in enumerate(wrapped_lines):
            line_surf = font_body.render(line, True, TEXT_COLOR)
            line_rect = line_surf.get_rect(center=(x + CARD_WIDTH // 2, start_text_y + i * line_height))
            self.screen.blit(line_surf, line_rect)
        
        # 6. Draw "Click to close" text at bottom
        close_text = font_small.render("Click anywhere to close", True, (100, 100, 100))
        self.screen.blit(close_text, close_text.get_rect(center=(x + CARD_WIDTH // 2, y + CARD_HEIGHT - 15)))

    def _draw_market_overlay(self):
        """
        Draws a temporary overlay showing market price updates.
        Auto-dismisses after market_overlay_duration milliseconds.
        """
        # Check if overlay should auto-hide
        elapsed = pygame.time.get_ticks() - self.market_overlay_start_time
        if elapsed > self.market_overlay_duration:
            self.show_market_overlay = False
            return
        
        # Calculate fade effect (fade out in last 500ms)
        alpha = 255
        fade_start = self.market_overlay_duration - 500
        if elapsed > fade_start:
            alpha = int(255 * (1 - (elapsed - fade_start) / 500))
        
        # Card dimensions
        CARD_WIDTH = 450
        CARD_HEIGHT = min(80 + len(self.market_price_changes) * 32, 400)
        
        # Center on screen
        x = self.width // 2 - CARD_WIDTH // 2
        y = self.height // 2 - CARD_HEIGHT // 2
        
        # Create semi-transparent surface
        overlay_surface = pygame.Surface((CARD_WIDTH, CARD_HEIGHT), pygame.SRCALPHA)
        
        # Background with gradient effect
        bg_color = (30, 35, 45, min(alpha, 240))
        pygame.draw.rect(overlay_surface, bg_color, (0, 0, CARD_WIDTH, CARD_HEIGHT), border_radius=12)
        
        # Border
        border_color = (100, 180, 255, alpha)
        pygame.draw.rect(overlay_surface, border_color, (0, 0, CARD_WIDTH, CARD_HEIGHT), 3, border_radius=12)
        
        # Header bar
        header_color = (50, 100, 180, min(alpha, 220))
        pygame.draw.rect(overlay_surface, header_color, (4, 4, CARD_WIDTH - 8, 45), border_radius=8)
        
        # Fonts
        font_title = pygame.font.SysFont("Arial", 22, bold=True)
        font_item = pygame.font.SysFont("Arial", 16)
        font_small = pygame.font.SysFont("Arial", 12)
        
        # Title with icons
        title_surf = font_title.render("📈 MARKET UPDATE 📉", True, (255, 255, 255))
        overlay_surface.blit(title_surf, (CARD_WIDTH // 2 - title_surf.get_width() // 2, 14))
        
        # Turn number
        turn_num = self.env.turn_count if hasattr(self.env, 'turn_count') else 0
        turn_text = font_small.render(f"Turn {turn_num}", True, (180, 180, 180))
        overlay_surface.blit(turn_text, (CARD_WIDTH - turn_text.get_width() - 15, 18))
        
        # Price changes list
        y_offset = 58
        for name, old_price, new_price, pct in self.market_price_changes:
            # Determine color based on change
            if pct > 0:
                arrow = "▲"
                change_color = (100, 255, 100)  # Green for increase
            else:
                arrow = "▼"
                change_color = (255, 100, 100)  # Red for decrease
            
            # Property name
            name_text = font_item.render(name[:20], True, (220, 220, 220))
            overlay_surface.blit(name_text, (15, y_offset))
            
            # Price change
            price_text = f"£{old_price} → £{new_price}"
            price_surf = font_item.render(price_text, True, (200, 200, 200))
            overlay_surface.blit(price_surf, (200, y_offset))
            
            # Percentage with arrow
            pct_text = f"{arrow} {abs(pct):.0f}%"
            pct_surf = font_item.render(pct_text, True, change_color)
            overlay_surface.blit(pct_surf, (CARD_WIDTH - pct_surf.get_width() - 15, y_offset))
            
            y_offset += 32
        
        # Progress bar at bottom (time remaining)
        progress = 1 - (elapsed / self.market_overlay_duration)
        bar_width = int((CARD_WIDTH - 30) * progress)
        bar_rect = pygame.Rect(15, CARD_HEIGHT - 12, bar_width, 4)
        pygame.draw.rect(overlay_surface, (100, 180, 255, min(alpha, 180)), bar_rect, border_radius=2)
        
        # Blit overlay to screen
        self.screen.blit(overlay_surface, (x, y))

    def _draw_house(self, surface, x, y, color, size=12):
        """
        Draws a single small house icon.
        """
        width = size
        total_height = size * 1.2
        roof_h = total_height * 0.4
        body_h = total_height - roof_h

        top_peak = (x, y - total_height / 2)
        roof_left = (x - width / 2, y - total_height / 2 + roof_h)
        roof_right = (x + width / 2, y - total_height / 2 + roof_h)
        
        # Green house
        house_color = (0, 128, 0)
        pygame.draw.polygon(surface, house_color, [top_peak, roof_left, roof_right])
        body_rect = pygame.Rect(roof_left[0], roof_left[1], width, body_h)
        pygame.draw.rect(surface, house_color, body_rect)
        pygame.draw.polygon(surface, self.BLACK, [top_peak, roof_left, roof_right], 1)
        pygame.draw.rect(surface, self.BLACK, body_rect, 1)
    
    def _draw_hotel(self, surface, x, y, size=18):
        """
        Draws a hotel icon (red, larger building).
        """
        width = size
        height = size * 1.3
        
        # Red hotel - taller rectangular building
        hotel_color = (200, 0, 0)
        hotel_rect = pygame.Rect(x - width/2, y - height/2, width, height)
        pygame.draw.rect(surface, hotel_color, hotel_rect)
        pygame.draw.rect(surface, self.BLACK, hotel_rect, 1)
        
        # Draw "H" on hotel
        h_surf = self.font_house_num.render("H", True, self.WHITE)
        h_rect = h_surf.get_rect(center=hotel_rect.center)
        surface.blit(h_surf, h_rect)
    
    def _draw_houses_on_property(self, surface, x, y, houses_built, orientation):
        """
        Draw 1-4 houses or 1 hotel on a property based on development level.
        """
        if houses_built == 0:
            return
        
        if houses_built == 5:
            # Hotel
            self._draw_hotel(surface, x, y, size=16)
        else:
            # Draw 1-4 houses in a row
            house_size = 10
            spacing = 12
            total_width = houses_built * spacing
            
            # Adjust positioning based on orientation
            if orientation in ["bottom", "top"]:
                start_x = x - total_width / 2 + spacing / 2
                for i in range(houses_built):
                    self._draw_house(surface, start_x + i * spacing, y, None, size=house_size)
            else:  # left, right
                start_y = y - total_width / 2 + spacing / 2
                for i in range(houses_built):
                    self._draw_house(surface, x, start_y + i * spacing, None, size=house_size)
    
    def _draw_owner_flag(self, surface, x, y, player_color, player_num):
        """
        Draw a flag/banner to indicate property ownership (distinct from player tokens).
        """
        # Flag pole
        pole_height = 20
        pygame.draw.line(surface, self.BLACK, (x, y), (x, y - pole_height), 2)
        
        # Flag (triangular pennant)
        flag_width = 14
        flag_height = 10
        flag_points = [
            (x, y - pole_height),  # Top of pole
            (x + flag_width, y - pole_height + flag_height / 2),  # Point
            (x, y - pole_height + flag_height),  # Bottom of flag
        ]
        pygame.draw.polygon(surface, player_color, flag_points)
        pygame.draw.polygon(surface, self.BLACK, flag_points, 1)
        
        # Player number on flag
        num_surf = self.font_house_num.render(str(player_num), True, self.WHITE)
        num_rect = num_surf.get_rect(center=(x + flag_width/2 - 1, y - pole_height + flag_height/2))
        surface.blit(num_surf, num_rect)

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------
    def _draw_tile(self, index):
        tile = self.tiles[index]
        info = self.layout[index]
        rect = info["rect"]
        orientation = info["orientation"]

        base_size = (rect.width, rect.height)
        if orientation in ("left", "right"):
            base_size = (rect.height, rect.width)
        surface = pygame.Surface(base_size, pygame.SRCALPHA)

        if tile["type"] == "corner":
            self._draw_corner_surface(surface, tile)
        else:
            self._draw_standard_surface(surface, tile, index)

        angle = self._angle_for_orientation(orientation)
        rotated = pygame.transform.rotate(surface, angle)
        rotated_rect = rotated.get_rect(center=rect.center)
        self.screen.blit(rotated, rotated_rect.topleft)

        if index == self.selected_tile:
            pygame.draw.rect(self.screen, (255, 215, 0), rotated_rect, 4)

    def _draw_highlight(self):
        layout = self.layout[self.selected_tile]
        highlight_rect = layout["rect"]
        overlay = pygame.Surface(highlight_rect.size, pygame.SRCALPHA)
        overlay.fill((255, 215, 0, 60))
        self.screen.blit(overlay, highlight_rect.topleft)
        pygame.draw.rect(self.screen, (255, 215, 0), highlight_rect, 3)

    def _draw_standard_surface(self, surface, tile, index):
        rect = surface.get_rect()
        pygame.draw.rect(surface, self.WHITE, rect)
        pygame.draw.rect(surface, self.BLACK, rect, 2)

        tint = self._price_tint(index)
        if tint[3] > 0:
            overlay = pygame.Surface(rect.size, pygame.SRCALPHA)
            overlay.fill(tint)
            surface.blit(overlay, (0, 0))

        tile_type = tile["type"]
        if tile_type == "property":
            color = self.PROPERTY_COLORS[tile["color"]]
            pygame.draw.rect(surface, color, (6, 6, rect.width - 12, 24))
        elif tile_type == "railroad":
            pygame.draw.rect(surface, (230, 230, 230), (6, 6, rect.width - 12, rect.height - 12))
            pygame.draw.rect(surface, self.BLACK, (14, rect.height // 2, rect.width - 28, 4))
            spacing = (rect.width - 36) // 3
            for offset in range(4):
                x = 18 + offset * spacing
                pygame.draw.circle(surface, self.BLACK, (x, rect.height // 2 + 8), 6, 2)
        elif tile_type == "utility":
            pygame.draw.rect(surface, (245, 245, 245), (6, 6, rect.width - 12, rect.height - 12))
        elif tile_type == "chance":
            pygame.draw.rect(surface, self.CHANCE_ORANGE, (6, 6, rect.width - 12, rect.height - 12))
        elif tile_type == "chest":
            pygame.draw.rect(surface, self.CHEST_BLUE, (6, 6, rect.width - 12, rect.height - 12))
        elif tile_type == "tax":
            pygame.draw.rect(surface, (250, 250, 210), (6, 6, rect.width - 12, rect.height - 12))

        name_lines = self._wrap_text(tile["name"], self.font_small, rect.width - 16)
        text_y = 38 if tile_type == "property" else 22
        for line in name_lines:
            text_surf = self.font_small.render(line, True, self.BLACK)
            text_rect = text_surf.get_rect(center=(rect.width // 2, text_y))
            surface.blit(text_surf, text_rect)
            text_y += 18

        if tile_type == "property":
            price = self.font_price.render(f"${tile['price']}", True, self.BLACK)
            price_rect = price.get_rect(center=(rect.width // 2, rect.height - 20))
            surface.blit(price, price_rect)
        elif tile_type == "railroad":
            price = self.font_price.render("Rent $25", True, self.BLACK)
            surface.blit(price, price.get_rect(center=(rect.width // 2, rect.height - 22)))
        elif tile_type == "utility":
            price = self.font_price.render("Rent 4x Dice", True, self.BLACK)
            surface.blit(price, price.get_rect(center=(rect.width // 2, rect.height - 22)))
        elif tile_type == "tax":
            price = self.font_price.render(f"Pay ${tile['price']}", True, self.BLACK)
            surface.blit(price, price.get_rect(center=(rect.width // 2, rect.height - 22)))
        elif tile_type == "chance":
            question = self.font_large.render("?", True, self.BLACK)
            surface.blit(question, question.get_rect(center=(rect.width // 2, rect.height // 2 + 6)))
        elif tile_type == "chest":
            chest = pygame.Rect(rect.width // 2 - 20, rect.height // 2 - 12, 40, 24)
            pygame.draw.rect(surface, (210, 140, 60), chest)
            pygame.draw.rect(surface, self.BLACK, chest, 2)
            latch = pygame.Rect(chest.centerx - 5, chest.centery - 4, 10, 12)
            pygame.draw.rect(surface, (240, 200, 120), latch)
            pygame.draw.rect(surface, self.BLACK, latch, 1)

        base = self.env.base_prices[index]
        current = self.env.current_prices[index]
        if base > 0:
            diff_pct = (current / base - 1.0) * 100.0
            change = self.font_price.render(f"{diff_pct:+.0f}%", True, self.BLACK)
            surface.blit(change, (rect.width - change.get_width() - 6, 6))

        owner = self.env.property_owners[index]
        if owner >= 0:
            player_color = self._get_player_color(owner)
            
            # Draw house icon for owned properties (below the color bar, centered)
            if tile_type == "property":
                # Get number of houses built (0-5, where 5 = hotel)
                houses_built = 0
                if hasattr(self.env, 'property_houses'):
                    houses_built = int(self.env.property_houses[index])
                
                # Position house below the property name, above the price
                house_x = rect.width // 2
                house_y = rect.height // 2 + 5
                
                # Draw house with number if houses are built, otherwise just empty house
                if houses_built > 0:
                    self._draw_house(surface, house_x, house_y, player_color, number=houses_built, size=18)
                else:
                    # Draw empty house (no number) for owned but undeveloped property
                    self._draw_house(surface, house_x, house_y, player_color, number=None, size=18)
            
            # Draw owner indicator circle in bottom right
            owner_surface = pygame.Surface((24, 24), pygame.SRCALPHA)
            pygame.draw.circle(owner_surface, player_color, (12, 12), 10)
            pygame.draw.circle(owner_surface, self.BLACK, (12, 12), 10, 2)
            owner_rect = owner_surface.get_rect(bottomright=(rect.width - 6, rect.height - 6))
            surface.blit(owner_surface, owner_rect)
            label = self.font_price.render(str(owner + 1), True, self.WHITE)
            label_rect = label.get_rect(center=owner_rect.center)
            surface.blit(label, label_rect)

    def _draw_corner_surface(self, surface, tile):
        rect = surface.get_rect()
        pygame.draw.rect(surface, self.WHITE, rect)
        pygame.draw.rect(surface, self.BLACK, rect, 2)
        label = tile["corner"]

        if label == "go":
            pygame.draw.polygon(
                surface,
                (220, 40, 40),
                [
                    (rect.width - 10, rect.height - 10),
                    (rect.width - 10, rect.height - 70),
                    (rect.width - 70, rect.height - 10),
                ],
            )
            go_text = self.font_corner.render("GO", True, self.BLACK)
            surface.blit(go_text, (rect.width - 60, rect.height - 90))
            collect = self.font_price.render("COLLECT $200", True, self.BLACK)
            surface.blit(collect, (rect.width - collect.get_width() - 12, rect.height - 120))
        elif label == "jail":
            jail_rect = pygame.Rect(12, 12, rect.width - 24, rect.height - 24)
            pygame.draw.rect(surface, (255, 165, 0), jail_rect)
            pygame.draw.rect(surface, self.BLACK, jail_rect, 2)
            jail_text = self.font_corner.render("IN JAIL", True, self.BLACK)
            surface.blit(jail_text, (rect.width - jail_text.get_width() - 16, 16))
            visit = self.font_price.render("JUST", True, self.BLACK)
            surface.blit(visit, (18, rect.height - 50))
            visit2 = self.font_price.render("VISITING", True, self.BLACK)
            surface.blit(visit2, (18, rect.height - 32))
        elif label == "free":
            pygame.draw.rect(surface, (240, 240, 240), (12, 12, rect.width - 24, rect.height - 24))
            free = self.font_corner.render("FREE", True, self.BLACK)
            surface.blit(free, (16, 18))
            parking = self.font_corner.render("PARKING", True, self.BLACK)
            surface.blit(parking, (16, 50))
            pygame.draw.polygon(surface, self.BLACK, [(rect.width - 40, rect.height - 30), (rect.width - 20, rect.height - 40), (rect.width - 20, rect.height - 20)])
        elif label == "goto":
            pygame.draw.rect(surface, (240, 240, 240), (12, 12, rect.width - 24, rect.height - 24))
            go = self.font_corner.render("GO", True, self.BLACK)
            surface.blit(go, (18, 16))
            to_jail = self.font_corner.render("TO JAIL", True, self.BLACK)
            surface.blit(to_jail, (18, 48))
            pygame.draw.line(surface, self.BLACK, (20, rect.height - 20), (rect.width - 30, rect.height - 60), 5)
            pygame.draw.polygon(surface, (220, 40, 40), [(rect.width - 20, rect.height - 60), (rect.width - 40, rect.height - 70), (rect.width - 40, rect.height - 50)])
    
    def render_board(self):
        self.screen.fill((22, 24, 29))
        self.screen.blit(self.board_surface, self.board_rect.topleft)
        pygame.draw.rect(self.screen, self.BOARD_EDGE, self.board_rect, 6)

        self._draw_property_indicators()  # Draw houses and owner markers on owned properties
        self._draw_highlight()
        self._draw_players()
        self._draw_info_panel()
        
        # Draw property card overlay if showing
        if self.show_property_card:
            self._draw_property_card(self.selected_tile)
        
        # Draw event card overlay if showing (Chance/Community Chest)
        if self.show_event_card:
            self._draw_event_card()
        
        # Draw market price update overlay
        if self.show_market_overlay:
            self._draw_market_overlay()
        
        # DEBUG
        # self._draw_debug_collision_rects()
        
        pygame.display.flip()
    
    def _draw_property_indicators(self):
        """Draw houses and owner markers on all owned properties directly on the board."""
        for index in range(40):
            tile = self.tiles[index]
            
            # Get owner - handle both env types
            if self.using_decision_env:
                owner = self.env.property_owner[index]
                houses_built = int(self.env.property_houses[index])
            else:
                owner = self.env.property_owners[index] if hasattr(self.env, 'property_owners') else -1
                houses_built = int(self.env.property_houses[index]) if hasattr(self.env, 'property_houses') else 0
            
            if owner < 0:
                continue  # Not owned, skip
            
            # Get the tile's layout info
            layout_info = self.layout[index]
            rect = layout_info["rect"]
            orientation = layout_info["orientation"]
            player_color = self._get_player_color(owner)
            
            # Calculate positions based on orientation
            if orientation == "bottom":
                house_x = rect.centerx
                house_y = rect.top + 22
                flag_x = rect.right - 12
                flag_y = rect.bottom - 8
            elif orientation == "top":
                house_x = rect.centerx
                house_y = rect.bottom - 22
                flag_x = rect.right - 12
                flag_y = rect.top + 28
            elif orientation == "left":
                house_x = rect.right - 22
                house_y = rect.centery
                flag_x = rect.left + 28
                flag_y = rect.bottom - 8
            elif orientation == "right":
                house_x = rect.left + 22
                house_y = rect.centery
                flag_x = rect.right - 12
                flag_y = rect.bottom - 8
            else:  # corners
                house_x = rect.centerx
                house_y = rect.centery
                flag_x = rect.right - 15
                flag_y = rect.bottom - 8
            
            # Draw houses/hotel for properties only
            if tile["type"] == "property" and houses_built > 0:
                self._draw_houses_on_property(self.screen, house_x, house_y, houses_built, orientation)
            
            # Draw owner flag (distinct from player tokens)
            self._draw_owner_flag(self.screen, flag_x, flag_y, player_color, owner + 1)
    
    def _get_collision_rect(self, idx):
        """
        Calculate the collision rectangle for tile at index idx.
        Uses the same direct positioning as _calculate_layout.
        """
        left = self.board_rect.left
        top = self.board_rect.top
        right = self.board_rect.right
        bottom = self.board_rect.bottom
        
        # Corners
        if idx == 0:  # GO - bottom right
            return pygame.Rect(right - self.corner_size, bottom - self.corner_size, self.corner_size, self.corner_size)
        elif idx == 10:  # Jail - bottom left
            return pygame.Rect(left, bottom - self.corner_size, self.corner_size, self.corner_size)
        elif idx == 20:  # Free Parking - top left
            return pygame.Rect(left, top, self.corner_size, self.corner_size)
        elif idx == 30:  # Go To Jail - top right
            return pygame.Rect(right - self.corner_size, top, self.corner_size, self.corner_size)
        
        # Bottom row (tiles 1-9): right to left
        elif 1 <= idx <= 9:
            offset = idx
            return pygame.Rect(
                right - self.corner_size - offset * self.edge_size,
                bottom - self.corner_size,
                self.edge_size,
                self.corner_size,
            )
        
        # Left column (tiles 11-19): bottom to top
        elif 11 <= idx <= 19:
            offset = idx - 10
            return pygame.Rect(
                left,
                bottom - self.corner_size - offset * self.edge_size,
                self.corner_size,
                self.edge_size,
            )
        
        # Top row (tiles 21-29): left to right, starting after the corner
        elif 21 <= idx <= 29:
            offset = idx - 20  # offset 1-9
            return pygame.Rect(
                left + self.corner_size + (offset - 1) * self.edge_size,
                top,
                self.edge_size,
                self.corner_size,
            )
        
        # Right column (tiles 31-39): top to bottom, starting after the corner
        elif 31 <= idx <= 39:
            offset = idx - 30  # offset 1-9
            return pygame.Rect(
                right - self.corner_size,
                top + self.corner_size + (offset - 1) * self.edge_size,
                self.corner_size,
                self.edge_size,
            )
        
        # Fallback
        return self.layout[idx]["rect"].copy()
    
    def _draw_debug_collision_rects(self):
        """Draw red outlines showing the actual clickable collision areas."""
        for idx in range(40):
            collision_rect = self._get_collision_rect(idx)
            # Draw red outline for clickable area
            pygame.draw.rect(self.screen, (255, 0, 0), collision_rect, 2)
    
    def _get_player_color(self, player_id):
        palette = [(220, 30, 33), (0, 115, 230), (46, 180, 75), (255, 215, 0)]
        return palette[player_id % len(palette)]
    
    def _draw_players(self):
        for pid in range(self.max_players):
            tile_index = self.player_positions[pid]
            layout = self.layout[tile_index]
            rect = layout["rect"]
            orientation = layout["orientation"]

            base_x, base_y = rect.center
            if orientation == "bottom":
                base_y = rect.bottom - 28
            elif orientation == "top":
                base_y = rect.top + 28
            elif orientation == "left":
                base_x = rect.left + 28
            elif orientation == "right":
                base_x = rect.right - 28

            if orientation.startswith("corner"):
                base_x = rect.centerx
                base_y = rect.centery

            base_x += (pid % 2) * 18 - 9
            base_y += (pid // 2) * 18 - 9

            pygame.draw.circle(self.screen, self._get_player_color(pid), (int(base_x), int(base_y)), 11)
            pygame.draw.circle(self.screen, self.BLACK, (int(base_x), int(base_y)), 11, 2)
            label = self.font_price.render(str(pid + 1), True, self.WHITE)
            self.screen.blit(label, label.get_rect(center=(int(base_x), int(base_y))))
    
    def _draw_centerpieces(self):
        inner = self.board_rect.inflate(-self.corner_size * 1.5, -self.corner_size * 1.5)
        pygame.draw.rect(self.screen, self.BOARD_CENTER, inner)
        pygame.draw.rect(self.screen, self.BOARD_EDGE, inner, 3)

        banner_surface = pygame.Surface((500, 140), pygame.SRCALPHA)
        pygame.draw.rect(banner_surface, self.BANNER_RED, (0, 0, 500, 140), border_radius=12)
        pygame.draw.rect(banner_surface, self.WHITE, (0, 0, 500, 140), 6, border_radius=12)
        text = self.font_banner.render("MONOPOLY", True, self.WHITE)
        banner_surface.blit(text, text.get_rect(center=(250, 70)))
        banner_rot = pygame.transform.rotate(banner_surface, 35)
        self.screen.blit(banner_rot, banner_rot.get_rect(center=self.board_rect.center))

        chest_surface = pygame.Surface((180, 240), pygame.SRCALPHA)
        pygame.draw.rect(chest_surface, self.CHEST_BLUE, (0, 0, 180, 240))
        pygame.draw.rect(chest_surface, self.BLACK, (0, 0, 180, 240), 4)
        chest_surface.blit(self.font_large.render("COMMUNITY", True, self.BLACK), (18, 40))
        chest_surface.blit(self.font_large.render("CHEST", True, self.BLACK), (40, 80))
        chest_surface.blit(self.font_banner.render("?", True, self.BLACK), (64, 108))
        self.screen.blit(chest_surface, chest_surface.get_rect(center=(self.board_rect.centerx - 140, self.board_rect.centery + 30)))

        chance_surface = pygame.Surface((180, 240), pygame.SRCALPHA)
        pygame.draw.rect(chance_surface, self.CHANCE_ORANGE, (0, 0, 180, 240))
        pygame.draw.rect(chance_surface, self.BLACK, (0, 0, 180, 240), 4)
        chance_surface.blit(self.font_large.render("CHANCE", True, self.BLACK), (36, 40))
        chance_surface.blit(self.font_banner.render("?", True, self.BLACK), (66, 108))
        self.screen.blit(chance_surface, chance_surface.get_rect(center=(self.board_rect.centerx + 140, self.board_rect.centery - 100)))

    def _draw_info_panel(self):
        pygame.draw.rect(self.screen, (248, 248, 248), self.info_rect)
        pygame.draw.rect(self.screen, self.BLACK, self.info_rect, 3)

        x = self.info_rect.left + 12
        y = self.info_rect.top + 12
        panel_width = self.info_rect.width - 24

        # Title - Player Console
        title = self.font_large.render("Player Console", True, self.BLACK)
        self.screen.blit(title, (x, y))
        y += 32

        current_player = self._get_current_player()
        player_type = "Human" if current_player == 0 else "AI"
        
        # Current Player with type
        player_label = self.font_stats.render(f"Turn: P{current_player + 1} ({player_type})", True, self.BLACK)
        self.screen.blit(player_label, (x, y))
        y += 22

        # Show both players' cash
        for pid in range(self.max_players):
            p_type = "You" if pid == 0 else "AI"
            cash = int(self._get_player_cash(pid))
            color = self._get_player_color(pid)
            
            # Player indicator circle
            pygame.draw.circle(self.screen, color, (x + 8, y + 9), 6)
            pygame.draw.circle(self.screen, self.BLACK, (x + 8, y + 9), 6, 1)
            
            money_text = self.font_stats.render(f"P{pid + 1} ({p_type}): ${cash}", True, self.BLACK)
            self.screen.blit(money_text, (x + 20, y))
            y += 20
        
        y += 5

        # Game State indicator
        state_text = ""
        state_color = self.BLACK
        
        # Show AI status if it's AI's turn
        if current_player == 1:  # AI player
            state_text = "🤖 AI Thinking..."
            state_color = (139, 69, 19)  # Brown/bronze color for AI
        elif self.game_state == GameState.AWAITING_ROLL:
            state_text = "🎲 Roll Dice"
            state_color = (0, 128, 0)
        elif self.game_state == GameState.AWAITING_DECISION:
            state_text = "⚡ Make Decision"
            state_color = (0, 0, 200)
        elif self.game_state == GameState.AWAITING_TILE_SELECT:
            action = self.pending_action_type or "action"
            state_text = f"👆 Click tile to {action}"
            state_color = (200, 100, 0)
        elif self.game_state == GameState.GAME_OVER:
            state_text = "🏆 Game Over"
            state_color = (128, 0, 128)
        
        state_label = self.font_stats.render(state_text, True, state_color)
        self.screen.blit(state_label, (x, y))
        y += 24

        # Owned Properties section
        self.screen.blit(self.font_medium.render("Your Properties", True, self.BLACK), (x, y))
        y += 22
        
        # Properties list box
        props_box_height = 80
        props_box = pygame.Rect(x, y, panel_width, props_box_height)
        pygame.draw.rect(self.screen, self.WHITE, props_box)
        pygame.draw.rect(self.screen, (180, 180, 180), props_box, 1)
        
        # Get human player's (P1) properties
        owned_props = []
        for idx, tile in enumerate(self.tiles):
            owner = self._get_property_owner(idx)
            if owner == 0:  # Human player
                houses = self._get_property_houses(idx)
                dev = f" ({houses}h)" if houses > 0 else ""
                owned_props.append(f"{tile['name']}{dev}")
        
        prop_y = y + 4
        for prop_name in owned_props[:4]:  # Show up to 4 properties
            prop_text = self.font_small.render(prop_name[:22], True, self.BLACK)
            self.screen.blit(prop_text, (x + 6, prop_y))
            prop_y += 18
        
        if len(owned_props) > 4:
            more_text = self.font_small.render(f"... +{len(owned_props) - 4} more", True, (100, 100, 100))
            self.screen.blit(more_text, (x + 6, prop_y))
        
        y += props_box_height + 8

        # Dice Rolls section
        self.screen.blit(self.font_medium.render("Last Roll", True, self.BLACK), (x, y))
        y += 22
        
        dice_box_width = panel_width
        dice_box_height = 52
        dice_box = pygame.Rect(x, y, dice_box_width, dice_box_height)
        pygame.draw.rect(self.screen, (230, 230, 230), dice_box, border_radius=6)
        pygame.draw.rect(self.screen, (180, 180, 180), dice_box, 2, border_radius=6)
        
        if self.last_dice:
            dice_total_width = 44 * 2 + 6
            dice_start_x = dice_box.centerx - dice_total_width // 2
            dice_y = dice_box.centery - 22
            for idx, value in enumerate(self.last_dice):
                dice_img = self.dice_images.get(value)
                if dice_img:
                    self.screen.blit(dice_img, (dice_start_x + idx * 47, dice_y))
        
        y += dice_box_height + 8
        
        # Current Tile section
        self.screen.blit(self.font_medium.render("Current Tile", True, self.BLACK), (x, y))
        y += 22
        
        tile = self.tiles[self.selected_tile]
        tile_name_box = pygame.Rect(x, y, panel_width, 28)
        pygame.draw.rect(self.screen, (230, 230, 230), tile_name_box, border_radius=4)
        pygame.draw.rect(self.screen, (180, 180, 180), tile_name_box, 1, border_radius=4)
        tile_name_text = self.font_stats.render(tile["name"][:20], True, self.BLACK)
        self.screen.blit(tile_name_text, (tile_name_box.left + 10, tile_name_box.centery - tile_name_text.get_height() // 2))

        self._draw_action_buttons()
        # self._draw_message_log()

    def _draw_action_buttons(self):
        if self.action_buttons:
            heading_pos = (self.action_buttons[0].rect.left, self.action_buttons[0].rect.top - 28)
            heading = self.font_medium.render("Actions", True, self.BLACK)
            self.screen.blit(heading, heading_pos)

        for button in self.action_buttons:
            button.draw(self.screen, self.font_medium)
        
        # Draw the sell building counter (+/- buttons and number)
        # Find the Sell Building button by label so layout is resilient to ordering
        sell_button = next((b for b in self.action_buttons if b.label == "Sell Building"), None)
        if sell_button is not None:
            counter_x = sell_button.rect.right + 8
            counter_y = sell_button.rect.top
            counter_height = sell_button.rect.height

            # Minus button
            self.minus_rect = pygame.Rect(counter_x, counter_y, 26, counter_height)
            pygame.draw.rect(self.screen, (220, 60, 60), self.minus_rect, border_radius=4)
            pygame.draw.rect(self.screen, (30, 30, 30), self.minus_rect, 2, border_radius=4)
            minus_text = self.font_stats.render("-", True, self.WHITE)
            self.screen.blit(minus_text, minus_text.get_rect(center=self.minus_rect.center))

            # Counter display
            counter_display_rect = pygame.Rect(counter_x + 28, counter_y, 26, counter_height)
            pygame.draw.rect(self.screen, self.WHITE, counter_display_rect)
            pygame.draw.rect(self.screen, (30, 30, 30), counter_display_rect, 2)
            count_text = self.font_stats.render(str(self.sell_count), True, self.BLACK)
            self.screen.blit(count_text, count_text.get_rect(center=counter_display_rect.center))

            # Plus button
            self.plus_rect = pygame.Rect(counter_x + 56, counter_y, 26, counter_height)
            pygame.draw.rect(self.screen, (100, 149, 237), self.plus_rect, border_radius=4)
            pygame.draw.rect(self.screen, (30, 30, 30), self.plus_rect, 2, border_radius=4)
            plus_text = self.font_stats.render("+", True, self.WHITE)
            self.screen.blit(plus_text, plus_text.get_rect(center=self.plus_rect.center))

    def _draw_message_log(self):
        
        """Draw activity log at bottom of info panel."""
        footer_height = 70
        footer_rect = pygame.Rect(
            self.info_rect.left + 12,
            self.info_rect.bottom - footer_height - 8,
            self.info_rect.width - 24,
            footer_height,
        )
        pygame.draw.rect(self.screen, (250, 250, 250), footer_rect, border_radius=6)
        pygame.draw.rect(self.screen, (180, 180, 180), footer_rect, 1, border_radius=6)
        y = footer_rect.top + 4
        self.screen.blit(self.font_small.render("Activity Log", True, (100, 100, 100)), (footer_rect.left + 8, y))
        y += 16
        for msg in self.message_log[-3:]:  # Show last 3 messages
            # Truncate long messages
            display_msg = msg[:40] + "..." if len(msg) > 40 else msg
            msg_text = self.font_small.render(display_msg, True, (60, 60, 60))
            self.screen.blit(msg_text, (footer_rect.left + 8, y))
            y += 16

    def _draw_dice_history_section(self, start_x, start_y):
        section_title = self.font_medium.render("Dice Rolls", True, self.BLACK)
        self.screen.blit(section_title, (start_x, start_y))
        y = start_y + 28
        row_height = 54
        for pid in range(self.max_players):
            row_label = self.font_stats.render(f"P{pid + 1}", True, self.BLACK)
            self.screen.blit(row_label, (start_x, y))
            dice = self.player_last_rolls[pid]
            if dice:
                for idx, value in enumerate(dice):
                    dice_img = self.dice_images.get(value)
                    if dice_img:
                        self.screen.blit(dice_img, (start_x + 70 + idx * 52, y - 6))
                total = sum(dice)
                total_text = self.font_stats.render(f"= {total}", True, (60, 60, 60))
                self.screen.blit(total_text, (start_x + 70 + 2 * 52 + 20, y + 6))
            else:
                placeholder = self.font_stats.render("--", True, (150, 150, 150))
                self.screen.blit(placeholder, (start_x + 70, y))
            y += row_height
        return y

    def _handle_tile_click(self, pos):
        """
        Check if a tile was clicked. Handle based on game state.
        """
        for idx in range(40):
            collision_rect = self._get_collision_rect(idx)
            
            if collision_rect.collidepoint(pos):
                tile = self.tiles[idx]
                current_player = self._get_current_player()
                
                # If we're waiting for tile selection for build/sell
                if self.game_state == GameState.AWAITING_TILE_SELECT:
                    if self.pending_action_type == "build":
                        if idx in self._get_buildable_tiles():
                            self._execute_build(idx)
                            return tile["name"]
                        else:
                            self._push_message(f"Cannot build on {tile['name']}")
                    elif self.pending_action_type == "sell_building":
                        if idx in self._get_sellable_building_tiles():
                            self._execute_sell_building(idx)
                            return tile["name"]
                        else:
                            self._push_message(f"Cannot sell building from {tile['name']}")
                    elif self.pending_action_type == "sell_property":
                        if idx in self._get_sellable_property_tiles():
                            self._execute_sell_property(idx)
                            return tile["name"]
                        else:
                            self._push_message(f"Cannot sell {tile['name']}")
                    return None
                
                # Normal click - show property card
                if tile["type"] in ["property", "railroad", "utility"]:
                    self.selected_tile = idx
                    self.show_property_card = True
                    self.last_clicked_property = tile["name"]
                    return tile["name"]
        return None

    def _handle_button_click(self, pos):
        # Close event card if showing (Chance/Community Chest)
        if self.show_event_card:
            self.show_event_card = False
            self.current_event_card = None
            self.current_event_type = None
            return
        
        # Close property card if showing
        if self.show_property_card:
            self.show_property_card = False
            return
        
        # Cancel tile selection mode on any non-tile click
        if self.game_state == GameState.AWAITING_TILE_SELECT:
            # Check if a tile was clicked
            clicked_property = self._handle_tile_click(pos)
            if clicked_property:
                return
            # Otherwise cancel selection mode
            self.game_state = GameState.AWAITING_DECISION
            self.pending_action_type = None
            self._push_message("Action cancelled")
            return
        
        # Check if plus/minus buttons clicked for sell count
        if hasattr(self, 'plus_rect') and self.plus_rect.collidepoint(pos):
            self.sell_count = min(self.sell_count + 1, 5)  # Max 5 buildings
            return
        if hasattr(self, 'minus_rect') and self.minus_rect.collidepoint(pos):
            self.sell_count = max(self.sell_count - 1, 0)  # Min 0
            return
        
        # Check if a tile was clicked
        clicked_property = self._handle_tile_click(pos)
        if clicked_property:
            self._push_message(f"Viewing: {clicked_property}")
            return
        
        for button in self.action_buttons:
            if button.handle_click(pos):
                break

    def _handle_roll(self):
        """Roll dice and move current player."""
        if self.game_state != GameState.AWAITING_ROLL:
            return
        
        current_player = self._get_current_player()
        
        # Roll dice
        die_one = random.randint(1, 6)
        die_two = random.randint(1, 6)
        dice_total = die_one + die_two
        self.last_roll_total = dice_total
        self.last_dice = (die_one, die_two)
        self.player_last_rolls[current_player] = self.last_dice
        
        # Move player
        old_position = self.player_positions[current_player]
        new_position = (old_position + dice_total) % len(self.tiles)
        self.player_positions[current_player] = new_position
        self.selected_tile = new_position
        
        # Update env positions
        if self.using_decision_env:
            passed_go = old_position + dice_total >= 40
            self.env.player_positions[current_player] = new_position
            if passed_go:
                self.env.player_cash[current_player] += 200
                self._push_message(f"P{current_player + 1} passed GO! Collected $200")
        
        tile = self.tiles[new_position]
        self._push_message(f"P{current_player + 1} rolled {dice_total} → {tile['name']}")
        
        # Handle tile effects
        self._handle_tile_landing(current_player, new_position, dice_total)
        
        # Transition to decision state
        self.game_state = GameState.AWAITING_DECISION
        self.awaiting_roll = False
        self.awaiting_decision = True
        
        # Show property card if landed on purchasable tile
        if tile["type"] in ["property", "railroad", "utility"]:
            self.show_property_card = True
        elif tile["type"] == "chance":
            self.current_event_card = random.choice(chance_cards)
            self.current_event_type = "CHANCE"
            self.show_event_card = True
            self._apply_card_effect(current_player, self.current_event_card)
        elif tile["type"] == "chest":
            self.current_event_card = random.choice(chest_cards)
            self.current_event_type = "COMMUNITY CHEST"
            self.show_event_card = True
            self._apply_card_effect(current_player, self.current_event_card)
    
    def _handle_tile_landing(self, player_id, tile_idx, dice_total):
        """Handle effects of landing on a tile."""
        tile = self.tiles[tile_idx]
        tile_type = tile.get("type")
        
        if self.using_decision_env:
            # Tax tiles
            if tile_type == "tax":
                amount = tile.get("price", 100)
                self.env.player_cash[player_id] -= amount
                self._push_message(f"P{player_id + 1} paid ${amount} tax")
            
            # Go To Jail
            elif tile.get("corner") == "goto":
                self._push_message(f"P{player_id + 1} goes to JAIL!")
                self.env.player_positions[player_id] = 10  # Jail position
                self.player_positions[player_id] = 10
                self.env.player_in_jail[player_id] = True
                self.env.jail_turns_remaining[player_id] = 3
            
            # Pay rent if landing on owned property
            elif tile_type in ["property", "railroad", "utility"]:
                owner = self._get_property_owner(tile_idx)
                if owner >= 0 and owner != player_id:
                    rent = self._calculate_rent(tile_idx, dice_total)
                    self.env.player_cash[player_id] -= rent
                    self.env.player_cash[owner] += rent
                    self._push_message(f"P{player_id + 1} paid ${rent} rent to P{owner + 1}")
    
    def _calculate_rent(self, tile_idx, dice_total):
        """Calculate rent for a property."""
        if self.using_decision_env:
            return self.env._calculate_rent(tile_idx, dice_total)
        
        tile = self.tiles[tile_idx]
        houses = self._get_property_houses(tile_idx)
        rent_data = tile.get("rent", {})
        
        if tile.get("type") == "property":
            if houses == 0:
                return rent_data.get("base", 0)
            elif houses == 5:
                return rent_data.get("hotel", 0)
            else:
                return rent_data.get(f"{houses}h", 0)
        return rent_data.get("base", 0)
    
    def _apply_card_effect(self, player_id, card):
        """Apply effect of Chance/Community Chest card."""
        card_type = card.get("type")
        
        if self.using_decision_env:
            if card_type == "money":
                self.env.player_cash[player_id] += card.get("amount", 0)
                amt = card.get("amount", 0)
                if amt >= 0:
                    self._push_message(f"P{player_id + 1} received ${amt}")
                else:
                    self._push_message(f"P{player_id + 1} paid ${-amt}")
            elif card_type == "move":
                target = card.get("target", 0)
                self.env.player_positions[player_id] = target
                self.player_positions[player_id] = target
                if target == 0:
                    self.env.player_cash[player_id] += 200
                self._push_message(f"P{player_id + 1} moved to {self.tiles[target]['name']}")
            elif card_type == "go_to_jail":
                self.env.player_positions[player_id] = 10
                self.player_positions[player_id] = 10
                self.env.player_in_jail[player_id] = True
                self.env.jail_turns_remaining[player_id] = 3
                self._push_message(f"P{player_id + 1} goes to JAIL!")
            elif card_type == "jail_card":
                self.env.player_jail_cards[player_id] += 1
                self._push_message(f"P{player_id + 1} got a Get Out of Jail Free card!")

    def _handle_buy(self):
        """Buy the tile the player is currently standing on."""
        if self.game_state != GameState.AWAITING_DECISION:
            return
        
        current_player = self._get_current_player()
        current_position = self.player_positions[current_player]
        tile = self.tiles[current_position]
        
        if tile["type"] not in {"property", "railroad", "utility"}:
            self._push_message("Cannot buy this tile")
            return
        
        owner = self._get_property_owner(current_position)
        if owner >= 0:
            self._push_message(f"Already owned by P{owner + 1}")
            return
        
        price = tile.get("price", 0)
        if self._get_player_cash(current_player) < price:
            self._push_message("Not enough cash!")
            return
        
        # Execute purchase
        if self.using_decision_env:
            self.env.player_cash[current_player] -= price
            self.env.property_owner[current_position] = current_player
        else:
            self.env.player_cash[current_player] -= price
            self.env.property_owners[current_position] = current_player
        
        self._push_message(f"P{current_player + 1} bought {tile['name']} for ${price}")
        self._end_turn()

    def _handle_skip(self):
        """End turn without buying."""
        if self.game_state not in [GameState.AWAITING_DECISION, GameState.AWAITING_TILE_SELECT]:
            return
        
        current_player = self._get_current_player()
        self._push_message(f"P{current_player + 1} ends turn")
        self._end_turn()

    def _handle_build(self):
        """Enter build mode - player selects tile to build on."""
        if self.game_state != GameState.AWAITING_DECISION:
            return
        
        buildable = self._get_buildable_tiles()
        if not buildable:
            self._push_message("No properties available to build on!")
            return
        
        # List buildable properties
        names = [self.tiles[idx]["name"] for idx in buildable[:3]]
        self._push_message(f"Click a property to build: {', '.join(names)}...")
        
        self.game_state = GameState.AWAITING_TILE_SELECT
        self.pending_action_type = "build"
        self.buildable_tiles = buildable
    
    def _execute_build(self, tile_idx):
        """Execute building a house on the selected tile."""
        current_player = self._get_current_player()
        tile = self.tiles[tile_idx]
        
        if self.using_decision_env:
            result = self.env._handle_build_property(tile_idx, current_player)
            if result > 0:
                houses = self._get_property_houses(tile_idx)
                dev = "hotel" if houses == 5 else f"{houses} house(s)"
                self._push_message(f"P{current_player + 1} built on {tile['name']} ({dev})")
            else:
                self._push_message(f"Failed to build on {tile['name']}")
        
        self.game_state = GameState.AWAITING_DECISION
        self.pending_action_type = None
    
    def _handle_sell_building(self):
        """Enter sell building mode."""
        if self.game_state != GameState.AWAITING_DECISION:
            return
        
        sellable = self._get_sellable_building_tiles()
        if not sellable:
            self._push_message("No buildings to sell!")
            return
        
        names = [self.tiles[idx]["name"] for idx in sellable[:3]]
        self._push_message(f"Click property to sell building: {', '.join(names)}...")
        
        self.game_state = GameState.AWAITING_TILE_SELECT
        self.pending_action_type = "sell_building"
        self.sellable_building_tiles = sellable
    
    def _execute_sell_building(self, tile_idx):
        """Execute selling a building from the selected tile."""
        current_player = self._get_current_player()
        tile = self.tiles[tile_idx]
        
        if self.using_decision_env:
            result = self.env._handle_sell_building(tile_idx, current_player)
            if result > 0:
                refund = tile.get("house_cost", 50) * 0.5
                self._push_message(f"P{current_player + 1} sold building from {tile['name']} (+${int(refund)})")
            else:
                self._push_message(f"Failed to sell building from {tile['name']}")
        
        self.game_state = GameState.AWAITING_DECISION
        self.pending_action_type = None
    
    def _handle_sell_property_btn(self):
        """Enter sell property mode."""
        if self.game_state != GameState.AWAITING_DECISION:
            return
        
        sellable = self._get_sellable_property_tiles()
        if not sellable:
            self._push_message("No properties to sell!")
            return
        
        names = [self.tiles[idx]["name"] for idx in sellable[:3]]
        self._push_message(f"Click property to sell: {', '.join(names)}...")
        
        self.game_state = GameState.AWAITING_TILE_SELECT
        self.pending_action_type = "sell_property"
        self.sellable_property_tiles = sellable
    
    def _execute_sell_property(self, tile_idx):
        """Execute selling the selected property."""
        current_player = self._get_current_player()
        tile = self.tiles[tile_idx]
        
        if self.using_decision_env:
            result = self.env._handle_sell_property(tile_idx, current_player)
            if result > 0:
                base_price = tile.get("price", 0)
                refund = base_price * 0.5
                self._push_message(f"P{current_player + 1} sold {tile['name']} (+${int(refund)})")
            else:
                self._push_message(f"Failed to sell {tile['name']}")
        
        self.game_state = GameState.AWAITING_DECISION
        self.pending_action_type = None

    def _end_turn(self):
        """End current player's turn and switch to next player."""
        current_player = self._get_current_player()
        
        # Check bankruptcy
        if self.using_decision_env:
            self.env._check_and_handle_bankruptcy(current_player)
            
            # Check for game over
            active_players = sum(1 for i in range(self.max_players) if not self.env.player_bankrupt[i])
            if active_players <= 1:
                self.game_state = GameState.GAME_OVER
                winner = next(i for i in range(self.max_players) if not self.env.player_bankrupt[i])
                self._push_message(f"GAME OVER! Player {winner + 1} wins!")
                return
        
        # Switch to next player
        next_player = (current_player + 1) % self.max_players
        
        if self.using_decision_env:
            self.env.current_player = next_player
            self.env.turn_count += 1
            
            # Check if it's time for market price update (every 5 turns)
            if self.env.turn_count % self.market_update_interval == 0 and self.market_model is not None:
                self._update_market_prices()
            
            # Skip bankrupt players
            while self.env.player_bankrupt[next_player]:
                next_player = (next_player + 1) % self.max_players
                self.env.current_player = next_player
        else:
            self.env.current_player = next_player
            if hasattr(self.env, 'turn_counter'):
                self.env.turn_counter += 1
                # Check if it's time for market price update
                if self.env.turn_counter % self.market_update_interval == 0 and self.market_model is not None:
                    self._update_market_prices()
        
        # Reset state for next turn
        self.game_state = GameState.AWAITING_ROLL
        self.awaiting_roll = True
        self.awaiting_decision = False
        self.pending_action_type = None
        
        player_type = "Human" if next_player == 0 else "AI"
        self._push_message(f"Player {next_player + 1}'s turn ({player_type})")

    def run(self):
        """Launch interactive UI for Human vs AI game."""
        running = True
        print("=" * 50)
        print("DYMONOPOLY - Human vs AI")
        print("=" * 50)
        print("Controls:")
        print("  - Left click on buttons to trigger actions")
        print("  - Click on tiles to view property cards")
        print("  - When building/selling, click on the target tile")
        print("  - ESC closes the window")
        print("=" * 50)
        
        if self.ai_policy is not None:
            print("[AI] Using trained DQN model for Player 2")
        else:
            print("[AI] Using random actions for Player 2 (model not loaded)")

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    # Only process human clicks if it's human's turn
                    current_player = self._get_current_player()
                    if current_player == 0:  # Human player
                        self._handle_button_click(event.pos)

            # Process AI turn if applicable
            if self.game_state != GameState.GAME_OVER:
                self._process_ai_turn()

            self.render_board()
            self.clock.tick(60)

        pygame.quit()

# Usage example
if __name__ == "__main__":
    print("Initializing Dymonopoly Decision Environment...")
    env = DymonopolyDecisionEnv(num_players=2)
    env.reset()
    
    print("Starting visualization...")
    visualizer = MonopolyVisualizer(env)
    visualizer.run()