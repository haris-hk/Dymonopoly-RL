import gymnasium as gym
from gymnasium import spaces
import numpy as np

class DymonopolyEnv(gym.Env):
    metadata = {"render_modes": ["human"]}
    
    def __init__(self, num_properties=40, num_players=4):
        super().__init__()
        
        #stuff i m adding, rough
        

        self.properties = [
        {"name": "GO", "type": "corner", "corner": "go"},

        {"name": "Mediterranean Avenue", "type": "property", "color": "brown",
        "price": 60, "owned_by": -1,
        "rent": {"base": 2, "1h": 10, "2h": 30, "3h": 90, "4h": 160, "hotel": 250},
        "visiting_freq": 0, "trading_frequency": 0},

        {"name": "Community Chest", "type": "chest"},

        {"name": "Baltic Avenue", "type": "property", "color": "brown",
        "price": 60, "owned_by": -1,
        "rent": {"base": 4, "1h": 20, "2h": 60, "3h": 180, "4h": 320, "hotel": 450},
        "visiting_freq": 0, "trading_frequency": 0},

        {"name": "Income Tax", "type": "tax", "price": 200},

        {"name": "Reading Railroad", "type": "railroad", "price": 200, "owned_by": -1,
        "rent": {"1_rr": 25, "2_rr": 50, "3_rr": 100, "4_rr": 200},
        "visiting_freq": 0, "trading_frequency": 0},

        {"name": "Oriental Avenue", "type": "property", "color": "light_blue",
        "price": 100, "owned_by": -1,
        "rent": {"base": 6, "1h": 30, "2h": 90, "3h": 270, "4h": 400, "hotel": 550},
        "visiting_freq": 0, "trading_frequency": 0},

        {"name": "Chance", "type": "chance"},

        {"name": "Vermont Avenue", "type": "property", "color": "light_blue",
        "price": 100, "owned_by": -1,
        "rent": {"base": 6, "1h": 30, "2h": 90, "3h": 270, "4h": 400, "hotel": 550},
        "visiting_freq": 0, "trading_frequency": 0},

        {"name": "Connecticut Avenue", "type": "property", "color": "light_blue",
        "price": 120, "owned_by": -1,
        "rent": {"base": 8, "1h": 40, "2h": 100, "3h": 300, "4h": 450, "hotel": 600},
        "visiting_freq": 0, "trading_frequency": 0},

        {"name": "In Jail / Just Visiting", "type": "corner", "corner": "jail"},

        {"name": "St. Charles Place", "type": "property", "color": "pink",
        "price": 140, "owned_by": -1,
        "rent": {"base": 10, "1h": 50, "2h": 150, "3h": 450, "4h": 625, "hotel": 750},
        "visiting_freq": 0, "trading_frequency": 0},

        {"name": "Electric Company", "type": "utility", "price": 150, "owned_by": -1,
        "rent": {"one_util": "4x dice", "both_utils": "10x dice"},
        "visiting_freq": 0, "trading_frequency": 0},

        {"name": "States Avenue", "type": "property", "color": "pink",
        "price": 140, "owned_by": -1,
        "rent": {"base": 10, "1h": 50, "2h": 150, "3h": 450, "4h": 625, "hotel": 750},
        "visiting_freq": 0, "trading_frequency": 0},

        {"name": "Virginia Avenue", "type": "property", "color": "pink",
        "price": 160, "owned_by": -1,
        "rent": {"base": 12, "1h": 60, "2h": 180, "3h": 500, "4h": 700, "hotel": 900},
        "visiting_freq": 0, "trading_frequency": 0},

        {"name": "Pennsylvania Railroad", "type": "railroad", "price": 200, "owned_by": -1,
        "rent": {"1_rr": 25, "2_rr": 50, "3_rr": 100, "4_rr": 200},
        "visiting_freq": 0, "trading_frequency": 0},

        {"name": "St. James Place", "type": "property", "color": "orange",
        "price": 180, "owned_by": -1,
        "rent": {"base": 14, "1h": 70, "2h": 200, "3h": 550, "4h": 750, "hotel": 950},
        "visiting_freq": 0, "trading_frequency": 0},

        {"name": "Community Chest", "type": "chest"},

        {"name": "Tennessee Avenue", "type": "property", "color": "orange",
        "price": 180, "owned_by": -1,
        "rent": {"base": 14, "1h": 70, "2h": 200, "3h": 550, "4h": 750, "hotel": 950},
        "visiting_freq": 0, "trading_frequency": 0},

        {"name": "New York Avenue", "type": "property", "color": "orange",
        "price": 200, "owned_by": -1,
        "rent": {"base": 16, "1h": 80, "2h": 220, "3h": 600, "4h": 800, "hotel": 1000},
        "visiting_freq": 0, "trading_frequency": 0},

        {"name": "Free Parking", "type": "corner", "corner": "free"},

        {"name": "Kentucky Avenue", "type": "property", "color": "red",
        "price": 220, "owned_by": -1,
        "rent": {"base": 18, "1h": 90, "2h": 250, "3h": 700, "4h": 875, "hotel": 1050},
        "visiting_freq": 0, "trading_frequency": 0},

        {"name": "Chance", "type": "chance"},

        {"name": "Indiana Avenue", "type": "property", "color": "red",
        "price": 220, "owned_by": -1,
        "rent": {"base": 18, "1h": 90, "2h": 250, "3h": 700, "4h": 875, "hotel": 1050},
        "visiting_freq": 0, "trading_frequency": 0},

        {"name": "Illinois Avenue", "type": "property", "color": "red",
        "price": 240, "owned_by": -1,
        "rent": {"base": 20, "1h": 100, "2h": 300, "3h": 750, "4h": 925, "hotel": 1100},
        "visiting_freq": 0, "trading_frequency": 0},

        {"name": "B. & O. Railroad", "type": "railroad", "price": 200, "owned_by": -1,
        "rent": {"1_rr": 25, "2_rr": 50, "3_rr": 100, "4_rr": 200},
        "visiting_freq": 0, "trading_frequency": 0},

        {"name": "Atlantic Avenue", "type": "property", "color": "yellow",
        "price": 260, "owned_by": -1,
        "rent": {"base": 22, "1h": 110, "2h": 330, "3h": 800, "4h": 975, "hotel": 1150},
        "visiting_freq": 0, "trading_frequency": 0},

        {"name": "Ventnor Avenue", "type": "property", "color": "yellow",
        "price": 260, "owned_by": -1,
        "rent": {"base": 22, "1h": 110, "2h": 330, "3h": 800, "4h": 975, "hotel": 1150},
        "visiting_freq": 0, "trading_frequency": 0},

        {"name": "Water Works", "type": "utility", "price": 150, "owned_by": -1,
        "rent": {"one_util": "4x dice", "both_utils": "10x dice"},
        "visiting_freq": 0, "trading_frequency": 0},

        {"name": "Marvin Gardens", "type": "property", "color": "yellow",
        "price": 280, "owned_by": -1,
        "rent": {"base": 24, "1h": 120, "2h": 360, "3h": 850, "4h": 1025, "hotel": 1200},
        "visiting_freq": 0, "trading_frequency": 0},

        {"name": "Go To Jail", "type": "corner", "corner": "goto"},

        {"name": "Pacific Avenue", "type": "property", "color": "green",
        "price": 300, "owned_by": -1,
        "rent": {"base": 26, "1h": 130, "2h": 390, "3h": 900, "4h": 1100, "hotel": 1275},
        "visiting_freq": 0, "trading_frequency": 0},

        {"name": "North Carolina Avenue", "type": "property", "color": "green",
        "price": 300, "owned_by": -1,
        "rent": {"base": 26, "1h": 130, "2h": 390, "3h": 900, "4h": 1100, "hotel": 1275},
        "visiting_freq": 0, "trading_frequency": 0},

        {"name": "Community Chest", "type": "chest"},

        {"name": "Pennsylvania Avenue", "type": "property", "color": "green",
        "price": 320, "owned_by": -1,
        "rent": {"base": 28, "1h": 150, "2h": 450, "3h": 1000, "4h": 1200, "hotel": 1400},
        "visiting_freq": 0, "trading_frequency": 0},

        {"name": "Short Line", "type": "railroad", "price": 200, "owned_by": -1,
        "rent": {"1_rr": 25, "2_rr": 50, "3_rr": 100, "4_rr": 200},
        "visiting_freq": 0, "trading_frequency": 0},

        {"name": "Chance", "type": "chance"},

        {"name": "Park Place", "type": "property", "color": "dark_blue",
        "price": 350, "owned_by": -1,
        "rent": {"base": 35, "1h": 175, "2h": 500, "3h": 1100, "4h": 1300, "hotel": 1500},
        "visiting_freq": 0, "trading_frequency": 0},

        {"name": "Luxury Tax", "type": "tax", "price": 100},

        {"name": "Boardwalk", "type": "property", "color": "dark_blue",
        "price": 400, "owned_by": -1,
        "rent": {"base": 50, "1h": 200, "2h": 600, "3h": 1400, "4h": 1700, "hotel": 2000},
        "visiting_freq": 0, "trading_frequency": 0}
        ]


        self.num_properties = 40
        self.num_players = num_players
        self.turn_counter = 0
        self.price_update_interval = 5
        self.players = []

        for i in range (self.num_players):
            self.players.append([{"id": i , "owns": [], "money": 1500, "jail": False, "bankrupt" : False}])


        # Baseline market state
        self.base_prices = np.array([prop.get("price", 0) for prop in self.properties], dtype=float)
        self.current_prices = self.base_prices.copy()
        self.visiting_freq = np.zeros(self.num_properties)
        self.trading_freq = np.zeros(self.num_properties)
        self.property_owners = np.full(self.num_properties, -1)
        self.player_cash = np.full(self.num_players, 1500, dtype=float)
        self.current_player = 0

        # Reward weights (tunable)
        self.market_activity_weight = 1.0
        self.volatility_weight = 1.0
        self.market_depth_weight = 0.6
        self.monopoly_competition_weight = 0.2



        
        # Observation space: property prices, ownership, player positions, cash, etc.
        self.observation_space = spaces.Dict({
            "property_prices": spaces.Box(low=0, high=np.inf, shape=(num_properties,)),
            "property_owners": spaces.Box(low=-1, high=num_players-1, shape=(num_properties,)),
            "visiting_frequency": spaces.Box(low=0, high=np.inf, shape=(num_properties,)),
            "trading_frequency": spaces.Box(low=0, high=np.inf, shape=(num_properties,)),
            "player_cash": spaces.Box(low=0, high=np.inf, shape=(num_players,)),
            "current_player": spaces.Discrete(num_players)
        })
        
        # Action space: buy, sell, trade, pass
        self.action_space = spaces.Box(
            low=0.8,  # can decrease by 20%
            high=1.2,  # can increase by 20%
            shape=(num_properties,)  # one multiplier per property
)


        
    def _get_obs(self):
        """Return observation using numpy arrays (authoritative state)"""
        return {
            "property_prices": self.current_prices,
            "property_owners": self.property_owners,
            "visiting_frequency": self.visiting_freq,
            "trading_frequency": self.trading_freq,
            "player_cash": self.player_cash,
            "current_player": self.current_player
        }
    
    def _get_info(self):
        return {"turn": self.turn_counter}
    
    def _sync_property_dicts(self):
        """Sync numpy arrays back to property dicts for rendering/debugging"""
        for prop_id in range(self.num_properties):
            if "price" in self.properties[prop_id]:
                self.properties[prop_id]["price"] = float(self.current_prices[prop_id])
            self.properties[prop_id]["owned_by"] = int(self.property_owners[prop_id])
            self.properties[prop_id]["visiting_freq"] = float(self.visiting_freq[prop_id])
            self.properties[prop_id]["trading_frequency"] = float(self.trading_freq[prop_id])
    
    
    def _get_similar_property_factor(self):
        """Calculate competition reward across ALL color groups
        
        Returns aggregated reward for competition dynamics:
        - Rewards when multiple players own properties in same color group
        - Penalizes monopolies (one player owning 50%+ of a color group)
        """
        # Build color group mapping
        color_groups = {}
        for idx, prop in enumerate(self.properties):
            color = prop.get("color")
            if color:
                if color not in color_groups:
                    color_groups[color] = []
                color_groups[color].append(idx)
        
        if not color_groups:
            return 0.0
        
        total_competition_score = 0.0
        
        for color, group_indices in color_groups.items():
            if len(group_indices) <= 1:
                continue
            
            # Get owners for this color group
            group_owners = [self.property_owners[idx] for idx in group_indices]
            
            # Count owned properties (exclude unowned = -1)
            owned_properties = [owner for owner in group_owners if owner >= 0]
            
            if len(owned_properties) == 0:
                continue
            
            # Count unique owners in this color group
            from collections import Counter
            owner_counts = Counter(owned_properties)
            num_unique_owners = len(owner_counts)
            max_count = max(owner_counts.values())
            total_in_group = len(group_indices)
            num_owned = len(owned_properties)
            
            # Competition reward: multiple players own properties in same group
            if num_unique_owners > 1:
                # More unique owners = more competition = higher reward
                # Scale by how many properties are owned (more owned = more tense)
                competition_intensity = num_unique_owners / total_in_group
                ownership_density = num_owned / total_in_group
                total_competition_score += competition_intensity * ownership_density * 2.0
            
            # Monopoly penalty: one player owns 50%+ of group
            elif max_count / total_in_group >= 0.5:
                total_competition_score -= 1.0
            
            # Single owner with <50%: neutral (no change to score)
        
            # Average competition across all color groups
            avg_competition = total_competition_score / len(color_groups)
            return float(avg_competition)

    
    def _roll_dice(self):
        """Simulate Monopoly-style dice roll."""
        return np.random.randint(1, 7) + np.random.randint(1, 7)

    def step(self, action):
        """
        Action: array of price multipliers (0.8-1.2) for each property
        Every 5 turns: apply price adjustments and calculate reward
        """
        self.turn_counter += 1
        
        # Simulate ONE player's turn (simplified game logic)
        current_player = self.current_player
        
        # Initialize player positions if not exists
        if not hasattr(self, 'player_positions'):
            self.player_positions = [0] * self.num_players
        
        # 1. Roll dice and move
        dice_roll = self._roll_dice()
        self.player_positions[current_player] = (self.player_positions[current_player] + dice_roll) % self.num_properties
        landed_pos = self.player_positions[current_player]
        
        # 2. Update visiting frequency
        self.visiting_freq[landed_pos] += 1
        
        # 3. Handle property interaction (simplified)
        prop = self.properties[landed_pos]
        if prop.get("type") in ["property", "utility", "railroad"]:
            prop_price = self.current_prices[landed_pos]
            
            # If unowned and player can afford: 30% chance to buy
            if self.property_owners[landed_pos] == -1:
                if self.player_cash[current_player] >= prop_price and np.random.rand() < 0.3:
                    self.property_owners[landed_pos] = current_player
                    self.player_cash[current_player] -= prop_price
                    self.trading_freq[landed_pos] += 1
            
            # If owned by another player: pay rent (simplified)
            elif self.property_owners[landed_pos] != current_player:
                owner = int(self.property_owners[landed_pos])
                rent = prop.get("rent", {}).get("base", 0)
                if self.player_cash[current_player] >= rent:
                    self.player_cash[current_player] -= rent
                    self.player_cash[owner] += rent
        
        # 4. Check bankruptcy
        if self.player_cash[current_player] < 0:
            self.players[current_player][0]["bankrupt"] = True
        
        # 5. Next player
        self.current_player = (self.current_player + 1) % self.num_players
        
        # 6. Every 5 turns: RL bot adjusts prices
        reward = 0.0
        if self.turn_counter % self.price_update_interval == 0:
            # Apply action (price multipliers)
            for prop_id in range(self.num_properties):
                if self.base_prices[prop_id] == 0:  # Skip non-ownable
                    continue
                
                # Apply RL bot's price adjustment
                multiplier = np.clip(action[prop_id], 0.8, 1.2)
                self.current_prices[prop_id] *= multiplier
                
                # Keep prices reasonable (50%-200% of baseline)
                min_price = self.base_prices[prop_id] * 0.4
                max_price = self.base_prices[prop_id] * 2.2
                self.current_prices[prop_id] = np.clip(self.current_prices[prop_id], min_price, max_price)
            
            self._sync_property_dicts()
            reward = self.reward_function()  # Your existing reward function
        
        obs = self._get_obs()
        terminated = self.game_end_cond()
        truncated = False
        info = self._get_info()
        
        return obs, reward, terminated, truncated, info
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.turn_counter = 0
        self.visiting_freq = np.zeros(self.num_properties)
        self.trading_freq = np.zeros(self.num_properties)
        self.current_prices = self.base_prices.copy()
        self.property_owners = np.full(self.num_properties, -1)
        self.player_cash = np.full(self.num_players, 1500.0)
        self.current_player = 0
        self.player_positions = [0] * self.num_players  # All start at GO
        
        # Reset player data
        for i in range(self.num_players):
            self.players[i] = [{"id": i, "owns": [], "money": 1500, "jail": False, "bankrupt": False}]
        
        return self._get_obs(), self._get_info()
    
    def game_end_cond(self):
        bankrupt_players = 0
        for i in range(self.num_players):
            if self.players[i][0]["bankrupt"] == True:  # Fixed indexing
                bankrupt_players += 1
    
        # Game ends if 3+ players are bankrupt (only 1 or 0 active)
        if bankrupt_players >= self.num_players - 1:
            return True
        

    
    def _market_depth_reward(self):
        """Reward when most ownable properties are owned"""
        ownable_count = 0
        owned_count = 0
        
        for idx, prop in enumerate(self.properties):
            prop_type = prop.get("type")
            # Check if property is ownable
            if prop_type in ["property", "utility", "railroad"]:
                ownable_count += 1
                # Check if it's owned (owner != -1)
                if self.property_owners[idx] >= 0:
                    owned_count += 1
        
        if ownable_count == 0:
            return 0.0
        
        ownership_ratio = owned_count / ownable_count
        return ownership_ratio  # 0.0 (none owned) to 1.0 (all owned)

    def reward_function(self):
        """Liquidity Reward
            Volatility Penalty
            Inequality Penalty
            no.of turns"""
        base_prices = np.asarray(getattr(self, "base_prices", np.ones(self.num_properties)), dtype=float)
        current_prices = np.asarray(getattr(self, "current_prices", base_prices), dtype=float)
        trading_freq = np.asarray(getattr(self, "trading_freq", np.zeros(self.num_properties)), dtype=float)
        property_owners = np.asarray(getattr(self, "property_owners", np.full(self.num_properties, -1)))
        player_cash = np.asarray(getattr(self, "player_cash", np.zeros(self.num_players)), dtype=float)

        # 1) Liquidity reward (bounded 0-1)
        avg_trades = np.mean(trading_freq) if trading_freq.size else 0.0
        market_activity = np.tanh(avg_trades)

        # 2) Volatility penalty based on deviation from baseline prices
        safe_base = np.where(base_prices <= 0, 1.0, base_prices)
        price_delta = np.abs(current_prices - base_prices) / safe_base
        volatility_penalty = np.tanh(np.mean(price_delta))

        # 3) Reward when most ownable properties are owned
        market_depth = self._market_depth_reward()

        # 4) Longevity bonus encourages longer survival
        monopoly_competition = self._get_similar_property_factor()

        reward = (
            self.market_activity_weight * market_activity
            - self.volatility_weight * volatility_penalty
            + self.market_depth_weight * market_depth
            + self.monopoly_competition_weight * monopoly_competition
        )
        return float(reward)




    
    def render(self):
        if self.render_mode == "human":
            print("\n" + "=" * 80)
            print(f"{'DYMONOPOLY MARKET STATUS':^80}")
            print("=" * 80)
            
            # Turn info
            print(f"\n🎲 Turn: {self.turn_counter}")
            print(f"{'✅ PRICES UPDATED THIS TURN!' if self.turn_counter % self.price_update_interval == 0 else '⏳ Next price update in ' + str(self.price_update_interval - (self.turn_counter % self.price_update_interval)) + ' turns'}")
            
            # Player status
            print(f"\n{'PLAYER STATUS':^80}")
            print("-" * 80)
            print(f"{'Player':<10} {'Cash':<15} {'Properties':<15} {'Position':<15} {'Status':<15}")
            print("-" * 80)
            
            for i in range(self.num_players):
                cash = self.player_cash[i]
                owned_props = np.sum(self.property_owners == i)
                position = self.player_positions[i] if hasattr(self, 'player_positions') else 0
                status = "💀 Bankrupt" if self.players[i][0]["bankrupt"] else "✅ Active"
                current = "👉 " if i == self.current_player else "   "
                
                print(f"{current}Player {i+1:<3} ${cash:<14.2f} {owned_props:<15} {position:<15} {status:<15}")
            
            # Market statistics
            print(f"\n{'MARKET STATISTICS':^80}")
            print("-" * 80)
            
            # Calculate stats
            ownable_mask = self.base_prices > 0
            avg_price = np.mean(self.current_prices[ownable_mask])
            avg_base = np.mean(self.base_prices[ownable_mask])
            price_change_pct = ((avg_price / avg_base) - 1) * 100
            
            total_trades = int(np.sum(self.trading_freq))
            total_visits = int(np.sum(self.visiting_freq))
            owned_properties = int(np.sum(self.property_owners >= 0))
            total_ownable = int(np.sum(ownable_mask))
            ownership_pct = (owned_properties / total_ownable * 100) if total_ownable > 0 else 0
            
            print(f"📊 Average Property Price: ${avg_price:.2f} (Base: ${avg_base:.2f})")
            print(f"📈 Market Change: {price_change_pct:+.2f}%")
            print(f"💰 Total Market Cash: ${np.sum(self.player_cash):.2f}")
            print(f"🏠 Properties Owned: {owned_properties}/{total_ownable} ({ownership_pct:.1f}%)")
            print(f"🔄 Total Trades: {total_trades}")
            print(f"👣 Total Visits: {total_visits}")
            
            # Property sample (show top 5 most active)
            print(f"\n{'TOP 5 MOST ACTIVE PROPERTIES':^80}")
            print("-" * 80)
            print(f"{'Property':<25} {'Current':<12} {'Base':<12} {'Change':<12} {'Visits':<10} {'Owner':<10}")
            print("-" * 80)
            
            # Get indices of top 5 most visited properties
            visit_indices = np.argsort(self.visiting_freq)[-5:][::-1]
            
            for idx in visit_indices:
                if self.base_prices[idx] == 0:  # Skip non-ownable
                    continue
                    
                prop_name = self.properties[idx].get("name", f"Property {idx}")
                current_price = self.current_prices[idx]
                base_price = self.base_prices[idx]
                change_pct = ((current_price / base_price) - 1) * 100 if base_price > 0 else 0
                visits = int(self.visiting_freq[idx])
                owner = f"P{int(self.property_owners[idx])+1}" if self.property_owners[idx] >= 0 else "Unowned"
                
                print(f"{prop_name:<25} ${current_price:<11.2f} ${base_price:<11.2f} {change_pct:+.1f}%{' '*7} {visits:<10} {owner:<10}")
            
            # Reward components (if price was just updated)
            if self.turn_counter % self.price_update_interval == 0:
                print(f"\n{'REWARD BREAKDOWN (THIS UPDATE)':^80}")
                print("-" * 80)
                
                # Recalculate components
                avg_trades = np.mean(self.trading_freq)
                market_activity = np.tanh(avg_trades)
                
                safe_base = np.where(self.base_prices <= 0, 1.0, self.base_prices)
                price_delta = np.abs(self.current_prices - self.base_prices) / safe_base
                volatility_penalty = np.tanh(np.mean(price_delta))
                
                market_depth = self._market_depth_reward()
                monopoly_competition = self._get_similar_property_factor()
                
                total_reward = self.reward_function()
                
                print(f"💧 Liquidity (Market Activity):     {market_activity:+.4f} × {self.market_activity_weight} = {market_activity * self.market_activity_weight:+.4f}")
                print(f"📉 Volatility Penalty:             {-volatility_penalty:+.4f} × {self.volatility_weight} = {-volatility_penalty * self.volatility_weight:+.4f}")
                print(f"🏪 Market Depth:                   {market_depth:+.4f} × {self.market_depth_weight} = {market_depth * self.market_depth_weight:+.4f}")
                print(f"⚔️  Competition Factor:             {monopoly_competition:+.4f} × {self.monopoly_competition_weight} = {monopoly_competition * self.monopoly_competition_weight:+.4f}")
                print("-" * 80)
                print(f"{'TOTAL REWARD:':<35} {total_reward:+.4f}")
            
            print("=" * 80 + "\n")