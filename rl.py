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


        self.num_properties = num_properties
        self.num_players = num_players
        self.turn_counter = 0
        self.price_update_interval = 5
        
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
        self.action_space = spaces.Discrete(4)


        
    def _get_obs(self):
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
    
    def _update_prices(self):
        """Update prices every 5 turns based on market factors"""
        for prop_id in range(self.num_properties):
            # i) Visiting frequency: more visits = higher price
            visit_factor = 1 + (self.visiting_freq[prop_id] * 0.05)
            
            # ii) Similar properties: average price of property group
            similar_factor = self._get_similar_property_factor(prop_id)
            
            # iii) Cash liquidity: total market cash
            liquidity_factor = 1 + (np.sum(self.player_cash) / 10000 - 1) * 0.1
            
            # iv) Trading frequency: more trades = more volatility
            trade_factor = 1 + (self.trading_freq[prop_id] * 0.03)
            
            # v) Shock events: random events (can be expanded)
            shock_factor = np.random.choice([0.8, 1.0, 1.2], p=[0.1, 0.8, 0.1])
            
            # Update price
            self.current_prices[prop_id] = (
                self.base_prices[prop_id] * 
                visit_factor * similar_factor * liquidity_factor * 
                trade_factor * shock_factor
            )
    
    def _get_similar_property_factor(self, prop_id):
        """Calculate price influence from similar properties (same color group)"""
        # Implement grouping logic based on your game
        return 1.0
    
    def step(self, action):
        # Execute action, update game state
        self.turn_counter += 1
        
        # Update visiting frequency based on player movement
        # Update trading frequency on buy/sell actions
        
        # Update prices every 5 turns
        if self.turn_counter % self.price_update_interval == 0:
            self._update_prices()
        
        obs = self._get_obs()
        reward = 0  # Define your reward function
        terminated = False  # Game end condition
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
        self.player_cash = np.full(self.num_players, 1500)
        self.current_player = 0
        
        return self._get_obs(), self._get_info()
    
    def render(self):
        if self.render_mode == "human":
            print(f"Turn: {self.turn_counter}")
            print(f"Prices updated: {self.turn_counter % self.price_update_interval == 0}")
            # For terminal-based rendering
            for i in range(min(5, self.num_properties)):
                print(f"Property {i}: ${self.current_prices[i]:.2f} (Base: ${self.base_prices[i]:.2f})")