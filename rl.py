import gymnasium as gym
from gymnasium import spaces
import numpy as np

class DymonopolyEnv(gym.Env):
    metadata = {"render_modes": ["human"]}
    
    def __init__(self, num_properties=40, num_players=4):
        super().__init__()
        
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
        
        # Market state tracking
        self.visiting_freq = np.zeros(num_properties)
        self.trading_freq = np.zeros(num_properties)
        self.base_prices = np.random.uniform(100, 500, num_properties)
        self.current_prices = self.base_prices.copy()
        
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