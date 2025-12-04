import copy
import random
from collections import deque
from typing import Dict, List, Optional, Tuple
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

BASE_BOARD_TEMPLATE = [
    {"name": "GO", "type": "corner", "corner": "go"},

    {"name": "Old Kent Road", "type": "property", "color": "brown",
    "price": 60, "owned_by": -1,
    "rent": {"base": 2, "1h": 10, "2h": 30, "3h": 90, "4h": 160, "hotel": 250},
    "visiting_freq": 0, "trading_frequency": 0, "house_cost": 50},

    {"name": "Community Chest", "type": "chest"},

    {"name": "Whitechapel Road", "type": "property", "color": "brown",
    "price": 60, "owned_by": -1,
    "rent": {"base": 4, "1h": 20, "2h": 60, "3h": 180, "4h": 320, "hotel": 450},
    "visiting_freq": 0, "trading_frequency": 0, "house_cost": 50},

    {"name": "Income Tax", "type": "tax", "price": 200},

    {"name": "Kings Cross Station", "type": "railroad", "price": 200, "owned_by": -1,
    "rent": {"1_rr": 25, "2_rr": 50, "3_rr": 100, "4_rr": 200},
    "visiting_freq": 0, "trading_frequency": 0},

    {"name": "The Angel, Islington", "type": "property", "color": "light_blue",
    "price": 100, "owned_by": -1,
    "rent": {"base": 6, "1h": 30, "2h": 90, "3h": 270, "4h": 400, "hotel": 550},
    "visiting_freq": 0, "trading_frequency": 0, "house_cost": 50},

    {"name": "Chance", "type": "chance"},

    {"name": "Euston Road", "type": "property", "color": "light_blue",
    "price": 100, "owned_by": -1,
    "rent": {"base": 6, "1h": 30, "2h": 90, "3h": 270, "4h": 400, "hotel": 550},
    "visiting_freq": 0, "trading_frequency": 0, "house_cost": 50},

    {"name": "Pentonville Road", "type": "property", "color": "light_blue",
    "price": 120, "owned_by": -1,
    "rent": {"base": 8, "1h": 40, "2h": 100, "3h": 300, "4h": 450, "hotel": 600},
    "visiting_freq": 0, "trading_frequency": 0, "house_cost": 50},

    {"name": "In Jail / Just Visiting", "type": "corner", "corner": "jail"},

    {"name": "Pall Mall", "type": "property", "color": "pink",
    "price": 140, "owned_by": -1,
    "rent": {"base": 10, "1h": 50, "2h": 150, "3h": 450, "4h": 625, "hotel": 750},
    "visiting_freq": 0, "trading_frequency": 0, "house_cost": 100},

    {"name": "Electric Company", "type": "utility", "price": 150, "owned_by": -1,
    "rent": {"one_util": "4x dice", "both_utils": "10x dice"},
    "visiting_freq": 0, "trading_frequency": 0},

    {"name": "Whitehall", "type": "property", "color": "pink",
    "price": 140, "owned_by": -1,
    "rent": {"base": 10, "1h": 50, "2h": 150, "3h": 450, "4h": 625, "hotel": 750},
    "visiting_freq": 0, "trading_frequency": 0, "house_cost": 100},

    {"name": "Northumberland Avenue", "type": "property", "color": "pink",
    "price": 160, "owned_by": -1,
    "rent": {"base": 12, "1h": 60, "2h": 180, "3h": 500, "4h": 700, "hotel": 900},
    "visiting_freq": 0, "trading_frequency": 0, "house_cost": 100},

    {"name": "Marylebone Station", "type": "railroad", "price": 200, "owned_by": -1,
    "rent": {"1_rr": 25, "2_rr": 50, "3_rr": 100, "4_rr": 200},
    "visiting_freq": 0, "trading_frequency": 0},

    {"name": "Bow Street", "type": "property", "color": "orange",
    "price": 180, "owned_by": -1,
    "rent": {"base": 14, "1h": 70, "2h": 200, "3h": 550, "4h": 750, "hotel": 950},
    "visiting_freq": 0, "trading_frequency": 0, "house_cost": 100},

    {"name": "Community Chest", "type": "chest"},

    {"name": "Marlborough Street", "type": "property", "color": "orange",
    "price": 180, "owned_by": -1,
    "rent": {"base": 14, "1h": 70, "2h": 200, "3h": 550, "4h": 750, "hotel": 950},
    "visiting_freq": 0, "trading_frequency": 0, "house_cost": 100},

    {"name": "Vine Street", "type": "property", "color": "orange",
    "price": 200, "owned_by": -1,
    "rent": {"base": 16, "1h": 80, "2h": 220, "3h": 600, "4h": 800, "hotel": 1000},
    "visiting_freq": 0, "trading_frequency": 0, "house_cost": 100},

    {"name": "Free Parking", "type": "corner", "corner": "free"},

    {"name": "Strand", "type": "property", "color": "red",
    "price": 220, "owned_by": -1,
    "rent": {"base": 18, "1h": 90, "2h": 250, "3h": 700, "4h": 875, "hotel": 1050},
    "visiting_freq": 0, "trading_frequency": 0, "house_cost": 150},

    {"name": "Chance", "type": "chance"},

    {"name": "Fleet Street", "type": "property", "color": "red",
    "price": 220, "owned_by": -1,
    "rent": {"base": 18, "1h": 90, "2h": 250, "3h": 700, "4h": 875, "hotel": 1050},
    "visiting_freq": 0, "trading_frequency": 0, "house_cost": 150},

    {"name": "Trafalgar Square", "type": "property", "color": "red",
    "price": 240, "owned_by": -1,
    "rent": {"base": 20, "1h": 100, "2h": 300, "3h": 750, "4h": 925, "hotel": 1100},
    "visiting_freq": 0, "trading_frequency": 0, "house_cost": 150},

    {"name": "Fenchurch St. Station", "type": "railroad", "price": 200, "owned_by": -1,
    "rent": {"1_rr": 25, "2_rr": 50, "3_rr": 100, "4_rr": 200},
    "visiting_freq": 0, "trading_frequency": 0},

    {"name": "Leicester Square", "type": "property", "color": "yellow",
    "price": 260, "owned_by": -1,
    "rent": {"base": 22, "1h": 110, "2h": 330, "3h": 800, "4h": 975, "hotel": 1150},
    "visiting_freq": 0, "trading_frequency": 0, "house_cost": 150},

    {"name": "Coventry Street", "type": "property", "color": "yellow",
    "price": 260, "owned_by": -1,
    "rent": {"base": 22, "1h": 110, "2h": 330, "3h": 800, "4h": 975, "hotel": 1150},
    "visiting_freq": 0, "trading_frequency": 0, "house_cost": 150},

    {"name": "Water Works", "type": "utility", "price": 150, "owned_by": -1,
    "rent": {"one_util": "4x dice", "both_utils": "10x dice"},
    "visiting_freq": 0, "trading_frequency": 0},

    {"name": "Piccadilly", "type": "property", "color": "yellow",
    "price": 280, "owned_by": -1,
    "rent": {"base": 24, "1h": 120, "2h": 360, "3h": 850, "4h": 1025, "hotel": 1200},
    "visiting_freq": 0, "trading_frequency": 0, "house_cost": 150},

    {"name": "Go To Jail", "type": "corner", "corner": "goto"},

    {"name": "Regent Street", "type": "property", "color": "green",
    "price": 300, "owned_by": -1,
    "rent": {"base": 26, "1h": 130, "2h": 390, "3h": 900, "4h": 1100, "hotel": 1275},
    "visiting_freq": 0, "trading_frequency": 0, "house_cost": 200},

    {"name": "Oxford Street", "type": "property", "color": "green",
    "price": 300, "owned_by": -1,
    "rent": {"base": 26, "1h": 130, "2h": 390, "3h": 900, "4h": 1100, "hotel": 1275},
    "visiting_freq": 0, "trading_frequency": 0, "house_cost": 200},

    {"name": "Community Chest", "type": "chest"},

    {"name": "Bond Street", "type": "property", "color": "green",
    "price": 320, "owned_by": -1,
    "rent": {"base": 28, "1h": 150, "2h": 450, "3h": 1000, "4h": 1200, "hotel": 1400},
    "visiting_freq": 0, "trading_frequency": 0, "house_cost": 200},

    {"name": "Liverpool St. Station", "type": "railroad", "price": 200, "owned_by": -1,
    "rent": {"1_rr": 25, "2_rr": 50, "3_rr": 100, "4_rr": 200},
    "visiting_freq": 0, "trading_frequency": 0},

    {"name": "Chance", "type": "chance"},

    {"name": "Park Lane", "type": "property", "color": "dark_blue",
    "price": 350, "owned_by": -1,
    "rent": {"base": 35, "1h": 175, "2h": 500, "3h": 1100, "4h": 1300, "hotel": 1500},
    "visiting_freq": 0, "trading_frequency": 0, "house_cost": 200},

    {"name": "Super Tax", "type": "tax", "price": 100},

    {"name": "Mayfair", "type": "property", "color": "dark_blue",
    "price": 400, "owned_by": -1,
    "rent": {"base": 50, "1h": 200, "2h": 600, "3h": 1400, "4h": 1700, "hotel": 2000},
    "visiting_freq": 0, "trading_frequency": 0, "house_cost": 200}
]

def build_classic_board() -> List[Dict]:
    """Return a deep copy of the standard Monopoly board description."""
    return copy.deepcopy(BASE_BOARD_TEMPLATE)

class DymonopolyEnv(gym.Env):
    metadata = {"render_modes": ["human"]}
    
    def __init__(self, num_properties=40, num_players=4):
        super().__init__()
        
        #stuff i m adding, rough
        

        self.properties = [
    {"name": "GO", "type": "corner", "corner": "go"},

    {"name": "Old Kent Road", "type": "property", "color": "brown",
    "price": 60, "owned_by": -1,
    "rent": {"base": 2, "1h": 10, "2h": 30, "3h": 90, "4h": 160, "hotel": 250},
    "visiting_freq": 0, "trading_frequency": 0, "house_cost": 50},

    {"name": "Community Chest", "type": "chest"},

    {"name": "Whitechapel Road", "type": "property", "color": "brown",
    "price": 60, "owned_by": -1,
    "rent": {"base": 4, "1h": 20, "2h": 60, "3h": 180, "4h": 320, "hotel": 450},
    "visiting_freq": 0, "trading_frequency": 0, "house_cost": 50},

    {"name": "Income Tax", "type": "tax", "price": 200},

    {"name": "Kings Cross Station", "type": "railroad", "price": 200, "owned_by": -1,
    "rent": {"1_rr": 25, "2_rr": 50, "3_rr": 100, "4_rr": 200},
    "visiting_freq": 0, "trading_frequency": 0},

    {"name": "The Angel, Islington", "type": "property", "color": "light_blue",
    "price": 100, "owned_by": -1,
    "rent": {"base": 6, "1h": 30, "2h": 90, "3h": 270, "4h": 400, "hotel": 550},
    "visiting_freq": 0, "trading_frequency": 0, "house_cost": 50},

    {"name": "Chance", "type": "chance"},

    {"name": "Euston Road", "type": "property", "color": "light_blue",
    "price": 100, "owned_by": -1,
    "rent": {"base": 6, "1h": 30, "2h": 90, "3h": 270, "4h": 400, "hotel": 550},
    "visiting_freq": 0, "trading_frequency": 0, "house_cost": 50},

    {"name": "Pentonville Road", "type": "property", "color": "light_blue",
    "price": 120, "owned_by": -1,
    "rent": {"base": 8, "1h": 40, "2h": 100, "3h": 300, "4h": 450, "hotel": 600},
    "visiting_freq": 0, "trading_frequency": 0, "house_cost": 50},

    {"name": "In Jail / Just Visiting", "type": "corner", "corner": "jail"},

    {"name": "Pall Mall", "type": "property", "color": "pink",
    "price": 140, "owned_by": -1,
    "rent": {"base": 10, "1h": 50, "2h": 150, "3h": 450, "4h": 625, "hotel": 750},
    "visiting_freq": 0, "trading_frequency": 0, "house_cost": 100},

    {"name": "Electric Company", "type": "utility", "price": 150, "owned_by": -1,
    "rent": {"one_util": "4x dice", "both_utils": "10x dice"},
    "visiting_freq": 0, "trading_frequency": 0},

    {"name": "Whitehall", "type": "property", "color": "pink",
    "price": 140, "owned_by": -1,
    "rent": {"base": 10, "1h": 50, "2h": 150, "3h": 450, "4h": 625, "hotel": 750},
    "visiting_freq": 0, "trading_frequency": 0, "house_cost": 100},

    {"name": "Northumberland Avenue", "type": "property", "color": "pink",
    "price": 160, "owned_by": -1,
    "rent": {"base": 12, "1h": 60, "2h": 180, "3h": 500, "4h": 700, "hotel": 900},
    "visiting_freq": 0, "trading_frequency": 0, "house_cost": 100},

    {"name": "Marylebone Station", "type": "railroad", "price": 200, "owned_by": -1,
    "rent": {"1_rr": 25, "2_rr": 50, "3_rr": 100, "4_rr": 200},
    "visiting_freq": 0, "trading_frequency": 0},

    {"name": "Bow Street", "type": "property", "color": "orange",
    "price": 180, "owned_by": -1,
    "rent": {"base": 14, "1h": 70, "2h": 200, "3h": 550, "4h": 750, "hotel": 950},
    "visiting_freq": 0, "trading_frequency": 0, "house_cost": 100},

    {"name": "Community Chest", "type": "chest"},

    {"name": "Marlborough Street", "type": "property", "color": "orange",
    "price": 180, "owned_by": -1,
    "rent": {"base": 14, "1h": 70, "2h": 200, "3h": 550, "4h": 750, "hotel": 950},
    "visiting_freq": 0, "trading_frequency": 0, "house_cost": 100},

    {"name": "Vine Street", "type": "property", "color": "orange",
    "price": 200, "owned_by": -1,
    "rent": {"base": 16, "1h": 80, "2h": 220, "3h": 600, "4h": 800, "hotel": 1000},
    "visiting_freq": 0, "trading_frequency": 0, "house_cost": 100},

    {"name": "Free Parking", "type": "corner", "corner": "free"},

    {"name": "Strand", "type": "property", "color": "red",
    "price": 220, "owned_by": -1,
    "rent": {"base": 18, "1h": 90, "2h": 250, "3h": 700, "4h": 875, "hotel": 1050},
    "visiting_freq": 0, "trading_frequency": 0, "house_cost": 150},

    {"name": "Chance", "type": "chance"},

    {"name": "Fleet Street", "type": "property", "color": "red",
    "price": 220, "owned_by": -1,
    "rent": {"base": 18, "1h": 90, "2h": 250, "3h": 700, "4h": 875, "hotel": 1050},
    "visiting_freq": 0, "trading_frequency": 0, "house_cost": 150},

    {"name": "Trafalgar Square", "type": "property", "color": "red",
    "price": 240, "owned_by": -1,
    "rent": {"base": 20, "1h": 100, "2h": 300, "3h": 750, "4h": 925, "hotel": 1100},
    "visiting_freq": 0, "trading_frequency": 0, "house_cost": 150},

    {"name": "Fenchurch St. Station", "type": "railroad", "price": 200, "owned_by": -1,
    "rent": {"1_rr": 25, "2_rr": 50, "3_rr": 100, "4_rr": 200},
    "visiting_freq": 0, "trading_frequency": 0},

    {"name": "Leicester Square", "type": "property", "color": "yellow",
    "price": 260, "owned_by": -1,
    "rent": {"base": 22, "1h": 110, "2h": 330, "3h": 800, "4h": 975, "hotel": 1150},
    "visiting_freq": 0, "trading_frequency": 0, "house_cost": 150},

    {"name": "Coventry Street", "type": "property", "color": "yellow",
    "price": 260, "owned_by": -1,
    "rent": {"base": 22, "1h": 110, "2h": 330, "3h": 800, "4h": 975, "hotel": 1150},
    "visiting_freq": 0, "trading_frequency": 0, "house_cost": 150},

    {"name": "Water Works", "type": "utility", "price": 150, "owned_by": -1,
    "rent": {"one_util": "4x dice", "both_utils": "10x dice"},
    "visiting_freq": 0, "trading_frequency": 0},

    {"name": "Piccadilly", "type": "property", "color": "yellow",
    "price": 280, "owned_by": -1,
    "rent": {"base": 24, "1h": 120, "2h": 360, "3h": 850, "4h": 1025, "hotel": 1200},
    "visiting_freq": 0, "trading_frequency": 0, "house_cost": 150},

    {"name": "Go To Jail", "type": "corner", "corner": "goto"},

    {"name": "Regent Street", "type": "property", "color": "green",
    "price": 300, "owned_by": -1,
    "rent": {"base": 26, "1h": 130, "2h": 390, "3h": 900, "4h": 1100, "hotel": 1275},
    "visiting_freq": 0, "trading_frequency": 0, "house_cost": 200},

    {"name": "Oxford Street", "type": "property", "color": "green",
    "price": 300, "owned_by": -1,
    "rent": {"base": 26, "1h": 130, "2h": 390, "3h": 900, "4h": 1100, "hotel": 1275},
    "visiting_freq": 0, "trading_frequency": 0, "house_cost": 200},

    {"name": "Community Chest", "type": "chest"},

    {"name": "Bond Street", "type": "property", "color": "green",
    "price": 320, "owned_by": -1,
    "rent": {"base": 28, "1h": 150, "2h": 450, "3h": 1000, "4h": 1200, "hotel": 1400},
    "visiting_freq": 0, "trading_frequency": 0, "house_cost": 200},

    {"name": "Liverpool St. Station", "type": "railroad", "price": 200, "owned_by": -1,
    "rent": {"1_rr": 25, "2_rr": 50, "3_rr": 100, "4_rr": 200},
    "visiting_freq": 0, "trading_frequency": 0},

    {"name": "Chance", "type": "chance"},

    {"name": "Park Lane", "type": "property", "color": "dark_blue",
    "price": 350, "owned_by": -1,
    "rent": {"base": 35, "1h": 175, "2h": 500, "3h": 1100, "4h": 1300, "hotel": 1500},
    "visiting_freq": 0, "trading_frequency": 0, "house_cost": 200},

    {"name": "Super Tax", "type": "tax", "price": 100},

    {"name": "Mayfair", "type": "property", "color": "dark_blue",
    "price": 400, "owned_by": -1,
    "rent": {"base": 50, "1h": 200, "2h": 600, "3h": 1400, "4h": 1700, "hotel": 2000},
    "visiting_freq": 0, "trading_frequency": 0, "house_cost": 200}
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
            "property_prices": spaces.Box(low=0, high=1e6, shape=(num_properties,), dtype=np.float32),
            "property_owners": spaces.Box(low=-1, high=num_players-1, shape=(num_properties,), dtype=np.float32),
            "visiting_frequency": spaces.Box(low=0, high=1e6, shape=(num_properties,), dtype=np.float32),
            "trading_frequency": spaces.Box(low=0, high=1e6, shape=(num_properties,), dtype=np.float32),
            "player_cash": spaces.Box(low=0, high=1e8, shape=(num_players,), dtype=np.float32),
            "current_player": spaces.Discrete(num_players)
        })
        
        # Action space: normalized to [-1, 1] for SAC stability
        # Will be rescaled in step() to [0.4, 2.0] price multipliers
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(num_properties,),
            dtype=np.float32
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
        Action: array in [-1, 1] rescaled to price multipliers [0.4, 2.0]
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
                
                # Rescale action from [-1, 1] to [0.4, 2.0]
                # Formula: multiplier = 1.2 + 0.8 * action  (-1 -> 0.4, 0 -> 1.2, +1 -> 2.0)
                raw_action = np.clip(action[prop_id], -1.0, 1.0)
                multiplier = 1.2 + 0.8 * raw_action
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
        # End after max turns to prevent infinite episodes
        MAX_TURNS = 500
        if self.turn_counter >= MAX_TURNS:
            return True
        
        bankrupt_players = 0
        for i in range(self.num_players):
            if self.players[i][0]["bankrupt"] == True:  # Fixed indexing
                bankrupt_players += 1
    
        # Game ends if 3+ players are bankrupt (only 1 or 0 active)
        if bankrupt_players >= self.num_players - 1:
            return True
        
        return False  # Game continues

    
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


class DymonopolyDecisionEnv(gym.Env):
    """Environment that exposes only the strategic Monopoly decisions to a DQN agent."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        num_players: int = 4,
        starting_cash: int = 1500,
        max_turns: int = 200,
        dynamic_price_fn: Optional[callable] = None,
    ):
        super().__init__()

        self.properties = build_classic_board()
        self.board_size = len(self.properties)
        self.num_players = num_players
        self.starting_cash = starting_cash
        self.max_turns = max_turns
        self.dynamic_price_fn = dynamic_price_fn

        self.jail_position = next(
            idx for idx, tile in enumerate(self.properties) if tile.get("corner") == "jail"
        )
        self.go_to_jail_position = next(
            idx for idx, tile in enumerate(self.properties) if tile.get("corner") == "goto"
        )
        self.go_reward = 200

        self.base_action_meanings = [
            "end_turn",
            "buy_property",
            "pay_jail_fine",
            "use_get_out_of_jail_card",
            "accept_jail",
        ]
        self.END_TURN = 0
        self.BUY_PROPERTY = 1
        self.PAY_JAIL_FINE = 2
        self.USE_JAIL_CARD = 3
        self.ACCEPT_JAIL = 4

        self.build_offset = len(self.base_action_meanings)
        self.sell_building_offset = self.build_offset + self.board_size
        self.sell_property_offset = self.sell_building_offset + self.board_size
        total_actions = self.sell_property_offset + self.board_size
        self.action_space = spaces.Discrete(total_actions)

        obs_high_cash = 1e6
        self.observation_space = spaces.Dict(
            {
                "player_cash": spaces.Box(
                    low=0,
                    high=obs_high_cash,
                    shape=(self.num_players,),
                    dtype=np.float32,
                ),
                "player_positions": spaces.Box(
                    low=0,
                    high=self.board_size - 1,
                    shape=(self.num_players,),
                    dtype=np.int32,
                ),
                "player_jail": spaces.MultiBinary(self.num_players),
                "player_jail_cards": spaces.Box(
                    low=0,
                    high=2,
                    shape=(self.num_players,),
                    dtype=np.int32,
                ),
                "property_owner": spaces.Box(
                    low=-1,
                    high=self.num_players - 1,
                    shape=(self.board_size,),
                    dtype=np.int32,
                ),
                
                "property_houses": spaces.Box(
                    low=0,
                    high=5,
                    shape=(self.board_size,),
                    dtype=np.int32,
                ),
                "current_player": spaces.Discrete(self.num_players),
            }
        )

        self.flat_observation_size = (
            self.num_players * 4 + self.board_size * 2 + 1
        )  # cash, pos, jail, cards, owners, houses, current player index

        self.invalid_action_penalty = -2.0
        self.networth_reward_scale = 0.01
        self.survival_reward = 0.05
        self.cash_norm = float(self.starting_cash * 2)
       
        # Chance and Community Chest decks (extend as needed)
        self.chance_cards = [
            {"type": "money", "amount": 200, "text": "Bank error in your favour. Collect £200."},
            {"type": "money", "amount": -100, "text": "Pay speeding fine of £100."},
            {"type": "move", "target": 0, "text": "Advance to Go. Collect £200."},
            {"type": "go_to_jail", "text": "Go directly to Jail. Do not pass Go, do not collect £200."},
            {"type": "jail_card", "text": "Get Out of Jail Free. This card may be kept until needed or sold."},
        ]

        self.chest_cards = [
            {"type": "money", "amount": 150, "text": "Your building loan matures. Collect £150."},
            {"type": "money", "amount": -50, "text": "Doctor's fees. Pay £50."},
            {"type": "money", "amount": 100, "text": "Life insurance matures. Collect £100."},
            {"type": "go_to_jail", "text": "Go directly to Jail. Do not pass Go, do not collect £200."},
            {"type": "jail_card", "text": "Get Out of Jail Free. This card may be kept until needed or sold."},
        ]

        self._init_color_groups()
        self._reset_game_state()

    def _init_color_groups(self):
        self.color_groups: Dict[str, List[int]] = {}
        for idx, tile in enumerate(self.properties):
            if tile.get("type") == "property":
                color = tile.get("color")
                if not color:
                    continue
                if color not in self.color_groups:
                    self.color_groups[color] = []
                self.color_groups[color].append(idx)

    def _reset_game_state(self):
        self.player_positions = np.zeros(self.num_players, dtype=np.int32)
        self.player_cash = np.full(self.num_players, float(self.starting_cash), dtype=np.float32)
        self.player_in_jail = np.zeros(self.num_players, dtype=bool)
        self.jail_turns_remaining = np.zeros(self.num_players, dtype=np.int32)
        self.player_jail_cards = np.zeros(self.num_players, dtype=np.int32)
        self.player_bankrupt = np.zeros(self.num_players, dtype=bool)

        self.property_owner = np.full(self.board_size, -1, dtype=np.int32)
        self.property_houses = np.zeros(self.board_size, dtype=np.int32)

        self.current_player = 0
        self.turn_count = 0
        self.turn_reward_buffer = 0.0
        self.pending_context: Optional[Dict] = None
        self.pending_mask: Optional[np.ndarray] = None
        self.pending_jail_event: Optional[Dict] = None
        self.agent_last_worth = self._player_net_worth(0)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        self._reset_game_state()
        obs = self._prepare_agent_context()
        info = {"action_mask": self.pending_mask.copy(), "context": self.pending_context}
        return obs, info

    def _get_obs(self) -> Dict:
        return {
            "player_cash": self.player_cash.astype(np.float32),
            "player_positions": self.player_positions.astype(np.int32),
            "player_jail": self.player_in_jail.astype(np.int32),
            "player_jail_cards": self.player_jail_cards.astype(np.int32),
            "property_owner": self.property_owner.astype(np.int32),
            "property_houses": self.property_houses.astype(np.int32),
            "current_player": int(self.current_player),
        }

    def flatten_observation(self, obs: Dict) -> np.ndarray:
        cash = np.asarray(obs["player_cash"], dtype=np.float32) / max(self.cash_norm, 1.0)
        pos = np.asarray(obs["player_positions"], dtype=np.float32) / max(self.board_size - 1, 1)
        jail = np.asarray(obs["player_jail"], dtype=np.float32)
        jail_cards = np.clip(np.asarray(obs["player_jail_cards"], dtype=np.float32), 0, 2) / 2.0
        owners = (np.asarray(obs["property_owner"], dtype=np.float32) + 1.0) / max(self.num_players, 1)
        houses = np.asarray(obs["property_houses"], dtype=np.float32) / 5.0
        current = np.array([float(obs["current_player"]) / max(self.num_players - 1, 1)], dtype=np.float32)

        return np.concatenate([
            cash,
            pos,
            jail,
            jail_cards,
            owners,
            houses,
            current,
        ]).astype(np.float32)

    def _prepare_agent_context(self) -> Dict:
        if self.player_bankrupt[0]:
            ctx = {"type": "terminal", "valid_actions": [self.END_TURN]}
            self.pending_context = ctx
            self.pending_mask = self._context_to_mask(ctx)
            return self._get_obs()

        if self.pending_jail_event and self.pending_jail_event.get("player_id") == 0:
            ctx = self._build_jail_entry_context(0)
            self.pending_context = ctx
            self.pending_mask = self._context_to_mask(ctx)
            return self._get_obs()

        if self.player_in_jail[0]:
            if self.jail_turns_remaining[0] > 0:
                ctx = self._build_jail_skip_context(0)
                self.pending_context = ctx
                self.pending_mask = self._context_to_mask(ctx)
                return self._get_obs()
            self.player_in_jail[0] = False
            self.player_positions[0] = self.jail_position

        dice = self._roll_dice()
        passed_go = self._advance_position(0, dice)
        if passed_go:
            self.player_cash[0] += self.go_reward
        tile_idx = self.player_positions[0]
        self.turn_reward_buffer += self._handle_tile_effects(0, tile_idx, dice)

        if self.pending_jail_event and self.pending_jail_event.get("player_id") == 0:
            ctx = self._build_jail_entry_context(0)
            self.pending_context = ctx
            self.pending_mask = self._context_to_mask(ctx)
            return self._get_obs()

        ctx = self._build_decision_context(0, tile_idx)
        self.pending_context = ctx
        self.pending_mask = self._context_to_mask(ctx)
        return self._get_obs()

    def _build_jail_entry_context(self, player_id: int) -> Dict:
        valid_actions = [self.ACCEPT_JAIL]
        if self.player_cash[player_id] >= 50:
            valid_actions.append(self.PAY_JAIL_FINE)
        if self.player_jail_cards[player_id] > 0:
            valid_actions.append(self.USE_JAIL_CARD)
        return {
            "type": "jail_entry",
            "player": player_id,
            "valid_actions": valid_actions,
        }

    def _build_jail_skip_context(self, player_id: int) -> Dict:
        return {
            "type": "jail_skip",
            "player": player_id,
            "valid_actions": [self.END_TURN],
        }

    def _build_decision_context(self, player_id: int, tile_idx: int) -> Dict:
        context = {
            "type": "decision",
            "player": player_id,
            "buy_target": None,
            "buy_price": None,
            "build_targets": [],
            "sell_building_targets": [],
            "sell_property_targets": [],
            "valid_actions": [self.END_TURN],
        }

        tile = self.properties[tile_idx]
        if tile.get("type") in {"property", "railroad", "utility"}:
            owner = self.property_owner[tile_idx]
            price = self._get_property_price(tile_idx)
            if owner == -1 and self.player_cash[player_id] >= price:
                context["buy_target"] = tile_idx
                context["buy_price"] = price
                context["valid_actions"].append(self.BUY_PROPERTY)

        buildable = self._list_buildable_properties(player_id)
        if buildable:
            context["build_targets"] = buildable
            for prop_id in buildable:
                context["valid_actions"].append(self.build_offset + prop_id)

        sell_buildings = self._list_sell_building_candidates(player_id)
        if sell_buildings:
            context["sell_building_targets"] = sell_buildings
            for prop_id in sell_buildings:
                context["valid_actions"].append(self.sell_building_offset + prop_id)

        sell_properties = self._list_sell_property_candidates(player_id)
        if sell_properties:
            context["sell_property_targets"] = sell_properties
            for prop_id in sell_properties:
                context["valid_actions"].append(self.sell_property_offset + prop_id)

        return context

    def _list_buildable_properties(self, player_id: int) -> List[int]:
        buildable = []
        for prop_id in range(self.board_size):
            if self.property_owner[prop_id] != player_id:
                continue
            tile = self.properties[prop_id]
            if tile.get("type") != "property":
                continue
            color = tile.get("color")
            if not self._owns_full_color_set(player_id, color):
                continue
            if self.property_houses[prop_id] >= 5:
                continue
            cost = tile.get("house_cost", 0)
            if cost <= 0 or self.player_cash[player_id] < cost:
                continue
            buildable.append(prop_id)
        return buildable

    def _list_sell_building_candidates(self, player_id: int) -> List[int]:
        candidates = []
        for prop_id in range(self.board_size):
            if self.property_owner[prop_id] != player_id:
                continue
            if self.property_houses[prop_id] > 0:
                candidates.append(prop_id)
        return candidates

    def _list_sell_property_candidates(self, player_id: int) -> List[int]:
        candidates = []
        for prop_id in range(self.board_size):
            if self.property_owner[prop_id] != player_id:
                continue
            tile = self.properties[prop_id]
            if tile.get("type") not in {"property", "railroad", "utility"}:
                continue
            candidates.append(prop_id)
        return candidates

    def _context_to_mask(self, ctx: Dict) -> np.ndarray:
        mask = np.zeros(self.action_space.n, dtype=np.float32)
        for action_idx in ctx.get("valid_actions", []):
            if 0 <= action_idx < self.action_space.n:
                mask[action_idx] = 1.0

        for prop_id in ctx.get("build_targets", []):
            idx = self.build_offset + prop_id
            if 0 <= idx < self.action_space.n:
                mask[idx] = 1.0

        for prop_id in ctx.get("sell_building_targets", []):
            idx = self.sell_building_offset + prop_id
            if 0 <= idx < self.action_space.n:
                mask[idx] = 1.0

        for prop_id in ctx.get("sell_property_targets", []):
            idx = self.sell_property_offset + prop_id
            if 0 <= idx < self.action_space.n:
                mask[idx] = 1.0
        return mask

    def step(self, action: int):
        if self.pending_context is None:
            self._prepare_agent_context()

        reward = self.turn_reward_buffer
        self.turn_reward_buffer = 0.0
        mask = self.pending_mask
        info = {"action_mask": mask.copy(), "context": self.pending_context}

        if mask[action] == 0:
            reward += self.invalid_action_penalty
            action = self.END_TURN

        pre_worth = self._player_net_worth(0)
        reward += self._apply_action(action)
        self.pending_context = None

        self._check_and_handle_bankruptcy(0)

        if not self.player_bankrupt[0]:
            self._other_players_take_turns()
        post_worth = self._player_net_worth(0)
        reward += (post_worth - pre_worth) * self.networth_reward_scale
        self.agent_last_worth = post_worth
        reward += self.survival_reward

        self.turn_count += 1
        terminated = bool(self.player_bankrupt[0])
        truncated = self.turn_count >= self.max_turns

        if terminated or truncated:
            obs = self._get_obs()
            self.pending_context = {"type": "terminal", "valid_actions": [self.END_TURN]}
            self.pending_mask = self._context_to_mask(self.pending_context)
            info["action_mask"] = self.pending_mask.copy()
        else:
            obs = self._prepare_agent_context()
            info["action_mask"] = self.pending_mask.copy()

        return obs, reward, terminated, truncated, info

    def _apply_action(self, action: int) -> float:
        ctx = self.pending_context or {"type": "decision", "valid_actions": [self.END_TURN]}
        ctx_type = ctx.get("type")

        if ctx_type == "terminal":
            return 0.0

        if ctx_type == "jail_entry":
            player_id = ctx.get("player", 0)
            if action == self.PAY_JAIL_FINE:
                return self._handle_jail_payment(player_id)
            if action == self.USE_JAIL_CARD:
                return self._handle_jail_card_use(player_id)
            if action == self.ACCEPT_JAIL:
                return self._handle_accept_jail(player_id)
            return self.invalid_action_penalty

        if ctx_type == "jail_skip":
            player_id = ctx.get("player", 0)
            if action == self.END_TURN:
                return self._handle_jail_skip(player_id)
            return self.invalid_action_penalty

        if ctx_type == "decision":
            if action == self.END_TURN:
                return 0.0

            if action == self.BUY_PROPERTY and ctx.get("buy_target") is not None:
                return self._handle_buy(ctx["buy_target"], ctx.get("buy_price"))

            if self.build_offset <= action < self.sell_building_offset:
                prop_id = action - self.build_offset
                if prop_id in ctx.get("build_targets", []):
                    return self._handle_build_property(prop_id)
                return self.invalid_action_penalty

            if self.sell_building_offset <= action < self.sell_property_offset:
                prop_id = action - self.sell_building_offset
                if prop_id in ctx.get("sell_building_targets", []):
                    return self._handle_sell_building(prop_id)
                return self.invalid_action_penalty

            if self.sell_property_offset <= action < self.action_space.n:
                prop_id = action - self.sell_property_offset
                if prop_id in ctx.get("sell_property_targets", []):
                    return self._handle_sell_property(prop_id)
                return self.invalid_action_penalty

            return self.invalid_action_penalty

        return 0.0

    def _handle_buy(self, prop_id: Optional[int], price: Optional[float], player_id: int = 0) -> float:
        if prop_id is None:
            return self.invalid_action_penalty
        actual_price = price if price is not None else self._get_property_price(prop_id)
        if actual_price <= 0 or self.player_cash[player_id] < actual_price:
            return self.invalid_action_penalty
        if self.property_owner[prop_id] != -1:
            return self.invalid_action_penalty
        self.player_cash[player_id] -= actual_price
        self.property_owner[prop_id] = player_id
        return 1.0

    def _handle_build_property(self, prop_id: int, player_id: int = 0) -> float:
        if self.property_owner[prop_id] != player_id:
            return self.invalid_action_penalty
        tile = self.properties[prop_id]
        if tile.get("type") != "property":
            return self.invalid_action_penalty
        color = tile.get("color")
        if not self._owns_full_color_set(player_id, color):
            return self.invalid_action_penalty
        if self.property_houses[prop_id] >= 5:
            return self.invalid_action_penalty
        cost = tile.get("house_cost", 0)
        if cost <= 0 or self.player_cash[player_id] < cost:
            return self.invalid_action_penalty
        self.player_cash[player_id] -= cost
        self.property_houses[prop_id] += 1
        return 1.5

    def _handle_sell_building(self, prop_id: int, player_id: int = 0) -> float:
        if self.property_owner[prop_id] != player_id:
            return self.invalid_action_penalty
        if self.property_houses[prop_id] == 0:
            return self.invalid_action_penalty
        tile = self.properties[prop_id]
        refund = tile.get("house_cost", 0) * 0.5
        self.property_houses[prop_id] -= 1
        self.player_cash[player_id] += refund
        return 0.5

    def _handle_sell_property(self, prop_id: int, player_id: int = 0) -> float:
        if self.property_owner[prop_id] != player_id:
            return self.invalid_action_penalty
        tile = self.properties[prop_id]
        if tile.get("type") not in {"property", "railroad", "utility"}:
            return self.invalid_action_penalty
        base_price = self._get_property_price(prop_id)
        refund = base_price * 0.5
        if tile.get("type") == "property" and self.property_houses[prop_id] > 0:
            refund += self.property_houses[prop_id] * tile.get("house_cost", 0) * 0.5
        self.player_cash[player_id] += refund
        self.property_owner[prop_id] = -1
        self.property_houses[prop_id] = 0
        return 0.8

    def _handle_jail_payment(self, player_id: int) -> float:
        if self.player_cash[player_id] < 50:
            return self.invalid_action_penalty
        self.player_cash[player_id] -= 50
        self.player_in_jail[player_id] = False
        self.jail_turns_remaining[player_id] = 0
        self.pending_jail_event = None
        return 0.2

    def _handle_jail_card_use(self, player_id: int) -> float:
        if self.player_jail_cards[player_id] <= 0:
            return self.invalid_action_penalty
        self.player_jail_cards[player_id] -= 1
        self.player_in_jail[player_id] = False
        self.jail_turns_remaining[player_id] = 0
        self.pending_jail_event = None
        return 0.2

    def _handle_accept_jail(self, player_id: int) -> float:
        self._enforce_jail_sentence(player_id)
        self.pending_jail_event = None
        return -0.2

    def _handle_jail_skip(self, player_id: int) -> float:
        if self.jail_turns_remaining[player_id] <= 0:
            return self.invalid_action_penalty
        self.jail_turns_remaining[player_id] -= 1
        if self.jail_turns_remaining[player_id] == 0:
            self.player_in_jail[player_id] = False
            self.player_positions[player_id] = self.jail_position
        return -0.1

    def _enforce_jail_sentence(self, player_id: int):
        self.player_positions[player_id] = self.jail_position
        self.player_in_jail[player_id] = True
        self.jail_turns_remaining[player_id] = 3

    def _trigger_jail_entry(self, player_id: int, source: str):
        event = {"player_id": player_id, "source": source}
        if player_id == 0:
            self.pending_jail_event = event
        else:
            self._resolve_non_agent_jail_entry(event)

    def _resolve_non_agent_jail_entry(self, event: Dict):
        player_id = event.get("player_id", -1)
        if player_id < 0:
            return
        if self.player_jail_cards[player_id] > 0:
            self.player_jail_cards[player_id] -= 1
            return
        if self.player_cash[player_id] >= 50:
            self.player_cash[player_id] -= 50
            return
        self._enforce_jail_sentence(player_id)

    def _other_players_take_turns(self):
        for player_id in range(1, self.num_players):
            if self.player_bankrupt[player_id]:
                continue

            if self.player_in_jail[player_id]:
                if self.jail_turns_remaining[player_id] > 0:
                    self.jail_turns_remaining[player_id] -= 1
                    if self.jail_turns_remaining[player_id] == 0:
                        self.player_in_jail[player_id] = False
                        self.player_positions[player_id] = self.jail_position
                    continue
                self.player_in_jail[player_id] = False
                self.player_positions[player_id] = self.jail_position

            dice = self._roll_dice()
            passed_go = self._advance_position(player_id, dice)
            if passed_go:
                self.player_cash[player_id] += self.go_reward
            tile_idx = self.player_positions[player_id]
            self._handle_tile_effects(player_id, tile_idx, dice)

            if self.player_bankrupt[player_id] or self.player_in_jail[player_id]:
                self._check_and_handle_bankruptcy(player_id)
                continue

            self._heuristic_property_actions(player_id, tile_idx)
            self._check_and_handle_bankruptcy(player_id)

    def _heuristic_property_actions(self, player_id: int, tile_idx: int):
        tile = self.properties[tile_idx]
        price = self._get_property_price(tile_idx)
        if (
            tile.get("type") in {"property", "railroad", "utility"}
            and self.property_owner[tile_idx] == -1
            and self.player_cash[player_id] >= price
        ):
            self._handle_buy(tile_idx, price, player_id)

        buildable = self._list_buildable_properties(player_id)
        if buildable:
            choice = random.choice(buildable)
            self._handle_build_property(choice, player_id)

        if self.player_cash[player_id] < 0:
            building_candidates = self._list_sell_building_candidates(player_id)
            if building_candidates:
                choice = max(building_candidates, key=lambda idx: self.property_houses[idx])
                self._handle_sell_building(choice, player_id)
            else:
                property_candidates = self._list_sell_property_candidates(player_id)
                if property_candidates:
                    choice = random.choice(property_candidates)
                    self._handle_sell_property(choice, player_id)

    def _handle_tile_effects(self, player_id: int, tile_idx: int, dice: int) -> float:
        tile = self.properties[tile_idx]
        tile_type = tile.get("type")
        reward = 0.0

        if tile_type == "tax":
            amount = tile.get("price", 100)
            self.player_cash[player_id] -= amount
            reward -= amount * 0.01
        elif tile.get("corner") == "goto":
            self._trigger_jail_entry(player_id, "board")
            reward -= 0.5
        elif tile_type == "chance":
            reward += self._resolve_card(player_id, self.chance_cards)
        elif tile_type == "chest":
            reward += self._resolve_card(player_id, self.chest_cards)
        elif tile_type in {"property", "railroad", "utility"}:
            reward += self._resolve_property_interaction(player_id, tile_idx, dice)
        return reward

    def _resolve_property_interaction(self, player_id: int, tile_idx: int, dice: int) -> float:
        owner = self.property_owner[tile_idx]
        if owner == -1 or owner == player_id:
            return 0.0
        rent = self._calculate_rent(tile_idx, dice)
        self.player_cash[player_id] -= rent
        self.player_cash[owner] += rent
        return -rent * 0.01

    def _resolve_card(self, player_id: int, deck: List[Dict]) -> float:
        card = random.choice(deck)
        if card["type"] == "money":
            self.player_cash[player_id] += card["amount"]
            return card["amount"] * 0.01
        if card["type"] == "move":
            self.player_positions[player_id] = card["target"]
            if card["target"] == 0:
                self.player_cash[player_id] += self.go_reward
            return 0.5
        if card["type"] == "go_to_jail":
            self._trigger_jail_entry(player_id, "card")
            return -0.5
        if card["type"] == "jail_card":
            self.player_jail_cards[player_id] += 1
            return 0.2
        return 0.0

    def _calculate_rent(self, tile_idx: int, dice: int) -> int:
        tile = self.properties[tile_idx]
        tile_type = tile.get("type")
        if tile_type == "property":
            houses = self.property_houses[tile_idx]
            rent_key = "base"
            if 1 <= houses <= 4:
                rent_key = f"{houses}h"
            elif houses == 5:
                rent_key = "hotel"
            return tile.get("rent", {}).get(rent_key, tile.get("rent", {}).get("base", 0))
        if tile_type == "railroad":
            owner = self.property_owner[tile_idx]
            owned = sum(1 for idx, prop in enumerate(self.properties) if prop.get("type") == "railroad" and self.property_owner[idx] == owner)
            lookup = {1: "1_rr", 2: "2_rr", 3: "3_rr", 4: "4_rr"}
            return tile.get("rent", {}).get(lookup.get(owned, "1_rr"), 25)
        if tile_type == "utility":
            owner = self.property_owner[tile_idx]
            owned = sum(1 for idx, prop in enumerate(self.properties) if prop.get("type") == "utility" and self.property_owner[idx] == owner)
            multiplier = 10 if owned >= 2 else 4
            return multiplier * dice
        return 0

    def _roll_dice(self) -> int:
        return random.randint(1, 6) + random.randint(1, 6)

    def _advance_position(self, player_id: int, steps: int) -> bool:
        old_position = self.player_positions[player_id]
        new_position = (old_position + steps) % self.board_size
        self.player_positions[player_id] = new_position
        return old_position + steps >= self.board_size

    def _owns_full_color_set(self, player_id: int, color: Optional[str]) -> bool:
        if not color or color not in self.color_groups:
            return False
        return all(self.property_owner[idx] == player_id for idx in self.color_groups[color])

    def _select_properties_owned(self, player_id: int) -> List[int]:
        return [idx for idx, owner in enumerate(self.property_owner) if owner == player_id]

    def _get_property_price(self, prop_id: int) -> float:
        base_price = self.properties[prop_id].get("price", 0)
        if callable(self.dynamic_price_fn):
            return float(self.dynamic_price_fn(prop_id, base_price, self.turn_count))
        return float(base_price)

    def _check_and_handle_bankruptcy(self, player_id: int):
        if self.player_cash[player_id] >= 0 or self.player_bankrupt[player_id]:
            return
        self.player_bankrupt[player_id] = True
        for prop_id, owner in enumerate(self.property_owner):
            if owner == player_id:
                self.property_owner[prop_id] = -1
                self.property_houses[prop_id] = 0

    def _player_net_worth(self, player_id: int) -> float:
        cash = float(self.player_cash[player_id])
        property_value = 0.0
        for idx, owner in enumerate(self.property_owner):
            if owner != player_id:
                continue
            base = self._get_property_price(idx)
            house_value = self.property_houses[idx] * self.properties[idx].get("house_cost", 0)
            property_value += base + house_value * 0.5
        return cash + property_value

    def _get_house_cost(self, prop_id: int) -> float:
        return float(self.properties[prop_id].get("house_cost", 0))

    @property
    def action_meanings(self) -> List[str]:
        """Return list of human-readable action meanings."""
        meanings = self.base_action_meanings.copy()
        for i in range(self.board_size):
            meanings.append(f"build_on_{self.properties[i]['name']}")
        for i in range(self.board_size):
            meanings.append(f"sell_building_on_{self.properties[i]['name']}")
        for i in range(self.board_size):
            meanings.append(f"sell_property_{self.properties[i]['name']}")
        return meanings

    def _get_buildable_monopolies(self, player_id: int) -> List[int]:
        """Get list of properties where player can build houses/hotels."""
        return self._list_buildable_properties(player_id)

    def _get_sellable_buildings(self, player_id: int) -> List[int]:
        """Get list of properties where player can sell buildings."""
        return self._list_sell_building_candidates(player_id)

    def _get_all_owned_properties(self, player_id: int) -> List[int]:
        """Get list of all properties owned by player."""
        return self._select_properties_owned(player_id)

    def _prepare_turn_context(self, player_id: int):
        """Prepare context for a specific player's turn."""
        # If this is the agent (player 0), use existing logic
        if player_id == 0:
            self._prepare_agent_context()
            return
        
        # For other players, simulate their turn
        if self.player_bankrupt[player_id]:
            ctx = {"type": "terminal", "valid_actions": [self.END_TURN]}
            self.pending_context = ctx
            self.pending_mask = self._context_to_mask(ctx)
            return

        if self.pending_jail_event and self.pending_jail_event.get("player_id") == player_id:
            ctx = self._build_jail_entry_context(player_id)
            self.pending_context = ctx
            self.pending_mask = self._context_to_mask(ctx)
            return

        if self.player_in_jail[player_id]:
            if self.jail_turns_remaining[player_id] > 0:
                ctx = self._build_jail_skip_context(player_id)
                self.pending_context = ctx
                self.pending_mask = self._context_to_mask(ctx)
                return
            self.player_in_jail[player_id] = False
            self.player_positions[player_id] = self.jail_position

        dice = self._roll_dice()
        passed_go = self._advance_position(player_id, dice)
        if passed_go:
            self.player_cash[player_id] += self.go_reward
        tile_idx = self.player_positions[player_id]
        self.turn_reward_buffer += self._handle_tile_effects(player_id, tile_idx, dice)

        if self.pending_jail_event and self.pending_jail_event.get("player_id") == player_id:
            ctx = self._build_jail_entry_context(player_id)
            self.pending_context = ctx
            self.pending_mask = self._context_to_mask(ctx)
            return

        ctx = self._build_decision_context(player_id, tile_idx)
        self.pending_context = ctx
        self.pending_mask = self._context_to_mask(ctx)

    def _handle_build_for_player(self, ctx: Dict, player_id: int, prop_id: int) -> float:
        """Handle building action for any player."""
        return self._handle_build_property(prop_id, player_id)

    def _handle_sell_building_for_player(self, ctx: Dict, player_id: int, prop_id: int) -> float:
        """Handle selling building action for any player."""
        return self._handle_sell_building(prop_id, player_id)

    def _handle_sell_property_for_player(self, ctx: Dict, player_id: int, prop_id: int) -> float:
        """Handle selling property action for any player."""
        return self._handle_sell_property(prop_id, player_id)

    def _apply_action_for_player(self, action: int, player_id: int) -> float:
        """Apply action for a specific player."""
        # Temporarily set current_player to the acting player
        original_player = self.current_player
        self.current_player = player_id
        
        # Use existing _apply_action logic but adapt for the specific player
        ctx = self.pending_context or {"type": "decision", "valid_actions": [self.END_TURN]}
        ctx_type = ctx.get("type")

        if ctx_type == "terminal":
            self.current_player = original_player
            return 0.0

        if ctx_type == "jail_entry":
            if action == self.PAY_JAIL_FINE:
                result = self._handle_jail_payment(player_id)
            elif action == self.USE_JAIL_CARD:
                result = self._handle_jail_card_use(player_id)
            elif action == self.ACCEPT_JAIL:
                result = self._handle_accept_jail(player_id)
            else:
                result = self.invalid_action_penalty
            self.current_player = original_player
            return result

        if ctx_type == "jail_skip":
            if action == self.END_TURN:
                result = self._handle_jail_skip(player_id)
            else:
                result = self.invalid_action_penalty
            self.current_player = original_player
            return result

        if ctx_type == "decision":
            if action == self.END_TURN:
                self.current_player = original_player
                return 0.0

            if action == self.BUY_PROPERTY and ctx.get("buy_target") is not None:
                result = self._handle_buy(ctx["buy_target"], ctx.get("buy_price"), player_id)
                self.current_player = original_player
                return result

            if self.build_offset <= action < self.sell_building_offset:
                prop_id = action - self.build_offset
                if prop_id in ctx.get("build_targets", []):
                    result = self._handle_build_property(prop_id, player_id)
                    self.current_player = original_player
                    return result
                self.current_player = original_player
                return self.invalid_action_penalty

            if self.sell_building_offset <= action < self.sell_property_offset:
                prop_id = action - self.sell_building_offset
                if prop_id in ctx.get("sell_building_targets", []):
                    result = self._handle_sell_building(prop_id, player_id)
                    self.current_player = original_player
                    return result
                self.current_player = original_player
                return self.invalid_action_penalty

            if self.sell_property_offset <= action < self.action_space.n:
                prop_id = action - self.sell_property_offset
                if prop_id in ctx.get("sell_property_targets", []):
                    result = self._handle_sell_property(prop_id, player_id)
                    self.current_player = original_player
                    return result
                self.current_player = original_player
                return self.invalid_action_penalty

            self.current_player = original_player
            return self.invalid_action_penalty

        self.current_player = original_player
        return 0.0


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, mask, next_mask):
        self.buffer.append((state, action, reward, next_state, done, mask, next_mask))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done, mask, next_mask = map(np.array, zip(*batch))
        return state, action, reward, next_state, done, mask, next_mask

    def __len__(self):
        return len(self.buffer)


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


def select_action(policy_net: QNetwork, state: np.ndarray, mask: np.ndarray, epsilon: float, device) -> int:
    valid_actions = np.where(mask > 0)[0]
    if len(valid_actions) == 0:
        return 0
    if random.random() < epsilon:
        return int(random.choice(valid_actions))
    state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        q_values = policy_net(state_tensor).cpu().numpy()[0]
    q_values[mask == 0] = -np.inf
    return int(np.argmax(q_values))


def train_dqn(
    episodes: int = 500,
    max_steps_per_episode: int = 200,
    batch_size: int = 64,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.1,
    epsilon_decay: float = 0.995,
    learning_rate: float = 1e-3,
    target_update_interval: int = 25,
    buffer_capacity: int = 100_000,
    device: Optional[str] = None,
    num_players: int = 4,
):
    """Train a DQN agent on the decision-centric Monopoly environment."""

    env = DymonopolyDecisionEnv(num_players=num_players)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    obs, info = env.reset()
    state = env.flatten_observation(obs)
    action_mask = info["action_mask"]

    policy_net = QNetwork(env.flat_observation_size, env.action_space.n).to(device)
    target_net = QNetwork(env.flat_observation_size, env.action_space.n).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    replay_buffer = ReplayBuffer(capacity=buffer_capacity)

    epsilon = epsilon_start
    episode_rewards = []

    for episode in range(episodes):
        obs, info = env.reset()
        state = env.flatten_observation(obs)
        action_mask = info["action_mask"]
        total_reward = 0.0

        for step in range(max_steps_per_episode):
            action = select_action(policy_net, state, action_mask, epsilon, device)
            next_obs, reward, terminated, truncated, next_info = env.step(action)
            next_state = env.flatten_observation(next_obs)
            next_mask = next_info["action_mask"]

            replay_buffer.push(state, action, reward, next_state, terminated or truncated, action_mask, next_mask)

            state = next_state
            action_mask = next_mask
            total_reward += reward

            if len(replay_buffer) >= batch_size:
                batch = replay_buffer.sample(batch_size)
                _optimize_model(policy_net, target_net, optimizer, batch, gamma, device)

            if terminated or truncated:
                break

        episode_rewards.append(total_reward)
        epsilon = max(epsilon * epsilon_decay, epsilon_end)

        if (episode + 1) % target_update_interval == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {episode + 1}/{episodes} | Avg Reward (last 10): {avg_reward:.2f} | Epsilon: {epsilon:.3f}")

    return policy_net, episode_rewards


def _optimize_model(
    policy_net: QNetwork,
    target_net: QNetwork,
    optimizer: optim.Optimizer,
    batch,
    gamma: float,
    device,
):
    state, action, reward, next_state, done, mask, next_mask = batch

    state = torch.tensor(state, dtype=torch.float32, device=device)
    next_state = torch.tensor(next_state, dtype=torch.float32, device=device)
    action = torch.tensor(action, dtype=torch.int64, device=device).unsqueeze(1)
    reward = torch.tensor(reward, dtype=torch.float32, device=device).unsqueeze(1)
    done = torch.tensor(done.astype(np.float32), dtype=torch.float32, device=device).unsqueeze(1)
    next_mask = torch.tensor(next_mask, dtype=torch.float32, device=device)

    q_values = policy_net(state).gather(1, action)

    with torch.no_grad():
        next_q_values = target_net(next_state)
        invalid = next_mask == 0
        next_q_values[invalid] = -1e9
        best_next_q = next_q_values.max(1)[0].unsqueeze(1)
        target = reward + (1 - done) * gamma * best_next_q

    loss = nn.SmoothL1Loss()(q_values, target)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
    optimizer.step()



