import os
import pygame
import numpy as np
from rl import DymonopolyEnv
import random


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
    def __init__(self, env: DymonopolyEnv):
        pygame.init()
        self.env = env
        self.width = 1100
        self.height = 780
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Dymonopoly - Market Board")

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

        self.tiles = self._build_tiles()
        self.layout = self._calculate_layout()
        self.max_players = min(4, self.env.num_players)
        self.env.num_players = self.max_players
        self.player_positions = [0] * self.max_players
        self.env.current_player = 0
        self.env.turn_counter = 0

        self.collision_offsets = {}
        # example nudges: move top row a few px left, right column a few px up+       for i in range(21, 30):   # top row tiles 21..29
        for i in range(21, 30):   # top row tiles 21..29
            self.collision_offsets[i] = (20, 0)
        for i in range(31, 40):   # right column tiles 31..39
            self.collision_offsets[i] = (0, 20)

        self.awaiting_roll = True
        self.awaiting_decision = False
        self.last_roll_total = None
        self.last_dice = None
        self.player_last_rolls = [None] * self.max_players
        self.selected_tile = 0
        self.message_log = ["Click 'Roll Dice' to begin"]
        self.show_property_card = False  # Flag to show property card
        self.last_clicked_property = None  # Stores the name of the last clicked property

        self.images_dir = os.path.join(os.path.dirname(__file__), "images")
        self.board_surface = self._load_board_surface()
        self.dice_images = self._load_dice_images()

        self.clock = pygame.time.Clock()
        self.action_buttons = self._create_action_buttons()
        self.sell_count = 0  # Counter for sell building
    
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

        specs = [
            ("Roll Dice", self._handle_roll, lambda: self.awaiting_roll, (76, 175, 80)),  # Green
            ("Buy", self._handle_buy, lambda: self.awaiting_decision, (100, 149, 237)),  # Blue
            ("Skip", self._handle_skip, lambda: self.awaiting_decision, (189, 189, 189)),  # Gray
            ("Build", self._handle_build, lambda: True, (76, 175, 80)),  # Green
            ("Sell Property", self._handle_sell_property, lambda: True, (255, 165, 0)),  # Orange
        ]

        for idx, (label, callback, enabled_fn, color) in enumerate(specs):
            rect = pygame.Rect(
                start_x,
                start_y + idx * (button_height + 5),
                button_width,
                button_height,
            )
            buttons.append(ActionButton(rect, label, callback, color, enabled_fn))
        
        # Sell Building button with counter (special layout)
        # Position it after the regular specs so it appears below them vertically
        sell_y = start_y + len(specs) * (button_height + 5)
        sell_rect = pygame.Rect(start_x, sell_y, button_width - 90, button_height)
        buttons.append(ActionButton(sell_rect, "Sell Building", self._handle_sell, (210, 180, 140), lambda: True))
        
        return buttons
    
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
            owner_surface = pygame.Surface((24, 24), pygame.SRCALPHA)
            pygame.draw.circle(owner_surface, self._get_player_color(owner), (12, 12), 10)
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

        self._draw_highlight()
        self._draw_players()
        self._draw_info_panel()
        
        # Draw property card overlay if showing
        if self.show_property_card:
            self._draw_property_card(self.selected_tile)
        
        # DEBUG: Draw clickable collision rectangles (remove this after tuning)
        # self._draw_debug_collision_rects()
        
        pygame.display.flip()
    
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

        # Title - Player Console (italic style)
        title = self.font_large.render("Player Console", True, self.BLACK)
        self.screen.blit(title, (x, y))
        y += 32

        current_player = self.env.current_player % self.max_players
        
        # Current Player with colored circle
        player_label = self.font_stats.render("Current Player: ", True, self.BLACK)
        self.screen.blit(player_label, (x, y))
        player_id_text = self.font_stats.render(f"P{current_player + 1}", True, self.BLACK)
        self.screen.blit(player_id_text, (x + player_label.get_width(), y))
        y += 22

        # Money with money bag emoji
        cash = int(self.env.player_cash[current_player])
        money_text = self.font_stats.render(f"Money: ${cash} ", True, self.BLACK)
        self.screen.blit(money_text, (x, y))
        # Draw a small money bag icon
        bag_x = x + money_text.get_width()
        pygame.draw.circle(self.screen, (218, 165, 32), (bag_x + 8, y + 10), 8)
        pygame.draw.polygon(self.screen, (218, 165, 32), [(bag_x + 2, y + 6), (bag_x + 8, y), (bag_x + 14, y + 6)])
        y += 22

        # Roll info
        if self.last_dice:
            roll_label = f"Roll: --"
        else:
            roll_label = "Roll: --"
        self.screen.blit(self.font_stats.render(roll_label, True, self.BLACK), (x, y))
        y += 24

        # Owned Properties section with list box
        self.screen.blit(self.font_medium.render("Owned Properties", True, self.BLACK), (x, y))
        y += 22
        
        # Properties list box
        props_box_height = 100
        props_box = pygame.Rect(x, y, panel_width, props_box_height)
        pygame.draw.rect(self.screen, self.WHITE, props_box)
        pygame.draw.rect(self.screen, (180, 180, 180), props_box, 1)
        
        owned_props = [tile["name"] for idx, tile in enumerate(self.tiles) if self.env.property_owners[idx] == current_player]
        prop_y = y + 4
        for prop_name in owned_props[:5]:  # Show up to 5 properties
            prop_text = self.font_small.render(prop_name, True, self.BLACK)
            self.screen.blit(prop_text, (x + 6, prop_y))
            prop_y += 18
        
        # Scrollbar placeholder on right
        scrollbar_rect = pygame.Rect(props_box.right - 14, props_box.top, 12, props_box.height)
        pygame.draw.rect(self.screen, (220, 220, 220), scrollbar_rect)
        pygame.draw.rect(self.screen, (180, 180, 180), scrollbar_rect, 1)
        # Scroll thumb
        thumb_rect = pygame.Rect(scrollbar_rect.left + 2, scrollbar_rect.top + 2, 8, 24)
        pygame.draw.rect(self.screen, (160, 160, 160), thumb_rect)
        
        y += props_box_height + 10

        # Dice Rolls section
        self.screen.blit(self.font_medium.render("Dice Rolls", True, self.BLACK), (x, y))
        y += 22
        
        # Dice display box (centered, below Dice Rolls header)
        dice_box_width = panel_width
        dice_box_height = 52
        dice_box = pygame.Rect(x, y, dice_box_width, dice_box_height)
        pygame.draw.rect(self.screen, (230, 230, 230), dice_box, border_radius=6)
        pygame.draw.rect(self.screen, (180, 180, 180), dice_box, 2, border_radius=6)
        
        # Draw dice images inside the box (centered)
        if self.last_dice:
            dice_total_width = 44 * 2 + 6  # two dice + spacing
            dice_start_x = dice_box.centerx - dice_total_width // 2
            dice_y = dice_box.centery - 22
            for idx, value in enumerate(self.last_dice):
                dice_img = self.dice_images.get(value)
                if dice_img:
                    self.screen.blit(dice_img, (dice_start_x + idx * 47, dice_y))
        
        y += dice_box_height + 10
        
        # Current Tile section
        self.screen.blit(self.font_medium.render("Current Tile", True, self.BLACK), (x, y))
        y += 22
        
        # Current tile name in a box
        tile = self.tiles[self.selected_tile]
        tile_name_box = pygame.Rect(x, y, panel_width, 28)
        pygame.draw.rect(self.screen, (230, 230, 230), tile_name_box, border_radius=4)
        pygame.draw.rect(self.screen, (180, 180, 180), tile_name_box, 1, border_radius=4)
        tile_name_text = self.font_stats.render(tile["name"], True, self.BLACK)
        self.screen.blit(tile_name_text, (tile_name_box.left + 10, tile_name_box.centery - tile_name_text.get_height() // 2))

        self._draw_action_buttons()
        # self._draw_message_log()

    def _draw_action_buttons(self):
        if self.action_buttons:
            heading_pos = (self.action_buttons[0].rect.left, self.action_buttons[0].rect.top - 28)
            heading = self.font_medium.render("Move Options", True, self.BLACK)
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
        footer_height = 80
        footer_rect = pygame.Rect(
            self.info_rect.left + 12,
            self.info_rect.bottom - footer_height - 12,
            self.info_rect.width - 24,
            footer_height,
        )
        pygame.draw.rect(self.screen, (255, 255, 255), footer_rect, border_radius=6)
        pygame.draw.rect(self.screen, (180, 180, 180), footer_rect, 1, border_radius=6)
        y = footer_rect.top + 6
        self.screen.blit(self.font_medium.render("Activity Log", True, self.BLACK), (footer_rect.left + 8, y))
        y += 22
        for msg in self.message_log[-3:]:  # Show last 3 messages
            msg_text = self.font_small.render(msg, True, (60, 60, 60))
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
        Check if a tile was clicked. If it's a property/railroad/utility,
        show the property card and return the property name.
        Returns the property name if clicked, None otherwise.
        """
        for idx in range(40):
            collision_rect = self._get_collision_rect(idx)
            
            if collision_rect.collidepoint(pos):
                tile = self.tiles[idx]
                if tile["type"] in ["property", "railroad", "utility"]:
                    self.selected_tile = idx
                    self.show_property_card = True
                    self.last_clicked_property = tile["name"]
                    return tile["name"]
        return None

    def _handle_button_click(self, pos):
        # Close property card if showing
        if self.show_property_card:
            self.show_property_card = False
            return
        
        # Check +/- buttons for sell count
        if hasattr(self, 'minus_rect') and self.minus_rect.collidepoint(pos):
            self.sell_count = max(0, self.sell_count - 1)
            return
        if hasattr(self, 'plus_rect') and self.plus_rect.collidepoint(pos):
            self.sell_count = min(5, self.sell_count + 1)
            return
        
        # Check if a tile was clicked
        clicked_property = self._handle_tile_click(pos)
        if clicked_property:
            self._push_message(f"Viewing property: {clicked_property}")
            return
        
        for button in self.action_buttons:
            if button.handle_click(pos):
                break

    def _handle_roll(self):
        if not self.awaiting_roll:
            return
        die_one = random.randint(1, 6)
        die_two = random.randint(1, 6)
        dice_total = die_one + die_two
        self.last_roll_total = dice_total
        self.last_dice = (die_one, die_two)
        current_player = self.env.current_player % self.max_players
        self.player_last_rolls[current_player] = self.last_dice
        new_position = (self.player_positions[current_player] + dice_total) % len(self.tiles)
        self.player_positions[current_player] = new_position
        self.selected_tile = new_position
        tile = self.tiles[new_position]
        self._push_message(f"P{current_player + 1} rolled {dice_total} and landed on {tile['name']}")
        self.awaiting_roll = False
        self.awaiting_decision = True
        
        # Show property card if landed on a property, railroad, or utility
        if tile["type"] in ["property", "railroad", "utility"]:
            self.show_property_card = True
    
    def _handle_sell_property(self):
        pass

    def _handle_buy(self):
        if not self.awaiting_decision:
            return
        tile = self.tiles[self.selected_tile]
        current_player = self.env.current_player % self.max_players
        if tile["type"] not in {"property", "railroad", "utility"}:
            self._push_message("Nothing to buy on this tile.")
            return

        owner = self.env.property_owners[self.selected_tile]
        if owner >= 0 and owner != current_player:
            self._push_message(f"Already owned by P{owner + 1}.")
            return

        price = tile.get("price", 0)
        if owner == current_player:
            self._push_message("You already own this property.")
            return

        if price and self.env.player_cash[current_player] < price:
            self._push_message("Not enough cash to buy this property.")
            return

        if price:
            self.env.player_cash[current_player] -= price
        self.env.property_owners[self.selected_tile] = current_player
        self._push_message(f"P{current_player + 1} bought {tile['name']} for ${price}.")
        self._end_turn()

    def _handle_skip(self):
        if not self.awaiting_decision:
            return
        tile = self.tiles[self.selected_tile]
        current_player = self.env.current_player % self.max_players
        self._push_message(f"P{current_player + 1} skips buying {tile['name']}")
        self._end_turn()

    def _handle_build(self):
        current_player = self.env.current_player % self.max_players
        self._push_message(f"P{current_player + 1} selected Build House (logic pending)")

    def _handle_sell(self):
        current_player = self.env.current_player % self.max_players
        owner = self.env.property_owners[self.selected_tile]
        tile = self.tiles[self.selected_tile]
        if owner == current_player:
            self.env.property_owners[self.selected_tile] = -1
            refund = (tile.get("price", 0) or 0) // 2
            self.env.player_cash[current_player] += refund
            self._push_message(f"P{current_player + 1} sells {tile['name']} for ${refund}.")
        elif owner >= 0:
            self._push_message(f"Only P{owner + 1} can sell this property.")
        else:
            self._push_message("No owner registered on this tile.")

    def _end_turn(self):
        self.awaiting_roll = True
        self.awaiting_decision = False
        self.env.current_player = (self.env.current_player + 1) % self.max_players
        self.env.turn_counter += 1

    def run(self):
        """Launch interactive UI for up to four players."""
        running = True
        print("Controls:\n  - Left click buttons to trigger actions\n  - ESC closes the window")

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self._handle_button_click(event.pos)

            self.render_board()
            self.clock.tick(60)

        pygame.quit()

# Usage example
if __name__ == "__main__":
    print("Initializing Dymonopoly environment...")
    env = DymonopolyEnv(num_properties=40, num_players=4)
    env.reset()
    
    print("Starting visualization...")
    visualizer = MonopolyVisualizer(env)
    visualizer.run()