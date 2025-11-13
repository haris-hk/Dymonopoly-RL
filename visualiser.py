import pygame
import numpy as np
from rl import DymonopolyEnv
import random

class MonopolyVisualizer:
    def __init__(self, env: DymonopolyEnv):
        pygame.init()
        self.env = env
        self.width = 1240
        self.height = 920
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Dymonopoly - Market Board")

        self.corner_size = 120
        self.edge_size = 70
        self.board_size = self.corner_size * 2 + self.edge_size * 9
        self.margin_left = 40
        self.margin_top = (self.height - self.board_size) // 2
        self.board_rect = pygame.Rect(
            self.margin_left,
            self.margin_top,
            self.board_size,
            self.board_size,
        )
        self.info_rect = pygame.Rect(
            self.board_rect.right + 30,
            self.board_rect.top,
            self.width - (self.board_rect.right + 60),
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
        self.player_positions = [0] * self.env.num_players

        self.clock = pygame.time.Clock()
    
    def _build_tiles(self):
        return [
            {"name": "GO", "type": "corner", "corner": "go"},
            {"name": "Mediterranean Avenue", "type": "property", "color": "brown", "price": 60},
            {"name": "Community Chest", "type": "chest"},
            {"name": "Baltic Avenue", "type": "property", "color": "brown", "price": 60},
            {"name": "Income Tax", "type": "tax", "price": 200},
            {"name": "Reading Railroad", "type": "railroad", "price": 200},
            {"name": "Oriental Avenue", "type": "property", "color": "light_blue", "price": 100},
            {"name": "Chance", "type": "chance"},
            {"name": "Vermont Avenue", "type": "property", "color": "light_blue", "price": 100},
            {"name": "Connecticut Avenue", "type": "property", "color": "light_blue", "price": 120},
            {"name": "In Jail / Just Visiting", "type": "corner", "corner": "jail"},
            {"name": "St. Charles Place", "type": "property", "color": "pink", "price": 140},
            {"name": "Electric Company", "type": "utility", "price": 150},
            {"name": "States Avenue", "type": "property", "color": "pink", "price": 140},
            {"name": "Virginia Avenue", "type": "property", "color": "pink", "price": 160},
            {"name": "Pennsylvania Railroad", "type": "railroad", "price": 200},
            {"name": "St. James Place", "type": "property", "color": "orange", "price": 180},
            {"name": "Community Chest", "type": "chest"},
            {"name": "Tennessee Avenue", "type": "property", "color": "orange", "price": 180},
            {"name": "New York Avenue", "type": "property", "color": "orange", "price": 200},
            {"name": "Free Parking", "type": "corner", "corner": "free"},
            {"name": "Kentucky Avenue", "type": "property", "color": "red", "price": 220},
            {"name": "Chance", "type": "chance"},
            {"name": "Indiana Avenue", "type": "property", "color": "red", "price": 220},
            {"name": "Illinois Avenue", "type": "property", "color": "red", "price": 240},
            {"name": "B. & O. Railroad", "type": "railroad", "price": 200},
            {"name": "Atlantic Avenue", "type": "property", "color": "yellow", "price": 260},
            {"name": "Ventnor Avenue", "type": "property", "color": "yellow", "price": 260},
            {"name": "Water Works", "type": "utility", "price": 150},
            {"name": "Marvin Gardens", "type": "property", "color": "yellow", "price": 280},
            {"name": "Go To Jail", "type": "corner", "corner": "goto"},
            {"name": "Pacific Avenue", "type": "property", "color": "green", "price": 300},
            {"name": "North Carolina Avenue", "type": "property", "color": "green", "price": 300},
            {"name": "Community Chest", "type": "chest"},
            {"name": "Pennsylvania Avenue", "type": "property", "color": "green", "price": 320},
            {"name": "Short Line", "type": "railroad", "price": 200},
            {"name": "Chance", "type": "chance"},
            {"name": "Park Place", "type": "property", "color": "dark_blue", "price": 350},
            {"name": "Luxury Tax", "type": "tax", "price": 100},
            {"name": "Boardwalk", "type": "property", "color": "dark_blue", "price": 400},
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
                    left + offset * self.edge_size,
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
                    top + offset * self.edge_size,
                    self.corner_size,
                    self.edge_size,
                ),
                "orientation": "right",
            }

        return layout
    
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
        """Render the beautiful connected Monopoly board"""
        self.screen.fill((240, 240, 240))
        pygame.draw.rect(self.screen, self.WHITE, self.board_rect)
        pygame.draw.rect(self.screen, self.BOARD_EDGE, self.board_rect, 6)

        self._draw_centerpieces()
        for idx in range(40):
            self._draw_tile(idx)
        self._draw_players()
        self._draw_info_panel()
        pygame.display.flip()
    
    def _get_player_color(self, player_id):
        palette = [(220, 30, 33), (0, 115, 230), (46, 180, 75), (255, 215, 0)]
        return palette[player_id % len(palette)]
    
    def _draw_players(self):
        for pid in range(self.env.num_players):
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
        pygame.draw.rect(self.screen, (245, 245, 245), self.info_rect)
        pygame.draw.rect(self.screen, self.BLACK, self.info_rect, 3)

        x = self.info_rect.left + 16
        y = self.info_rect.top + 20

        title = self.font_large.render("Market", True, self.BLACK)
        self.screen.blit(title, (x, y))
        y += 36

        turn_text = self.font_stats.render(f"Turn: {self.env.turn_counter}", True, self.BLACK)
        self.screen.blit(turn_text, (x, y))
        y += 28

        if self.env.turn_counter % self.env.price_update_interval == 0 and self.env.turn_counter > 0:
            notice = self.font_stats.render("Prices refreshed!", True, (200, 0, 0))
        else:
            remaining = self.env.price_update_interval - (self.env.turn_counter % self.env.price_update_interval)
            notice = self.font_stats.render(f"Next update in {remaining}", True, (90, 90, 90))
        self.screen.blit(notice, (x, y))
        y += 34

        avg_price = np.mean(self.env.current_prices)
        avg_base = np.mean(self.env.base_prices)
        change_pct = (avg_price / avg_base - 1.0) * 100 if avg_base else 0
        avg_text = self.font_stats.render(f"Avg price: ${avg_price:.0f}", True, self.BLACK)
        pct_text = self.font_stats.render(f"Change: {change_pct:+.1f}%", True, (0, 120, 0) if change_pct >= 0 else (180, 0, 0))
        self.screen.blit(avg_text, (x, y))
        y += 26
        self.screen.blit(pct_text, (x, y))
        y += 30

        trades = int(np.sum(self.env.trading_freq))
        trades_text = self.font_stats.render(f"Trades: {trades}", True, self.BLACK)
        self.screen.blit(trades_text, (x, y))
        y += 32

        self.screen.blit(self.font_medium.render("Players", True, self.BLACK), (x, y))
        y += 26

        for pid in range(self.env.num_players):
            color = self._get_player_color(pid)
            pygame.draw.circle(self.screen, color, (x + 12, y + 10), 9)
            pygame.draw.circle(self.screen, self.BLACK, (x + 12, y + 10), 9, 2)
            cash = int(self.env.player_cash[pid])
            owned = int(np.sum(self.env.property_owners == pid))
            summary = self.font_stats.render(f"P{pid + 1}: ${cash} ({owned})", True, self.BLACK)
            self.screen.blit(summary, (x + 28, y))
            y += 28

    def _simulate_dice_roll(self):
        dice_total = random.randint(1, 6) + random.randint(1, 6)
        current_player = self.env.current_player
        new_position = (self.player_positions[current_player] + dice_total) % len(self.tiles)
        self.player_positions[current_player] = new_position
        self.env.visiting_freq[new_position] += 1
        return new_position

    def _simulate_action(self, tile_index):
        tile = self.tiles[tile_index]
        current_player = self.env.current_player
        action = 0

        if tile["type"] in {"property", "railroad", "utility"}:
            price = float(self.env.current_prices[tile_index])
            owner = self.env.property_owners[tile_index]

            if owner == -1 and self.env.player_cash[current_player] >= price:
                if random.random() < 0.65:
                    action = 1
                    self.env.player_cash[current_player] -= price
                    self.env.property_owners[tile_index] = current_player
                    self.env.trading_freq[tile_index] += 1
            elif owner == current_player and random.random() < 0.12:
                action = 2
                sale_price = price * 0.85
                self.env.player_cash[current_player] += sale_price
                self.env.property_owners[tile_index] = -1
                self.env.trading_freq[tile_index] += 1

        self.env.current_player = (self.env.current_player + 1) % self.env.num_players
        return action
    
    def run(self, num_steps=200):
        """Run visualization with proper game simulation"""
        running = True
        step = 0
        paused = False
        
        print("Controls:")
        print("  SPACE - Pause/Resume")
        print("  RIGHT ARROW - Step one turn (when paused)")
        print("  ESC - Quit")
        
        while running and step < num_steps:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                        print("Paused" if paused else "Resumed")
                    elif event.key == pygame.K_RIGHT and paused:
                        # Manual step when paused
                        self._take_turn()
                        step += 1
            
            if not paused:
                # Auto-play
                self._take_turn()
                step += 1
            
            self.render_board()
            self.clock.tick(3 if not paused else 30)  # 3 FPS when running, 30 when paused
        
        print(f"\nSimulation complete! {step} turns played.")
        print("Close the window to exit.")
        
        # Keep window open
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
        
        pygame.quit()
    
    def _take_turn(self):
        """Execute one complete turn"""
        tile_index = self._simulate_dice_roll()
        action = self._simulate_action(tile_index)
        self.env.step(action)

# Usage example
if __name__ == "__main__":
    print("Initializing Dymonopoly environment...")
    env = DymonopolyEnv(num_properties=40, num_players=4)
    env.reset()
    
    print("Starting visualization...")
    visualizer = MonopolyVisualizer(env)
    visualizer.run(num_steps=200)