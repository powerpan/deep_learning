from __future__ import annotations

import warnings

warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*",
    category=UserWarning,
)

import pygame

from config import (
    FPS,
    PANEL_HEIGHT,
    TILE_DOOR,
    TILE_EMPTY,
    TILE_EXIT,
    TILE_KEY,
    TILE_SIZE,
    TILE_START,
    TILE_TRAP,
    TILE_WALL,
)
from vision import visible_cells

COLORS = {
    TILE_WALL: (38, 42, 50),
    TILE_EMPTY: (232, 236, 242),
    TILE_START: (88, 166, 255),
    TILE_EXIT: (56, 180, 111),
    TILE_TRAP: (222, 75, 75),
    TILE_KEY: (245, 191, 66),
    TILE_DOOR: (137, 95, 190),
}

BACKGROUND = (18, 21, 28)
GRID_LINE = (186, 193, 204)
AGENT = (20, 105, 220)
TEXT = (244, 247, 251)
SUBTLE_TEXT = (190, 198, 212)
HIDDEN_OVERLAY = (0, 0, 0, 135)


class MazeRenderer:
    def __init__(
        self,
        tile_size: int = TILE_SIZE,
        panel_height: int = PANEL_HEIGHT,
        show_vision: bool = True,
    ) -> None:
        pygame.init()
        pygame.font.init()
        self.tile_size = tile_size
        self.panel_height = panel_height
        self.show_vision = show_vision
        self.screen: pygame.Surface | None = None
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 18)
        self.small_font = pygame.font.SysFont("Arial", 15)
        self.current_size: tuple[int, int] | None = None

    def render(self, env, info: dict, episode: int | None = None) -> bool:
        if self.screen is None or self.current_size != (env.rows, env.cols):
            width = max(env.cols * self.tile_size, 720)
            height = env.rows * self.tile_size + self.panel_height
            self.screen = pygame.display.set_mode((width, height))
            pygame.display.set_caption("AI Maze PPO Lab")
            self.current_size = (env.rows, env.cols)

        if not self._handle_events():
            return False

        assert self.screen is not None
        self.screen.fill(BACKGROUND)
        visible = visible_cells(env.grid, env.agent_pos, env.view_range)

        for row in range(env.rows):
            for col in range(env.cols):
                tile = env.grid[row][col]
                draw_tile = TILE_EMPTY if tile == TILE_KEY and env.key_collected else tile
                color = COLORS.get(draw_tile, COLORS[TILE_EMPTY])
                rect = pygame.Rect(
                    col * self.tile_size,
                    row * self.tile_size,
                    self.tile_size,
                    self.tile_size,
                )
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, GRID_LINE, rect, 1)

                if self.show_vision and (row, col) not in visible:
                    overlay = pygame.Surface((self.tile_size, self.tile_size), pygame.SRCALPHA)
                    overlay.fill(HIDDEN_OVERLAY)
                    self.screen.blit(overlay, rect.topleft)

                self._draw_tile_label(tile, rect, env.key_collected)

        self._draw_agent(env.agent_pos)
        self._draw_panel(env, info, episode)
        pygame.display.flip()
        self.clock.tick(FPS)
        return True

    def close(self) -> None:
        pygame.quit()

    def _handle_events(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        return True

    def _draw_agent(self, pos: tuple[int, int]) -> None:
        assert self.screen is not None
        row, col = pos
        center = (
            col * self.tile_size + self.tile_size // 2,
            row * self.tile_size + self.tile_size // 2,
        )
        pygame.draw.circle(self.screen, AGENT, center, self.tile_size // 3)
        pygame.draw.circle(self.screen, (255, 255, 255), center, self.tile_size // 3, 2)

    def _draw_tile_label(self, tile: str, rect: pygame.Rect, key_collected: bool) -> None:
        if tile not in {TILE_START, TILE_EXIT, TILE_TRAP, TILE_KEY, TILE_DOOR}:
            return
        if tile == TILE_KEY and key_collected:
            return
        assert self.screen is not None
        label = self.small_font.render(tile, True, (20, 24, 32))
        self.screen.blit(label, label.get_rect(center=rect.center))

    def _draw_panel(self, env, info: dict, episode: int | None) -> None:
        assert self.screen is not None
        top = env.rows * self.tile_size
        panel_rect = pygame.Rect(0, top, self.screen.get_width(), self.panel_height)
        pygame.draw.rect(self.screen, (28, 32, 42), panel_rect)

        title = "AI Maze PPO Lab"
        if episode is not None:
            title += f" | Episode {episode}"
        lines = [
            title,
            f"Map: {env.map_name}",
            (
                f"Step: {info.get('steps', 0)} / {env.max_steps}   "
                f"Reward: {info.get('total_reward', 0.0):.1f}   "
                f"Event: {info.get('event', '-')}"
            ),
            (
                f"Key: {info.get('has_key', False)}   "
                f"Door: {info.get('passed_door', False)}   "
                f"Success: {info.get('success', False)}   "
                f"Vision: straight rays, range {env.view_range}"
            ),
        ]
        for index, line in enumerate(lines):
            font = self.font if index == 0 else self.small_font
            color = TEXT if index == 0 else SUBTLE_TEXT
            surface = font.render(line, True, color)
            self.screen.blit(surface, (16, top + 14 + index * 27))
