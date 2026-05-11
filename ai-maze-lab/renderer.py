import pygame

from config import (
    CELL_SIZE,
    END_PAUSE_MS,
    FPS,
    INFO_PANEL_HEIGHT,
    MIN_WINDOW_WIDTH,
    WINDOW_TITLE,
)
from text_renderer import TextRenderer


COLORS = {
    "background": (22, 24, 29),
    "panel": (31, 35, 42),
    "grid": (187, 194, 205),
    "wall": (58, 64, 74),
    "floor": (238, 241, 245),
    "start": (74, 144, 226),
    "exit": (72, 187, 120),
    "trap": (221, 87, 87),
    "key": (245, 214, 83),
    "door": (139, 98, 62),
    "agent": (252, 190, 74),
    "agent_outline": (72, 45, 16),
    "text": (244, 247, 251),
    "muted_text": (177, 185, 198),
}


class Renderer:
    def __init__(self, env, cell_size=CELL_SIZE):
        pygame.display.init()
        self.text = TextRenderer()
        self.cell_size = cell_size
        self.grid_width = env.cols * cell_size
        self.grid_height = env.rows * cell_size
        self.width = max(MIN_WINDOW_WIDTH, self.grid_width)
        self.height = self.grid_height + INFO_PANEL_HEIGHT
        self.grid_offset_x = (self.width - self.grid_width) // 2
        self.grid_offset_y = 0
        self.running = True

        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("AI 迷宫实验室")
        self.clock = pygame.time.Clock()

    def render(self, env, stats, delay_ms=0):
        if not self._handle_events():
            return False

        self.screen.fill(COLORS["background"])
        self._draw_maze(env)
        self._draw_panel(stats)
        pygame.display.flip()

        if delay_ms > 0:
            pygame.time.delay(delay_ms)
        self.clock.tick(FPS)
        return self.running

    def wait(self, milliseconds=END_PAUSE_MS):
        start = pygame.time.get_ticks()
        while self.running and pygame.time.get_ticks() - start < milliseconds:
            self._handle_events()
            self.clock.tick(FPS)

    def close(self):
        pygame.quit()

    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
        return self.running

    def _draw_maze(self, env):
        for row in range(env.rows):
            for col in range(env.cols):
                tile = env.tile_at(row, col)
                rect = pygame.Rect(
                    self.grid_offset_x + col * self.cell_size,
                    self.grid_offset_y + row * self.cell_size,
                    self.cell_size,
                    self.cell_size,
                )
                pygame.draw.rect(self.screen, self._tile_color(tile), rect)
                pygame.draw.rect(self.screen, COLORS["grid"], rect, 1)

                if tile in {"S", "E", "T", "K", "D"}:
                    tile_label = {"S": "起", "E": "出", "T": "陷", "K": "钥", "D": "门"}[tile]
                    self.text.draw_centered(self.screen, tile_label, rect.center, COLORS["text"], size=20)

        self._draw_agent(env.agent_pos)

    def _draw_agent(self, agent_pos):
        row, col = agent_pos
        center = (
            self.grid_offset_x + col * self.cell_size + self.cell_size // 2,
            self.grid_offset_y + row * self.cell_size + self.cell_size // 2,
        )
        radius = max(8, self.cell_size // 3)
        pygame.draw.circle(self.screen, COLORS["agent_outline"], center, radius + 3)
        pygame.draw.circle(self.screen, COLORS["agent"], center, radius)

    def _draw_panel(self, stats):
        panel_rect = pygame.Rect(0, self.grid_height, self.width, INFO_PANEL_HEIGHT)
        pygame.draw.rect(self.screen, COLORS["panel"], panel_rect)

        phase = stats.get("phase", "training")
        episode = stats.get("episode", 0)
        step = stats.get("step", 0)
        total_reward = stats.get("total_reward", 0.0)
        epsilon = stats.get("epsilon", 0.0)
        success = stats.get("success", False)
        reason = stats.get("reason", "-")
        has_key = stats.get("has_key", False)
        passed_door = stats.get("passed_door", False)

        line_1 = f"{self._phase_label(phase)} | 第 {episode} 局 | 第 {step} 步"
        line_2 = f"奖励 {total_reward:.1f} | ε {epsilon:.3f} | 钥匙 {self._yes_no(has_key)} | 过门 {self._yes_no(passed_door)}"
        line_3 = f"成功 {self._yes_no(success)} | 状态 {self._reason_label(reason)} | 关闭窗口后会继续在终端完成训练。"

        self.text.draw(self.screen, line_1, 18, self.grid_height + 18, COLORS["text"], size=16)
        self.text.draw(self.screen, line_2, 18, self.grid_height + 48, COLORS["text"], size=16)
        self.text.draw(self.screen, line_3, 18, self.grid_height + 78, COLORS["muted_text"], size=14)

    def _tile_color(self, tile):
        if tile == "#":
            return COLORS["wall"]
        if tile == "S":
            return COLORS["start"]
        if tile == "E":
            return COLORS["exit"]
        if tile == "T":
            return COLORS["trap"]
        if tile == "K":
            return COLORS["key"]
        if tile == "D":
            return COLORS["door"]
        return COLORS["floor"]

    def _phase_label(self, phase):
        labels = {
            "training": "训练中",
            "greedy replay": "最优路线回放",
        }
        return labels.get(phase, phase)

    def _reason_label(self, reason):
        labels = {
            "move": "移动",
            "wall": "撞墙",
            "locked_door": "门锁住",
            "key": "拿到钥匙",
            "door": "通过门",
            "trap": "踩到陷阱",
            "exit": "到达出口",
            "timeout": "步数耗尽",
            "-": "-",
        }
        return labels.get(reason, reason)

    def _yes_no(self, value):
        return "是" if value else "否"
