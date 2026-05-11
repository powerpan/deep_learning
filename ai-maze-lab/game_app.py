from pathlib import Path

import numpy as np
import pygame

from analytics import (
    DEFAULT_OUTPUT_DIR,
    save_best_path,
    save_training_curves,
    save_training_log,
    save_visitation_heatmap,
)
from config import (
    ALPHA,
    EPSILON_DECAY,
    EPSILON_MIN,
    EPSILON_START,
    GAMMA,
    MAX_STEPS_PER_EPISODE,
    REPLAY_DELAY_MS,
)
from maze_env import MazeEnv
from q_agent import QLearningAgent
from text_renderer import TextRenderer


PROJECT_ROOT = Path(__file__).resolve().parent
MAP_FILES = [
    PROJECT_ROOT / "maps" / "level_1.txt",
    PROJECT_ROOT / "maps" / "level_2.txt",
    PROJECT_ROOT / "maps" / "level_3.txt",
    PROJECT_ROOT / "maps" / "level_4_key_door.txt",
    PROJECT_ROOT / "maps" / "level_5_trap_key_door.txt",
    PROJECT_ROOT / "maps" / "level_6_dead_end.txt",
]
APP_OUTPUT_DIR = PROJECT_ROOT / DEFAULT_OUTPUT_DIR

WINDOW_WIDTH = 980
WINDOW_HEIGHT = 780
SIDE_PANEL_WIDTH = 300
MARGIN = 24
HEADER_HEIGHT = 72
FPS = 60
TRAIN_STEPS_PER_FRAME = 8

COLORS = {
    "background": (20, 22, 27),
    "surface": (31, 35, 43),
    "surface_2": (42, 48, 58),
    "line": (80, 90, 105),
    "text": (242, 246, 250),
    "muted": (174, 184, 198),
    "button": (52, 97, 157),
    "button_hover": (65, 119, 190),
    "button_disabled": (64, 69, 78),
    "wall": (59, 66, 78),
    "floor": (236, 239, 244),
    "start": (75, 143, 225),
    "exit": (74, 186, 118),
    "trap": (219, 83, 83),
    "key": (245, 214, 83),
    "door": (139, 98, 62),
    "agent": (248, 190, 75),
    "agent_outline": (83, 51, 18),
}

ACTION_KEYS = {
    pygame.K_UP: 0,
    pygame.K_w: 0,
    pygame.K_DOWN: 1,
    pygame.K_s: 1,
    pygame.K_LEFT: 2,
    pygame.K_a: 2,
    pygame.K_RIGHT: 3,
    pygame.K_d: 3,
}


class Button:
    def __init__(self, rect, label, callback, enabled=True):
        self.rect = pygame.Rect(rect)
        self.label = label
        self.callback = callback
        self.enabled = enabled

    def draw(self, screen, text_renderer, mouse_pos):
        if not self.enabled:
            color = COLORS["button_disabled"]
        elif self.rect.collidepoint(mouse_pos):
            color = COLORS["button_hover"]
        else:
            color = COLORS["button"]

        pygame.draw.rect(screen, color, self.rect, border_radius=7)
        pygame.draw.rect(screen, COLORS["line"], self.rect, 1, border_radius=7)
        text_renderer.draw_centered(screen, self.label, self.rect.center, COLORS["text"], size=17)

    def handle_click(self, mouse_pos):
        if self.enabled and self.rect.collidepoint(mouse_pos):
            self.callback()
            return True
        return False


class MazeLabApp:
    def __init__(self):
        pygame.display.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("AI 迷宫实验室")
        self.clock = pygame.time.Clock()
        self.text = TextRenderer()

        self.running = True
        self.mode = "ready"
        self.map_index = 0
        self.env = None
        self.agent = None
        self.buttons = []
        self.cell_size = 40
        self.board_origin = (MARGIN, HEADER_HEIGHT)

        self.episode_stats = []
        self.visitation_counts = None
        self.training_stop_episode = 0
        self.current_state = None
        self.current_reward = 0.0
        self.current_step = 0
        self.current_success = False
        self.current_reason = "ready"
        self.last_export_status = "暂无导出"
        self.replay_last_tick = 0

        self.load_map(0)

    def load_map(self, map_index):
        self.map_index = map_index
        self.env = MazeEnv(MAP_FILES[map_index], max_steps=MAX_STEPS_PER_EPISODE)
        self.agent = QLearningAgent(
            rows=self.env.rows,
            cols=self.env.cols,
            alpha=ALPHA,
            gamma=GAMMA,
            epsilon=EPSILON_START,
            epsilon_decay=EPSILON_DECAY,
            epsilon_min=EPSILON_MIN,
        )
        self.episode_stats = []
        self.visitation_counts = np.zeros((self.env.rows, self.env.cols), dtype=np.int64)
        self.mode = "ready"
        self.current_state = self.env.reset()
        self.current_reward = 0.0
        self.current_step = 0
        self.current_success = False
        self.current_reason = "ready"
        self.last_export_status = "暂无导出"
        self._layout_board()

    def run(self):
        while self.running:
            self._handle_events()
            self._update()
            self._draw()
            self.clock.tick(FPS)

        pygame.quit()

    def _layout_board(self):
        available_width = WINDOW_WIDTH - SIDE_PANEL_WIDTH - MARGIN * 3
        available_height = WINDOW_HEIGHT - HEADER_HEIGHT - MARGIN
        self.cell_size = max(24, min(54, available_width // self.env.cols, available_height // self.env.rows))
        board_width = self.cell_size * self.env.cols
        board_height = self.cell_size * self.env.rows
        self.board_origin = (
            MARGIN + (available_width - board_width) // 2,
            HEADER_HEIGHT + (available_height - board_height) // 2,
        )

    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                self._handle_key(event.key)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mouse_pos = pygame.mouse.get_pos()
                for button in self.buttons:
                    if button.handle_click(mouse_pos):
                        break

    def _handle_key(self, key):
        if key == pygame.K_ESCAPE:
            self.mode = "ready"
            return

        if self.mode == "manual" and key in ACTION_KEYS and self.current_reason == "running":
            self._manual_step(ACTION_KEYS[key])

    def _update(self):
        if self.mode == "training":
            for _ in range(TRAIN_STEPS_PER_FRAME):
                if self.mode != "training":
                    break
                self._training_step()
        elif self.mode == "replay":
            now = pygame.time.get_ticks()
            if now - self.replay_last_tick >= REPLAY_DELAY_MS:
                self.replay_last_tick = now
                self._replay_step()

    def _start_training(self, episodes):
        if self.mode == "training":
            self.mode = "ready"
            self.current_reason = "paused"
            return

        self.training_stop_episode = len(self.episode_stats) + episodes
        self.mode = "training"
        self._begin_training_episode()

    def _begin_training_episode(self):
        self.current_state = self.env.reset()
        self.visitation_counts[self.current_state[:2]] += 1
        self.current_reward = 0.0
        self.current_step = 0
        self.current_success = False
        self.current_reason = "running"

    def _training_step(self):
        if len(self.episode_stats) >= self.training_stop_episode:
            self.mode = "ready"
            self.current_reason = "training complete"
            return

        action = self.agent.choose_action(self.current_state, explore=True)
        next_state, reward, done, info = self.env.step(action)
        self.agent.update(self.current_state, action, reward, next_state, done)

        self.current_state = next_state
        self.visitation_counts[self.current_state[:2]] += 1
        self.current_reward += reward
        self.current_step = info["steps"]
        self.current_success = info["success"]
        self.current_reason = info["reason"]

        if done:
            self.episode_stats.append(
                {
                    "episode": len(self.episode_stats) + 1,
                    "steps": self.current_step,
                    "total_reward": self.current_reward,
                    "success": self.current_success,
                    "reason": self.current_reason,
                    "epsilon": self.agent.epsilon,
                    "has_key": self.env.has_key,
                    "passed_door": self.env.passed_door,
                }
            )
            self.agent.decay_epsilon()

            if len(self.episode_stats) >= self.training_stop_episode:
                self.mode = "ready"
                self.current_reason = "training complete"
                self._export_outputs()
            else:
                self._begin_training_episode()

    def _start_manual(self):
        self.mode = "manual"
        self.current_state = self.env.reset()
        self.current_reward = 0.0
        self.current_step = 0
        self.current_success = False
        self.current_reason = "running"

    def _manual_step(self, action):
        next_state, reward, done, info = self.env.step(action)
        self.current_state = next_state
        self.current_reward += reward
        self.current_step = info["steps"]
        self.current_success = info["success"]
        self.current_reason = info["reason"] if done else "running"

    def _start_replay(self):
        self.mode = "replay"
        self.current_state = self.env.reset()
        self.current_reward = 0.0
        self.current_step = 0
        self.current_success = False
        self.current_reason = "running"
        self.replay_last_tick = 0

    def _replay_step(self):
        if self.current_reason != "running":
            return

        action = self.agent.choose_action(self.current_state, explore=False)
        next_state, reward, done, info = self.env.step(action)
        self.current_state = next_state
        self.current_reward += reward
        self.current_step = info["steps"]
        self.current_success = info["success"]
        self.current_reason = info["reason"] if done else "running"

    def _reset_agent(self):
        self.load_map(self.map_index)

    def _draw(self):
        self.screen.fill(COLORS["background"])
        self._draw_header()
        self._draw_maze()
        self._draw_side_panel()
        pygame.display.flip()

    def _draw_header(self):
        self.text.draw(self.screen, "AI 迷宫实验室", MARGIN, 18, COLORS["text"], size=30)
        self.text.draw(self.screen, "从零学会逃出迷宫的 Q-learning 智能体", MARGIN + 230, 34, COLORS["muted"], size=16)

    def _draw_maze(self):
        origin_x, origin_y = self.board_origin

        for row in range(self.env.rows):
            for col in range(self.env.cols):
                tile = self.env.tile_at(row, col)
                rect = pygame.Rect(
                    origin_x + col * self.cell_size,
                    origin_y + row * self.cell_size,
                    self.cell_size,
                    self.cell_size,
                )
                pygame.draw.rect(self.screen, self._tile_color(tile), rect)
                pygame.draw.rect(self.screen, COLORS["line"], rect, 1)

                if tile in {"S", "E", "T", "K", "D"}:
                    tile_label = {"S": "起", "E": "出", "T": "陷", "K": "钥", "D": "门"}[tile]
                    self.text.draw_centered(self.screen, tile_label, rect.center, COLORS["text"], size=20)

        self._draw_agent()

    def _draw_agent(self):
        row, col = self.env.agent_pos
        origin_x, origin_y = self.board_origin
        center = (
            origin_x + col * self.cell_size + self.cell_size // 2,
            origin_y + row * self.cell_size + self.cell_size // 2,
        )
        radius = max(8, self.cell_size // 3)
        pygame.draw.circle(self.screen, COLORS["agent_outline"], center, radius + 3)
        pygame.draw.circle(self.screen, COLORS["agent"], center, radius)

    def _draw_side_panel(self):
        panel_x = WINDOW_WIDTH - SIDE_PANEL_WIDTH - MARGIN
        panel = pygame.Rect(panel_x, HEADER_HEIGHT, SIDE_PANEL_WIDTH, WINDOW_HEIGHT - HEADER_HEIGHT - MARGIN)
        pygame.draw.rect(self.screen, COLORS["surface"], panel, border_radius=8)
        pygame.draw.rect(self.screen, COLORS["line"], panel, 1, border_radius=8)

        self.buttons = []
        y = panel.y + 18
        y = self._draw_section_title("地图", panel.x + 18, y)
        y = self._add_button_grid(
            panel.x + 18,
            y,
            [(f"第{index + 1}关", lambda idx=index: self.load_map(idx), True) for index in range(len(MAP_FILES))],
            columns=3,
        )

        y += 8
        y = self._draw_section_title("操作", panel.x + 18, y)
        y = self._add_button(panel.x + 18, y, "手动试玩", self._start_manual)
        y = self._add_button(panel.x + 18, y, "训练 +500 局", lambda: self._start_training(500))
        y = self._add_button(panel.x + 18, y, "训练 +2000 局", lambda: self._start_training(2000))
        y = self._add_button(panel.x + 18, y, "回放最优路线", self._start_replay)
        y = self._add_button(panel.x + 18, y, "导出结果", self._export_outputs, enabled=bool(self.episode_stats))
        y = self._add_button(panel.x + 18, y, "重置智能体", self._reset_agent)

        y += 8
        y = self._draw_section_title("状态", panel.x + 18, y)
        self._draw_stats(panel.x + 18, y)

    def _draw_section_title(self, title, x, y):
        self.text.draw(self.screen, title, x, y, COLORS["text"], size=18)
        return y + 24

    def _add_button_row(self, x, y, items):
        button_width = 82
        gap = 10
        for index, (label, callback, enabled) in enumerate(items):
            rect = (x + index * (button_width + gap), y, button_width, 32)
            button = Button(rect, label, callback, enabled=enabled)
            button.draw(self.screen, self.text, pygame.mouse.get_pos())
            self.buttons.append(button)
        return y + 38

    def _add_button_grid(self, x, y, items, columns=3):
        button_width = 82
        gap = 10
        row_height = 38
        for index, (label, callback, enabled) in enumerate(items):
            row = index // columns
            col = index % columns
            rect = (x + col * (button_width + gap), y + row * row_height, button_width, 32)
            button = Button(rect, label, callback, enabled=enabled)
            button.draw(self.screen, self.text, pygame.mouse.get_pos())
            self.buttons.append(button)
        rows = (len(items) + columns - 1) // columns
        return y + rows * row_height

    def _add_button(self, x, y, label, callback, enabled=True):
        button = Button((x, y, SIDE_PANEL_WIDTH - 36, 32), label, callback, enabled=enabled)
        button.draw(self.screen, self.text, pygame.mouse.get_pos())
        self.buttons.append(button)
        return y + 38

    def _draw_stats(self, x, y):
        recent_success_rate, recent_avg_steps = self._recent_metrics()
        stats = [
            f"地图：第 {self.map_index + 1} 关",
            f"模式：{self._mode_label()}",
            f"已训练：{len(self.episode_stats)} 局",
            f"当前步数：{self.current_step}",
            f"累计奖励：{self.current_reward:.1f}",
            f"探索率 ε：{self.agent.epsilon:.3f}",
            f"钥匙：{self._yes_no(self.env.has_key)}",
            f"已过门：{self._yes_no(self.env.passed_door)}",
            f"近100局成功率：{recent_success_rate:.1f}%",
            f"近100局平均步数：{recent_avg_steps:.1f}",
            f"本局成功：{self._yes_no(self.current_success)}",
            f"状态：{self._reason_label()}",
            f"导出：{self.last_export_status}",
        ]

        for line in stats:
            self.text.draw(self.screen, line, x, y, COLORS["muted"], size=14)
            y += 18

        y += 2
        help_lines = [
            "手动试玩：方向键 / WASD",
            "训练完成会自动导出 outputs/",
        ]
        for line in help_lines:
            self.text.draw(self.screen, line, x, y, COLORS["muted"], size=14)
            y += 18

    def _recent_metrics(self):
        recent = self.episode_stats[-100:]
        if not recent:
            return 0.0, 0.0
        success_rate = sum(1 for item in recent if item["success"]) / len(recent) * 100
        avg_steps = sum(item["steps"] for item in recent) / len(recent)
        return success_rate, avg_steps

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

    def _export_outputs(self):
        if not self.episode_stats:
            self.last_export_status = "还没有训练数据"
            return

        try:
            replay_result = self._greedy_path_result()
            save_training_log(self.episode_stats, APP_OUTPUT_DIR)
            save_training_curves(self.episode_stats, APP_OUTPUT_DIR)
            save_visitation_heatmap(self.env, self.visitation_counts, APP_OUTPUT_DIR)
            save_best_path(MAP_FILES[self.map_index], replay_result, APP_OUTPUT_DIR)
            self.last_export_status = "已保存到 outputs/"
        except Exception as exc:
            self.last_export_status = f"导出失败：{type(exc).__name__}"
            print(f"Export failed: {exc}")

    def _greedy_path_result(self):
        eval_env = MazeEnv(MAP_FILES[self.map_index], max_steps=MAX_STEPS_PER_EPISODE)
        state = eval_env.reset()
        path = [state]
        visited_states = {state}
        total_reward = 0.0
        success = False
        reason = "running"

        for _ in range(1, eval_env.max_steps + 1):
            action = self.agent.choose_action(state, explore=False)
            next_state, reward, done, info = eval_env.step(action)
            path.append(next_state)
            total_reward += reward
            state = next_state
            success = info["success"]
            reason = info["reason"]

            if done:
                break

            if state in visited_states:
                reason = "loop"
                break
            visited_states.add(state)
        else:
            reason = "timeout"

        return {
            "path": path,
            "success": success,
            "steps": max(0, len(path) - 1),
            "total_reward": total_reward,
            "reason": reason,
            "has_key": eval_env.has_key,
            "passed_door": eval_env.passed_door,
            "reached_exit": success,
        }

    def _mode_label(self):
        labels = {
            "ready": "待机",
            "training": "训练中",
            "manual": "手动试玩",
            "replay": "最优路线回放",
        }
        return labels.get(self.mode, self.mode)

    def _reason_label(self):
        labels = {
            "ready": "待机",
            "running": "进行中",
            "move": "移动",
            "wall": "撞墙",
            "locked_door": "门锁住",
            "key": "拿到钥匙",
            "door": "通过门",
            "trap": "踩到陷阱",
            "exit": "到达出口",
            "timeout": "步数耗尽",
            "paused": "已暂停",
            "training complete": "训练完成",
        }
        return labels.get(self.current_reason, self.current_reason)

    def _yes_no(self, value):
        return "是" if value else "否"


def main():
    MazeLabApp().run()


if __name__ == "__main__":
    main()
