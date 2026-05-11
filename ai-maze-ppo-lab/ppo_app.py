from __future__ import annotations

import argparse
import os
import queue
import subprocess
import sys
import threading
import time
import warnings
from collections import deque
from pathlib import Path

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*",
    category=UserWarning,
)

import pygame

from config import (
    ACTIONS,
    ACTION_NAMES,
    LOCAL_VIEW_SIZE,
    MAPS_DIR,
    MAX_STEPS,
    MODELS_DIR,
    OBSERVATION_MODE,
    OUTPUTS_DIR,
    RANDOM_DOOR_ORIENTATION,
    RANDOM_ENDPOINT_MODE,
    RANDOM_MAP_COLS,
    RANDOM_MAP_ROWS,
    RANDOM_MAP_STYLE,
    RANDOM_TRAP_DENSITY,
    RANDOM_WALL_DENSITY,
    PPO_ALGO,
    PPO_CURRICULUM,
    PPO_N_ENVS,
    TILE_DOOR,
    TILE_EMPTY,
    TILE_EXIT,
    TILE_KEY,
    TILE_START,
    TILE_TRAP,
    TILE_WALL,
    VIEW_RANGE,
    VIEW_WIDTH,
)
from maze_env import MazePPOEnv
from model_utils import initial_recurrent_state, load_trained_model, predict_action
from random_maps import generate_random_key_door_map
from text_renderer import TextRenderer
from vision import visible_cells


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = MODELS_DIR / "ppo_maze.zip"

(OUTPUTS_DIR / ".matplotlib").mkdir(parents=True, exist_ok=True)
(OUTPUTS_DIR / ".cache").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(OUTPUTS_DIR / ".matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(OUTPUTS_DIR / ".cache"))

WINDOW_WIDTH = 1120
WINDOW_HEIGHT = 780
PANEL_WIDTH = 350
MARGIN = 24
HEADER_HEIGHT = 74
FPS = 60
REPLAY_DELAY_MS = 110

STYLE_LABELS = {
    "mixed": "混合",
    "open": "开放",
    "split": "分割",
    "rooms": "房间",
    "deadends": "死胡同",
    "maze": "迷宫",
}
DOOR_LABELS = {
    "mixed": "混合",
    "horizontal": "横墙",
    "vertical": "竖墙",
}
ENDPOINT_LABELS = {
    "mixed": "混合",
    "corners": "角落",
    "edges": "边缘",
    "interior": "内部",
}
ALGO_LABELS = {
    "recurrent-ppo": "Recurrent",
    "ppo": "普通PPO",
}
CURRICULUM_LABELS = {
    "basic-to-keydoor": "开启",
    "none": "关闭",
}
OBS_MODE_LABELS = {
    "grid": "网格CNN",
    "strips": "四向带",
}
STYLE_OPTIONS = tuple(STYLE_LABELS)
DOOR_OPTIONS = tuple(DOOR_LABELS)
ENDPOINT_OPTIONS = tuple(ENDPOINT_LABELS)
ALGO_OPTIONS = tuple(ALGO_LABELS)
CURRICULUM_OPTIONS = tuple(CURRICULUM_LABELS)
OBS_MODE_OPTIONS = tuple(OBS_MODE_LABELS)
LOCAL_VIEW_OPTIONS = (7, 9)

COLORS = {
    "background": (18, 21, 27),
    "surface": (30, 35, 44),
    "surface_2": (42, 48, 60),
    "line": (80, 91, 108),
    "text": (242, 246, 250),
    "muted": (174, 184, 198),
    "button": (48, 99, 158),
    "button_hover": (62, 122, 192),
    "button_disabled": (62, 68, 78),
    "danger": (145, 61, 70),
    "wall": (54, 61, 74),
    "floor": (236, 239, 244),
    "start": (75, 143, 225),
    "exit": (74, 186, 118),
    "trap": (219, 83, 83),
    "key": (245, 214, 83),
    "door": (137, 95, 190),
    "agent": (248, 190, 75),
    "agent_outline": (83, 51, 18),
    "hidden": (0, 0, 0, 135),
}

TILE_COLORS = {
    TILE_WALL: COLORS["wall"],
    TILE_EMPTY: COLORS["floor"],
    TILE_START: COLORS["start"],
    TILE_EXIT: COLORS["exit"],
    TILE_TRAP: COLORS["trap"],
    TILE_KEY: COLORS["key"],
    TILE_DOOR: COLORS["door"],
}


class Button:
    def __init__(self, rect, label, callback, enabled=True):
        self.rect = pygame.Rect(rect)
        self.label = label
        self.callback = callback
        self.enabled = enabled

    def draw(self, screen, text, mouse_pos):
        if not self.enabled:
            color = COLORS["button_disabled"]
        elif self.rect.collidepoint(mouse_pos):
            color = COLORS["button_hover"]
        else:
            color = COLORS["button"]
        pygame.draw.rect(screen, color, self.rect, border_radius=7)
        pygame.draw.rect(screen, COLORS["line"], self.rect, 1, border_radius=7)
        text.draw_centered(screen, self.label, self.rect.center, COLORS["text"], size=13)

    def handle_click(self, mouse_pos):
        if self.enabled and self.rect.collidepoint(mouse_pos):
            self.callback()
            return True
        return False


class MazePPOApp:
    def __init__(self):
        pygame.display.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("AI Maze PPO Lab")
        self.clock = pygame.time.Clock()
        self.text = TextRenderer()

        self.running = True
        self.fixed_maps = self._discover_fixed_maps()
        self.map_index = 0
        self.random_history = []
        self.random_map_index = -1
        self.env: MazePPOEnv | None = None
        self.obs = None
        self.info = {}
        self.model = None
        self.model_is_recurrent = False
        self.replay_recurrent_state = None
        self.replay_episode_start = None
        self.screen_mode = "training"
        self.mode = "ready"
        self.status = "准备就绪"
        self.current_map_label = ""
        self.current_map_kind = "fixed"
        self.task_name = ""
        self.task_process: subprocess.Popen | None = None
        self.task_queue: queue.Queue[str] = queue.Queue()
        self.task_log = deque(maxlen=9)
        self.training_timesteps = 50_000
        self.random_maps = 50
        self.random_tests = 10
        self.random_rows = RANDOM_MAP_ROWS
        self.random_cols = RANDOM_MAP_COLS
        self.wall_density = RANDOM_WALL_DENSITY
        self.trap_density = RANDOM_TRAP_DENSITY
        self.random_style = RANDOM_MAP_STYLE
        self.door_orientation = RANDOM_DOOR_ORIENTATION
        self.endpoint_mode = RANDOM_ENDPOINT_MODE
        self.ent_coef = 0.03
        self.n_envs = PPO_N_ENVS
        self.ppo_algo = PPO_ALGO
        self.curriculum = PPO_CURRICULUM
        self.obs_mode = OBSERVATION_MODE
        self.local_view_size = LOCAL_VIEW_SIZE
        self.memory_assist = False
        self.deterministic_experiment = False
        self.random_experiment_index = 0
        self.buttons: list[Button] = []
        self.cell_size = 40
        self.board_origin = (MARGIN, HEADER_HEIGHT)
        self.replay_last_tick = 0
        self.replay_trace: list[str] = []
        self.replay_state_visits: dict[tuple[int, int, bool], int] = {}
        self.replay_recent_states = deque(maxlen=8)
        self.remembered_key: tuple[int, int] | None = None
        self.remembered_door: tuple[int, int] | None = None
        self.remembered_exit: tuple[int, int] | None = None

        if not self.fixed_maps:
            raise RuntimeError(f"No maps found in {MAPS_DIR}")
        self.load_map(0)

    def run(self):
        while self.running:
            self._handle_events()
            self._update()
            self._draw()
            self.clock.tick(FPS)
        self._stop_task()
        pygame.quit()

    def smoke_check(self):
        self._draw()
        pygame.display.flip()
        pygame.quit()
        print("PPO app smoke OK")

    def load_map(self, index: int):
        self.fixed_maps = self._discover_fixed_maps()
        self.map_index = index % len(self.fixed_maps)
        self.env = MazePPOEnv(
            map_path=self.fixed_maps[self.map_index],
            max_steps=MAX_STEPS,
            observation_mode=self.obs_mode,
            local_view_size=self.local_view_size,
        )
        self.obs, self.info = self.env.reset()
        self.model = None
        self.mode = "ready"
        self.current_map_label = self.fixed_maps[self.map_index].name
        self.current_map_kind = "fixed"
        self.status = "已载入固定地图"
        self._layout_board()

    def load_random_experiment_map(self, index: int | None = None):
        if self.task_process is not None:
            self.status = "后台任务运行中，暂不能换图"
            return
        if index is None:
            generated = self._new_random_map()
            self.random_history.append(generated)
            self.random_map_index = len(self.random_history) - 1
            status = "已生成随机实验地图"
        else:
            if not self.random_history:
                generated = self._new_random_map()
                self.random_history.append(generated)
                self.random_map_index = 0
                status = "已生成随机实验地图"
            else:
                self.random_map_index = index % len(self.random_history)
                generated = self.random_history[self.random_map_index]
                status = "已载入随机实验地图"
        self.env = MazePPOEnv(
            map_lines=generated.lines,
            max_steps=MAX_STEPS,
            observation_mode=self.obs_mode,
            local_view_size=self.local_view_size,
        )
        self.obs, self.info = self.env.reset()
        self.model = None
        self.mode = "ready"
        self.current_map_label = generated.name
        self.current_map_kind = "random"
        self.status = status
        self._layout_board()

    def _discover_fixed_maps(self) -> list[Path]:
        return sorted(MAPS_DIR.glob("*.txt"))

    def _new_random_map(self):
        self.random_experiment_index += 1
        return generate_random_key_door_map(
            rows=self.random_rows,
            cols=self.random_cols,
            wall_density=self.wall_density,
            trap_density=self.trap_density,
            style=self.random_style,
            door_orientation=self.door_orientation,
            endpoint_mode=self.endpoint_mode,
            name=f"random_experiment_{self.random_experiment_index:03d}"
        )

    def _layout_board(self):
        assert self.env is not None
        available_width = WINDOW_WIDTH - PANEL_WIDTH - MARGIN * 3
        available_height = WINDOW_HEIGHT - HEADER_HEIGHT - MARGIN
        self.cell_size = max(
            16,
            min(58, available_width // self.env.cols, available_height // self.env.rows),
        )
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
                if event.key == pygame.K_ESCAPE:
                    self.mode = "ready"
                elif event.key == pygame.K_LEFT and self.screen_mode == "experiment":
                    self._prev_map()
                elif event.key == pygame.K_RIGHT and self.screen_mode == "experiment":
                    self._next_map()
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mouse_pos = pygame.mouse.get_pos()
                for button in self.buttons:
                    if button.handle_click(mouse_pos):
                        break

    def _update(self):
        self._drain_task_log()
        self._check_task_finished()
        if self.mode == "replay":
            now = pygame.time.get_ticks()
            if now - self.replay_last_tick >= REPLAY_DELAY_MS:
                self.replay_last_tick = now
                self._replay_step()

    def _prev_map(self):
        if self.task_process is None:
            self.screen_mode = "experiment"
            self.load_map(self.map_index - 1)

    def _next_map(self):
        if self.task_process is None:
            self.screen_mode = "experiment"
            self.load_map(self.map_index + 1)

    def _prev_random_map(self):
        if self.task_process is not None:
            return
        self.screen_mode = "experiment"
        if not self.random_history:
            self.load_random_experiment_map()
            return
        self.load_random_experiment_map(self.random_map_index - 1)

    def _next_random_map(self):
        if self.task_process is not None:
            return
        self.screen_mode = "experiment"
        if not self.random_history or self.random_map_index >= len(self.random_history) - 1:
            self.load_random_experiment_map()
            return
        self.load_random_experiment_map(self.random_map_index + 1)

    def _show_training_ground(self):
        self.screen_mode = "training"
        if self.mode == "replay":
            self.mode = "ready"
        self.status = "训练场：调整参数后开始训练"

    def _show_experiment_ground(self):
        self.screen_mode = "experiment"
        if self.mode == "task":
            self.status = "后台任务仍在运行，可查看日志"
        else:
            self.status = "试验场：选择固定或随机地图实验"

    def _cycle_timesteps(self):
        options = [2_000, 50_000, 200_000, 500_000, 1_000_000]
        next_index = (options.index(self.training_timesteps) + 1) % len(options)
        self.training_timesteps = options[next_index]
        self.status = f"训练步数设为 {self.training_timesteps:,}"

    def _cycle_random_maps(self):
        options = [0, 20, 50, 200, 500]
        next_index = (options.index(self.random_maps) + 1) % len(options)
        self.random_maps = options[next_index]
        self.status = f"随机地图数量设为 {self.random_maps}"

    def _cycle_random_tests(self):
        options = [5, 10, 50, 100, 200]
        next_index = (options.index(self.random_tests) + 1) % len(options)
        self.random_tests = options[next_index]
        self.status = f"随机评估次数设为 {self.random_tests}"

    def _cycle_random_size(self):
        options = [(9, 11), (11, 15), (13, 17), (15, 21), (17, 25), (21, 31)]
        current = (self.random_rows, self.random_cols)
        next_index = (options.index(current) + 1) % len(options) if current in options else 0
        self.random_rows, self.random_cols = options[next_index]
        self.status = f"随机图尺寸设为 {self.random_rows}x{self.random_cols}"

    def _cycle_wall_density(self):
        options = [0.0, 0.05, 0.10, 0.12, 0.18, 0.24, 0.30, 0.36]
        next_index = (options.index(self.wall_density) + 1) % len(options)
        self.wall_density = options[next_index]
        self.status = f"墙密度设为 {self.wall_density:.2f}"

    def _cycle_trap_density(self):
        options = [0.0, 0.02, 0.04, 0.08, 0.12, 0.16, 0.20]
        next_index = (options.index(self.trap_density) + 1) % len(options)
        self.trap_density = options[next_index]
        self.status = f"陷阱密度设为 {self.trap_density:.2f}"

    def _cycle_random_style(self):
        self.random_style = self._next_option(self.random_style, STYLE_OPTIONS)
        self.status = f"地图风格设为 {STYLE_LABELS[self.random_style]}"

    def _cycle_door_orientation(self):
        self.door_orientation = self._next_option(self.door_orientation, DOOR_OPTIONS)
        self.status = f"门方向设为 {DOOR_LABELS[self.door_orientation]}"

    def _cycle_endpoint_mode(self):
        self.endpoint_mode = self._next_option(self.endpoint_mode, ENDPOINT_OPTIONS)
        self.status = f"出入口位置设为 {ENDPOINT_LABELS[self.endpoint_mode]}"

    def _cycle_ent_coef(self):
        options = [0.005, 0.01, 0.03, 0.06, 0.10]
        next_index = (options.index(self.ent_coef) + 1) % len(options)
        self.ent_coef = options[next_index]
        self.status = f"探索强度设为 {self.ent_coef:.2f}"

    def _cycle_n_envs(self):
        options = [1, 2, 4, 6]
        next_index = (options.index(self.n_envs) + 1) % len(options)
        self.n_envs = options[next_index]
        self.status = f"并行环境数设为 {self.n_envs}"

    def _cycle_ppo_algo(self):
        self.ppo_algo = self._next_option(self.ppo_algo, ALGO_OPTIONS)
        self.status = f"训练算法设为 {ALGO_LABELS[self.ppo_algo]}"

    def _cycle_curriculum(self):
        self.curriculum = self._next_option(self.curriculum, CURRICULUM_OPTIONS)
        self.status = f"课程学习已{CURRICULUM_LABELS[self.curriculum]}"

    def _cycle_obs_mode(self):
        if self.task_process is not None or self.mode == "replay":
            self.status = "训练或回放中，暂不能切换观察格式"
            return
        self.obs_mode = self._next_option(self.obs_mode, OBS_MODE_OPTIONS)
        self._reload_current_env()
        self.status = f"观察格式设为 {OBS_MODE_LABELS[self.obs_mode]}"

    def _cycle_local_view_size(self):
        if self.task_process is not None or self.mode == "replay":
            self.status = "训练或回放中，暂不能切换视野大小"
            return
        index = LOCAL_VIEW_OPTIONS.index(self.local_view_size) if self.local_view_size in LOCAL_VIEW_OPTIONS else 0
        self.local_view_size = LOCAL_VIEW_OPTIONS[(index + 1) % len(LOCAL_VIEW_OPTIONS)]
        self._reload_current_env()
        self.status = f"局部视野设为 {self.local_view_size}x{self.local_view_size}"

    def _reload_current_env(self):
        if self.current_map_kind == "fixed":
            self.load_map(self.map_index)
        else:
            self.load_random_experiment_map(self.random_map_index)

    def _toggle_memory_assist(self):
        self.memory_assist = not self.memory_assist
        self.status = f"记忆辅助已{'开启' if self.memory_assist else '关闭'}"

    def _toggle_action_mode(self):
        self.deterministic_experiment = not self.deterministic_experiment
        mode = "贪心" if self.deterministic_experiment else "采样"
        self.status = f"试验动作模式：{mode}"

    def _reset_agent(self):
        if self.task_process is not None:
            self.status = "训练或评估运行中，不能重置"
            return
        self.model = None
        if not MODEL_PATH.exists():
            self.status = "智能体已是未训练状态"
            return

        backup_dir = MODELS_DIR / "reset_backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        backup_path = backup_dir / f"ppo_maze_{timestamp}.zip"
        MODEL_PATH.rename(backup_path)
        self.status = f"智能体已重置，旧模型已备份"

    def _start_training(self):
        args = [
            sys.executable,
            "train_ppo.py",
            "--timesteps",
            str(self.training_timesteps),
            "--random-maps",
            str(self.random_maps),
            "--random-rows",
            str(self.random_rows),
            "--random-cols",
            str(self.random_cols),
            "--wall-density",
            str(self.wall_density),
            "--trap-density",
            str(self.trap_density),
            "--random-style",
            self.random_style,
            "--door-orientation",
            self.door_orientation,
            "--endpoint-mode",
            self.endpoint_mode,
            "--ent-coef",
            str(self.ent_coef),
            "--n-envs",
            str(self.n_envs),
            "--algo",
            self.ppo_algo,
            "--curriculum",
            self.curriculum,
            "--obs-mode",
            self.obs_mode,
            "--local-view-size",
            str(self.local_view_size),
            "--model-path",
            str(MODEL_PATH),
            "--outputs-dir",
            str(OUTPUTS_DIR),
        ]
        self._start_task("训练 PPO", args)

    def _start_evaluation(self):
        if not MODEL_PATH.exists():
            self.status = "还没有模型，请先训练"
            return
        args = [
            sys.executable,
            "evaluate_ppo.py",
            "--model",
            str(MODEL_PATH),
            "--random-tests",
            str(self.random_tests),
            "--random-rows",
            str(self.random_rows),
            "--random-cols",
            str(self.random_cols),
            "--wall-density",
            str(self.wall_density),
            "--trap-density",
            str(self.trap_density),
            "--random-style",
            self.random_style,
            "--door-orientation",
            self.door_orientation,
            "--endpoint-mode",
            self.endpoint_mode,
            "--obs-mode",
            self.obs_mode,
            "--local-view-size",
            str(self.local_view_size),
            "--outputs-dir",
            str(OUTPUTS_DIR),
        ]
        if not self.deterministic_experiment:
            args.append("--sample-actions")
        self._start_task("评估模型", args)

    def _start_fixed_experiment(self):
        self.screen_mode = "experiment"
        if self.current_map_kind != "fixed":
            self.load_map(self.map_index)
        self._start_replay()

    def _start_random_experiment(self):
        self.screen_mode = "experiment"
        if self.current_map_kind != "random":
            if self.random_history:
                self.load_random_experiment_map(self.random_map_index)
            else:
                self.load_random_experiment_map()
        self._start_replay()

    def _save_current_random_as_fixed(self):
        if self.task_process is not None:
            self.status = "后台任务运行中，暂不能保存地图"
            return
        if self.current_map_kind != "random" or self.env is None or self.env.map_data is None:
            self.status = "当前不是随机地图，不能加入固定图"
            return

        MAPS_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base_name = f"saved_random_{timestamp}"
        path = MAPS_DIR / f"{base_name}.txt"
        suffix = 1
        while path.exists():
            path = MAPS_DIR / f"{base_name}_{suffix}.txt"
            suffix += 1

        path.write_text("\n".join(self.env.map_data.lines) + "\n")
        self.fixed_maps = self._discover_fixed_maps()
        self.map_index = self.fixed_maps.index(path)
        self.current_map_label = path.name
        self.current_map_kind = "fixed"
        self.status = f"随机图已加入固定图：{path.name}"

    def _start_task(self, task_name: str, args: list[str]):
        if self.task_process is not None:
            self.status = "已有后台任务正在运行"
            return
        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("MPLCONFIGDIR", str(OUTPUTS_DIR / ".matplotlib"))
        env.setdefault("XDG_CACHE_HOME", str(OUTPUTS_DIR / ".cache"))
        self.task_name = task_name
        self.task_log.clear()
        self.status = f"{task_name} 已启动"
        self.mode = "task"
        self.task_process = subprocess.Popen(
            args,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        threading.Thread(target=self._read_task_output, daemon=True).start()

    def _read_task_output(self):
        assert self.task_process is not None
        assert self.task_process.stdout is not None
        for line in self.task_process.stdout:
            stripped = line.strip()
            if stripped:
                self.task_queue.put(stripped)

    def _drain_task_log(self):
        while True:
            try:
                line = self.task_queue.get_nowait()
            except queue.Empty:
                break
            display_line = self._format_task_log_line(line)
            if display_line:
                self.task_log.append(display_line)

    def _format_task_log_line(self, line: str) -> str | None:
        if self.task_name != "训练 PPO":
            return line

        if line.startswith("训练进度："):
            return line
        return None

    def _check_task_finished(self):
        if self.task_process is None:
            return
        code = self.task_process.poll()
        if code is None:
            return
        name = self.task_name
        self.task_process = None
        self.task_name = ""
        self.status = f"{name} 完成" if code == 0 else f"{name} 失败，退出码 {code}"
        self.mode = "ready"

    def _stop_task(self):
        if self.task_process is None:
            return
        self.task_process.terminate()
        try:
            self.task_process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self.task_process.kill()
        self.task_process = None
        self.status = "后台任务已停止"

    def _stop_current_activity(self):
        if self.task_process is not None:
            self._stop_task()
            return
        if self.mode == "replay":
            self.mode = "ready"
            self.status = "实验已停止"

    def _start_replay(self):
        if not MODEL_PATH.exists():
            self.status = "还没有模型，请先训练"
            return
        try:
            self.model, self.model_is_recurrent = load_trained_model(str(MODEL_PATH))
        except SystemExit:
            self.status = "缺少 PPO 依赖，请先安装 requirements.txt"
            return
        assert self.env is not None
        if self.model.observation_space.shape != self.env.observation_space.shape:
            self.model = None
            self.status = "模型视野格式已过期，请重置并重新训练"
            return
        self.obs, self.info = self.env.reset()
        self.replay_recurrent_state, self.replay_episode_start = initial_recurrent_state(
            self.model_is_recurrent
        )
        self.replay_state_visits = {}
        self.replay_recent_states.clear()
        self.remembered_key = None
        self.remembered_door = None
        self.remembered_exit = None
        self._remember_visible_targets()
        self._record_replay_state()
        self.replay_trace = [
            f"Replay for {self.current_map_label}",
            f"Map kind: {self.current_map_kind}",
            f"Model: {MODEL_PATH}",
            f"Model type: {'RecurrentPPO' if self.model_is_recurrent else 'PPO'}",
            f"Memory assist: {self.memory_assist}",
            f"Action mode: {'greedy' if self.deterministic_experiment else 'sample'}",
            "",
            f"0: pos={self.info['position']} reward=0.00 has_key={self.info['has_key']} event={self.info['event']}",
        ]
        self.mode = "replay"
        self.status = "正在回放当前模型"
        self.replay_last_tick = 0

    def _replay_step(self):
        if self.model is None or self.obs is None:
            self.mode = "ready"
            return
        assert self.env is not None
        self._remember_visible_targets()
        action, self.replay_recurrent_state, self.replay_episode_start = predict_action(
            self.model,
            self.obs,
            deterministic=self.deterministic_experiment,
            is_recurrent=self.model_is_recurrent,
            state=self.replay_recurrent_state,
            episode_start=self.replay_episode_start,
        )
        model_action = int(action)
        assisted_action = self._assist_action(model_action)
        assist_note = ""
        if assisted_action != model_action:
            assist_note = f" assist={ACTION_NAMES[assisted_action]}"
        action = assisted_action
        self.obs, reward, terminated, truncated, self.info = self.env.step(int(action))
        self._remember_visible_targets()
        self._record_replay_state()
        action_name = ACTION_NAMES[int(action)]
        self.replay_trace.append(
            (
                f"{self.info['steps']}: action={action_name} pos={self.info['position']} "
                f"reward={reward:.2f} total={self.info['total_reward']:.2f} "
                f"has_key={self.info['has_key']} door={self.info['passed_door']} "
                f"event={self.info['event']}{assist_note}"
            )
        )
        if terminated or truncated:
            self._save_replay_trace()
            self.mode = "ready"
            self.status = "回放成功" if self.info["success"] else "回放结束，未到出口"

    def _remember_visible_targets(self):
        assert self.env is not None
        for row, col in visible_cells(
            self.env.grid,
            self.env.agent_pos,
            self.env.view_range,
            self.env.view_width,
            observation_mode=self.env.observation_mode,
            local_view_size=self.env.local_view_size,
        ):
            tile = self.env.grid[row][col]
            if tile == TILE_KEY and not self.env.key_collected:
                self.remembered_key = (row, col)
            elif tile == TILE_DOOR:
                self.remembered_door = (row, col)
            elif tile == TILE_EXIT:
                self.remembered_exit = (row, col)

    def _record_replay_state(self):
        assert self.env is not None
        state = (self.env.agent_pos[0], self.env.agent_pos[1], self.env.has_key)
        self.replay_recent_states.append(state)
        self.replay_state_visits[state] = self.replay_state_visits.get(state, 0) + 1

    def _assist_action(self, model_action: int) -> int:
        if not self.memory_assist:
            return model_action
        assert self.env is not None

        legal_actions = self._legal_actions()
        if not legal_actions:
            return model_action
        if model_action not in legal_actions:
            return self._choose_exploration_action(legal_actions)

        target = self._current_memory_target()
        if target and self._is_looping(model_action):
            return self._choose_target_action(legal_actions, target)
        if self._is_looping(model_action):
            return self._choose_exploration_action(legal_actions)
        return model_action

    def _current_memory_target(self) -> tuple[int, int] | None:
        assert self.env is not None
        if not self.env.has_key and self.remembered_key is not None:
            return self.remembered_key
        if self.env.has_key and not self.env.passed_door and self.remembered_door is not None:
            return self.remembered_door
        if self.remembered_exit is not None:
            return self.remembered_exit
        return None

    def _legal_actions(self) -> list[int]:
        assert self.env is not None
        legal = []
        row, col = self.env.agent_pos
        for action, (dr, dc) in ACTIONS.items():
            nr = row + dr
            nc = col + dc
            if nr < 0 or nr >= self.env.rows or nc < 0 or nc >= self.env.cols:
                continue
            tile = self.env.grid[nr][nc]
            if tile == TILE_WALL:
                continue
            if tile == TILE_DOOR and not self.env.has_key:
                continue
            legal.append(action)

        safe = [
            action
            for action in legal
            if self._next_tile(action) != TILE_TRAP or len(legal) == 1
        ]
        return safe or legal

    def _choose_target_action(self, legal_actions: list[int], target: tuple[int, int]) -> int:
        assert self.env is not None

        def score(action: int) -> tuple[int, int]:
            nr, nc = self._next_position(action)
            distance = abs(nr - target[0]) + abs(nc - target[1])
            visits = self.replay_state_visits.get((nr, nc, self.env.has_key), 0)
            return distance, visits

        return min(legal_actions, key=score)

    def _choose_exploration_action(self, legal_actions: list[int]) -> int:
        assert self.env is not None

        def score(action: int) -> tuple[int, int]:
            nr, nc = self._next_position(action)
            visits = self.replay_state_visits.get((nr, nc, self.env.has_key), 0)
            backtrack = int(
                len(self.replay_recent_states) >= 2
                and (nr, nc, self.env.has_key) == self.replay_recent_states[-2]
            )
            return visits, backtrack

        return min(legal_actions, key=score)

    def _is_looping(self, action: int) -> bool:
        assert self.env is not None
        nr, nc = self._next_position(action)
        next_state = (nr, nc, self.env.has_key)
        if self.replay_state_visits.get(next_state, 0) >= 3:
            return True
        if len(self.replay_recent_states) >= 4:
            recent = list(self.replay_recent_states)
            two_cell_loop = recent[-1] == recent[-3] and recent[-2] == recent[-4]
            if two_cell_loop and next_state == recent[-2]:
                return True
        return False

    def _next_position(self, action: int) -> tuple[int, int]:
        assert self.env is not None
        dr, dc = ACTIONS[action]
        return self.env.agent_pos[0] + dr, self.env.agent_pos[1] + dc

    def _next_tile(self, action: int) -> str:
        assert self.env is not None
        row, col = self._next_position(action)
        return self.env.grid[row][col]

    def _save_replay_trace(self):
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        self.replay_trace.extend(
            [
                "",
                f"Success: {self.info['success']}",
                f"Has key: {self.info['has_key']}",
                f"Passed door: {self.info['passed_door']}",
                f"Steps: {self.info['steps']}",
                f"Total reward: {self.info['total_reward']:.2f}",
            ]
        )
        (OUTPUTS_DIR / "replay_trace.txt").write_text("\n".join(self.replay_trace) + "\n")

    def _draw(self):
        self.screen.fill(COLORS["background"])
        self._draw_header()
        self._draw_board()
        self._draw_panel()
        pygame.display.flip()

    def _draw_header(self):
        self.text.draw(self.screen, "AI Maze PPO Lab", MARGIN, 20, COLORS["text"], size=28)
        self.text.draw(
            self.screen,
            "训练场调参数和重置智能体，试验场用训练好的模型跑固定/随机地图",
            MARGIN + 310,
            28,
            COLORS["muted"],
            size=16,
        )

    def _draw_board(self):
        assert self.env is not None
        visible = visible_cells(
            self.env.grid,
            self.env.agent_pos,
            self.env.view_range,
            self.env.view_width,
            observation_mode=self.env.observation_mode,
            local_view_size=self.env.local_view_size,
        )
        ox, oy = self.board_origin

        for row in range(self.env.rows):
            for col in range(self.env.cols):
                tile = self.env.grid[row][col]
                draw_tile = TILE_EMPTY if tile == TILE_KEY and self.env.key_collected else tile
                rect = pygame.Rect(
                    ox + col * self.cell_size,
                    oy + row * self.cell_size,
                    self.cell_size,
                    self.cell_size,
                )
                pygame.draw.rect(self.screen, TILE_COLORS.get(draw_tile, COLORS["floor"]), rect)
                pygame.draw.rect(self.screen, COLORS["line"], rect, 1)

                if (row, col) not in visible:
                    overlay = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
                    overlay.fill(COLORS["hidden"])
                    self.screen.blit(overlay, rect.topleft)

                if tile in {TILE_START, TILE_EXIT, TILE_TRAP, TILE_KEY, TILE_DOOR}:
                    if not (tile == TILE_KEY and self.env.key_collected):
                        self.text.draw_centered(
                            self.screen,
                            tile,
                            rect.center,
                            (24, 28, 35),
                            size=max(14, self.cell_size // 2),
                        )

        self._draw_agent()

    def _draw_agent(self):
        assert self.env is not None
        ox, oy = self.board_origin
        row, col = self.env.agent_pos
        center = (
            ox + col * self.cell_size + self.cell_size // 2,
            oy + row * self.cell_size + self.cell_size // 2,
        )
        pygame.draw.circle(self.screen, COLORS["agent"], center, max(8, self.cell_size // 3))
        pygame.draw.circle(
            self.screen,
            COLORS["agent_outline"],
            center,
            max(8, self.cell_size // 3),
            2,
        )

    def _draw_panel(self):
        assert self.env is not None
        panel_x = WINDOW_WIDTH - PANEL_WIDTH - MARGIN
        pygame.draw.rect(
            self.screen,
            COLORS["surface"],
            pygame.Rect(panel_x, HEADER_HEIGHT, PANEL_WIDTH, WINDOW_HEIGHT - HEADER_HEIGHT - MARGIN),
            border_radius=8,
        )

        scene = "训练场" if self.screen_mode == "training" else "试验场"
        map_kind = "固定地图" if self.current_map_kind == "fixed" else "随机地图"
        key_text = "不需要" if not self.info.get("requires_key", True) else str(self.info.get("has_key", False))
        door_text = "无门" if not self.info.get("has_door", True) else str(self.info.get("passed_door", False))
        lines = [
            f"区域：{scene}",
            f"地图：{self._shorten(self.current_map_label, 21)}（{map_kind}）",
            f"状态：{self._shorten(self.status, 24)}",
            f"模型：{'已存在' if MODEL_PATH.exists() else '未训练'}  步数：{self.info.get('steps', 0)}/{self.env.max_steps}",
            f"奖励：{self.info.get('total_reward', 0.0):.1f}  钥匙：{key_text}  门：{door_text}",
            (
                f"观察：{OBS_MODE_LABELS[self.obs_mode]}  "
                f"{self.local_view_size}x{self.local_view_size}  朝向/上步/重复"
            ),
            f"随机：{self.random_rows}x{self.random_cols}  墙{self.wall_density:.2f}  陷{self.trap_density:.2f}",
            (
                f"风格：{STYLE_LABELS[self.random_style]}  门方向：{DOOR_LABELS[self.door_orientation]}  "
                f"位置：{ENDPOINT_LABELS[self.endpoint_mode]}"
            ),
        ]
        if self.screen_mode == "training":
            lines.append(
                f"训练：{ALGO_LABELS[self.ppo_algo]}  课程{CURRICULUM_LABELS[self.curriculum]}  "
                f"并行 {self.n_envs}"
            )
            lines.append(f"参数：entropy {self.ent_coef:.2f}  + 新格奖励")
        else:
            action_mode = "贪心" if self.deterministic_experiment else "采样"
            assist = "开" if self.memory_assist else "关"
            lines.append(f"试验策略：记忆辅助{assist} / 动作{action_mode}")

        y = HEADER_HEIGHT + 18
        for line in lines:
            self.text.draw(self.screen, line, panel_x + 18, y, COLORS["text"], size=14)
            y += 21

        buttons_bottom = self._build_buttons(panel_x, y + 8)
        mouse_pos = pygame.mouse.get_pos()
        for button in self.buttons:
            button.draw(self.screen, self.text, mouse_pos)

        log_top = max(buttons_bottom + 8, WINDOW_HEIGHT - 150)
        self.text.draw(self.screen, "后台日志", panel_x + 18, log_top, COLORS["muted"], size=15)
        y = log_top + 28
        for line in list(self.task_log)[-3:]:
            self.text.draw(
                self.screen,
                self._shorten(line, 38),
                panel_x + 18,
                y,
                COLORS["muted"],
                size=13,
            )
            y += 23

        hint = "提示：试验场可用左右方向键切换固定地图，Esc 停止回放"
        self.text.draw(self.screen, hint, panel_x + 18, WINDOW_HEIGHT - 48, COLORS["muted"], size=13)

    def _build_buttons(self, panel_x: int, top: int):
        task_busy = self.task_process is not None
        replay_busy = self.mode == "replay"
        busy = task_busy or replay_busy
        button_width = 148
        gap = 4
        x1 = panel_x + 18
        x2 = x1 + button_width + gap
        y = top
        height = 24

        self.buttons = [
            Button((x1, y, button_width, height), "训练场", self._show_training_ground, True),
            Button((x2, y, button_width, height), "试验场", self._show_experiment_ground, True),
        ]
        y += height + gap

        if self.screen_mode == "training":
            self.buttons.extend(
                [
                    Button(
                        (x1, y, button_width, height),
                        f"步数 {self.training_timesteps:,}",
                        self._cycle_timesteps,
                        not busy,
                    ),
                    Button(
                        (x2, y, button_width, height),
                        f"随机图 {self.random_maps}",
                        self._cycle_random_maps,
                        not busy,
                    ),
                ]
            )
            y += height + gap
            self.buttons.extend(
                [
                    Button((x1, y, button_width, height), f"尺寸 {self.random_rows}x{self.random_cols}", self._cycle_random_size, not busy),
                    Button((x2, y, button_width, height), f"风格 {STYLE_LABELS[self.random_style]}", self._cycle_random_style, not busy),
                ]
            )
            y += height + gap
            self.buttons.extend(
                [
                    Button((x1, y, button_width, height), f"观察 {OBS_MODE_LABELS[self.obs_mode]}", self._cycle_obs_mode, not busy),
                    Button((x2, y, button_width, height), f"视野 {self.local_view_size}x{self.local_view_size}", self._cycle_local_view_size, not busy),
                ]
            )
            y += height + gap
            self.buttons.extend(
                [
                    Button((x1, y, button_width, height), f"门方向 {DOOR_LABELS[self.door_orientation]}", self._cycle_door_orientation, not busy),
                    Button((x2, y, button_width, height), f"位置 {ENDPOINT_LABELS[self.endpoint_mode]}", self._cycle_endpoint_mode, not busy),
                ]
            )
            y += height + gap
            self.buttons.extend(
                [
                    Button((x1, y, button_width, height), f"墙 {self.wall_density:.2f}", self._cycle_wall_density, not busy),
                    Button((x2, y, button_width, height), f"陷阱 {self.trap_density:.2f}", self._cycle_trap_density, not busy),
                ]
            )
            y += height + gap
            self.buttons.extend(
                [
                    Button((x1, y, button_width, height), f"探索 {self.ent_coef:.2f}", self._cycle_ent_coef, not busy),
                    Button((x2, y, button_width, height), f"并行 {self.n_envs}", self._cycle_n_envs, not busy),
                ]
            )
            y += height + gap
            self.buttons.extend(
                [
                    Button((x1, y, button_width, height), f"算法 {ALGO_LABELS[self.ppo_algo]}", self._cycle_ppo_algo, not busy),
                    Button((x2, y, button_width, height), f"课程 {CURRICULUM_LABELS[self.curriculum]}", self._cycle_curriculum, not busy),
                ]
            )
            y += height + gap
            self.buttons.extend(
                [
                    Button((x1, y, button_width, height), "开始训练", self._start_training, not busy),
                    Button((x2, y, button_width, height), "重置智能体", self._reset_agent, not busy),
                ]
            )
            y += height + gap
            self.buttons.extend(
                [
                    Button((x1, y, button_width, height), "停止训练", self._stop_current_activity, task_busy),
                    Button((x2, y, button_width, height), "去试验场", self._show_experiment_ground, True),
                ]
            )
            y += height
        else:
            self.buttons.extend(
                [
                    Button((x1, y, button_width, height), "上一张固定图", self._prev_map, not busy),
                    Button((x2, y, button_width, height), "下一张固定图", self._next_map, not busy),
                ]
            )
            y += height + gap
            self.buttons.extend(
                [
                    Button((x1, y, button_width, height), "上一张随机图", self._prev_random_map, not busy),
                    Button((x2, y, button_width, height), "下一张随机图", self._next_random_map, not busy),
                ]
            )
            y += height + gap
            self.buttons.extend(
                [
                    Button((x1, y, button_width, height), "试验固定图", self._start_fixed_experiment, not busy),
                    Button((x2, y, button_width, height), "试验随机图", self._start_random_experiment, not busy),
                ]
            )
            y += height + gap
            self.buttons.append(
                Button(
                    (x1, y, button_width * 2 + gap, height),
                    "当前随机图加入固定图",
                    self._save_current_random_as_fixed,
                    not busy,
                )
            )
            y += height + gap
            self.buttons.extend(
                [
                    Button(
                        (x1, y, button_width, height),
                        f"尺寸 {self.random_rows}x{self.random_cols}",
                        self._cycle_random_size,
                        not busy,
                    ),
                    Button((x2, y, button_width, height), f"风格 {STYLE_LABELS[self.random_style]}", self._cycle_random_style, not busy),
                ]
            )
            y += height + gap
            self.buttons.extend(
                [
                    Button((x1, y, button_width, height), f"观察 {OBS_MODE_LABELS[self.obs_mode]}", self._cycle_obs_mode, not busy),
                    Button((x2, y, button_width, height), f"视野 {self.local_view_size}x{self.local_view_size}", self._cycle_local_view_size, not busy),
                ]
            )
            y += height + gap
            self.buttons.extend(
                [
                    Button((x1, y, button_width, height), f"门方向 {DOOR_LABELS[self.door_orientation]}", self._cycle_door_orientation, not busy),
                    Button((x2, y, button_width, height), f"位置 {ENDPOINT_LABELS[self.endpoint_mode]}", self._cycle_endpoint_mode, not busy),
                ]
            )
            y += height + gap
            self.buttons.extend(
                [
                    Button((x1, y, button_width, height), f"墙 {self.wall_density:.2f}", self._cycle_wall_density, not busy),
                    Button((x2, y, button_width, height), f"陷阱 {self.trap_density:.2f}", self._cycle_trap_density, not busy),
                ]
            )
            y += height + gap
            self.buttons.extend(
                [
                    Button(
                        (x1, y, button_width, height),
                        "记忆辅助 开" if self.memory_assist else "记忆辅助 关",
                        self._toggle_memory_assist,
                        not busy,
                    ),
                    Button(
                        (x2, y, button_width, height),
                        "动作 贪心" if self.deterministic_experiment else "动作 采样",
                        self._toggle_action_mode,
                        not busy,
                    ),
                ]
            )
            y += height + gap
            self.buttons.extend(
                [
                    Button((x1, y, button_width, height), f"评估 {self.random_tests}", self._cycle_random_tests, not busy),
                    Button((x2, y, button_width, height), "批量评估", self._start_evaluation, not busy),
                ]
            )
            y += height + gap
            self.buttons.append(
                Button((x1, y, button_width * 2 + gap, height), "停止实验", self._stop_current_activity, busy)
            )
            y += height

        return y

    @staticmethod
    def _shorten(text: str, max_len: int) -> str:
        return text if len(text) <= max_len else text[: max_len - 3] + "..."

    @staticmethod
    def _format_target(target: tuple[int, int] | None) -> str:
        return "-" if target is None else f"({target[0]},{target[1]})"

    @staticmethod
    def _next_option(current: str, options: tuple[str, ...]) -> str:
        index = options.index(current) if current in options else 0
        return options[(index + 1) % len(options)]


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="AI Maze PPO Lab desktop app.")
    parser.add_argument("--smoke", action="store_true", help="Open one frame and exit.")
    args = parser.parse_args(argv)

    app = MazePPOApp()
    if args.smoke:
        app.smoke_check()
    else:
        app.run()


if __name__ == "__main__":
    main()
