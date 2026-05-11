from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover - exercised only before install.
    raise ImportError(
        "Missing Gymnasium. Install dependencies with: pip install -r requirements.txt"
    ) from exc

from config import (
    ACTIONS,
    DISCOVERY_REWARD,
    DOOR_REWARD,
    EXIT_REWARD,
    KEY_REWARD,
    LOCAL_VIEW_SIZE,
    LOCKED_DOOR_REWARD,
    MAX_STEPS,
    MAX_STEPS_PER_CELL,
    OBSERVATION_MODE,
    RECENT_VISIT_WINDOW,
    REVISIT_PENALTY,
    STEP_REWARD,
    TILE_DOOR,
    TILE_EMPTY,
    TILE_EXIT,
    TILE_KEY,
    TILE_START,
    TILE_TRAP,
    TILE_WALL,
    TRAP_REWARD,
    VIEW_RANGE,
    VIEW_WIDTH,
    WALL_REWARD,
)
from random_maps import GeneratedMap
from vision import encode_line_of_sight, encode_local_grid, format_observation_shape


@dataclass(frozen=True)
class MapData:
    name: str
    lines: list[str]
    grid: list[list[str]]
    start: tuple[int, int]
    exit: tuple[int, int]
    has_key_tile: bool
    has_door_tile: bool


RandomMapFactory = Callable[[np.random.Generator], GeneratedMap]


def load_map_file(path: str | Path) -> list[str]:
    map_path = Path(path)
    lines = [line.rstrip("\n") for line in map_path.read_text().splitlines() if line]
    if not lines:
        raise ValueError(f"Map is empty: {map_path}")
    return lines


def parse_map(lines: list[str], name: str = "map") -> MapData:
    width = len(lines[0])
    if any(len(line) != width for line in lines):
        raise ValueError(f"Map {name} must be rectangular")

    start_positions: list[tuple[int, int]] = []
    exit_positions: list[tuple[int, int]] = []
    has_key_tile = False
    has_door_tile = False
    valid_tiles = {
        TILE_START,
        TILE_EXIT,
        TILE_WALL,
        TILE_EMPTY,
        TILE_TRAP,
        TILE_KEY,
        TILE_DOOR,
    }

    for row, line in enumerate(lines):
        for col, tile in enumerate(line):
            if tile not in valid_tiles:
                raise ValueError(f"Invalid tile {tile!r} at {(row, col)} in {name}")
            if tile == TILE_START:
                start_positions.append((row, col))
            elif tile == TILE_EXIT:
                exit_positions.append((row, col))
            elif tile == TILE_KEY:
                has_key_tile = True
            elif tile == TILE_DOOR:
                has_door_tile = True

    if len(start_positions) != 1:
        raise ValueError(f"Map {name} must contain exactly one S")
    if len(exit_positions) != 1:
        raise ValueError(f"Map {name} must contain exactly one E")

    return MapData(
        name=name,
        lines=list(lines),
        grid=[list(line) for line in lines],
        start=start_positions[0],
        exit=exit_positions[0],
        has_key_tile=has_key_tile,
        has_door_tile=has_door_tile,
    )


class MazePPOEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        map_path: str | Path | None = None,
        map_lines: list[str] | None = None,
        map_sources: list[str | Path] | None = None,
        random_map_factory: RandomMapFactory | None = None,
        random_map_probability: float = 0.0,
        max_steps: int = MAX_STEPS,
        view_range: int = VIEW_RANGE,
        view_width: int = VIEW_WIDTH,
        observation_mode: str = OBSERVATION_MODE,
        local_view_size: int = LOCAL_VIEW_SIZE,
        recent_visit_window: int = RECENT_VISIT_WINDOW,
        exploration_reward: bool = False,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.rng = np.random.default_rng(seed)
        self.base_max_steps = max_steps
        self.max_steps = max_steps
        self.view_range = view_range
        self.view_width = view_width
        self.observation_mode = observation_mode
        self.local_view_size = local_view_size
        self.recent_visit_window = recent_visit_window
        self.exploration_reward = exploration_reward
        self.random_map_factory = random_map_factory
        self.random_map_probability = float(np.clip(random_map_probability, 0.0, 1.0))
        self.map_lines = map_lines
        self.map_sources = [Path(source) for source in (map_sources or [])]
        if map_path is not None:
            self.map_sources.insert(0, Path(map_path))

        if self.map_lines is None and not self.map_sources and random_map_factory is None:
            raise ValueError("Provide map_path, map_lines, map_sources, or random_map_factory")

        self.action_space = spaces.Discrete(len(ACTIONS))
        obs_shape = format_observation_shape(
            view_range,
            view_width,
            observation_mode,
            local_view_size,
        )
        obs_dtype = np.uint8 if observation_mode == "grid" else np.float32
        self.observation_space = spaces.Box(
            low=0,
            high=255 if observation_mode == "grid" else 1.0,
            shape=obs_shape,
            dtype=obs_dtype,
        )

        self.map_data: MapData | None = None
        self.grid: list[list[str]] = []
        self.agent_pos = (0, 0)
        self.has_key = False
        self.key_collected = False
        self.passed_door = False
        self.reached_exit = False
        self.step_count = 0
        self.total_reward = 0.0
        self.last_event = "reset"
        self.visit_counts: np.ndarray | None = None
        self.facing_action = 3
        self.last_action = -1
        self.last_reward = 0.0
        self.recent_positions = deque(maxlen=max(1, int(recent_visit_window)))

    @property
    def rows(self) -> int:
        return len(self.grid)

    @property
    def cols(self) -> int:
        return len(self.grid[0]) if self.grid else 0

    @property
    def map_name(self) -> str:
        return self.map_data.name if self.map_data else "unloaded"

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.map_data = self._choose_map()
        self.max_steps = self._max_steps_for_map(self.map_data)
        self.grid = [row[:] for row in self.map_data.grid]
        self.agent_pos = self.map_data.start
        self.has_key = not self.map_data.has_key_tile
        self.key_collected = not self.map_data.has_key_tile
        self.passed_door = not self.map_data.has_door_tile
        self.reached_exit = False
        self.step_count = 0
        self.total_reward = 0.0
        self.last_event = "reset"
        self.visit_counts = np.zeros((self.rows, self.cols), dtype=np.int32)
        self.visit_counts[self.agent_pos] = 1
        self.facing_action = 3
        self.last_action = -1
        self.last_reward = 0.0
        self.recent_positions = deque(maxlen=max(1, int(self.recent_visit_window)))
        self.recent_positions.append(self.agent_pos)

        return self._observation(), self._info()

    def step(self, action: int):
        action = int(action)
        if action not in ACTIONS:
            raise ValueError(f"Invalid action: {action}")

        self.step_count += 1
        self.facing_action = action
        reward = STEP_REWARD
        terminated = False
        truncated = False
        self.last_event = "move"

        dr, dc = ACTIONS[action]
        next_pos = (self.agent_pos[0] + dr, self.agent_pos[1] + dc)
        row, col = next_pos

        if row < 0 or row >= self.rows or col < 0 or col >= self.cols:
            reward = WALL_REWARD
            self.last_event = "wall"
        else:
            tile = self.grid[row][col]
            if tile == TILE_WALL:
                reward = WALL_REWARD
                self.last_event = "wall"
            elif tile == TILE_DOOR and not self.has_key:
                reward = LOCKED_DOOR_REWARD
                self.last_event = "locked_door"
            else:
                self.agent_pos = next_pos
                if tile == TILE_KEY and not self.key_collected:
                    self.has_key = True
                    self.key_collected = True
                    reward += KEY_REWARD
                    self.last_event = "key"
                elif tile == TILE_DOOR:
                    if not self.passed_door:
                        reward += DOOR_REWARD
                    self.passed_door = True
                    self.last_event = "door"
                elif tile == TILE_TRAP:
                    reward = TRAP_REWARD
                    terminated = True
                    self.last_event = "trap"
                elif tile == TILE_EXIT:
                    reward = EXIT_REWARD
                    self.reached_exit = True
                    terminated = True
                    self.last_event = "exit"

                if self.exploration_reward:
                    reward += self._mark_visit_and_get_exploration_reward()

        if self.step_count >= self.max_steps and not terminated:
            truncated = True
            self.last_event = "max_steps"

        self.total_reward += reward
        self.last_action = action
        self.last_reward = reward
        self.recent_positions.append(self.agent_pos)
        return self._observation(), reward, terminated, truncated, self._info()

    def _mark_visit_and_get_exploration_reward(self) -> float:
        if self.visit_counts is None:
            return 0.0
        row, col = self.agent_pos
        previous_visits = int(self.visit_counts[row, col])
        self.visit_counts[row, col] += 1
        if previous_visits == 0:
            return DISCOVERY_REWARD
        return REVISIT_PENALTY

    def _max_steps_for_map(self, map_data: MapData) -> int:
        base_steps = self.base_max_steps if self.base_max_steps > 0 else MAX_STEPS
        map_scaled_steps = int(len(map_data.lines) * len(map_data.lines[0]) * MAX_STEPS_PER_CELL)
        return max(base_steps, map_scaled_steps)

    def _choose_map(self) -> MapData:
        use_random = (
            self.random_map_factory is not None
            and (not self.map_sources or float(self.rng.random()) < self.random_map_probability)
        )
        if use_random:
            generated = self.random_map_factory(self.rng)
            return parse_map(generated.lines, generated.name)

        if self.map_lines is not None and not self.map_sources:
            return parse_map(self.map_lines, "inline_map")

        if self.map_sources:
            index = int(self.rng.integers(0, len(self.map_sources)))
            path = self.map_sources[index]
            return parse_map(load_map_file(path), str(path))

        if self.map_lines is not None:
            return parse_map(self.map_lines, "inline_map")

        raise RuntimeError("No map source available")

    def _observation(self) -> np.ndarray:
        if self.observation_mode == "grid":
            return encode_local_grid(
                self.grid,
                self.agent_pos,
                self.has_key,
                self.key_collected,
                self.step_count,
                self.max_steps,
                self.local_view_size,
                self.facing_action,
                self.last_action,
                self.last_reward,
                list(self.recent_positions),
            )
        return encode_line_of_sight(
            self.grid,
            self.agent_pos,
            self.has_key,
            self.key_collected,
            self.step_count,
            self.max_steps,
            self.view_range,
            self.view_width,
        )

    def _info(self) -> dict:
        return {
            "map_name": self.map_name,
            "position": self.agent_pos,
            "has_key": self.has_key,
            "key_collected": self.key_collected,
            "requires_key": self.map_data.has_key_tile if self.map_data else False,
            "has_door": self.map_data.has_door_tile if self.map_data else False,
            "passed_door": self.passed_door,
            "success": self.reached_exit,
            "steps": self.step_count,
            "total_reward": self.total_reward,
            "event": self.last_event,
            "facing_action": self.facing_action,
            "last_action": self.last_action,
            "last_reward": self.last_reward,
            "visit_count": int(self.visit_counts[self.agent_pos]) if self.visit_counts is not None else 0,
        }
