from pathlib import Path

from config import (
    ACTION_DELTAS,
    DOOR_REWARD,
    EXIT_REWARD,
    KEY_REWARD,
    LOCKED_DOOR_REWARD,
    MAX_STEPS_PER_EPISODE,
    STEP_REWARD,
    TRAP_REWARD,
    WALL_REWARD,
)


class MazeEnv:
    """Grid maze environment with a small Q-learning friendly API."""

    VALID_TILES = {"S", "E", "#", ".", "T", "K", "D"}

    def __init__(self, map_path, max_steps=MAX_STEPS_PER_EPISODE):
        self.map_path = Path(map_path)
        self.max_steps = max_steps
        self.grid = self._load_map(self.map_path)
        self.rows = len(self.grid)
        self.cols = len(self.grid[0])
        self.agent_pos = self.start_pos
        self.has_key = False
        self.passed_door = False
        self.steps = 0

    def _load_map(self, map_path):
        if not map_path.exists():
            raise FileNotFoundError(f"Map file not found: {map_path}")

        lines = [
            line.rstrip("\n")
            for line in map_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not lines:
            raise ValueError(f"Map file is empty: {map_path}")

        width = len(lines[0])
        start_positions = []
        exit_positions = []
        grid = []

        for row, line in enumerate(lines):
            if len(line) != width:
                raise ValueError("Map must be rectangular: all rows need the same length.")

            grid_row = []
            for col, tile in enumerate(line):
                if tile not in self.VALID_TILES:
                    raise ValueError(f"Invalid tile {tile!r} at row {row}, col {col}.")
                if tile == "S":
                    start_positions.append((row, col))
                elif tile == "E":
                    exit_positions.append((row, col))
                grid_row.append(tile)
            grid.append(grid_row)

        if len(start_positions) != 1:
            raise ValueError("Map must contain exactly one start tile S.")
        if len(exit_positions) != 1:
            raise ValueError("Map must contain exactly one exit tile E.")

        self.start_pos = start_positions[0]
        self.exit_pos = exit_positions[0]
        return grid

    def reset(self):
        self.agent_pos = self.start_pos
        self.has_key = False
        self.passed_door = False
        self.steps = 0
        return self.get_state()

    def get_state(self):
        row, col = self.agent_pos
        return row, col, int(self.has_key)

    def step(self, action):
        if action not in ACTION_DELTAS:
            raise ValueError(f"Unknown action {action}. Expected one of {list(ACTION_DELTAS)}.")

        self.steps += 1
        row, col = self.agent_pos
        delta_row, delta_col = ACTION_DELTAS[action]
        next_row = row + delta_row
        next_col = col + delta_col

        reward = STEP_REWARD
        done = False
        success = False
        picked_key = False
        door_passed_this_step = False
        reason = "move"

        if self._is_blocked(next_row, next_col):
            reward = WALL_REWARD
            next_row, next_col = row, col
            reason = "wall"
        elif self.tile_at(next_row, next_col) == "D" and not self.has_key:
            reward = LOCKED_DOOR_REWARD
            next_row, next_col = row, col
            reason = "locked_door"
        else:
            self.agent_pos = (next_row, next_col)
            tile = self.tile_at(next_row, next_col)
            if tile == "K" and not self.has_key:
                self.has_key = True
                picked_key = True
                reward += KEY_REWARD
                reason = "key"
            elif tile == "D" and self.has_key:
                door_passed_this_step = True
                reason = "door"
                if not self.passed_door:
                    reward += DOOR_REWARD
                self.passed_door = True

            if tile == "T":
                reward = TRAP_REWARD
                done = True
                reason = "trap"
            elif tile == "E":
                reward = EXIT_REWARD
                done = True
                success = True
                reason = "exit"

        if not done and self.steps >= self.max_steps:
            done = True
            reason = "timeout"

        info = {
            "success": success,
            "reason": reason,
            "steps": self.steps,
            "has_key": self.has_key,
            "picked_key": picked_key,
            "passed_door": self.passed_door,
            "door_passed_this_step": door_passed_this_step,
        }
        return self.get_state(), reward, done, info

    def tile_at(self, row, col):
        return self.grid[row][col]

    def _is_blocked(self, row, col):
        if row < 0 or row >= self.rows or col < 0 or col >= self.cols:
            return True
        return self.grid[row][col] == "#"
