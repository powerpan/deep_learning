from __future__ import annotations

from typing import Iterable

import numpy as np

from config import (
    ACTIONS,
    MAX_STEPS,
    TILE_CHANNELS,
    TILE_DOOR,
    TILE_EMPTY,
    TILE_EXIT,
    TILE_KEY,
    TILE_START,
    TILE_TRAP,
    TILE_WALL,
    VIEW_RANGE,
)

CHANNEL_INDEX = {name: idx for idx, name in enumerate(TILE_CHANNELS)}


def observation_size(view_range: int = VIEW_RANGE) -> int:
    return len(ACTIONS) * view_range * len(TILE_CHANNELS) + 2


def _tile_channel(tile: str, key_collected: bool) -> str:
    if tile == TILE_WALL:
        return "wall"
    if tile == TILE_TRAP:
        return "trap"
    if tile == TILE_KEY and not key_collected:
        return "key"
    if tile == TILE_DOOR:
        return "door"
    if tile == TILE_EXIT:
        return "exit"
    if tile in {TILE_EMPTY, TILE_START, TILE_KEY}:
        return "empty"
    return "unknown"


def _one_hot(channel: str) -> np.ndarray:
    encoded = np.zeros(len(TILE_CHANNELS), dtype=np.float32)
    encoded[CHANNEL_INDEX[channel]] = 1.0
    return encoded


def encode_line_of_sight(
    grid: list[list[str]],
    agent_pos: tuple[int, int],
    has_key: bool,
    key_collected: bool,
    step_count: int,
    max_steps: int = MAX_STEPS,
    view_range: int = VIEW_RANGE,
) -> np.ndarray:
    """Encode four straight-line rays around the agent.

    The wall tile itself is visible. Tiles behind a wall are encoded as unknown.
    """
    rows = len(grid)
    cols = len(grid[0])
    encoded = np.zeros(
        (len(ACTIONS), view_range, len(TILE_CHANNELS)), dtype=np.float32
    )

    for direction_index, (_, (dr, dc)) in enumerate(ACTIONS.items()):
        blocked = False
        for distance in range(1, view_range + 1):
            ray_index = distance - 1
            row = agent_pos[0] + dr * distance
            col = agent_pos[1] + dc * distance

            if blocked:
                encoded[direction_index, ray_index] = _one_hot("unknown")
                continue

            if row < 0 or row >= rows or col < 0 or col >= cols:
                encoded[direction_index, ray_index] = _one_hot("wall")
                blocked = True
                continue

            tile = grid[row][col]
            encoded[direction_index, ray_index] = _one_hot(
                _tile_channel(tile, key_collected)
            )
            if tile == TILE_WALL:
                blocked = True

    step_ratio = min(float(step_count) / max(float(max_steps), 1.0), 1.0)
    extras = np.array([1.0 if has_key else 0.0, step_ratio], dtype=np.float32)
    return np.concatenate([encoded.reshape(-1), extras])


def visible_cells(
    grid: list[list[str]],
    agent_pos: tuple[int, int],
    view_range: int = VIEW_RANGE,
) -> set[tuple[int, int]]:
    rows = len(grid)
    cols = len(grid[0])
    cells = {agent_pos}

    for dr, dc in ACTIONS.values():
        for distance in range(1, view_range + 1):
            row = agent_pos[0] + dr * distance
            col = agent_pos[1] + dc * distance
            if row < 0 or row >= rows or col < 0 or col >= cols:
                break
            cells.add((row, col))
            if grid[row][col] == TILE_WALL:
                break

    return cells


def format_observation_shape(view_range: int = VIEW_RANGE) -> tuple[int, ...]:
    return (observation_size(view_range),)
