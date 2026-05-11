from __future__ import annotations

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
    VIEW_WIDTH,
)

CHANNEL_INDEX = {name: idx for idx, name in enumerate(TILE_CHANNELS)}


def observation_size(view_range: int = VIEW_RANGE, view_width: int = VIEW_WIDTH) -> int:
    return len(ACTIONS) * view_range * view_width * len(TILE_CHANNELS) + 2


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
    view_width: int = VIEW_WIDTH,
) -> np.ndarray:
    """Encode four directional 3-wide strips around the agent.

    Each direction sees a narrow forward strip. A wall tile is visible, and
    tiles behind that wall in the same strip lane are encoded as unknown.
    """
    rows = len(grid)
    cols = len(grid[0])
    side_offsets = _side_offsets(view_width)
    encoded = np.zeros(
        (len(ACTIONS), view_range, view_width, len(TILE_CHANNELS)), dtype=np.float32
    )

    for direction_index, (_, (dr, dc)) in enumerate(ACTIONS.items()):
        blocked = [False for _ in side_offsets]
        for distance in range(1, view_range + 1):
            distance_index = distance - 1
            for side_index, side_offset in enumerate(side_offsets):
                row, col = _strip_cell(agent_pos, dr, dc, distance, side_offset)

                if blocked[side_index]:
                    encoded[direction_index, distance_index, side_index] = _one_hot("unknown")
                    continue

                if row < 0 or row >= rows or col < 0 or col >= cols:
                    encoded[direction_index, distance_index, side_index] = _one_hot("wall")
                    blocked[side_index] = True
                    continue

                tile = grid[row][col]
                encoded[direction_index, distance_index, side_index] = _one_hot(
                    _tile_channel(tile, key_collected)
                )
                if tile == TILE_WALL:
                    blocked[side_index] = True

    step_ratio = min(float(step_count) / max(float(max_steps), 1.0), 1.0)
    extras = np.array([1.0 if has_key else 0.0, step_ratio], dtype=np.float32)
    return np.concatenate([encoded.reshape(-1), extras])


def visible_cells(
    grid: list[list[str]],
    agent_pos: tuple[int, int],
    view_range: int = VIEW_RANGE,
    view_width: int = VIEW_WIDTH,
) -> set[tuple[int, int]]:
    rows = len(grid)
    cols = len(grid[0])
    cells = {agent_pos}
    side_offsets = _side_offsets(view_width)

    for dr, dc in ACTIONS.values():
        blocked = [False for _ in side_offsets]
        for distance in range(1, view_range + 1):
            for side_index, side_offset in enumerate(side_offsets):
                if blocked[side_index]:
                    continue
                row, col = _strip_cell(agent_pos, dr, dc, distance, side_offset)
                if row < 0 or row >= rows or col < 0 or col >= cols:
                    blocked[side_index] = True
                    continue
                cells.add((row, col))
                if grid[row][col] == TILE_WALL:
                    blocked[side_index] = True

    return cells


def format_observation_shape(
    view_range: int = VIEW_RANGE,
    view_width: int = VIEW_WIDTH,
) -> tuple[int, ...]:
    return (observation_size(view_range, view_width),)


def _side_offsets(view_width: int) -> list[int]:
    view_width = max(1, int(view_width))
    half = view_width // 2
    return list(range(-half, -half + view_width))


def _strip_cell(
    agent_pos: tuple[int, int],
    dr: int,
    dc: int,
    distance: int,
    side_offset: int,
) -> tuple[int, int]:
    row, col = agent_pos
    if dr != 0:
        return row + dr * distance, col + side_offset
    return row + side_offset, col + dc * distance
