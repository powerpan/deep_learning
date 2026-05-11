from __future__ import annotations

import numpy as np

from config import (
    ACTIONS,
    LOCAL_VIEW_SIZE,
    MAX_STEPS,
    OBSERVATION_MODE,
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
LOCAL_GRID_CHANNELS = (
    "tile",
    "recent_visited",
    "facing_action",
    "last_action",
    "last_reward",
    "has_key",
    "step_repeat",
)
LOCAL_GRID_CHANNEL_INDEX = {
    name: index for index, name in enumerate(LOCAL_GRID_CHANNELS)
}
TILE_VALUES = {
    "unknown": 0,
    "wall": 42,
    "empty": 85,
    "trap": 128,
    "key": 170,
    "door": 212,
    "exit": 255,
}


def observation_size(view_range: int = VIEW_RANGE, view_width: int = VIEW_WIDTH) -> int:
    return len(ACTIONS) * view_range * view_width * len(TILE_CHANNELS) + 2


def local_grid_channels() -> int:
    return len(LOCAL_GRID_CHANNELS)


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


def encode_local_grid(
    grid: list[list[str]],
    agent_pos: tuple[int, int],
    has_key: bool,
    key_collected: bool,
    step_count: int,
    max_steps: int = MAX_STEPS,
    local_view_size: int = LOCAL_VIEW_SIZE,
    facing_action: int = 3,
    last_action: int = -1,
    last_reward: float = 0.0,
    recent_positions: list[tuple[int, int]] | tuple[tuple[int, int], ...] | None = None,
) -> np.ndarray:
    """Encode an agent-centered local square for CnnPolicy.

    The grid is channel-first and uint8 so Stable-Baselines3 treats it as an
    image. A compact seven-channel layout avoids SB3's small-image channel
    heuristic from misidentifying the channel axis.
    """
    size = _normalize_local_view_size(local_view_size)
    center = size // 2
    rows = len(grid)
    cols = len(grid[0])
    recent = set(recent_positions or [])
    obs = np.zeros((len(LOCAL_GRID_CHANNELS), size, size), dtype=np.uint8)

    for local_row in range(size):
        for local_col in range(size):
            row = agent_pos[0] + local_row - center
            col = agent_pos[1] + local_col - center
            if row < 0 or row >= rows or col < 0 or col >= cols:
                channel = "wall"
            else:
                channel = _tile_channel(grid[row][col], key_collected)
            obs[LOCAL_GRID_CHANNEL_INDEX["tile"], local_row, local_col] = TILE_VALUES[channel]
            if (row, col) in recent:
                obs[LOCAL_GRID_CHANNEL_INDEX["recent_visited"], local_row, local_col] = 255

    obs[LOCAL_GRID_CHANNEL_INDEX["tile"], center, center] = 255
    obs[LOCAL_GRID_CHANNEL_INDEX["facing_action"], :, :] = _action_to_uint8(facing_action)
    obs[LOCAL_GRID_CHANNEL_INDEX["last_action"], :, :] = _action_to_uint8(last_action)
    obs[LOCAL_GRID_CHANNEL_INDEX["last_reward"], :, :] = _reward_to_uint8(last_reward)
    obs[LOCAL_GRID_CHANNEL_INDEX["has_key"], :, :] = 255 if has_key else 0
    obs[LOCAL_GRID_CHANNEL_INDEX["step_repeat"], :, :] = _ratio_to_uint8(
        float(step_count) / max(float(max_steps), 1.0)
    )
    repeated = bool(recent_positions and list(recent_positions).count(agent_pos) > 1)
    if repeated:
        obs[LOCAL_GRID_CHANNEL_INDEX["step_repeat"], center, center] = 255
    return obs


def visible_cells(
    grid: list[list[str]],
    agent_pos: tuple[int, int],
    view_range: int = VIEW_RANGE,
    view_width: int = VIEW_WIDTH,
    observation_mode: str = "strips",
    local_view_size: int = LOCAL_VIEW_SIZE,
) -> set[tuple[int, int]]:
    rows = len(grid)
    cols = len(grid[0])
    cells = {agent_pos}
    if observation_mode == "grid":
        radius = _normalize_local_view_size(local_view_size) // 2
        for row in range(agent_pos[0] - radius, agent_pos[0] + radius + 1):
            for col in range(agent_pos[1] - radius, agent_pos[1] + radius + 1):
                if 0 <= row < rows and 0 <= col < cols:
                    cells.add((row, col))
        return cells

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
    observation_mode: str = OBSERVATION_MODE,
    local_view_size: int = LOCAL_VIEW_SIZE,
) -> tuple[int, ...]:
    if observation_mode == "grid":
        size = _normalize_local_view_size(local_view_size)
        return (len(LOCAL_GRID_CHANNELS), size, size)
    return (observation_size(view_range, view_width),)


def _ratio_to_uint8(value: float) -> int:
    return int(np.clip(value, 0.0, 1.0) * 255)


def _action_to_uint8(action: int) -> int:
    if action not in ACTIONS:
        return 0
    return int((int(action) + 1) / len(ACTIONS) * 255)


def _reward_to_uint8(reward: float) -> int:
    # Most rewards currently live in [-20, 80]. Clipping keeps the image range stable.
    return _ratio_to_uint8((float(reward) + 20.0) / 100.0)


def _normalize_local_view_size(local_view_size: int) -> int:
    size = max(3, int(local_view_size))
    return size if size % 2 == 1 else size + 1


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
