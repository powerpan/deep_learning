from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from config import (
    ACTIONS,
    RANDOM_MAP_COLS,
    RANDOM_MAP_ROWS,
    RANDOM_TRAP_DENSITY,
    RANDOM_WALL_DENSITY,
    TILE_DOOR,
    TILE_EMPTY,
    TILE_EXIT,
    TILE_KEY,
    TILE_START,
    TILE_TRAP,
    TILE_WALL,
)


@dataclass(frozen=True)
class GeneratedMap:
    name: str
    lines: list[str]


def find_tile(lines: list[str], tile: str) -> tuple[int, int] | None:
    for row, line in enumerate(lines):
        col = line.find(tile)
        if col != -1:
            return row, col
    return None


def is_solvable_key_door(lines: list[str]) -> bool:
    start = find_tile(lines, TILE_START)
    exit_pos = find_tile(lines, TILE_EXIT)
    if start is None or exit_pos is None:
        return False

    rows = len(lines)
    cols = len(lines[0])
    queue: deque[tuple[int, int, bool]] = deque([(start[0], start[1], False)])
    seen = {(start[0], start[1], False)}

    while queue:
        row, col, has_key = queue.popleft()
        if (row, col) == exit_pos:
            return True

        for dr, dc in ACTIONS.values():
            nr = row + dr
            nc = col + dc
            if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                continue
            tile = lines[nr][nc]
            if tile in {TILE_WALL, TILE_TRAP}:
                continue
            if tile == TILE_DOOR and not has_key:
                continue
            next_has_key = has_key or tile == TILE_KEY
            state = (nr, nc, next_has_key)
            if state not in seen:
                seen.add(state)
                queue.append(state)

    return False


def _path_between(
    start: tuple[int, int], end: tuple[int, int], rng: np.random.Generator
) -> set[tuple[int, int]]:
    row, col = start
    end_row, end_col = end
    cells = {(row, col)}

    horizontal_first = bool(rng.integers(0, 2))
    axes = ("col", "row") if horizontal_first else ("row", "col")

    for axis in axes:
        if axis == "row":
            step = 1 if end_row >= row else -1
            while row != end_row:
                row += step
                cells.add((row, col))
        else:
            step = 1 if end_col >= col else -1
            while col != end_col:
                col += step
                cells.add((row, col))

    return cells


def generate_random_key_door_map(
    rng: np.random.Generator | int | None = None,
    rows: int = RANDOM_MAP_ROWS,
    cols: int = RANDOM_MAP_COLS,
    wall_density: float = RANDOM_WALL_DENSITY,
    trap_density: float = RANDOM_TRAP_DENSITY,
    name: str = "random",
) -> GeneratedMap:
    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)

    rows = max(rows, 9)
    cols = max(cols, 11)
    barrier_row = rows // 2
    door_col = int(rng.integers(2, cols - 2))

    grid = [[TILE_EMPTY for _ in range(cols)] for _ in range(rows)]
    for row in range(rows):
        grid[row][0] = TILE_WALL
        grid[row][cols - 1] = TILE_WALL
    for col in range(cols):
        grid[0][col] = TILE_WALL
        grid[rows - 1][col] = TILE_WALL

    for col in range(1, cols - 1):
        grid[barrier_row][col] = TILE_WALL
    grid[barrier_row][door_col] = TILE_DOOR

    start = (1, 1)
    key = (int(rng.integers(1, barrier_row)), int(rng.integers(2, cols - 2)))
    exit_pos = (rows - 2, cols - 2)

    protected = set()
    protected |= _path_between(start, key, rng)
    protected |= _path_between(key, (barrier_row - 1, door_col), rng)
    protected.add((barrier_row, door_col))
    protected |= _path_between((barrier_row + 1, door_col), exit_pos, rng)
    protected.update({start, key, exit_pos})

    for row in range(1, rows - 1):
        if row == barrier_row:
            continue
        for col in range(1, cols - 1):
            if (row, col) in protected:
                continue
            roll = float(rng.random())
            if roll < wall_density:
                grid[row][col] = TILE_WALL
            elif roll < wall_density + trap_density:
                grid[row][col] = TILE_TRAP

    sr, sc = start
    kr, kc = key
    er, ec = exit_pos
    grid[sr][sc] = TILE_START
    grid[kr][kc] = TILE_KEY
    grid[er][ec] = TILE_EXIT
    grid[barrier_row][door_col] = TILE_DOOR

    lines = ["".join(row) for row in grid]
    if not is_solvable_key_door(lines):
        return generate_random_key_door_map(
            rng=rng,
            rows=rows,
            cols=cols,
            wall_density=max(0.0, wall_density * 0.8),
            trap_density=max(0.0, trap_density * 0.8),
            name=name,
        )

    return GeneratedMap(name=name, lines=lines)


def build_random_map_pool(
    count: int,
    seed: int | None = None,
    rows: int = RANDOM_MAP_ROWS,
    cols: int = RANDOM_MAP_COLS,
) -> list[GeneratedMap]:
    rng = np.random.default_rng(seed)
    return [
        generate_random_key_door_map(
            rng=rng,
            rows=rows,
            cols=cols,
            name=f"random_{index:04d}",
        )
        for index in range(max(0, count))
    ]
