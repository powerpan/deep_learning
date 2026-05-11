from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from config import (
    ACTIONS,
    RANDOM_MAP_COLS,
    RANDOM_MAP_ROWS,
    RANDOM_DOOR_ORIENTATION,
    RANDOM_ENDPOINT_MODE,
    RANDOM_MAP_STYLE,
    RANDOM_SIMPLE_MAP_PROB,
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

STYLE_OPTIONS = ("mixed", "open", "split", "rooms", "deadends", "maze")
DOOR_ORIENTATION_OPTIONS = ("mixed", "horizontal", "vertical")
ENDPOINT_MODE_OPTIONS = ("mixed", "corners", "edges", "interior")


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
    style: str = RANDOM_MAP_STYLE,
    door_orientation: str = RANDOM_DOOR_ORIENTATION,
    endpoint_mode: str = RANDOM_ENDPOINT_MODE,
    name: str = "random",
    _attempt: int = 0,
) -> GeneratedMap:
    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)

    rows = max(rows, 9)
    cols = max(cols, 11)
    style = _resolve_option(style, STYLE_OPTIONS, rng)
    door_orientation = _resolve_option(door_orientation, DOOR_ORIENTATION_OPTIONS, rng)
    endpoint_mode = _resolve_option(endpoint_mode, ENDPOINT_MODE_OPTIONS, rng)
    wall_density, trap_density = _style_densities(style, wall_density, trap_density)

    grid = [[TILE_EMPTY for _ in range(cols)] for _ in range(rows)]
    for row in range(rows):
        grid[row][0] = TILE_WALL
        grid[row][cols - 1] = TILE_WALL
    for col in range(cols):
        grid[0][col] = TILE_WALL
        grid[rows - 1][col] = TILE_WALL

    barrier = _build_barrier(grid, rng, door_orientation)
    start_region, exit_region = _start_exit_regions(rows, cols, barrier, rng)
    start = _choose_point(start_region, endpoint_mode, rng)
    exit_pos = _choose_point(exit_region, endpoint_mode, rng)
    key = _choose_distinct_point(start_region, {start}, rng)

    protected = set()
    protected |= _path_between(start, key, rng)
    protected |= _path_between(key, barrier["approach"], rng)
    protected.add(barrier["door"])
    protected |= _path_between(barrier["exit_side"], exit_pos, rng)
    protected.update({start, key, exit_pos})

    _add_style_walls(grid, protected, rng, style)

    for row in range(1, rows - 1):
        if barrier["orientation"] == "horizontal" and row == barrier["line"]:
            continue
        for col in range(1, cols - 1):
            if barrier["orientation"] == "vertical" and col == barrier["line"]:
                continue
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
    dr, dc = barrier["door"]
    grid[dr][dc] = TILE_DOOR

    lines = ["".join(row) for row in grid]
    if not is_solvable_key_door(lines):
        if _attempt >= 12:
            return generate_random_key_door_map(
                rng=rng,
                rows=rows,
                cols=cols,
                wall_density=0.0,
                trap_density=0.0,
                style="split",
                door_orientation=door_orientation,
                endpoint_mode=endpoint_mode,
                name=name,
                _attempt=_attempt + 1,
            )
        return generate_random_key_door_map(
            rng=rng,
            rows=rows,
            cols=cols,
            wall_density=max(0.0, wall_density * 0.8),
            trap_density=max(0.0, trap_density * 0.8),
            style=style,
            door_orientation=door_orientation,
            endpoint_mode=endpoint_mode,
            name=name,
            _attempt=_attempt + 1,
        )

    return GeneratedMap(name=name, lines=lines)


def generate_random_exit_map(
    rng: np.random.Generator | int | None = None,
    rows: int = RANDOM_MAP_ROWS,
    cols: int = RANDOM_MAP_COLS,
    wall_density: float = RANDOM_WALL_DENSITY,
    trap_density: float = RANDOM_TRAP_DENSITY,
    style: str = RANDOM_MAP_STYLE,
    endpoint_mode: str = RANDOM_ENDPOINT_MODE,
    name: str = "random_simple",
    _attempt: int = 0,
) -> GeneratedMap:
    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)

    rows = max(rows, 7)
    cols = max(cols, 9)
    style = _resolve_option(style, STYLE_OPTIONS, rng)
    endpoint_mode = _resolve_option(endpoint_mode, ENDPOINT_MODE_OPTIONS, rng)
    wall_density, trap_density = _style_densities(style, wall_density, trap_density)

    grid = [[TILE_EMPTY for _ in range(cols)] for _ in range(rows)]
    for row in range(rows):
        grid[row][0] = TILE_WALL
        grid[row][cols - 1] = TILE_WALL
    for col in range(cols):
        grid[0][col] = TILE_WALL
        grid[rows - 1][col] = TILE_WALL

    region = (1, rows - 2, 1, cols - 2)
    start = _choose_point(region, endpoint_mode, rng)
    exit_pos = _choose_distinct_mode_point(region, {start}, endpoint_mode, rng)
    protected = _path_between(start, exit_pos, rng)
    protected.update({start, exit_pos})

    _add_style_walls(grid, protected, rng, style)

    for row in range(1, rows - 1):
        for col in range(1, cols - 1):
            if (row, col) in protected:
                continue
            roll = float(rng.random())
            if roll < wall_density:
                grid[row][col] = TILE_WALL
            elif roll < wall_density + trap_density:
                grid[row][col] = TILE_TRAP

    sr, sc = start
    er, ec = exit_pos
    grid[sr][sc] = TILE_START
    grid[er][ec] = TILE_EXIT

    lines = ["".join(row) for row in grid]
    if not is_solvable_key_door(lines):
        if _attempt >= 12:
            return generate_random_exit_map(
                rng=rng,
                rows=rows,
                cols=cols,
                wall_density=0.0,
                trap_density=0.0,
                style="open",
                endpoint_mode=endpoint_mode,
                name=name,
                _attempt=_attempt + 1,
            )
        return generate_random_exit_map(
            rng=rng,
            rows=rows,
            cols=cols,
            wall_density=max(0.0, wall_density * 0.8),
            trap_density=max(0.0, trap_density * 0.8),
            style=style,
            endpoint_mode=endpoint_mode,
            name=name,
            _attempt=_attempt + 1,
        )

    return GeneratedMap(name=name, lines=lines)


def _resolve_option(value: str, options: tuple[str, ...], rng: np.random.Generator) -> str:
    if value not in options:
        value = options[0]
    if value == "mixed":
        candidates = [item for item in options if item != "mixed"]
        return str(rng.choice(candidates))
    return value


def _style_densities(style: str, wall_density: float, trap_density: float) -> tuple[float, float]:
    multipliers = {
        "open": (0.45, 0.65),
        "split": (1.0, 1.0),
        "rooms": (0.85, 0.85),
        "deadends": (1.25, 1.1),
        "maze": (1.55, 0.75),
    }
    wall_multiplier, trap_multiplier = multipliers.get(style, (1.0, 1.0))
    wall = float(np.clip(wall_density * wall_multiplier, 0.0, 0.45))
    trap = float(np.clip(trap_density * trap_multiplier, 0.0, 0.28))
    return wall, trap


def _build_barrier(grid: list[list[str]], rng: np.random.Generator, orientation: str) -> dict:
    rows = len(grid)
    cols = len(grid[0])
    if orientation == "horizontal":
        line = int(rng.integers(max(3, rows // 3), min(rows - 3, rows * 2 // 3) + 1))
        door_col = int(rng.integers(2, cols - 2))
        for col in range(1, cols - 1):
            grid[line][col] = TILE_WALL
        door = (line, door_col)
        grid[line][door_col] = TILE_DOOR
        return {
            "orientation": orientation,
            "line": line,
            "door": door,
            "before": (line - 1, door_col),
            "after": (line + 1, door_col),
        }

    line = int(rng.integers(max(3, cols // 3), min(cols - 3, cols * 2 // 3) + 1))
    door_row = int(rng.integers(2, rows - 2))
    for row in range(1, rows - 1):
        grid[row][line] = TILE_WALL
    door = (door_row, line)
    grid[door_row][line] = TILE_DOOR
    return {
        "orientation": orientation,
        "line": line,
        "door": door,
        "before": (door_row, line - 1),
        "after": (door_row, line + 1),
    }


def _start_exit_regions(
    rows: int,
    cols: int,
    barrier: dict,
    rng: np.random.Generator,
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    if barrier["orientation"] == "horizontal":
        before = (1, barrier["line"] - 1, 1, cols - 2)
        after = (barrier["line"] + 1, rows - 2, 1, cols - 2)
    else:
        before = (1, rows - 2, 1, barrier["line"] - 1)
        after = (1, rows - 2, barrier["line"] + 1, cols - 2)

    if bool(rng.integers(0, 2)):
        barrier["approach"] = barrier["before"]
        barrier["exit_side"] = barrier["after"]
        return before, after

    barrier["approach"] = barrier["after"]
    barrier["exit_side"] = barrier["before"]
    return after, before


def _choose_point(
    region: tuple[int, int, int, int],
    mode: str,
    rng: np.random.Generator,
) -> tuple[int, int]:
    rmin, rmax, cmin, cmax = region
    mode = _resolve_option(mode, ENDPOINT_MODE_OPTIONS, rng) if mode == "mixed" else mode
    if mode == "corners":
        candidates = [(rmin, cmin), (rmin, cmax), (rmax, cmin), (rmax, cmax)]
        return candidates[int(rng.integers(0, len(candidates)))]
    if mode == "edges":
        if bool(rng.integers(0, 2)):
            row = rmin if bool(rng.integers(0, 2)) else rmax
            col = int(rng.integers(cmin, cmax + 1))
        else:
            row = int(rng.integers(rmin, rmax + 1))
            col = cmin if bool(rng.integers(0, 2)) else cmax
        return row, col
    row = int(rng.integers(rmin, rmax + 1))
    col = int(rng.integers(cmin, cmax + 1))
    return row, col


def _choose_distinct_point(
    region: tuple[int, int, int, int],
    blocked: set[tuple[int, int]],
    rng: np.random.Generator,
) -> tuple[int, int]:
    for _ in range(40):
        point = _choose_point(region, "interior", rng)
        if point not in blocked:
            return point
    rmin, _rmax, cmin, _cmax = region
    return rmin, cmin


def _choose_distinct_mode_point(
    region: tuple[int, int, int, int],
    blocked: set[tuple[int, int]],
    mode: str,
    rng: np.random.Generator,
) -> tuple[int, int]:
    for _ in range(40):
        point = _choose_point(region, mode, rng)
        if point not in blocked:
            return point
    return _choose_distinct_point(region, blocked, rng)


def _add_style_walls(
    grid: list[list[str]],
    protected: set[tuple[int, int]],
    rng: np.random.Generator,
    style: str,
) -> None:
    if style == "open":
        return
    if style == "rooms":
        _add_wall_segments(grid, protected, rng, count=4, min_len=4, max_len=8, gap=True)
    elif style == "deadends":
        _add_wall_segments(grid, protected, rng, count=7, min_len=3, max_len=6, gap=False)
    elif style == "maze":
        _add_wall_segments(grid, protected, rng, count=10, min_len=3, max_len=9, gap=True)


def _add_wall_segments(
    grid: list[list[str]],
    protected: set[tuple[int, int]],
    rng: np.random.Generator,
    count: int,
    min_len: int,
    max_len: int,
    gap: bool,
) -> None:
    rows = len(grid)
    cols = len(grid[0])
    for _ in range(count):
        horizontal = bool(rng.integers(0, 2))
        length = int(rng.integers(min_len, max_len + 1))
        if horizontal:
            row = int(rng.integers(2, rows - 2))
            col = int(rng.integers(1, max(2, cols - length - 1)))
            cells = [(row, c) for c in range(col, min(cols - 1, col + length))]
        else:
            row = int(rng.integers(1, max(2, rows - length - 1)))
            col = int(rng.integers(2, cols - 2))
            cells = [(r, col) for r in range(row, min(rows - 1, row + length))]
        if gap and len(cells) > 2:
            gap_index = int(rng.integers(1, len(cells) - 1))
            cells.pop(gap_index)
        for row, col in cells:
            if (row, col) not in protected and grid[row][col] == TILE_EMPTY:
                grid[row][col] = TILE_WALL


def build_random_map_pool(
    count: int,
    seed: int | None = None,
    rows: int = RANDOM_MAP_ROWS,
    cols: int = RANDOM_MAP_COLS,
    wall_density: float = RANDOM_WALL_DENSITY,
    trap_density: float = RANDOM_TRAP_DENSITY,
    style: str = RANDOM_MAP_STYLE,
    door_orientation: str = RANDOM_DOOR_ORIENTATION,
    endpoint_mode: str = RANDOM_ENDPOINT_MODE,
    simple_map_probability: float = RANDOM_SIMPLE_MAP_PROB,
) -> list[GeneratedMap]:
    rng = np.random.default_rng(seed)
    simple_map_probability = float(np.clip(simple_map_probability, 0.0, 1.0))
    pool = []
    for index in range(max(0, count)):
        if float(rng.random()) < simple_map_probability:
            pool.append(
                generate_random_exit_map(
                    rng=rng,
                    rows=rows,
                    cols=cols,
                    wall_density=wall_density,
                    trap_density=trap_density,
                    style=style,
                    endpoint_mode=endpoint_mode,
                    name=f"random_simple_{index:04d}",
                )
            )
        else:
            pool.append(
                generate_random_key_door_map(
                    rng=rng,
                    rows=rows,
                    cols=cols,
                    wall_density=wall_density,
                    trap_density=trap_density,
                    style=style,
                    door_orientation=door_orientation,
                    endpoint_mode=endpoint_mode,
                    name=f"random_key_door_{index:04d}",
                )
            )
    return pool
