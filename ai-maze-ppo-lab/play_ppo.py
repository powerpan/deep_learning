from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from config import ACTION_NAMES, MAPS_DIR, MAX_STEPS, MODELS_DIR, OUTPUTS_DIR, VIEW_RANGE, VIEW_WIDTH

(OUTPUTS_DIR / ".matplotlib").mkdir(parents=True, exist_ok=True)
(OUTPUTS_DIR / ".cache").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(OUTPUTS_DIR / ".matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(OUTPUTS_DIR / ".cache"))

from maze_env import MazePPOEnv
from renderer import MazeRenderer


def _load_ppo():
    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise SystemExit(
            "Missing PPO dependencies. Run: pip install -r requirements.txt"
        ) from exc
    return PPO


def write_trace(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Replay a trained PPO maze agent.")
    parser.add_argument("--model", type=str, default=str(MODELS_DIR / "ppo_maze.zip"))
    parser.add_argument("--map", type=str, default=str(MAPS_DIR / "level_1.txt"))
    parser.add_argument("--outputs-dir", type=str, default=str(OUTPUTS_DIR))
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--view-range", type=int, default=VIEW_RANGE)
    parser.add_argument("--view-width", type=int, default=VIEW_WIDTH)
    parser.add_argument("--delay", type=float, default=0.08)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args(argv)

    PPO = _load_ppo()
    model = PPO.load(args.model)
    env = MazePPOEnv(
        map_path=args.map,
        max_steps=args.max_steps,
        view_range=args.view_range,
        view_width=args.view_width,
    )
    renderer = None if args.no_render else MazeRenderer(show_vision=True)

    obs, info = env.reset()
    trace = [
        f"Replay for {args.map}",
        f"Model: {args.model}",
        "",
        f"0: pos={info['position']} reward=0.00 has_key={info['has_key']} event={info['event']}",
    ]

    try:
        for step in range(1, args.max_steps + 1):
            if renderer and not renderer.render(env, info):
                break
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            action_name = ACTION_NAMES[int(action)]
            trace.append(
                (
                    f"{step}: action={action_name} pos={info['position']} "
                    f"reward={reward:.2f} total={info['total_reward']:.2f} "
                    f"has_key={info['has_key']} door={info['passed_door']} "
                    f"event={info['event']}"
                )
            )
            if renderer:
                time.sleep(args.delay)
            if terminated or truncated:
                break

        if renderer:
            renderer.render(env, info)
            time.sleep(1.0)
    finally:
        if renderer:
            renderer.close()

    trace.extend(
        [
            "",
            f"Success: {info['success']}",
            f"Has key: {info['has_key']}",
            f"Passed door: {info['passed_door']}",
            f"Steps: {info['steps']}",
            f"Total reward: {info['total_reward']:.2f}",
        ]
    )
    output_path = Path(args.outputs_dir) / "replay_trace.txt"
    write_trace(output_path, trace)
    print(f"Saved replay trace: {output_path}")


if __name__ == "__main__":
    main()
