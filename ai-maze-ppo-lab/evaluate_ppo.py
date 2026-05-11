from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from config import MAPS_DIR, MAX_STEPS, MODELS_DIR, OUTPUTS_DIR, VIEW_RANGE

os.environ.setdefault("MPLCONFIGDIR", str(OUTPUTS_DIR / ".matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(OUTPUTS_DIR / ".cache"))

from maze_env import MazePPOEnv
from random_maps import generate_random_key_door_map
from train_ppo import discover_map_files


def _load_ppo():
    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise SystemExit(
            "Missing PPO dependencies. Run: pip install -r requirements.txt"
        ) from exc
    return PPO


def run_episode(model, env: MazePPOEnv) -> dict:
    obs, info = env.reset()
    terminated = False
    truncated = False
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, _reward, terminated, truncated, info = env.step(int(action))
    return dict(info)


def summarize(results: list[dict]) -> dict:
    if not results:
        return {
            "episodes": 0,
            "success_rate": 0.0,
            "avg_steps": 0.0,
            "avg_reward": 0.0,
            "has_key_rate": 0.0,
            "passed_door_rate": 0.0,
        }
    return {
        "episodes": len(results),
        "success_rate": float(np.mean([bool(item["success"]) for item in results])),
        "avg_steps": float(np.mean([float(item["steps"]) for item in results])),
        "avg_reward": float(np.mean([float(item["total_reward"]) for item in results])),
        "has_key_rate": float(np.mean([bool(item["has_key"]) for item in results])),
        "passed_door_rate": float(np.mean([bool(item["passed_door"]) for item in results])),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO maze agent.")
    parser.add_argument("--model", type=str, default=str(MODELS_DIR / "ppo_maze.zip"))
    parser.add_argument("--map", type=str, default=None)
    parser.add_argument("--maps-dir", type=str, default=str(MAPS_DIR))
    parser.add_argument("--random-tests", type=int, default=50)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--view-range", type=int, default=VIEW_RANGE)
    parser.add_argument("--outputs-dir", type=str, default=str(OUTPUTS_DIR))
    args = parser.parse_args(argv)

    PPO = _load_ppo()
    model = PPO.load(args.model)
    rng = np.random.default_rng(args.seed)

    fixed_paths = [Path(args.map)] if args.map else discover_map_files(args.maps_dir)
    fixed_results = []
    for path in fixed_paths:
        env = MazePPOEnv(
            map_path=path,
            max_steps=args.max_steps,
            view_range=args.view_range,
            seed=args.seed,
        )
        result = run_episode(model, env)
        result["map_name"] = str(path)
        fixed_results.append(result)
        env.close()

    random_results = []
    for index in range(args.random_tests):
        generated = generate_random_key_door_map(rng=rng, name=f"eval_random_{index:04d}")
        env = MazePPOEnv(
            map_lines=generated.lines,
            max_steps=args.max_steps,
            view_range=args.view_range,
            seed=args.seed + index,
        )
        result = run_episode(model, env)
        result["map_name"] = generated.name
        random_results.append(result)
        env.close()

    summary = {
        "model": args.model,
        "fixed_maps": summarize(fixed_results),
        "random_maps": summarize(random_results),
        "fixed_results": fixed_results,
        "random_results": random_results,
    }

    outputs_dir = Path(args.outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    output_path = outputs_dir / "eval_summary.json"
    output_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k.endswith("maps")}, indent=2))
    print(f"Saved eval summary: {output_path}")


if __name__ == "__main__":
    main()
