from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np

from config import (
    DEFAULT_RANDOM_MAP_PROB,
    MAPS_DIR,
    MAX_STEPS,
    MODELS_DIR,
    OUTPUTS_DIR,
    PPO_BATCH_SIZE,
    PPO_ENT_COEF,
    PPO_GAMMA,
    PPO_LEARNING_RATE,
    PPO_N_STEPS,
    VIEW_RANGE,
)

os.environ.setdefault("MPLCONFIGDIR", str(OUTPUTS_DIR / ".matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(OUTPUTS_DIR / ".cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from maze_env import MazePPOEnv
from random_maps import GeneratedMap, build_random_map_pool


def _load_sb3():
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.monitor import Monitor
    except ImportError as exc:
        raise SystemExit(
            "Missing PPO dependencies. Run: pip install -r requirements.txt"
        ) from exc
    return PPO, Monitor


def discover_map_files(maps_dir: str | Path) -> list[Path]:
    path = Path(maps_dir)
    return sorted(path.glob("*.txt"))


def rolling_mean(values: list[float], window: int = 100) -> list[float]:
    if not values:
        return []
    result = []
    for index in range(len(values)):
        start = max(0, index - window + 1)
        result.append(float(np.mean(values[start : index + 1])))
    return result


def parse_monitor_csv(path: str | Path) -> dict[str, list[float]]:
    rows: list[dict[str, str]] = []
    with Path(path).open(newline="") as handle:
        content = [line for line in handle if not line.startswith("#")]
    reader = csv.DictReader(content)
    for row in reader:
        rows.append(row)

    rewards = [float(row["r"]) for row in rows]
    lengths = [float(row["l"]) for row in rows]
    successes = [
        1.0 if str(row.get("success", "")).lower() in {"true", "1"} else 0.0
        for row in rows
    ]
    return {"reward": rewards, "steps": lengths, "success": successes}


def save_training_curve(monitor_path: str | Path, output_path: str | Path) -> None:
    data = parse_monitor_csv(monitor_path)
    if not data["reward"]:
        return

    episodes = list(range(1, len(data["reward"]) + 1))
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    axes[0].plot(episodes, rolling_mean(data["success"]), color="#2ca02c")
    axes[0].set_ylabel("Success rate")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].set_title("Rolling Success Rate (100 episodes)")

    axes[1].plot(episodes, rolling_mean(data["steps"]), color="#1f77b4")
    axes[1].set_ylabel("Steps")
    axes[1].set_title("Rolling Average Steps (100 episodes)")

    axes[2].plot(episodes, rolling_mean(data["reward"]), color="#d62728")
    axes[2].set_ylabel("Reward")
    axes[2].set_xlabel("Episode")
    axes[2].set_title("Rolling Average Reward (100 episodes)")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def make_random_factory(pool: list[GeneratedMap]):
    def factory(rng: np.random.Generator) -> GeneratedMap:
        index = int(rng.integers(0, len(pool)))
        return pool[index]

    return factory


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train PPO on partial-view maze maps.")
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--map", type=str, default=None)
    parser.add_argument("--maps-dir", type=str, default=str(MAPS_DIR))
    parser.add_argument("--random-maps", type=int, default=200)
    parser.add_argument("--random-map-prob", type=float, default=DEFAULT_RANDOM_MAP_PROB)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--view-range", type=int, default=VIEW_RANGE)
    parser.add_argument("--model-path", type=str, default=str(MODELS_DIR / "ppo_maze.zip"))
    parser.add_argument("--outputs-dir", type=str, default=str(OUTPUTS_DIR))
    args = parser.parse_args(argv)

    PPO, Monitor = _load_sb3()

    outputs_dir = Path(args.outputs_dir)
    models_dir = Path(args.model_path).parent
    outputs_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    fixed_maps = [Path(args.map)] if args.map else discover_map_files(args.maps_dir)
    random_pool = build_random_map_pool(args.random_maps, seed=args.seed)
    random_factory = make_random_factory(random_pool) if random_pool else None

    env = MazePPOEnv(
        map_sources=fixed_maps,
        random_map_factory=random_factory,
        random_map_probability=args.random_map_prob if random_pool else 0.0,
        max_steps=args.max_steps,
        view_range=args.view_range,
        seed=args.seed,
    )
    monitor_path = outputs_dir / "training_monitor.csv"
    env = Monitor(
        env,
        filename=str(monitor_path),
        info_keywords=("success", "has_key", "passed_door"),
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        seed=args.seed,
        learning_rate=PPO_LEARNING_RATE,
        gamma=PPO_GAMMA,
        n_steps=PPO_N_STEPS,
        batch_size=PPO_BATCH_SIZE,
        ent_coef=PPO_ENT_COEF,
    )
    model.learn(total_timesteps=args.timesteps)
    model.save(args.model_path)
    env.close()

    curve_path = outputs_dir / "ppo_training_curve.png"
    save_training_curve(monitor_path, curve_path)

    print(f"Saved model: {args.model_path}")
    print(f"Saved monitor CSV: {monitor_path}")
    print(f"Saved training curve: {curve_path}")


if __name__ == "__main__":
    main()
