from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

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
    PPO_N_ENVS,
    PPO_N_STEPS,
    PPO_TORCH_THREADS,
    PPO_VEC_ENV,
    RANDOM_MAP_COLS,
    RANDOM_MAP_ROWS,
    RANDOM_DOOR_ORIENTATION,
    RANDOM_ENDPOINT_MODE,
    RANDOM_MAP_STYLE,
    RANDOM_TRAP_DENSITY,
    RANDOM_WALL_DENSITY,
    VIEW_RANGE,
    VIEW_WIDTH,
)

(OUTPUTS_DIR / ".matplotlib").mkdir(parents=True, exist_ok=True)
(OUTPUTS_DIR / ".cache").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(OUTPUTS_DIR / ".matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(OUTPUTS_DIR / ".cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from maze_env import MazePPOEnv
from random_maps import (
    DOOR_ORIENTATION_OPTIONS,
    ENDPOINT_MODE_OPTIONS,
    STYLE_OPTIONS,
    GeneratedMap,
    build_random_map_pool,
)


def _load_sb3():
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import BaseCallback
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
    except ImportError as exc:
        raise SystemExit(
            "Missing PPO dependencies. Run: pip install -r requirements.txt"
        ) from exc
    return PPO, BaseCallback, DummyVecEnv, SubprocVecEnv, VecMonitor


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


def make_training_env(
    *,
    fixed_maps: list[Path],
    random_pool: list[GeneratedMap],
    random_map_probability: float,
    max_steps: int,
    view_range: int,
    view_width: int,
    exploration_reward: bool,
    seed: int,
):
    def init_env() -> MazePPOEnv:
        random_factory = make_random_factory(random_pool) if random_pool else None
        return MazePPOEnv(
            map_sources=fixed_maps,
            random_map_factory=random_factory,
            random_map_probability=random_map_probability if random_pool else 0.0,
            max_steps=max_steps,
            view_range=view_range,
            view_width=view_width,
            exploration_reward=exploration_reward,
            seed=seed,
        )

    return init_env


def rollout_steps_per_env(base_steps: int, n_envs: int, batch_size: int) -> int:
    n_envs = max(1, n_envs)
    n_steps = max(64, base_steps // n_envs)
    while (n_steps * n_envs) % batch_size != 0:
        n_steps += 1
    return n_steps


def format_progress(
    steps: int,
    total_timesteps: int,
    episodes: int,
    elapsed_seconds: float | None = None,
) -> str:
    capped_steps = min(steps, total_timesteps)
    percent = 100.0 if total_timesteps <= 0 else capped_steps / total_timesteps * 100.0
    speed_text = "-"
    if elapsed_seconds and elapsed_seconds > 0 and capped_steps > 0:
        speed_text = f"{capped_steps / elapsed_seconds:.0f}步/s"
    return (
        f"训练进度：{capped_steps:,}/{total_timesteps:,} "
        f"{percent:.0f}% 回合{episodes} {speed_text}"
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train PPO on partial-view maze maps.")
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--map", type=str, default=None)
    parser.add_argument("--maps-dir", type=str, default=str(MAPS_DIR))
    parser.add_argument("--random-maps", type=int, default=200)
    parser.add_argument("--random-map-prob", type=float, default=DEFAULT_RANDOM_MAP_PROB)
    parser.add_argument("--random-rows", type=int, default=RANDOM_MAP_ROWS)
    parser.add_argument("--random-cols", type=int, default=RANDOM_MAP_COLS)
    parser.add_argument("--wall-density", type=float, default=RANDOM_WALL_DENSITY)
    parser.add_argument("--trap-density", type=float, default=RANDOM_TRAP_DENSITY)
    parser.add_argument(
        "--random-style",
        type=str,
        choices=STYLE_OPTIONS,
        default=RANDOM_MAP_STYLE,
    )
    parser.add_argument(
        "--door-orientation",
        type=str,
        choices=DOOR_ORIENTATION_OPTIONS,
        default=RANDOM_DOOR_ORIENTATION,
    )
    parser.add_argument(
        "--endpoint-mode",
        type=str,
        choices=ENDPOINT_MODE_OPTIONS,
        default=RANDOM_ENDPOINT_MODE,
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--view-range", type=int, default=VIEW_RANGE)
    parser.add_argument("--view-width", type=int, default=VIEW_WIDTH)
    parser.add_argument("--ent-coef", type=float, default=PPO_ENT_COEF)
    parser.add_argument("--n-envs", type=int, default=PPO_N_ENVS)
    parser.add_argument(
        "--vec-env",
        choices=("auto", "dummy", "subproc"),
        default=PPO_VEC_ENV,
    )
    parser.add_argument("--torch-threads", type=int, default=PPO_TORCH_THREADS)
    parser.add_argument("--no-exploration-reward", action="store_true")
    parser.add_argument("--model-path", type=str, default=str(MODELS_DIR / "ppo_maze.zip"))
    parser.add_argument("--outputs-dir", type=str, default=str(OUTPUTS_DIR))
    args = parser.parse_args(argv)

    PPO, BaseCallback, DummyVecEnv, SubprocVecEnv, VecMonitor = _load_sb3()

    try:
        import torch

        torch_threads = max(1, int(args.torch_threads))
        torch.set_num_threads(torch_threads)
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    class TrainingProgressCallback(BaseCallback):
        def __init__(self, total_timesteps: int):
            super().__init__()
            self.total_timesteps = total_timesteps
            self.episodes = 0
            self.recent_successes: list[float] = []
            self._last_progress_line = ""
            self._start_time = 0.0

        def _on_training_start(self) -> None:
            self._start_time = time.perf_counter()
            self._print_progress(0)

        def _on_step(self) -> bool:
            for info in self.locals.get("infos", []):
                episode_info = info.get("episode")
                if episode_info is None:
                    continue
                self.episodes += 1
                success = episode_info.get("success")
                if success is not None:
                    if isinstance(success, str):
                        success = success.lower() in {"true", "1", "yes"}
                    self.recent_successes.append(1.0 if bool(success) else 0.0)
                    self.recent_successes = self.recent_successes[-100:]
            return True

        def _on_rollout_end(self) -> None:
            self._print_progress(self.num_timesteps)

        def _on_training_end(self) -> None:
            return None

        def _print_progress(self, steps: int) -> None:
            line = format_progress(
                steps,
                self.total_timesteps,
                self.episodes,
                time.perf_counter() - self._start_time if self._start_time else None,
            )
            if line != self._last_progress_line:
                print(line, flush=True)
                self._last_progress_line = line

    outputs_dir = Path(args.outputs_dir)
    models_dir = Path(args.model_path).parent
    outputs_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    n_envs = max(1, int(args.n_envs))
    fixed_maps = [Path(args.map)] if args.map else discover_map_files(args.maps_dir)
    random_pool = build_random_map_pool(
        args.random_maps,
        seed=args.seed,
        rows=args.random_rows,
        cols=args.random_cols,
        wall_density=args.wall_density,
        trap_density=args.trap_density,
        style=args.random_style,
        door_orientation=args.door_orientation,
        endpoint_mode=args.endpoint_mode,
    )
    monitor_path = outputs_dir / "training_monitor.csv"
    env_fns = [
        make_training_env(
            fixed_maps=fixed_maps,
            random_pool=random_pool,
            random_map_probability=args.random_map_prob,
            max_steps=args.max_steps,
            view_range=args.view_range,
            view_width=args.view_width,
            exploration_reward=not args.no_exploration_reward,
            seed=args.seed + index,
        )
        for index in range(n_envs)
    ]
    vec_env_kind = args.vec_env
    if vec_env_kind == "auto":
        vec_env_kind = "dummy"
    if vec_env_kind == "subproc" and n_envs > 1:
        env = SubprocVecEnv(env_fns, start_method="spawn")
    else:
        env = DummyVecEnv(env_fns)
    env = VecMonitor(
        env,
        filename=str(monitor_path),
        info_keywords=("success", "has_key", "passed_door"),
    )
    n_steps = rollout_steps_per_env(PPO_N_STEPS, n_envs, PPO_BATCH_SIZE)
    print(
        f"训练配置：并行环境 {n_envs} | 采样后端 {vec_env_kind} | "
        f"每环境 rollout {n_steps} | torch threads {max(1, int(args.torch_threads))}",
        flush=True,
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        seed=args.seed,
        learning_rate=PPO_LEARNING_RATE,
        gamma=PPO_GAMMA,
        n_steps=n_steps,
        batch_size=PPO_BATCH_SIZE,
        ent_coef=args.ent_coef,
    )
    model.learn(
        total_timesteps=args.timesteps,
        callback=TrainingProgressCallback(args.timesteps),
    )
    model.save(args.model_path)
    env.close()

    curve_path = outputs_dir / "ppo_training_curve.png"
    save_training_curve(monitor_path, curve_path)

    print(f"模型已保存：{args.model_path}")
    print(f"监控日志：{monitor_path}")
    print(f"训练曲线：{curve_path}")


if __name__ == "__main__":
    main()
