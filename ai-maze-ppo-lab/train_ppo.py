from __future__ import annotations

import argparse
import csv
import os
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np

from config import (
    DEFAULT_RANDOM_MAP_PROB,
    LOCAL_VIEW_SIZE,
    MAPS_DIR,
    MAX_STEPS,
    MODELS_DIR,
    OBSERVATION_MODE,
    OUTPUTS_DIR,
    PPO_ALGO,
    PPO_BATCH_SIZE,
    PPO_CURRICULUM,
    PPO_ENT_COEF,
    PPO_GAMMA,
    PPO_LEARNING_RATE,
    PPO_N_EPOCHS,
    PPO_N_ENVS,
    PPO_N_STEPS,
    PPO_TORCH_THREADS,
    PPO_VEC_ENV,
    RANDOM_MAP_COLS,
    RANDOM_MAP_ROWS,
    RANDOM_DOOR_ORIENTATION,
    RANDOM_ENDPOINT_MODE,
    RANDOM_MAP_STYLE,
    RANDOM_SIMPLE_MAP_PROB,
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
from policy import SmallGridCNN
from random_maps import (
    DOOR_ORIENTATION_OPTIONS,
    ENDPOINT_MODE_OPTIONS,
    STYLE_OPTIONS,
    GeneratedMap,
    build_random_map_pool,
)


@dataclass(frozen=True)
class TrainingPhase:
    name: str
    timesteps: int
    fixed_maps: list[Path]
    random_pool: list[GeneratedMap]
    random_map_probability: float


def _load_sb3():
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import BaseCallback
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
        from sb3_contrib import RecurrentPPO
    except ImportError as exc:
        raise SystemExit(
            "Missing PPO dependencies. Run: pip install -r requirements.txt"
        ) from exc
    return PPO, RecurrentPPO, BaseCallback, DummyVecEnv, SubprocVecEnv, VecMonitor


def discover_map_files(maps_dir: str | Path) -> list[Path]:
    path = Path(maps_dir)
    return sorted(path.glob("*.txt"))


def map_has_key_or_door(path: str | Path) -> bool:
    text = Path(path).read_text()
    return "K" in text or "D" in text


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


def combine_monitor_csv(parts: list[Path], output_path: Path) -> None:
    rows: list[str] = []
    header = ""
    for path in parts:
        if not path.exists():
            monitor_suffix_path = Path(str(path) + ".monitor.csv")
            path = monitor_suffix_path if monitor_suffix_path.exists() else path
        if not path.exists():
            continue
        with path.open() as handle:
            for line in handle:
                if line.startswith("#"):
                    continue
                if not header:
                    header = line
                    continue
                if line == header:
                    continue
                rows.append(line)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        handle.write('#{"t_start": 0, "env_id": "None"}\n')
        if header:
            handle.write(header)
        for row in rows:
            handle.write(row)


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
    observation_mode: str,
    local_view_size: int,
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
            observation_mode=observation_mode,
            local_view_size=local_view_size,
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


def allocate_phase_timesteps(total_timesteps: int, rollout_batch: int) -> list[int]:
    if total_timesteps < rollout_batch * 3:
        return [total_timesteps]

    ratios = (0.25, 0.35, 0.40)
    allocated = []
    used = 0
    for ratio in ratios[:-1]:
        value = max(rollout_batch, int(total_timesteps * ratio))
        value = (value // rollout_batch) * rollout_batch
        allocated.append(value)
        used += value
    allocated.append(max(rollout_batch, total_timesteps - used))
    return allocated


def build_training_phases(
    *,
    curriculum: str,
    total_timesteps: int,
    rollout_batch: int,
    fixed_maps: list[Path],
    random_maps: int,
    seed: int,
    rows: int,
    cols: int,
    wall_density: float,
    trap_density: float,
    style: str,
    door_orientation: str,
    endpoint_mode: str,
    simple_map_probability: float,
    random_map_probability: float,
) -> list[TrainingPhase]:
    if curriculum == "none":
        return [
            TrainingPhase(
                name="full",
                timesteps=total_timesteps,
                fixed_maps=fixed_maps,
                random_pool=build_random_map_pool(
                    random_maps,
                    seed=seed,
                    rows=rows,
                    cols=cols,
                    wall_density=wall_density,
                    trap_density=trap_density,
                    style=style,
                    door_orientation=door_orientation,
                    endpoint_mode=endpoint_mode,
                    simple_map_probability=simple_map_probability,
                ),
                random_map_probability=random_map_probability,
            )
        ]

    amounts = allocate_phase_timesteps(total_timesteps, rollout_batch)
    if len(amounts) == 1:
        return build_training_phases(
            curriculum="none",
            total_timesteps=total_timesteps,
            rollout_batch=rollout_batch,
            fixed_maps=fixed_maps,
            random_maps=random_maps,
            seed=seed,
            rows=rows,
            cols=cols,
            wall_density=wall_density,
            trap_density=trap_density,
            style=style,
            door_orientation=door_orientation,
            endpoint_mode=endpoint_mode,
            simple_map_probability=simple_map_probability,
            random_map_probability=random_map_probability,
        )

    simple_fixed_maps = [path for path in fixed_maps if not map_has_key_or_door(path)]
    phase_specs = [
        (
            "exit-only",
            amounts[0],
            simple_fixed_maps,
            min(wall_density, 0.08),
            0.0,
            "open",
            1.0,
        ),
        (
            "key-door",
            amounts[1],
            fixed_maps,
            min(wall_density, 0.12),
            0.0,
            style,
            0.0,
        ),
        (
            "full-mix",
            amounts[2],
            fixed_maps,
            wall_density,
            trap_density,
            style,
            simple_map_probability,
        ),
    ]

    phases = []
    for index, (
        name,
        timesteps,
        phase_fixed_maps,
        phase_wall_density,
        phase_trap_density,
        phase_style,
        phase_simple_probability,
    ) in enumerate(phase_specs):
        phases.append(
            TrainingPhase(
                name=name,
                timesteps=timesteps,
                fixed_maps=phase_fixed_maps,
                random_pool=build_random_map_pool(
                    random_maps,
                    seed=seed + index * 1000,
                    rows=rows,
                    cols=cols,
                    wall_density=phase_wall_density,
                    trap_density=phase_trap_density,
                    style=phase_style,
                    door_orientation=door_orientation,
                    endpoint_mode=endpoint_mode,
                    simple_map_probability=phase_simple_probability,
                ),
                random_map_probability=1.0 if not phase_fixed_maps else random_map_probability,
            )
        )
    return phases


def create_vec_env(
    *,
    phase: TrainingPhase,
    n_envs: int,
    vec_env_kind: str,
    max_steps: int,
    view_range: int,
    view_width: int,
    observation_mode: str,
    local_view_size: int,
    exploration_reward: bool,
    seed: int,
    DummyVecEnv,
    SubprocVecEnv,
    VecMonitor,
    monitor_path: Path,
):
    env_fns = [
        make_training_env(
            fixed_maps=phase.fixed_maps,
            random_pool=phase.random_pool,
            random_map_probability=phase.random_map_probability,
            max_steps=max_steps,
            view_range=view_range,
            view_width=view_width,
            observation_mode=observation_mode,
            local_view_size=local_view_size,
            exploration_reward=exploration_reward,
            seed=seed + index,
        )
        for index in range(n_envs)
    ]
    if vec_env_kind == "subproc" and n_envs > 1:
        env = SubprocVecEnv(env_fns, start_method="spawn")
    else:
        env = DummyVecEnv(env_fns)
    return VecMonitor(
        env,
        filename=str(monitor_path),
        info_keywords=("success", "has_key", "passed_door"),
    )


def format_progress(
    steps: int,
    total_timesteps: int,
    episodes: int,
    elapsed_seconds: float | None = None,
    speed_steps: int | None = None,
) -> str:
    capped_steps = min(steps, total_timesteps)
    percent = 100.0 if total_timesteps <= 0 else capped_steps / total_timesteps * 100.0
    speed_text = "-"
    speed_steps = capped_steps if speed_steps is None else max(0, speed_steps)
    if elapsed_seconds and elapsed_seconds > 0 and speed_steps > 0:
        speed_text = f"{speed_steps / elapsed_seconds:.0f}步/s"
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
    parser.add_argument("--simple-map-prob", type=float, default=RANDOM_SIMPLE_MAP_PROB)
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
    parser.add_argument(
        "--obs-mode",
        choices=("grid", "strips"),
        default=OBSERVATION_MODE,
    )
    parser.add_argument("--local-view-size", type=int, default=LOCAL_VIEW_SIZE)
    parser.add_argument("--ent-coef", type=float, default=PPO_ENT_COEF)
    parser.add_argument("--batch-size", type=int, default=PPO_BATCH_SIZE)
    parser.add_argument("--n-epochs", type=int, default=PPO_N_EPOCHS)
    parser.add_argument("--algo", choices=("recurrent-ppo", "ppo"), default=PPO_ALGO)
    parser.add_argument(
        "--curriculum",
        choices=("basic-to-keydoor", "none"),
        default=PPO_CURRICULUM,
    )
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

    PPO, RecurrentPPO, BaseCallback, DummyVecEnv, SubprocVecEnv, VecMonitor = _load_sb3()

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
            self._start_steps = 0

        def _on_training_start(self) -> None:
            self._start_time = time.perf_counter()
            self._start_steps = self.num_timesteps
            if self.num_timesteps == 0:
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
                steps - self._start_steps,
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
    monitor_path = outputs_dir / "training_monitor.csv"
    vec_env_kind = args.vec_env
    if vec_env_kind == "auto":
        vec_env_kind = "dummy"
    batch_size = max(1, int(args.batch_size))
    n_epochs = max(1, int(args.n_epochs))
    n_steps = rollout_steps_per_env(PPO_N_STEPS, n_envs, batch_size)
    rollout_batch = n_steps * n_envs
    phases = build_training_phases(
        curriculum=args.curriculum,
        total_timesteps=args.timesteps,
        rollout_batch=rollout_batch,
        fixed_maps=fixed_maps,
        random_maps=args.random_maps,
        seed=args.seed,
        rows=args.random_rows,
        cols=args.random_cols,
        wall_density=args.wall_density,
        trap_density=args.trap_density,
        style=args.random_style,
        door_orientation=args.door_orientation,
        endpoint_mode=args.endpoint_mode,
        simple_map_probability=args.simple_map_prob,
        random_map_probability=args.random_map_prob,
    )
    observation_label = (
        f"grid {args.local_view_size}x{args.local_view_size}"
        if args.obs_mode == "grid"
        else f"strips {args.view_range}x{args.view_width}"
    )
    print(
        f"训练配置：算法 {args.algo} | 课程 {args.curriculum} | "
        f"观察 {observation_label} | 并行环境 {n_envs} | "
        f"采样后端 {vec_env_kind} | 每环境 rollout {n_steps} | "
        f"batch {batch_size} | epochs {n_epochs} | "
        f"torch threads {max(1, int(args.torch_threads))}",
        flush=True,
    )

    model_class = RecurrentPPO if args.algo == "recurrent-ppo" else PPO
    if args.obs_mode == "grid":
        policy_name = "CnnLstmPolicy" if args.algo == "recurrent-ppo" else "CnnPolicy"
        policy_kwargs = {
            "features_extractor_class": SmallGridCNN,
            "features_extractor_kwargs": {"features_dim": 64},
        }
    else:
        policy_name = "MlpLstmPolicy" if args.algo == "recurrent-ppo" else "MlpPolicy"
        policy_kwargs = None
    model = None
    phase_monitor_paths: list[Path] = []

    for phase_index, phase in enumerate(phases):
        phase_monitor_path = outputs_dir / f"training_monitor_{phase_index + 1}_{phase.name}.csv"
        phase_monitor_paths.append(phase_monitor_path)
        env = create_vec_env(
            phase=phase,
            n_envs=n_envs,
            vec_env_kind=vec_env_kind,
            max_steps=args.max_steps,
            view_range=args.view_range,
            view_width=args.view_width,
            observation_mode=args.obs_mode,
            local_view_size=args.local_view_size,
            exploration_reward=not args.no_exploration_reward,
            seed=args.seed + phase_index * 10000,
            DummyVecEnv=DummyVecEnv,
            SubprocVecEnv=SubprocVecEnv,
            VecMonitor=VecMonitor,
            monitor_path=phase_monitor_path,
        )
        print(
            f"课程阶段 {phase_index + 1}/{len(phases)}：{phase.name} "
            f"{phase.timesteps:,} steps",
            flush=True,
        )

        if model is None:
            model = model_class(
                policy_name,
                env,
                verbose=0,
                seed=args.seed,
                learning_rate=PPO_LEARNING_RATE,
                gamma=PPO_GAMMA,
                n_steps=n_steps,
                batch_size=batch_size,
                n_epochs=n_epochs,
                ent_coef=args.ent_coef,
                policy_kwargs=policy_kwargs,
            )
            reset_num_timesteps = True
        else:
            model.set_env(env)
            reset_num_timesteps = False

        model.learn(
            total_timesteps=phase.timesteps,
            callback=TrainingProgressCallback(args.timesteps),
            reset_num_timesteps=reset_num_timesteps,
        )
        env.close()

    assert model is not None
    model.save(args.model_path)
    combine_monitor_csv(phase_monitor_paths, monitor_path)

    curve_path = outputs_dir / "ppo_training_curve.png"
    save_training_curve(monitor_path, curve_path)

    print(f"模型已保存：{args.model_path}")
    print(f"监控日志：{monitor_path}")
    print(f"训练曲线：{curve_path}")


if __name__ == "__main__":
    main()
