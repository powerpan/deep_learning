import csv
import os
from pathlib import Path

import numpy as np


DEFAULT_OUTPUT_DIR = Path("outputs")
ROLLING_WINDOW = 100


def ensure_output_dir(output_dir=DEFAULT_OUTPUT_DIR):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def save_training_log(stats, output_dir=DEFAULT_OUTPUT_DIR):
    output_path = ensure_output_dir(output_dir) / "training_log.csv"
    fieldnames = ["episode", "total_reward", "steps", "success", "epsilon"]

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for item in stats:
            writer.writerow(
                {
                    "episode": item["episode"],
                    "total_reward": f"{item['total_reward']:.6f}",
                    "steps": item["steps"],
                    "success": int(item["success"]),
                    "epsilon": f"{item['epsilon']:.6f}",
                }
            )

    return output_path


def save_training_curves(stats, output_dir=DEFAULT_OUTPUT_DIR, window=ROLLING_WINDOW):
    output_dir = ensure_output_dir(output_dir)
    configure_matplotlib_cache(output_dir)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = output_dir / "training_curve.png"

    episodes = np.array([item["episode"] for item in stats], dtype=np.int32)
    rewards = np.array([item["total_reward"] for item in stats], dtype=np.float32)
    steps = np.array([item["steps"] for item in stats], dtype=np.float32)
    success = np.array([1.0 if item["success"] else 0.0 for item in stats], dtype=np.float32)

    success_rate = rolling_mean(success, window) * 100.0
    avg_steps = rolling_mean(steps, window)
    avg_reward = rolling_mean(rewards, window)

    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    fig.suptitle("AI Maze Lab Training Curves", fontsize=16)

    axes[0].plot(episodes, success_rate, color="#2e8b57", linewidth=1.8)
    axes[0].set_title(f"Rolling Success Rate ({window} episodes)")
    axes[0].set_ylabel("Success Rate (%)")
    axes[0].set_ylim(0, 105)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(episodes, avg_steps, color="#4169e1", linewidth=1.8)
    axes[1].set_title(f"Rolling Average Steps ({window} episodes)")
    axes[1].set_ylabel("Steps")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(episodes, avg_reward, color="#b45f06", linewidth=1.8)
    axes[2].set_title(f"Rolling Average Reward ({window} episodes)")
    axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("Reward")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def save_visitation_heatmap(env, visitation_counts, output_dir=DEFAULT_OUTPUT_DIR):
    output_dir = ensure_output_dir(output_dir)
    configure_matplotlib_cache(output_dir)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = output_dir / "visitation_heatmap.png"
    heatmap = np.array(visitation_counts, dtype=np.float32)

    for row in range(env.rows):
        for col in range(env.cols):
            if env.tile_at(row, col) == "#":
                heatmap[row, col] = 0.0

    fig, ax = plt.subplots(figsize=(max(7, env.cols * 0.7), max(5, env.rows * 0.7)))
    image = ax.imshow(heatmap, cmap="YlOrRd")
    fig.colorbar(image, ax=ax, label="Visit Count")
    ax.set_title("State Visitation Heatmap")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_xticks(np.arange(env.cols))
    ax.set_yticks(np.arange(env.rows))
    ax.set_xticks(np.arange(-0.5, env.cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, env.rows, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.8, alpha=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    for row in range(env.rows):
        for col in range(env.cols):
            tile = env.tile_at(row, col)
            if tile in {"S", "E", "T", "K", "D", "#"}:
                ax.text(col, row, tile, ha="center", va="center", color="black", fontsize=10)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def run_greedy_path(env, agent, max_steps=None):
    max_steps = max_steps or env.max_steps
    state = env.reset()
    path = [state]
    visited_states = {state}
    total_reward = 0.0
    success = False
    reason = "running"

    for _ in range(1, max_steps + 1):
        action = agent.choose_action(state, explore=False)
        next_state, reward, done, info = env.step(action)
        path.append(next_state)
        total_reward += reward
        state = next_state
        success = info["success"]
        reason = info["reason"]

        if done:
            break

        if state in visited_states:
            reason = "loop"
            break
        visited_states.add(state)
    else:
        reason = "timeout"

    return {
        "path": path,
        "success": success,
        "steps": max(0, len(path) - 1),
        "total_reward": total_reward,
        "reason": reason,
        "has_key": env.has_key,
        "passed_door": env.passed_door,
        "reached_exit": success,
    }


def save_best_path(map_path, best_path_result, output_dir=DEFAULT_OUTPUT_DIR):
    output_path = ensure_output_dir(output_dir) / "best_path.txt"
    path = best_path_result["path"]

    with output_path.open("w", encoding="utf-8") as file:
        file.write(f"Best path for {map_path}\n")
        file.write(f"Success: {best_path_result['success']}\n")
        file.write(f"Steps: {best_path_result['steps']}\n")
        file.write(f"Total reward: {best_path_result['total_reward']:.1f}\n")
        file.write(f"Reason: {best_path_result['reason']}\n")
        file.write(f"Has key: {best_path_result.get('has_key', False)}\n")
        file.write(f"Passed door: {best_path_result.get('passed_door', False)}\n")
        file.write(f"Reached exit: {best_path_result.get('reached_exit', best_path_result['success'])}\n\n")
        for index, position in enumerate(path):
            file.write(f"{index}: {position}\n")

    return output_path


def rolling_mean(values, window):
    if len(values) == 0:
        return values

    result = np.zeros_like(values, dtype=np.float32)
    for index in range(len(values)):
        start = max(0, index - window + 1)
        result[index] = float(np.mean(values[start : index + 1]))
    return result


def configure_matplotlib_cache(output_dir):
    cache_dir = Path(output_dir) / ".matplotlib-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    xdg_cache_dir = cache_dir / "xdg"
    xdg_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache_dir))
