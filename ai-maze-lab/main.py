import argparse
from pathlib import Path

import numpy as np

from analytics import (
    DEFAULT_OUTPUT_DIR,
    save_best_path,
    save_training_curves,
    save_training_log,
    save_visitation_heatmap,
)
from config import (
    ALPHA,
    EPSILON_DECAY,
    EPSILON_MIN,
    EPSILON_START,
    GAMMA,
    MAX_STEPS_PER_EPISODE,
    RENDER_EVERY,
    REPLAY_DELAY_MS,
    TRAIN_EPISODES,
    TRAIN_RENDER_DELAY_MS,
)
from maze_env import MazeEnv
from q_agent import QLearningAgent


def parse_args():
    parser = argparse.ArgumentParser(description="AI Maze Lab: Q-learning maze visualizer")
    parser.add_argument("--map", default="maps/level_1.txt", help="Path to a text maze map.")
    parser.add_argument("--episodes", type=int, default=TRAIN_EPISODES, help="Training episodes.")
    parser.add_argument(
        "--render-every",
        type=int,
        default=RENDER_EVERY,
        help="Render one training episode every N episodes. Use 0 to skip training render.",
    )
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS_PER_EPISODE)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--gamma", type=float, default=GAMMA)
    parser.add_argument("--epsilon", type=float, default=EPSILON_START)
    parser.add_argument("--epsilon-decay", type=float, default=EPSILON_DECAY)
    parser.add_argument("--epsilon-min", type=float, default=EPSILON_MIN)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-render", action="store_true", help="Train and print stats without Pygame.")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for CSV logs, plots, heatmaps, and best path output.",
    )
    return parser.parse_args()


def resolve_map_path(map_arg):
    map_path = Path(map_arg)
    if map_path.exists():
        return map_path

    project_relative = Path(__file__).resolve().parent / map_arg
    if project_relative.exists():
        return project_relative

    return map_path


def should_render_episode(episode, render_every):
    return render_every > 0 and episode % render_every == 0


def train(env, agent, renderer, episodes, render_every):
    episode_stats = []
    visitation_counts = np.zeros((env.rows, env.cols), dtype=np.int64)

    for episode in range(1, episodes + 1):
        state = env.reset()
        visitation_counts[state[:2]] += 1
        total_reward = 0.0
        success = False
        reason = "running"
        steps = 0
        episode_epsilon = agent.epsilon
        render_this_episode = renderer is not None and should_render_episode(episode, render_every)

        for step in range(1, env.max_steps + 1):
            action = agent.choose_action(state, explore=True)
            next_state, reward, done, info = env.step(action)
            agent.update(state, action, reward, next_state, done)

            state = next_state
            visitation_counts[state[:2]] += 1
            total_reward += reward
            steps = step
            success = info["success"]
            reason = info["reason"]

            if render_this_episode and renderer is not None:
                keep_rendering = renderer.render(
                    env,
                    {
                        "phase": "training",
                        "episode": episode,
                        "step": step,
                        "total_reward": total_reward,
                        "epsilon": agent.epsilon,
                        "success": success,
                        "reason": reason,
                        "has_key": info["has_key"],
                        "passed_door": info["passed_door"],
                    },
                    delay_ms=TRAIN_RENDER_DELAY_MS,
                )
                if not keep_rendering:
                    renderer.close()
                    renderer = None
                    render_this_episode = False

            if done:
                break

        agent.decay_epsilon()
        episode_stats.append(
            {
                "episode": episode,
                "steps": steps,
                "total_reward": total_reward,
                "success": success,
                "reason": reason,
                "epsilon": episode_epsilon,
                "has_key": env.has_key,
                "passed_door": env.passed_door,
            }
        )

    return episode_stats, visitation_counts, renderer


def replay_greedy_policy(env, agent, renderer, episode_number):
    state = env.reset()
    path = [state]
    visited_states = {state}
    total_reward = 0.0
    success = False
    reason = "running"
    steps = 0

    if renderer is not None:
        renderer.render(
            env,
            {
                "phase": "greedy replay",
                "episode": episode_number,
                "step": 0,
                "total_reward": total_reward,
                "epsilon": 0.0,
                "success": success,
                "reason": reason,
                "has_key": env.has_key,
                "passed_door": env.passed_door,
            },
            delay_ms=REPLAY_DELAY_MS,
        )

    for step in range(1, env.max_steps + 1):
        action = agent.choose_action(state, explore=False)
        next_state, reward, done, info = env.step(action)
        state = next_state
        path.append(state)
        total_reward += reward
        steps = step
        success = info["success"]
        reason = info["reason"]

        if renderer is not None:
            keep_rendering = renderer.render(
                env,
                {
                    "phase": "greedy replay",
                    "episode": episode_number,
                    "step": step,
                    "total_reward": total_reward,
                    "epsilon": 0.0,
                    "success": success,
                    "reason": reason,
                    "has_key": info["has_key"],
                    "passed_door": info["passed_door"],
                },
                delay_ms=REPLAY_DELAY_MS,
            )
            if not keep_rendering:
                renderer.close()
                renderer = None

        if done:
            break

        if state in visited_states:
            reason = "loop"
            break
        visited_states.add(state)

    if renderer is not None:
        renderer.wait()

    return {
        "steps": steps,
        "total_reward": total_reward,
        "success": success,
        "reason": reason,
        "path": path,
        "has_key": env.has_key,
        "passed_door": env.passed_door,
        "reached_exit": success,
    }


def print_summary(stats, replay_stats, final_epsilon):
    recent = stats[-100:]
    recent_success_rate = 0.0
    recent_avg_steps = 0.0

    if recent:
        recent_success_rate = sum(1 for item in recent if item["success"]) / len(recent) * 100
        recent_avg_steps = sum(item["steps"] for item in recent) / len(recent)

    print("\nTraining summary")
    print("----------------")
    print(f"Total episodes: {len(stats)}")
    print(f"Recent 100 success rate: {recent_success_rate:.1f}%")
    print(f"Recent 100 average steps: {recent_avg_steps:.1f}")
    print(f"Final epsilon: {final_epsilon:.3f}")
    print(
        "Greedy replay: "
        f"success={replay_stats['success']}, "
        f"steps={replay_stats['steps']}, "
        f"reward={replay_stats['total_reward']:.1f}, "
        f"reason={replay_stats['reason']}, "
        f"has_key={replay_stats.get('has_key', False)}, "
        f"passed_door={replay_stats.get('passed_door', False)}"
    )


def save_outputs(env, map_path, stats, visitation_counts, replay_stats, output_dir):
    output_paths = {
        "training_log": save_training_log(stats, output_dir),
        "training_curve": save_training_curves(stats, output_dir),
        "visitation_heatmap": save_visitation_heatmap(env, visitation_counts, output_dir),
        "best_path": save_best_path(map_path, replay_stats, output_dir),
    }

    print("\nSaved outputs")
    print("-------------")
    for label, path in output_paths.items():
        print(f"{label}: {path}")

    return output_paths


def main():
    args = parse_args()
    map_path = resolve_map_path(args.map)

    env = MazeEnv(map_path, max_steps=args.max_steps)
    agent = QLearningAgent(
        rows=env.rows,
        cols=env.cols,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon,
        epsilon_decay=args.epsilon_decay,
        epsilon_min=args.epsilon_min,
        seed=args.seed,
    )

    renderer = None
    if not args.no_render:
        from renderer import Renderer

        renderer = Renderer(env)

    stats, visitation_counts, renderer = train(
        env=env,
        agent=agent,
        renderer=renderer,
        episodes=args.episodes,
        render_every=args.render_every,
    )
    replay_stats = replay_greedy_policy(env, agent, renderer, args.episodes)

    if renderer is not None:
        renderer.close()

    save_outputs(env, map_path, stats, visitation_counts, replay_stats, args.output_dir)
    print_summary(stats, replay_stats, agent.epsilon)


if __name__ == "__main__":
    main()
