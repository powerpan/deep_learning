from __future__ import annotations

import sys


COMMANDS = {
    "train": ("train_ppo", "Train PPO"),
    "play": ("play_ppo", "Replay a trained model"),
    "evaluate": ("evaluate_ppo", "Evaluate fixed and random maps"),
}


def print_help() -> None:
    print("AI Maze PPO Lab")
    print("")
    print("Usage:")
    print("  python main.py train --timesteps 500000 --n-envs 4 --random-maps 200 --obs-mode grid --local-view-size 7")
    print("  python main.py play --model models/ppo_maze.zip --map maps/level_1.txt")
    print("  python main.py evaluate --model models/ppo_maze.zip --random-tests 50 --endpoint-mode mixed")
    print("")
    print("Commands:")
    for name, (_, description) in COMMANDS.items():
        print(f"  {name:<9} {description}")


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        print_help()
        return

    command = args.pop(0)
    if command not in COMMANDS:
        print(f"Unknown command: {command}")
        print_help()
        raise SystemExit(2)

    module_name = COMMANDS[command][0]
    module = __import__(module_name)
    module.main(args)


if __name__ == "__main__":
    main()
