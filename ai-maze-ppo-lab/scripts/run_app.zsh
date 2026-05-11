#!/bin/zsh
set -e

PROJECT_ROOT="${0:A:h:h}"
CONDA_BIN="/Users/ericpan/anaconda3/bin/conda"
CONDA_ENV_NAME="ai-maze-lab"
PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"
REQUIRED_IMPORTS="import pygame, numpy, matplotlib, gymnasium, stable_baselines3, torch"

open_terminal_bootstrap() {
  local terminal_cmd

  if [ -x "$CONDA_BIN" ]; then
    terminal_cmd="cd ${(q)PROJECT_ROOT} && ${(q)CONDA_BIN} env list | grep -qE '^${CONDA_ENV_NAME}[[:space:]]' || ${(q)CONDA_BIN} env create -f environment.yml; ${(q)CONDA_BIN} run -n ${CONDA_ENV_NAME} python -m pip install -r requirements.txt; ${(q)CONDA_BIN} run -n ${CONDA_ENV_NAME} python ppo_app.py"
  else
    terminal_cmd="cd ${(q)PROJECT_ROOT} && python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt && python ppo_app.py"
  fi

  if command -v osascript >/dev/null 2>&1; then
    osascript -e 'display dialog "首次启动需要安装 PPO 依赖：pygame、numpy、matplotlib、gymnasium、stable-baselines3、torch。完成后会自动启动界面。" buttons {"继续"} default button "继续" with title "AI Maze PPO Lab"'
    osascript \
      -e 'tell application "Terminal" to activate' \
      -e "tell application \"Terminal\" to do script \"$terminal_cmd\""
  else
    echo "Please run these commands:"
    echo "cd ${(q)PROJECT_ROOT}"
    echo "python3 -m venv .venv"
    echo "source .venv/bin/activate"
    echo "pip install -r requirements.txt"
    echo "python ppo_app.py"
  fi
}

cd "$PROJECT_ROOT"

if [ -x "$CONDA_BIN" ] && "$CONDA_BIN" env list | grep -qE "^${CONDA_ENV_NAME}[[:space:]]"; then
  if "$CONDA_BIN" run -n "$CONDA_ENV_NAME" python -c "$REQUIRED_IMPORTS" >/dev/null 2>&1; then
    exec "$CONDA_BIN" run -n "$CONDA_ENV_NAME" python "$PROJECT_ROOT/ppo_app.py"
  fi
  open_terminal_bootstrap
  exit 0
fi

if [ ! -x "$PYTHON_BIN" ]; then
  open_terminal_bootstrap
  exit 0
fi

if ! "$PYTHON_BIN" -c "$REQUIRED_IMPORTS" >/dev/null 2>&1; then
  open_terminal_bootstrap
  exit 0
fi

exec "$PYTHON_BIN" "$PROJECT_ROOT/ppo_app.py"
