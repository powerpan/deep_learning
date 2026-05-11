#!/bin/zsh
set -e

PROJECT_ROOT="${0:A:h:h}"
CONDA_BIN="/Users/ericpan/anaconda3/bin/conda"
CONDA_ENV_NAME="ai-maze-lab"
PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"

open_terminal_bootstrap() {
  local terminal_cmd

  if [ -x "$CONDA_BIN" ]; then
    terminal_cmd="cd ${(q)PROJECT_ROOT} && ${(q)CONDA_BIN} env list | grep -qE '^${CONDA_ENV_NAME}[[:space:]]' || ${(q)CONDA_BIN} env create -f environment.yml; ${(q)CONDA_BIN} run -n ${CONDA_ENV_NAME} python game_app.py"
  else
    terminal_cmd="cd ${(q)PROJECT_ROOT} && python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt && python game_app.py"
  fi

  if command -v osascript >/dev/null 2>&1; then
    osascript -e 'display dialog "首次启动需要创建运行环境并安装 pygame、numpy、matplotlib。推荐使用 conda 的 Python 3.12 环境，完成后自动启动。" buttons {"继续"} default button "继续" with title "AI Maze Lab"'
    osascript \
      -e 'tell application "Terminal" to activate' \
      -e "tell application \"Terminal\" to do script \"$terminal_cmd\""
  else
    echo "Please run these commands:"
    echo "cd ${(q)PROJECT_ROOT}"
    echo "python3 -m venv .venv"
    echo "source .venv/bin/activate"
    echo "pip install -r requirements.txt"
    echo "python game_app.py"
  fi
}

cd "$PROJECT_ROOT"

if [ -x "$CONDA_BIN" ] && "$CONDA_BIN" env list | grep -qE "^${CONDA_ENV_NAME}[[:space:]]"; then
  exec "$CONDA_BIN" run -n "$CONDA_ENV_NAME" python "$PROJECT_ROOT/game_app.py"
fi

if [ ! -x "$PYTHON_BIN" ]; then
  open_terminal_bootstrap
  exit 0
fi

if ! "$PYTHON_BIN" -c "import pygame, numpy, matplotlib" >/dev/null 2>&1; then
  open_terminal_bootstrap
  exit 0
fi

exec "$PYTHON_BIN" "$PROJECT_ROOT/game_app.py"
