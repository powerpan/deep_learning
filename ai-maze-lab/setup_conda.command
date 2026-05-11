#!/bin/zsh
set -e

SCRIPT_DIR="${0:A:h}"
CONDA_BIN="/Users/ericpan/anaconda3/bin/conda"

if [ ! -x "$CONDA_BIN" ]; then
  echo "未找到 conda：$CONDA_BIN"
  echo "请确认 Anaconda 安装路径，或手动修改本脚本里的 CONDA_BIN。"
  exit 1
fi

cd "$SCRIPT_DIR"
"$CONDA_BIN" env list | grep -qE '^ai-maze-lab[[:space:]]' || "$CONDA_BIN" env create -f environment.yml
"$CONDA_BIN" run -n ai-maze-lab python game_app.py
