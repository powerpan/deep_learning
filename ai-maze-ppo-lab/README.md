# AI Maze PPO Lab

AI Maze PPO Lab 是一个独立于 `ai-maze-lab` 的新项目，用来实验“局部视野 + PPO”的迷宫智能体。

旧项目继续保留 Q-learning 版本；这个项目使用 `Gymnasium + Stable-Baselines3 + PyTorch`，目标不是复用单关卡 Q-table，而是训练一个能在多张固定地图和随机地图中学习通用规则的策略网络。

## 核心特性

- 迷宫元素：`S` 起点、`E` 出口、`#` 墙、`.` 空地、`T` 陷阱、`K` 钥匙、`D` 门。
- 智能体初始没有钥匙，必须先走到 `K`，之后才能通过 `D`。
- 观察不是完整地图，而是上下左右四个方向的直线视野。
- 墙本身可见，墙后的格子不可见，拐角后的内容也不可见。
- 第一版使用 PPO 的 `MlpPolicy`，不引入 LSTM，先保证能训练、能保存、能回放、能评估。
- 训练时可以混合固定关卡和随机可解关卡，避免只记住单张地图。

## 安装

建议使用 Python 3.10 到 3.12。PyTorch 和 Stable-Baselines3 对 Python 3.14 的支持通常会滞后，所以不建议直接使用系统 Python 3.14。

如果你继续使用已有 conda 环境：

```bash
cd /Users/ericpan/game_project/deep_learning/ai-maze-ppo-lab
/Users/ericpan/anaconda3/envs/ai-maze-lab/bin/python -m pip install -r requirements.txt
```

如果使用普通 venv：

```bash
cd /Users/ericpan/game_project/deep_learning/ai-maze-ppo-lab
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 训练

```bash
cd /Users/ericpan/game_project/deep_learning/ai-maze-ppo-lab
python train_ppo.py --timesteps 500000 --random-maps 200
```

输出：

- `models/ppo_maze.zip`：训练后的 PPO 模型。
- `outputs/training_monitor.csv`：每个 episode 的 reward、步数、成功状态等。
- `outputs/ppo_training_curve.png`：最近 100 局滚动成功率、平均步数、平均 reward。

## 回放

```bash
cd /Users/ericpan/game_project/deep_learning/ai-maze-ppo-lab
python play_ppo.py --model models/ppo_maze.zip --map maps/level_1.txt
```

回放窗口会显示完整地图，并用暗色遮罩标出当前智能体看不到的区域。PPO 模型实际收到的 observation 只包含直线视野，不包含完整地图。

输出：

- `outputs/replay_trace.txt`：每一步位置、动作、奖励、钥匙状态、门状态。

## 评估

```bash
cd /Users/ericpan/game_project/deep_learning/ai-maze-ppo-lab
python evaluate_ppo.py --model models/ppo_maze.zip --random-tests 50
```

输出：

- `outputs/eval_summary.json`：固定地图和随机地图上的成功率、平均步数、平均 reward、拿钥匙率和过门率。

## 统一入口

也可以通过 `main.py` 调用：

```bash
python main.py train --timesteps 500000 --random-maps 200
python main.py play --model models/ppo_maze.zip --map maps/level_2_key_door.txt
python main.py evaluate --model models/ppo_maze.zip --random-tests 50
```

## 为什么需要 PPO

Q-learning 的 Q-table 通常绑定具体状态。如果状态是完整坐标，例如 `(row, col, has_key)`，它学到的是当前地图上的位置价值。换一张地图后，坐标含义变了，旧 Q-table 很难直接泛化。

PPO 使用神经网络策略。这里输入的是局部视野，而不是绝对地图坐标，因此模型有机会学习更通用的行为规则，例如：

- 看到钥匙时靠近钥匙。
- 没钥匙时不要撞门。
- 有钥匙后通过门。
- 避开陷阱。
- 沿通道探索出口。

## 当前限制

- 第一版 PPO 使用无记忆 `MlpPolicy`，在长走廊、回头路、复杂死胡同里可能表现不稳定。
- 局部视野导致任务变成部分可观测问题，后续可以升级到 Recurrent PPO 或加入短期记忆。
- 随机地图生成器只保证有一条合法路径，不保证每张图都难度均衡。

## 测试

安装依赖后可以运行：

```bash
cd /Users/ericpan/game_project/deep_learning/ai-maze-ppo-lab
python -m unittest discover tests
```

测试覆盖：

- 没钥匙不能过门。
- 钥匙奖励不会重复刷。
- 到达出口会成功结束。
- 墙后视野会变成 unknown。
- 随机地图包含 `S / K / D / E` 且可解。
