# AI Maze Lab

AI Maze Lab 是一个小型强化学习可视化项目：一个智能体从零开始探索迷宫，通过基础 Q-learning 学会从起点 `S` 走到出口 `E`。

项目目标是轻量、能跑、能看见训练过程。核心仍然只用 `pygame` 做 2D 可视化、`numpy` 维护 Q-table，训练分析图使用 `matplotlib` 输出，不引入 Gymnasium、Stable-Baselines3、PyTorch 或 TensorFlow。

## 项目结构

```text
ai-maze-lab/
  README.md
  requirements.txt
  main.py
  maze_env.py
  q_agent.py
  renderer.py
  analytics.py
  config.py
  maps/
    level_1.txt
    level_2.txt
    level_3.txt
    level_4_key_door.txt
    level_5_trap_key_door.txt
    level_6_dead_end.txt
  outputs/
```

## 安装方式

推荐使用 conda，避免 Python 3.14 下部分 pygame 字体模块兼容问题：

```bash
cd /Users/ericpan/game_project/deep_learning/ai-maze-lab
conda env create -f environment.yml
conda run -n ai-maze-lab python game_app.py
```

也可以使用 Python venv：

```bash
cd /Users/ericpan/game_project/deep_learning/ai-maze-lab
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 运行方式

启动简易游戏软件模式：

```bash
python game_app.py
```

macOS 双击启动：

```text
双击项目根目录里的 AI Maze Lab.app
```

如果本机有 Anaconda，双击启动器会优先创建并使用 `ai-maze-lab` conda 环境。

如果没有 Anaconda，启动器会退回到 `.venv` 方案，打开终端安装 `pygame`、`numpy`、`matplotlib`，完成后自动启动。

当前这台机器已经创建好了 conda 环境，后续双击会直接使用：

```text
/Users/ericpan/anaconda3/envs/ai-maze-lab
```

运行默认关卡：

```bash
python main.py --map maps/level_1.txt
```

运行第二关，并训练 3000 局：

```bash
python main.py --map maps/level_2.txt --episodes 3000
```

每 50 局渲染一次训练过程：

```bash
python main.py --map maps/level_2.txt --episodes 3000 --render-every 50
```

只看终端统计、不打开 Pygame 窗口：

```bash
python main.py --map maps/level_1.txt --episodes 1000 --no-render
```

第二版实验记录示例：

```bash
python main.py --map maps/level_1.txt --episodes 3000
```

训练结束后会自动生成 `outputs/` 目录和训练结果文件。

第三版钥匙门关卡示例：

```bash
python main.py --map maps/level_4_key_door.txt --episodes 5000
python main.py --map maps/level_5_trap_key_door.txt --episodes 8000 --render-every 100
python main.py --map maps/level_6_dead_end.txt --episodes 10000 --render-every 100
```

如果使用双击 `.app` 或 `python game_app.py` 的中文界面：

- 点击 `训练 +500 局` 或 `训练 +2000 局`。
- 当前批次训练完成后会自动导出到项目根目录的 `outputs/`。
- 也可以手动点击 `导出结果` 重新生成最新结果文件。

## 简易游戏软件模式

`game_app.py` 提供一个更接近小游戏的 Pygame 界面：

- `第1关` 到 `第6关`：切换地图，并重新创建当前智能体。
- `手动试玩`：手动试玩地图，可用方向键或 `WASD` 移动。
- `训练 +500 局` / `训练 +2000 局`：在当前地图继续训练。
- `回放最优路线`：按当前 Q-table 使用 greedy 策略回放最优路径。
- `导出结果`：把当前训练日志、曲线图、热力图和最优路径写入 `outputs/`。
- `重置智能体`：重置当前地图和 Q-table。

## 如何观察训练效果

程序训练时会每隔若干 episode 渲染一局，窗口底部会显示：

- 当前阶段：训练或最优路径回放
- 当前 episode
- 当前步数
- 当前累计奖励
- 当前 epsilon
- 本局是否成功到达出口
- 当前结束原因，例如 `exit`、`wall`、`locked_door`、`key`、`door`、`trap`、`timeout`
- 当前是否持有钥匙、是否已经通过门

训练早期，智能体主要靠 epsilon-greedy 随机探索，可能频繁撞墙、绕路或掉进陷阱。随着 Q-table 更新，epsilon 逐渐衰减，智能体会更多选择已学到的高价值动作。训练结束后，程序会自动用 greedy 策略回放当前学到的最优路径。

终端最后会输出：

- 总 episode 数
- 最近 100 局成功率
- 最近 100 局平均步数
- 最终 epsilon
- 最优路径回放结果

## 第二版新增功能

第二版继续使用 Python + Pygame + NumPy + Q-learning，不引入 DQN、PPO、Gymnasium 或 Stable-Baselines3。新增内容主要用于训练过程分析，图像输出由 matplotlib 负责：

- 记录每一局训练日志。
- 输出最近 100 局滚动成功率、平均步数、平均奖励曲线。
- 统计训练期间每个格子的访问次数，并生成访问热力图。
- 训练结束后用当前 Q-table 的 greedy 策略保存一条最优路径。

## 输出文件说明

默认输出目录是 `outputs/`，如果不存在会自动创建。

训练结束后会生成：

- `outputs/training_log.csv`：每个 episode 的 `episode`、`total_reward`、`steps`、`success`、`epsilon`。
- `outputs/training_curve.png`：训练曲线图，包含最近 100 局成功率、平均步数、平均 reward。
- `outputs/visitation_heatmap.png`：访问热力图，颜色越深表示训练过程中越常访问。
- `outputs/best_path.txt`：训练后 greedy 策略走出的路径坐标。如果失败，会写入 `Success: False` 和实际走过的路径。第三版还会记录 `Has key`、`Passed door`、`Reached exit`。

查看方式：

```bash
cd /Users/ericpan/game_project/deep_learning/ai-maze-lab
open outputs/training_curve.png
open outputs/visitation_heatmap.png
open outputs/best_path.txt
```

也可以直接在 Finder 中打开 `outputs/` 文件夹查看。

## 当前算法说明

当前版本采用的是表格型 Q-learning，不是深度强化学习。

第三版开始，状态使用 `(row, col, has_key)`，其中 `has_key` 用 `0/1` 表示是否已经拿到钥匙。

这样做是必要的。因为同一个格子在“没有钥匙”和“已经有钥匙”时，后续最优动作可能完全不同。没有这个状态维度时，Q-table 会把两种情况混在一起，智能体很难稳定学会“先拿钥匙，再开门，再到出口”。

动作使用：

- `0`：上
- `1`：下
- `2`：左
- `3`：右

奖励函数：

- 每走一步：`-0.1`
- 撞墙：`-2`
- 没有钥匙撞门：`-2`
- 第一次拿到钥匙：`+10`
- 已经有钥匙后再次经过钥匙格：不重复加分
- 第一次通过门：`+5`
- 踩陷阱：`-20`，并结束本局
- 到达出口：`+80`，并结束本局
- 超过最大步数：结束本局

## 第三版钥匙和门

新增地图元素：

- `K`：钥匙。智能体走到钥匙格后，`has_key=True`。
- `D`：门。没有钥匙时不能通过，有钥匙后可以通过。

新关卡：

- `maps/level_4_key_door.txt`：基础钥匙门关卡，推荐训练 `5000` 局。
- `maps/level_5_trap_key_door.txt`：钥匙、门、陷阱组合关卡，推荐训练 `8000` 局。
- `maps/level_6_dead_end.txt`：钥匙、门、死胡同组合关卡，推荐训练 `10000` 局。

观察是否学会了钥匙门顺序：

1. 看终端最后的 greedy replay 输出，应出现 `has_key=True, passed_door=True`。
2. 打开 `outputs/best_path.txt`，确认 `Has key: True`、`Passed door: True`、`Reached exit: True`。
3. 打开 `outputs/visitation_heatmap.png`，观察高频访问区域是否从起点延伸到钥匙、门和出口。
4. 如果用可视化训练，窗口底部会显示钥匙和过门状态。

## 后续可扩展方向

- 加入移动怪物
- 加入随机地图生成
- 加入 GIF/MP4 导出
- 加入 DQN 或 PPO
