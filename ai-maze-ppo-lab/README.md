# AI Maze PPO Lab

AI Maze PPO Lab 是一个独立于 `ai-maze-lab` 的新项目，用来实验“局部视野 + PPO”的迷宫智能体。

旧项目继续保留 Q-learning 版本；这个项目使用 `Gymnasium + Stable-Baselines3/sb3-contrib + PyTorch`，目标不是复用单关卡 Q-table，而是训练一个能在多张固定地图和随机地图中学习通用规则的策略网络。

## 核心特性

- 迷宫元素：`S` 起点、`E` 出口、`#` 墙、`.` 空地、`T` 陷阱、`K` 钥匙、`D` 门。
- 如果地图包含 `K`，智能体初始没有钥匙，必须先走到 `K`，之后才能通过 `D`。如果地图没有 `K/D`，它会被视为普通“找出口”关卡。
- 观察不是完整地图，而是上下左右四个方向的窄视野带，默认每个方向看 `3 格深 x 3 格宽`。
- 墙本身可见，墙后的格子不可见，拐角后的内容也不可见。
- 默认使用 `RecurrentPPO + MlpLstmPolicy`，让模型在局部视野下保留隐藏状态；仍可通过 `--algo ppo` 退回普通 PPO。
- 训练时可以混合固定关卡和随机可解关卡，随机图支持开放、分割、房间、死胡同、迷宫等风格，门可以横向或纵向出现，出入口也不再固定在左上和右下。

## 安装

建议使用 Python 3.10 到 3.12。PyTorch 和 Stable-Baselines3 对 Python 3.14 的支持通常会滞后，所以不建议直接使用系统 Python 3.14。

## 双击启动界面

macOS 上可以直接双击：

```bash
/Users/ericpan/game_project/deep_learning/ai-maze-ppo-lab/AI Maze PPO Lab.app
```

也可以双击：

```bash
/Users/ericpan/game_project/deep_learning/ai-maze-ppo-lab/start_ai_maze_ppo_lab.command
```

首次启动时，如果依赖没有装好，启动器会打开 Terminal，自动安装 `pygame、numpy、matplotlib、gymnasium、stable-baselines3、sb3-contrib、torch`，完成后再启动界面。

界面分成两个区域：

训练场：

- `步数`：切换训练步数预设，包含 `2,000 / 50,000 / 200,000 / 500,000 / 1,000,000`。
- `随机图`：切换训练时混入的随机地图数量，最高预设到 `500`。
- `探索`：切换 PPO 的 entropy 系数，范围从 `0.005` 到 `0.10`。数值越大，训练时越愿意尝试不同动作。
- `并行`：切换 PPO 向量化采样环境数量，包含 `1 / 2 / 4 / 6`。MacBook Air M4 16GB 推荐先用 `4`，需要留出系统余量时不要开到更高。
- `尺寸`：切换随机地图尺寸，例如 `9x11 / 11x15 / 13x17 / 15x21 / 17x25 / 21x31`。
- `风格`：切换随机地图布局，包含混合、开放、分割、房间、死胡同、迷宫。
- `门向`：切换门所在墙的方向，包含混合、横墙、竖墙。
- `位置`：切换起点和出口分布，包含混合、角落、边缘、内部。
- `墙 / 陷阱`：切换随机地图中的墙体密度和陷阱密度。
- `开始训练`：后台启动 PPO 训练，并保存 `models/ppo_maze.zip`。
- `重置智能体`：把当前 `models/ppo_maze.zip` 移到 `models/reset_backups/`，让智能体回到未训练状态。
- `停止训练`：终止当前后台训练任务。

试验场：

- `上一张固定图 / 下一张固定图`：切换 `maps/` 下的固定地图。
- `上一张随机图 / 下一张随机图`：在本次打开界面后生成过的随机地图历史中切换；如果还没有随机地图，`下一张随机图` 会新生成一张。
- `试验固定图`：用训练好的模型在当前固定地图上跑一次实验。
- `试验随机图`：用训练好的模型在当前随机地图上跑一次实验；如果还没有随机地图，会先生成一张。
- `当前随机图加入固定图`：把当前随机地图保存到 `maps/saved_random_*.txt`，之后它会出现在固定地图列表里。
- `尺寸 / 风格 / 门向 / 位置 / 墙 / 陷阱`：调整之后新生成的随机图参数；已经生成的随机图历史不会被改写。
- `动作 采样 / 动作 贪心`：切换试验时的动作选择。采样更容易跳出局部循环，贪心更稳定但更容易卡住。
- `记忆辅助`：记录本局看见过的钥匙、门、出口和访问次数；检测到两格打转时，会优先朝记住的目标或低访问方向走。
- `评估`：切换批量评估时的随机测试次数，最高预设到 `200`。
- `批量评估`：后台运行固定地图和随机地图评估。
- `停止实验`：停止当前回放或后台评估任务。

批量评估默认每次使用新的随机地图；如果需要复现同一批地图，可以命令行显式传 `--seed`。界面处于 `动作 采样` 时，批量评估也会采样动作；处于 `动作 贪心` 时则使用确定性动作。

试验场回放时，暗色区域代表智能体当前看不到的位置。PPO 模型实际收到的 observation 也只包含局部窄视野带。

视野规则：智能体不会看到完整地图；它在上、下、左、右四个方向分别看到一个 `3x3` 的前方窄带。这样比单条直线更接近人的侧向余光，也能减少“门刚离开直线视野就完全丢失”的问题。墙体仍会遮挡同一条窄带通道后面的格子。

注意：当前默认模型已经是 Recurrent PPO，模型本身会维护 LSTM 隐状态。界面里的“记忆辅助”仍然只是试验场回放层的短期辅助逻辑，用来对比或兜底；如果要观察模型本身能力，可以在试验场关闭“记忆辅助”。

如果之前训练过旧版“四向单线视野”的模型，需要在训练场点击 `重置智能体` 后重新训练。新版 observation 维度已经变成四向 `3x3` 窄视野带，旧模型不能直接用于新版试验场。

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
python train_ppo.py --timesteps 1000000 --n-envs 4 --random-maps 200 --ent-coef 0.03 --view-range 3 --view-width 3 --random-rows 11 --random-cols 15 --wall-density 0.12 --trap-density 0.04
```

更强调泛化训练时，建议显式使用混合随机图：

```bash
python train_ppo.py --timesteps 3000000 --n-envs 4 --random-maps 300 --random-style mixed --door-orientation mixed --endpoint-mode mixed --random-rows 15 --random-cols 21 --wall-density 0.18 --trap-density 0.08
```

训练默认使用课程学习：

- `exit-only`：先训练无钥匙、无门、无陷阱的小难度找出口任务。
- `key-door`：再训练钥匙门任务，暂时不加陷阱。
- `full-mix`：最后混合普通出口图、钥匙门、陷阱、死胡同和不同门方向。

可以用 `--curriculum none` 关闭课程学习，用 `--algo ppo` 退回无记忆普通 PPO。

并行训练说明：

- `--n-envs 1`：最稳，速度较慢。
- `--n-envs 2`：保守加速。
- `--n-envs 4`：MacBook Air M4 16GB 推荐默认值，通常不会把机器完全跑满。
- `--n-envs 6`：更快但更占 CPU，适合插电、暂时不做其他重任务时使用。

这个项目的迷宫环境很轻，默认使用 `DummyVecEnv`。它不是多进程，但会把多个环境的 observation 批量送入 PPO，实际测试比 `SubprocVecEnv` 更快；`SubprocVecEnv` 的进程通信开销在这里反而容易拖慢训练。只有以后环境计算变重时才建议手动试 `--vec-env subproc`。

训练脚本会默认限制 PyTorch/OpenMP 线程数，避免每个环境再额外开很多线程造成过载。右下角界面日志只显示训练进度、回合数和当前步速，不显示 PPO 内部 loss 表格。

步数上限会随地图尺寸自动放宽。基础值仍是 `260`，但大地图会按格子数量提高 episode 上限，避免 `21x31` 这类地图还没探索完就被截断。

输出：

- `models/ppo_maze.zip`：训练后的 PPO 模型。
- `outputs/training_monitor.csv`：每个 episode 的 reward、步数、成功状态等。
- `outputs/ppo_training_curve.png`：最近 100 局滚动成功率、平均步数、平均 reward。

## 回放

```bash
cd /Users/ericpan/game_project/deep_learning/ai-maze-ppo-lab
python play_ppo.py --model models/ppo_maze.zip --map maps/level_1.txt
```

回放窗口会显示完整地图，并用暗色遮罩标出当前智能体看不到的区域。PPO 模型实际收到的 observation 只包含局部 `3x3` 窄视野带，不包含完整地图。

输出：

- `outputs/replay_trace.txt`：每一步位置、动作、奖励、钥匙状态、门状态。

## 评估

```bash
cd /Users/ericpan/game_project/deep_learning/ai-maze-ppo-lab
python evaluate_ppo.py --model models/ppo_maze.zip --random-tests 50 --random-rows 11 --random-cols 15 --wall-density 0.12 --trap-density 0.04
```

评估更复杂随机图：

```bash
python evaluate_ppo.py --model models/ppo_maze.zip --random-tests 50 --random-style mixed --door-orientation mixed --endpoint-mode mixed --random-rows 15 --random-cols 21 --wall-density 0.18 --trap-density 0.08 --sample-actions
```

输出：

- `outputs/eval_summary.json`：固定地图和随机地图上的成功率、平均步数、平均 reward、拿钥匙率和过门率。

## 随机地图参数

随机地图生成器会先保证 `S -> K -> D -> E` 至少有一条合法路径，再按参数补充墙和陷阱。

- `--random-style`：`mixed / open / split / rooms / deadends / maze`。`mixed` 会每张图随机挑一种风格。
- `--door-orientation`：`mixed / horizontal / vertical`。横墙表示门在一条水平隔断上，竖墙表示门在一条垂直隔断上。
- `--endpoint-mode`：`mixed / corners / edges / interior`。控制起点和出口更偏向角落、边缘或内部位置。
- `--random-rows / --random-cols`：随机地图行列数。界面预设最大到 `21x31`，命令行也可以继续加大，但越大训练越慢。
- `--wall-density / --trap-density`：额外随机墙和陷阱密度。密度越高，地图越难；生成器会在失败时自动降一点密度，尽量保持可解。
- `--simple-map-prob`：训练随机池里无钥匙/无门“直接找出口”地图的比例，默认 `0.25`。这样模型不会只学会钥匙门任务，也能保留普通迷宫能力。

这些参数同时影响训练场、试验场和批量评估。训练时建议用 `mixed`，试验时可以固定某一种风格观察模型短板。

## 统一入口

也可以通过 `main.py` 调用：

```bash
python main.py train --timesteps 500000 --random-maps 200
python main.py play --model models/ppo_maze.zip --map maps/level_2_key_door.txt
python main.py evaluate --model models/ppo_maze.zip --random-tests 50
```

图形界面也可以用命令启动：

```bash
python ppo_app.py
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

- Recurrent PPO 比普通 MLP PPO 更适合局部视野，但依然需要足够训练步数；复杂随机泛化通常要百万级到数百万步。
- 局部视野导致任务变成部分可观测问题，LSTM 只能缓解，不能保证自动学会全局最短路径算法。
- 随机地图生成器只保证有一条合法路径，不保证每张图都难度均衡；复杂风格和大尺寸通常需要更多训练步数。

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
- 随机地图支持横向门、纵向门和多种地图风格。
