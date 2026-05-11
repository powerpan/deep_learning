# Deep Learning Playground

这个仓库目前包含两个本地强化学习迷宫实验项目：

- `ai-maze-lab/`：轻量 Q-learning + NumPy + Pygame 版本，包含训练日志、曲线、热力图、钥匙和门规则，以及 macOS 双击启动包装。
- `ai-maze-ppo-lab/`：局部直线视野 + Gymnasium + Stable-Baselines3 PPO 版本，用来训练更通用的迷宫策略网络。

## 快速入口

Q-learning 版本：

```bash
cd ai-maze-lab
python main.py --map maps/level_1.txt
```

PPO 版本：

```bash
cd ai-maze-ppo-lab
python train_ppo.py --timesteps 500000 --random-maps 200
```

具体安装和运行方式见各子项目内的 `README.md`。
