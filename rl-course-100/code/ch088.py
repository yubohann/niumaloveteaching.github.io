"""
第088章 多智能体环境：捕食者-猎物

实验内容（12x12 网格上的 3 捕食者 vs 2 猎物，纯 numpy 内联实现）：
  - 捕食者：3 个，学习体，用"共享网络的独立 Q 学习"（IQL + 参数共享 + 经验回放
    + 目标网络 + ε-贪婪探索）训练
  - 猎物：2 个，脚本策略——朝"离最近捕食者最远"的邻格逃跑，受墙壁限制
  - 捕获判定：任一捕食者与猎物切比雪夫距离 ≤1 即捕获该猎物
  - 奖励：每步 -0.02 - 0.05×（到最近猎物距离），任意猎物被捕获时所有捕食者 +1
    （共享捕获奖励 = 合作信号）
  - 指标：全歼率、平均捕获数、随机基线对比、ε 与损失

运行：
    python code/ch088.py           # 完整训练（CPU 约 2-5 分钟）
    python code/ch088.py --quick   # 快速跑通（约 20-40 秒）

预期：全歼率从随机基线的接近 0 上升到 0.3～0.7（以实跑为准）；捕食者会
学会"分头包抄 + 把猎物逼向墙角"。
"""

import argparse
import collections
import copy
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# 环境：捕食者-猎物网格世界
# ---------------------------------------------------------------------------
class PredatorPreyEnv:
    """3 个捕食者在地图上追捕 2 个脚本猎物。

    观测（10 维/捕食者，归一化到 [0,1]）：自己的坐标(2)、最近猎物的相对
    坐标(2)、另一猎物的相对坐标(2)、另外两个捕食者的相对坐标(4)。
    被捕获的猎物其相对坐标置为角落(1,1)以区分。
    动作：5 个离散动作（原地/上下左右）。猎物 8 邻域移动。
    """

    obs_dim = 10
    n_actions = 5
    n_predators = 3
    n_preys = 2
    move_delta = [(-1, 0), (1, 0), (0, -1), (0, 1)]   # 上下左右（捕食者）
    prey_delta = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)]

    def __init__(self, size: int = 12, capture_dist: int = 1, max_steps: int = 60):
        self.size = size
        self.capture_dist = capture_dist
        self.max_steps = max_steps
        self.rng = np.random.default_rng(0)

    def reset(self, rng: np.random.Generator):
        self.rng = rng
        # 捕食者集中在下半场，猎物在上半场，避免一开始就贴脸
        self.pred_pos = self._sample_cells(self.n_predators, rng,
                                           row_range=(6, self.size - 1))
        self.prey_pos = self._sample_cells(self.n_preys, rng, row_range=(0, 5))
        self.prey_alive = [True] * self.n_preys
        self.steps = 0
        self.captures = 0
        return self._obs()

    def _sample_cells(self, n, rng, row_range):
        cells = []
        used = set()
        while len(cells) < n:
            r = int(rng.integers(row_range[0], row_range[1] + 1))
            c = int(rng.integers(0, self.size))
            if (r, c) in used:
                continue
            used.add((r, c))
            cells.append([r, c])
        return np.asarray(cells, dtype=np.int64)

    def _obs(self) -> np.ndarray:
        scale = float(self.size - 1)
        obs = []
        for i in range(self.n_predators):
            others = [self.pred_pos[j] - self.pred_pos[i]
                      for j in range(self.n_predators) if j != i]
            prey_rel = []
            for j in range(self.n_preys):
                if self.prey_alive[j]:
                    prey_rel.append(self.prey_pos[j] - self.pred_pos[i])
                else:
                    prey_rel.append(np.array([self.size, self.size]))
            o = np.concatenate([
                self.pred_pos[i] / scale,
                np.asarray(prey_rel[0]) / scale,
                np.asarray(prey_rel[1]) / scale,
                np.asarray(others).reshape(-1) / scale,
            ])
            obs.append(o.astype(np.float32))
        return np.stack(obs)

    def _chebyshev(self, a, b) -> int:
        return int(max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1]))))

    def _move_preys(self):
        """脚本猎物：在 8 邻域里选"离最近捕食者最远"的格子，带少量随机。"""
        for j in range(self.n_preys):
            if not self.prey_alive[j]:
                continue
            best_score, best_cell = -1e9, tuple(self.prey_pos[j])
            for dr, dc in self.prey_delta:
                r = int(np.clip(self.prey_pos[j][0] + dr, 0, self.size - 1))
                c = int(np.clip(self.prey_pos[j][1] + dc, 0, self.size - 1))
                d_min = min(self._chebyshev([r, c], p) for p in self.pred_pos)
                score = d_min + self.rng.uniform(0.0, 0.4)   # 打破对称的噪声
                if score > best_score:
                    best_score, best_cell = score, (r, c)
            self.prey_pos[j] = np.asarray(best_cell, dtype=np.int64)

    def step(self, actions: np.ndarray):
        """actions: (3,) 离散动作。返回 obs, rewards, terminated, truncated, info。"""
        # 1) 捕食者移动
        for i in range(self.n_predators):
            dr, dc = self.move_delta[int(actions[i])] if int(actions[i]) > 0 else (0, 0)
            self.pred_pos[i, 0] = int(np.clip(self.pred_pos[i, 0] + dr, 0, self.size - 1))
            self.pred_pos[i, 1] = int(np.clip(self.pred_pos[i, 1] + dc, 0, self.size - 1))
        # 2) 猎物逃跑
        self._move_preys()
        # 3) 捕获判定（切比雪夫距离 ≤ capture_dist）
        captured_now = 0
        for j in range(self.n_preys):
            if not self.prey_alive[j]:
                continue
            for p in self.pred_pos:
                if self._chebyshev(p, self.prey_pos[j]) <= self.capture_dist:
                    self.prey_alive[j] = False
                    captured_now += 1
                    break
        self.captures += captured_now
        self.steps += 1

        # 4) 奖励：聚集塑形 + 共享捕获奖励
        rewards = np.zeros(self.n_predators, dtype=np.float32)
        alive = [self.prey_pos[j] for j in range(self.n_preys) if self.prey_alive[j]]
        if alive:
            for i in range(self.n_predators):
                d_min = min(self._chebyshev(self.pred_pos[i], q) for q in alive)
                rewards[i] = -0.02 - 0.05 * d_min
        rewards += 1.0 * captured_now                # 全队共享的捕获奖励

        terminated = not any(self.prey_alive)
        truncated = (not terminated) and self.steps >= self.max_steps
        info = {"captures": captured_now,
                "n_captured_total": self.captures,
                "all_dead": terminated}
        return self._obs(), rewards, terminated, truncated, info


# ---------------------------------------------------------------------------
# Q 网络（参数共享：3 个捕食者共用一个网络）
# ---------------------------------------------------------------------------
class QNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SharedAgent:
    """共享网络的独立 Q 学习：每个捕食者独立探索，数据进同一个回放池。"""

    def __init__(self, env: PredatorPreyEnv, args, device):
        self.args = args
        self.device = device
        self.q = QNet(env.obs_dim, env.n_actions, args.hidden).to(device)
        self.q_target = copy.deepcopy(self.q)
        for p in self.q_target.parameters():
            p.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.q.parameters(), lr=args.lr)
        self.buffer = collections.deque(maxlen=args.buffer_capacity)
        self.updates = 0

    @torch.no_grad()
    def act(self, obs_np: np.ndarray, epsilon: float, rng: np.random.Generator):
        """逐智能体的 ε-贪婪：每个捕食者独立决定是否随机探索。"""
        n = obs_np.shape[0]
        actions = rng.integers(0, self.args.n_actions, size=n).astype(np.int64)
        explore = rng.uniform(size=n) < epsilon
        if not explore.all():
            obs = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device)
            greedy = np.argmax(self.q(obs).cpu().numpy(), axis=1).astype(np.int64)
            actions[~explore] = greedy[~explore]
        return actions

    def push(self, obs, act, rew, next_obs, done):
        for i in range(obs.shape[0]):
            self.buffer.append((obs[i].copy(), int(act[i]), float(rew[i]),
                                next_obs[i].copy(), float(done)))

    def update(self):
        args = self.args
        batch = random.sample(self.buffer, args.batch_size)
        obs = torch.as_tensor(np.stack([b[0] for b in batch]),
                              dtype=torch.float32, device=self.device)
        act = torch.as_tensor([b[1] for b in batch],
                              dtype=torch.int64, device=self.device).unsqueeze(-1)
        rew = torch.as_tensor([b[2] for b in batch],
                              dtype=torch.float32, device=self.device).unsqueeze(-1)
        next_obs = torch.as_tensor(np.stack([b[3] for b in batch]),
                                   dtype=torch.float32, device=self.device)
        done = torch.as_tensor([b[4] for b in batch],
                               dtype=torch.float32, device=self.device).unsqueeze(-1)

        with torch.no_grad():
            q_next = self.q_target(next_obs).max(dim=1, keepdim=True).values
            y = rew + args.gamma * (1.0 - done) * q_next     # 双 Q 的朴素版：单 Q 目标
        q = self.q(obs).gather(1, act)
        loss = F.smooth_l1_loss(q, y)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 5.0)
        self.optimizer.step()

        self.updates += 1
        if self.updates % args.target_update == 0:
            self.q_target.load_state_dict(self.q.state_dict())
        return float(loss.item())


# ---------------------------------------------------------------------------
# 评估（学习策略与随机基线共用）
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(agent, episodes, device, seed, random_policy=False):
    env = PredatorPreyEnv()
    rng = np.random.default_rng(seed)
    all_dead, n_captures, first_capture_steps = [], [], []
    for _ in range(episodes):
        obs = env.reset(rng)
        done = False
        first_step = None
        step = 0
        while not done:
            if random_policy:
                actions = rng.integers(0, env.n_actions, size=env.n_predators)
            else:
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
                actions = np.argmax(agent.q(obs_t).cpu().numpy(), axis=1)
            obs, rew, terminated, truncated, info = env.step(actions)
            step += 1
            if info["n_captured_total"] > 0 and first_step is None:
                first_step = step
            done = terminated or truncated
        all_dead.append(1.0 if terminated else 0.0)
        n_captures.append(info["n_captured_total"])
        first_capture_steps.append(first_step if first_step is not None
                                   else env.max_steps)
    return (float(np.mean(all_dead)), float(np.mean(n_captures)),
            float(np.mean(first_capture_steps)))


def parse_args():
    p = argparse.ArgumentParser(description="第088章：捕食者-猎物（独立 Q 学习）")
    p.add_argument("--max-steps", type=int, default=60000, help="环境步数上限")
    p.add_argument("--warmup", type=int, default=2000, help="随机动作预热步数")
    p.add_argument("--update-every", type=int, default=2, help="每多少步做一次梯度更新")
    p.add_argument("--batch-size", type=int, default=64, help="采样批量")
    p.add_argument("--buffer-capacity", type=int, default=80000, help="回放池容量")
    p.add_argument("--lr", type=float, default=1e-3, help="学习率")
    p.add_argument("--gamma", type=float, default=0.95, help="折扣因子")
    p.add_argument("--hidden", type=int, default=128, help="隐藏层宽度")
    p.add_argument("--eps-start", type=float, default=1.0, help="初始探索率")
    p.add_argument("--eps-end", type=float, default=0.05, help="最终探索率")
    p.add_argument("--eps-decay-steps", type=int, default=25000, help="探索率衰减步数")
    p.add_argument("--target-update", type=int, default=200, help="目标网络更新间隔（更新次数）")
    p.add_argument("--eval-every", type=int, default=10000, help="评估间隔（步）")
    p.add_argument("--eval-episodes", type=int, default=20, help="评估回合数")
    p.add_argument("--seed", type=int, default=0, help="随机种子")
    p.add_argument("--save-dir", type=str, default="runs/ch088", help="模型保存目录")
    p.add_argument("--quick", action="store_true", help="快速模式：缩小训练规模")
    return p.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.max_steps = min(args.max_steps, 16000)
        args.eps_decay_steps = min(args.eps_decay_steps, 6000)
        args.eval_every = 4000
        args.eval_episodes = 10
    set_seed(args.seed)

    device = torch.device("cpu")
    env = PredatorPreyEnv()
    args.n_actions = env.n_actions
    rng = np.random.default_rng(args.seed)
    agent = SharedAgent(env, args, device)

    print(f"设备 {device} | 网格 {env.size}x{env.size} | 捕食者 {env.n_predators} | "
          f"猎物 {env.n_preys} | 种子 {args.seed}")

    # 随机基线
    base_all_dead, base_caps, base_first = evaluate(
        agent, args.eval_episodes, device, args.seed + 111, random_policy=True)
    print(f"随机基线：全歼率 {base_all_dead:.2f} | 平均捕获 {base_caps:.2f} | "
          f"首次捕获步数 {base_first:.1f}")

    obs = env.reset(rng)
    ep_stats = []
    ep_ret = np.zeros(env.n_predators, dtype=np.float32)
    ep_len, losses = 0, []
    t_start = time.time()

    for step in range(1, args.max_steps + 1):
        # 1) ε-贪婪选择动作
        frac = min(1.0, step / max(args.eps_decay_steps, 1))
        epsilon = args.eps_start + frac * (args.eps_end - args.eps_start)
        if step <= args.warmup:
            actions = rng.integers(0, env.n_actions, size=env.n_predators)
        else:
            actions = agent.act(obs, epsilon, rng)

        # 2) 环境步进 + 存转移
        next_obs, rew, terminated, truncated, info = env.step(actions)
        done = terminated or truncated
        if step > args.warmup:
            agent.push(obs, actions, rew, next_obs, float(terminated))
        ep_ret += rew
        ep_len += 1

        # 3) 训练
        if step > args.warmup and step % args.update_every == 0 \
                and len(agent.buffer) >= args.batch_size:
            losses.append(agent.update())

        # 4) 回合结束
        if done:
            ep_stats.append({"return": float(ep_ret.mean()), "captures": info["n_captured_total"],
                             "all_dead": float(info["all_dead"]), "len": ep_len})
            obs = env.reset(rng)
            ep_ret = np.zeros(env.n_predators, dtype=np.float32)
            ep_len = 0
        else:
            obs = next_obs

        # 5) 日志与评估
        if step % args.eval_every == 0:
            recent = ep_stats[-30:]
            all_dead, caps, first_step = evaluate(
                agent, args.eval_episodes, device, args.seed + 777)
            m_loss = float(np.mean(losses[-500:])) if losses else float("nan")
            print(f"[步 {step:6d}] ε {epsilon:.2f} | 评估全歼率 {all_dead:.2f} | "
                  f"平均捕获 {caps:.2f} | 首次捕获步数 {first_step:5.1f} | "
                  f"损失 {m_loss:.4f} | 近30局回报 "
                  f"{np.mean([s['return'] for s in recent]) if recent else float('nan'):.2f}")

    elapsed = time.time() - t_start

    # 最终评估（贪婪）
    all_dead, caps, first_step = evaluate(agent, args.eval_episodes,
                                          device, args.seed + 999)
    print("-" * 74)
    print(f"训练结束 | 用时 {elapsed:.1f}s | 环境步 {args.max_steps} | "
          f"梯度更新 {agent.updates} 次 | 回放池 {len(agent.buffer)}")
    print(f"评估（{args.eval_episodes} 回合，贪婪策略）：全歼率 {all_dead:.2f} | "
          f"平均捕获 {caps:.2f} | 首次捕获步数 {first_step:.1f}")
    print(f"对照随机基线：全歼率 {base_all_dead:.2f} → {all_dead:.2f}")

    os.makedirs(args.save_dir, exist_ok=True)
    path = os.path.join(args.save_dir, "iql_predator_prey.pt")
    torch.save({"q": agent.q.state_dict(), "args": vars(args)}, path)
    print(f"模型已保存到 {path}")


if __name__ == "__main__":
    main()
