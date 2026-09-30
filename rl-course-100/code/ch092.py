"""
第092章 多智能体环境：仓储机器人

实验内容（10x10 网格仓库里的 3 台拣货机器人，纯 numpy 内联环境）：
  - 场景：场上始终保持 3 个待取货物；机器人走到货物格即取货（进入携带
    状态，货物在别处补生），把货物送到任一发货站即完成一单
  - 机器人之间不能同格、不能对穿：冲突时双方原地等待（返工惩罚）
  - 算法：共享网络的双重 DQN（Double DQN + 参数共享 + 经验回放），
    每台机器人独立 ε-贪婪
  - 奖励：完成一单 +1（仅送达者）；每步 -0.01 - 0.02×（到任务目标距离）；
    被堵 -0.02
  - 指标：每 1000 步完成订单数、平均单耗步数、冲突频率；与脚本贪心、
    随机策略对照

运行：
    python code/ch092.py           # 完整训练（CPU 约 2-4 分钟）
    python code/ch092.py --quick   # 快速跑通（约 20-40 秒）

预期：学习的吞吐量从随机策略的接近 0 提升到接近脚本贪心水平（以实跑为准），
并在冲突频率上优于贪心——因为它能学会绕行而不是硬挤。
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
# 环境：网格仓库
# ---------------------------------------------------------------------------
class WarehouseEnv:
    """3 台机器人、2 个发货站、3 个常驻货物的网格仓库。

    观测（11 维/机器人）：自己的坐标(2)、是否携带货物(1)、最近货物相对
    坐标(2)、最近发货站相对坐标(2)、另外两台机器人的相对坐标(4)。
    动作：5 个离散动作（原地/上下左右）。冲突：同格或对穿时双方原地等待。
    """

    obs_dim = 11
    n_actions = 5
    n_agents = 3
    move_delta = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def __init__(self, size: int = 10, n_items: int = 3, max_steps: int = 200):
        self.size = size
        self.n_items = n_items
        self.max_steps = max_steps
        self.stations = np.array([[0, 0], [0, size - 1]], dtype=np.int64)
        self.rng = np.random.default_rng(0)

    def reset(self, rng: np.random.Generator):
        self.rng = rng
        base = np.array([[self.size - 1, self.size // 2 - 1],
                         [self.size - 1, self.size // 2],
                         [self.size - 1, self.size // 2 + 1]], dtype=np.int64)
        self.pos = np.clip(base + rng.integers(-1, 2, size=(3, 2)),
                           0, self.size - 1).astype(np.int64)
        self.carrying = np.zeros(self.n_agents, dtype=np.int64)
        self.items = []
        for _ in range(self.n_items):
            self.items.append(self._spawn_item())
        self.items = np.stack(self.items).astype(np.int64)
        self.steps = 0
        self.deliveries = 0
        self.conflicts = 0
        return self._obs()

    def _spawn_item(self):
        """随机找一个没有被占用的格子生成货物。"""
        for _ in range(200):
            cell = self.rng.integers(0, self.size, size=2)
            if any((cell == self.stations).all(axis=1)):
                continue
            if any((cell == p).all() for p in self.pos):
                continue
            if any((cell == it).all() for it in self.items):
                continue
            return cell.astype(np.int64)
        return self.rng.integers(0, self.size, size=2).astype(np.int64)

    def _nearest(self, target_set, origin):
        d = np.abs(target_set - origin).sum(axis=1)
        k = int(np.argmin(d))
        return target_set[k], float(d[k])

    def _obs(self) -> np.ndarray:
        scale = float(self.size - 1)
        obs = []
        for i in range(self.n_agents):
            item, _ = self._nearest(self.items, self.pos[i]) if len(self.items) else \
                (np.zeros(2, dtype=np.int64), 0.0)
            station, _ = self._nearest(self.stations, self.pos[i])
            others = [self.pos[j] - self.pos[i]
                      for j in range(self.n_agents) if j != i]
            o = np.concatenate([
                self.pos[i] / scale,
                [float(self.carrying[i])],
                (item - self.pos[i]) / scale,
                (station - self.pos[i]) / scale,
                np.asarray(others).reshape(-1) / scale,
            ])
            obs.append(o.astype(np.float32))
        return np.stack(obs)

    def step(self, actions: np.ndarray):
        """actions: (3,) 离散动作。返回 obs, rewards, terminated, truncated, info。"""
        # 1) 计算期望位置
        desired = self.pos.copy()
        for i in range(self.n_agents):
            act = int(actions[i])
            if act > 0:
                dr, dc = self.move_delta[act - 1]
                desired[i, 0] = int(np.clip(self.pos[i, 0] + dr, 0, self.size - 1))
                desired[i, 1] = int(np.clip(self.pos[i, 1] + dc, 0, self.size - 1))

        # 2) 冲突消解：同格冲突、对穿冲突，以及撞进"原地等待者"的格子都算被堵
        blocked = np.zeros(self.n_agents, dtype=bool)
        for i in range(self.n_agents):
            for j in range(i + 1, self.n_agents):
                if (desired[i] == desired[j]).all():
                    blocked[i] = blocked[j] = True
                if (desired[i] == self.pos[j]).all() and (desired[j] == self.pos[i]).all():
                    blocked[i] = blocked[j] = True
        changed = True
        while changed:                       # 被堵者会级联影响其他人的目标格
            changed = False
            for i in range(self.n_agents):
                if blocked[i]:
                    continue
                for j in range(self.n_agents):
                    if i == j:
                        continue
                    target_j = self.pos[j] if blocked[j] else desired[j]
                    if (desired[i] == target_j).all():
                        blocked[i] = True
                        changed = True
                        break
        self.conflicts += int(blocked.sum())
        new_pos = self.pos.copy()
        for i in range(self.n_agents):
            if not blocked[i]:
                new_pos[i] = desired[i]
        self.pos = new_pos
        self.steps += 1

        # 3) 取货与送货
        rewards = np.full(self.n_agents, -0.01, dtype=np.float32)
        delivered_now = 0
        for i in range(self.n_agents):
            if blocked[i]:
                rewards[i] -= 0.02                      # 被堵的返工惩罚
            if not self.carrying[i] and len(self.items):
                k = int(np.argmin(np.abs(self.items - self.pos[i]).sum(axis=1)))
                if (self.items[k] == self.pos[i]).all():
                    self.carrying[i] = 1                # 取货
                    self.items[k] = self._spawn_item()  # 货物在别处补生
            if self.carrying[i]:
                if any((self.pos[i] == s).all() for s in self.stations):
                    self.carrying[i] = 0
                    rewards[i] += 1.0                   # 完成一单
                    self.deliveries += 1
                    delivered_now += 1

        # 4) 塑形：朝"任务目标"（携带时去站点，否则去货物）靠近
        for i in range(self.n_agents):
            if self.carrying[i]:
                target, d = self._nearest(self.stations, self.pos[i])
            else:
                target, d = self._nearest(self.items, self.pos[i])
            rewards[i] -= 0.02 * d

        truncated = self.steps >= self.max_steps
        info = {"deliveries": self.deliveries, "step_deliveries": delivered_now,
                "conflicts": self.conflicts}
        return self._obs(), rewards, False, truncated, info


# ---------------------------------------------------------------------------
# 网络与双重 DQN
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


class DoubleDQNAgent:
    """共享网络的双重 DQN：动作选择用在线网络，动作评估用目标网络。"""

    def __init__(self, env: WarehouseEnv, args, device):
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
            # 双重 DQN：在线网络选动作，目标网络评估价值
            next_actions = self.q(next_obs).argmax(dim=1, keepdim=True)
            q_next = self.q_target(next_obs).gather(1, next_actions)
            y = rew + args.gamma * (1.0 - done) * q_next
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
# 评估：学习策略 / 脚本贪心 / 随机
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(agent, episodes, device, seed, mode="learned"):
    env = WarehouseEnv()
    rng = np.random.default_rng(seed)
    per_1000, per_order, conflict_rate = [], [], []
    for _ in range(episodes):
        obs = env.reset(rng)
        done, n = False, 0
        info = {"deliveries": 0, "conflicts": 0}
        while not done:
            if mode == "learned":
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
                a = np.argmax(agent.q(obs_t).cpu().numpy(), axis=1)
            elif mode == "random":
                a = rng.integers(0, env.n_actions, size=env.n_agents)
            else:   # 脚本贪心：朝最近任务目标走，挑差距大的轴
                a = np.zeros(env.n_agents, dtype=np.int64)
                for i in range(env.n_agents):
                    if env.carrying[i]:
                        target, _ = env._nearest(env.stations, env.pos[i])
                    else:
                        target, _ = env._nearest(env.items, env.pos[i])
                    d = target - env.pos[i]
                    if abs(d[0]) >= abs(d[1]) and d[0] != 0:
                        a[i] = 1 if d[0] < 0 else 2
                    elif d[1] != 0:
                        a[i] = 3 if d[1] < 0 else 4
                    else:
                        a[i] = 0
            obs, rew, terminated, truncated, info = env.step(a)
            n += 1
            done = terminated or truncated
        per_1000.append(info["deliveries"] / max(n, 1) * 1000)
        per_order.append(n / max(info["deliveries"], 1))
        conflict_rate.append(info["conflicts"] / max(n, 1))
    return (float(np.mean(per_1000)), float(np.mean(per_order)),
            float(np.mean(conflict_rate)))


def parse_args():
    p = argparse.ArgumentParser(description="第092章：仓储机器人（双重 DQN）")
    p.add_argument("--max-steps", type=int, default=60000, help="环境步数上限")
    p.add_argument("--warmup", type=int, default=2000, help="随机动作预热步数")
    p.add_argument("--update-every", type=int, default=2, help="每多少步一次更新")
    p.add_argument("--batch-size", type=int, default=64, help="采样批量")
    p.add_argument("--buffer-capacity", type=int, default=80000, help="回放池容量")
    p.add_argument("--lr", type=float, default=1e-3, help="学习率")
    p.add_argument("--gamma", type=float, default=0.97, help="折扣因子")
    p.add_argument("--hidden", type=int, default=128, help="隐藏层宽度")
    p.add_argument("--eps-start", type=float, default=1.0, help="初始探索率")
    p.add_argument("--eps-end", type=float, default=0.05, help="最终探索率")
    p.add_argument("--eps-decay-steps", type=int, default=25000, help="探索率衰减步数")
    p.add_argument("--target-update", type=int, default=200, help="目标网络更新间隔")
    p.add_argument("--eval-every", type=int, default=10000, help="评估间隔（步）")
    p.add_argument("--eval-episodes", type=int, default=5, help="评估回合数")
    p.add_argument("--seed", type=int, default=0, help="随机种子")
    p.add_argument("--save-dir", type=str, default="runs/ch092", help="模型保存目录")
    p.add_argument("--quick", action="store_true", help="快速模式：缩小训练规模")
    return p.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.max_steps = min(args.max_steps, 16000)
        args.eps_decay_steps = min(args.eps_decay_steps, 6000)
        args.eval_every = 4000
        args.eval_episodes = 3
    set_seed(args.seed)

    device = torch.device("cpu")
    env = WarehouseEnv()
    args.n_actions = env.n_actions
    rng = np.random.default_rng(args.seed)
    agent = DoubleDQNAgent(env, args, device)

    print(f"设备 {device} | 网格 {env.size}x{env.size} | 机器人 {env.n_agents} | "
          f"常驻货物 {env.n_items} | 种子 {args.seed}")

    for mode, name in (("random", "随机策略"), ("greedy", "脚本贪心")):
        per_1000, per_order, conf = evaluate(agent, args.eval_episodes, device,
                                             args.seed + 111, mode=mode)
        print(f"{name}基线：每1000步订单 {per_1000:5.1f} | 平均单耗 {per_order:5.1f} 步 | "
              f"冲突/步 {conf:.3f}")

    obs = env.reset(rng)
    ep_stats = []
    ep_ret, ep_len = 0.0, 0
    losses = []
    t_start = time.time()

    for step in range(1, args.max_steps + 1):
        frac = min(1.0, step / max(args.eps_decay_steps, 1))
        epsilon = args.eps_start + frac * (args.eps_end - args.eps_start)
        if step <= args.warmup:
            actions = rng.integers(0, env.n_actions, size=env.n_agents)
        else:
            actions = agent.act(obs, epsilon, rng)

        next_obs, rew, terminated, truncated, info = env.step(actions)
        done = terminated or truncated
        if step > args.warmup:
            agent.push(obs, actions, rew, next_obs, float(terminated))
        ep_ret += float(rew.sum())
        ep_len += 1

        if step > args.warmup and step % args.update_every == 0 \
                and len(agent.buffer) >= args.batch_size:
            losses.append(agent.update())

        if done:
            ep_stats.append({"deliveries": info["deliveries"],
                             "conflicts": info["conflicts"], "len": ep_len})
            obs = env.reset(rng)
            ep_ret, ep_len = 0.0, 0
        else:
            obs = next_obs

        if step % args.eval_every == 0:
            per_1000, per_order, conf = evaluate(agent, args.eval_episodes,
                                                 device, args.seed + 777)
            recent = ep_stats[-20:]
            train_dph = np.mean([s["deliveries"] / max(s["len"], 1) * 1000
                                 for s in recent]) if recent else float("nan")
            m_loss = float(np.mean(losses[-500:])) if losses else float("nan")
            print(f"[步 {step:6d}] ε {epsilon:.2f} | 评估每1000步订单 {per_1000:5.1f} | "
                  f"单耗 {per_order:5.1f} 步 | 冲突/步 {conf:.3f} | "
                  f"训练近20局 {train_dph:5.1f} | 损失 {m_loss:.4f}")

    elapsed = time.time() - t_start
    per_1000, per_order, conf = evaluate(agent, args.eval_episodes, device,
                                         args.seed + 999)
    print("-" * 74)
    print(f"训练结束 | 用时 {elapsed:.1f}s | 环境步 {args.max_steps} | "
          f"梯度更新 {agent.updates} 次")
    print(f"评估（{args.eval_episodes} 回合，贪婪策略）：每1000步订单 {per_1000:.1f} | "
          f"平均单耗 {per_order:.1f} 步 | 冲突/步 {conf:.3f}")

    os.makedirs(args.save_dir, exist_ok=True)
    path = os.path.join(args.save_dir, "double_dqn_warehouse.pt")
    torch.save({"q": agent.q.state_dict(), "args": vars(args)}, path)
    print(f"模型已保存到 {path}")


if __name__ == "__main__":
    main()
