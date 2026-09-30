"""
第091章 多智能体环境：无人机编队

实验内容（4 架无人机在三维空间保持四面体编队，纯 numpy 内联环境）：
  - 一个虚拟长机沿李萨如曲线飞行，4 架无人机要分别保持在长机周围的
    四面体顶点上；彼此太近会碰撞惩罚
  - 算法：简化版 IDDPG（确定性策略梯度 + 双 Q 目标网络 + 目标动作平滑
    + 高斯探索噪声），共享网络（观测里含各自的编队偏移，天然可泛化）
  - 基线：P 控制器（比例-微分跟踪自己的目标点）
  - 指标：平均编队误差、编队保持率（误差 < 0.15 的步数占比）、碰撞率

运行：
    python code/ch091.py           # 完整训练（CPU 约 2-5 分钟）
    python code/ch091.py --quick   # 快速跑通（约 30-60 秒）

预期：训练后平均编队误差通常低于 P 控制器基线（以实跑为准），且碰撞率
接近 0；P 控制器在急转弯处会出现明显滞后误差。
"""

import argparse
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
# 环境：三维无人机编队
# ---------------------------------------------------------------------------
class DroneFormationEnv:
    """4 架无人机跟踪一个沿李萨如曲线运动的长机，保持四面体编队。

    观测（21 维/无人机）：相对长机的位置(3)、自己的速度(3)、自己的编队
    偏移(3)、另外 3 架无人机的相对位置(9)、长机速度(3)。
    动作：3 维加速度 [-1,1]。
    奖励（每机独立）：-编队误差 - 0.005‖a‖²，与他机距离 < 0.15 时每对 -0.5。
    """

    obs_dim = 21
    act_dim = 3
    n_agents = 4

    def __init__(self, max_steps: int = 120, accel: float = 0.15,
                 friction: float = 0.85, v_max: float = 1.2,
                 collision_radius: float = 0.15, bound: float = 2.0):
        self.max_steps = max_steps
        self.accel = accel
        self.friction = friction
        self.v_max = v_max
        self.collision_radius = collision_radius
        self.bound = bound
        # 四面体编队偏移（4 个顶点）
        base = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]],
                        dtype=np.float32)
        self.offsets = base / np.sqrt(3.0) * 0.25

    def _leader(self, t: float) -> np.ndarray:
        return np.array([
            0.7 * np.sin(0.08 * t),
            0.4 * np.sin(0.16 * t),
            0.15 * np.sin(0.08 * t + 1.0),
        ], dtype=np.float32)

    def reset(self, rng: np.random.Generator):
        self.steps = 0
        self.leader_pos = self._leader(0.0)
        self.leader_vel = np.zeros(3, dtype=np.float32)
        self.pos = self.leader_pos[None, :] + self.offsets \
            + rng.uniform(-0.15, 0.15, size=(self.n_agents, 3)).astype(np.float32)
        self.vel = np.zeros((self.n_agents, 3), dtype=np.float32)
        self.prev_leader = self.leader_pos.copy()
        return self._obs()

    def _obs(self) -> np.ndarray:
        obs = []
        for i in range(self.n_agents):
            others = [self.pos[j] - self.pos[i]
                      for j in range(self.n_agents) if j != i]
            o = np.concatenate([
                self.pos[i] - self.leader_pos,
                self.vel[i],
                self.offsets[i],
                np.concatenate(others),
                self.leader_vel,
            ])
            obs.append(o.astype(np.float32))
        return np.stack(obs)

    def step(self, actions: np.ndarray):
        a = np.clip(actions, -1.0, 1.0).astype(np.float32)
        self.vel = np.clip(self.friction * self.vel + self.accel * a,
                           -self.v_max, self.v_max)
        self.pos = np.clip(self.pos + self.vel, -self.bound, self.bound)

        # 长机按解析轨迹前进，速度用有限差分
        self.steps += 1
        new_leader = self._leader(float(self.steps))
        self.leader_vel = (new_leader - self.leader_pos).astype(np.float32)
        self.leader_pos = new_leader

        # 奖励：编队误差 + 控制代价 + 碰撞
        targets = self.leader_pos[None, :] + self.offsets
        errors = np.linalg.norm(self.pos - targets, axis=1)
        rewards = -errors.astype(np.float32) - 0.005 * (a ** 2).sum(axis=1)
        n_collisions = 0
        for i in range(self.n_agents):
            for j in range(i + 1, self.n_agents):
                if np.linalg.norm(self.pos[i] - self.pos[j]) < self.collision_radius:
                    n_collisions += 1
                    rewards[i] -= 0.5
                    rewards[j] -= 0.5

        truncated = self.steps >= self.max_steps
        info = {"mean_error": float(errors.mean()), "max_error": float(errors.max()),
                "collisions": n_collisions}
        return self._obs(), rewards, False, truncated, info


# ---------------------------------------------------------------------------
# 网络：确定性 Actor + 双 Q
# ---------------------------------------------------------------------------
class DeterministicActor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, act_dim), nn.Tanh(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class QNet(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class IDDPG:
    """简化版独立 DDPG：共享 Actor/双 Q，目标动作平滑，共享回放池。"""

    def __init__(self, env: DroneFormationEnv, args, device):
        self.args = args
        self.device = device
        self.actor = DeterministicActor(env.obs_dim, env.act_dim, args.hidden).to(device)
        q_in = env.obs_dim + env.act_dim
        self.q1 = QNet(q_in, args.hidden).to(device)
        self.q2 = QNet(q_in, args.hidden).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.q1_target = copy.deepcopy(self.q1)
        self.q2_target = copy.deepcopy(self.q2)
        for net in (self.actor_target, self.q1_target, self.q2_target):
            for p in net.parameters():
                p.requires_grad_(False)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=args.lr)
        self.critic_opt = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=args.lr)
        self.buffer = []
        self.updates = 0

    @torch.no_grad()
    def act(self, obs_np: np.ndarray):
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device)
        return self.actor(obs).cpu().numpy().astype(np.float32)

    def update(self):
        args = self.args
        idx = np.random.randint(0, len(self.buffer), size=args.batch_size)
        batch = [self.buffer[i] for i in idx]

        def stack(key):
            return torch.as_tensor(np.stack([b[key] for b in batch]),
                                   dtype=torch.float32, device=self.device)

        obs, act, rew = stack("obs"), stack("act"), stack("rew")
        next_obs, done = stack("next_obs"), stack("done")

        with torch.no_grad():
            next_a = self.actor_target(next_obs)
            noise = (torch.randn_like(next_a) * 0.2).clamp(-0.5, 0.5)  # 目标动作平滑
            next_a = torch.clamp(next_a + noise, -1.0, 1.0)
            q_in = torch.cat([next_obs, next_a], dim=-1)
            y = rew + args.gamma * (1.0 - done) * torch.min(
                self.q1_target(q_in), self.q2_target(q_in)).squeeze(-1)

        q_in = torch.cat([obs, act], dim=-1)
        critic_loss = (F.mse_loss(self.q1(q_in).squeeze(-1), y)
                       + F.mse_loss(self.q2(q_in).squeeze(-1), y))
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        pi = self.actor(obs)
        q_in = torch.cat([obs, pi], dim=-1)
        actor_loss = -self.q1(q_in).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        with torch.no_grad():
            for net, tnet in ((self.actor, self.actor_target),
                              (self.q1, self.q1_target), (self.q2, self.q2_target)):
                for p, tp in zip(net.parameters(), tnet.parameters()):
                    tp.data.mul_(1.0 - args.tau).add_(args.tau * p.data)
        self.updates += 1
        return float(critic_loss.item())


# ---------------------------------------------------------------------------
# 评估：学习策略与 P 控制器基线
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(agent, episodes, device, seed, mode="learned"):
    env = DroneFormationEnv()
    rng = np.random.default_rng(seed)
    mean_errors, keep_rates, collisions, rets = [], [], [], []
    for _ in range(episodes):
        obs = env.reset(rng)
        done, err_sum, n, col_sum, ep_ret = False, 0.0, 0, 0, 0.0
        info = {"mean_error": 1.0, "collisions": 0}
        while not done:
            if mode == "learned":
                a = agent.act(obs)
            else:  # P 控制器：朝目标点做比例-微分控制
                targets = env.leader_pos[None, :] + env.offsets
                a = np.clip(3.0 * (targets - env.pos) - 2.5 * env.vel, -1.0, 1.0)
            obs, rew, terminated, truncated, info = env.step(a)
            err_sum += info["mean_error"]
            col_sum += info["collisions"]
            ep_ret += float(rew.mean())
            n += 1
            done = terminated or truncated
        mean_errors.append(err_sum / max(n, 1))
        keep_rates.append(1.0 if err_sum / max(n, 1) < 0.15 else 0.0)
        collisions.append(col_sum / max(n, 1))
        rets.append(ep_ret)
    return (float(np.mean(mean_errors)), float(np.mean(keep_rates)),
            float(np.mean(collisions)), float(np.mean(rets)))


def parse_args():
    p = argparse.ArgumentParser(description="第091章：无人机三维编队（IDDPG）")
    p.add_argument("--max-steps", type=int, default=30000, help="环境步数上限")
    p.add_argument("--warmup", type=int, default=2000, help="随机动作预热步数")
    p.add_argument("--batch-size", type=int, default=256, help="采样批量")
    p.add_argument("--buffer-capacity", type=int, default=120000, help="回放池容量")
    p.add_argument("--lr", type=float, default=3e-4, help="学习率")
    p.add_argument("--gamma", type=float, default=0.97, help="折扣因子")
    p.add_argument("--tau", type=float, default=0.005, help="软更新系数")
    p.add_argument("--hidden", type=int, default=128, help="隐藏层宽度")
    p.add_argument("--noise-start", type=float, default=0.3, help="探索噪声初始标准差")
    p.add_argument("--noise-end", type=float, default=0.05, help="探索噪声最终标准差")
    p.add_argument("--noise-decay-steps", type=int, default=15000, help="噪声衰减步数")
    p.add_argument("--eval-every", type=int, default=6000, help="评估间隔（步）")
    p.add_argument("--eval-episodes", type=int, default=5, help="评估回合数")
    p.add_argument("--seed", type=int, default=0, help="随机种子")
    p.add_argument("--save-dir", type=str, default="runs/ch091", help="模型保存目录")
    p.add_argument("--quick", action="store_true", help="快速模式：缩小训练规模")
    return p.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.max_steps = min(args.max_steps, 10000)
        args.warmup = min(args.warmup, 800)
        args.noise_decay_steps = min(args.noise_decay_steps, 5000)
        args.eval_every = 3000
        args.eval_episodes = 3
    set_seed(args.seed)

    device = torch.device("cpu")
    env = DroneFormationEnv()
    rng = np.random.default_rng(args.seed)
    agent = IDDPG(env, args, device)

    print(f"设备 {device} | 4 架无人机三维编队 | 种子 {args.seed}")

    e_p, k_p, c_p, r_p = evaluate(agent, args.eval_episodes, device,
                                  args.seed + 111, mode="pd")
    print(f"P 控制器基线：平均编队误差 {e_p:.3f} | 保持率 {k_p:.2f} | "
          f"碰撞/步 {c_p:.3f} | 平均回报 {r_p:.2f}")

    obs = env.reset(rng)
    t_start = time.time()
    losses = []

    for step in range(1, args.max_steps + 1):
        # 1) 选动作：预热随机；随后策略 + 高斯噪声
        if step <= args.warmup:
            actions = rng.uniform(-1.0, 1.0,
                                  size=(env.n_agents, env.act_dim)).astype(np.float32)
        else:
            frac = min(1.0, (step - args.warmup) / max(args.noise_decay_steps, 1))
            sigma = args.noise_start + frac * (args.noise_end - args.noise_start)
            actions = agent.act(obs) + rng.normal(
                0.0, sigma, size=(env.n_agents, env.act_dim)).astype(np.float32)
            actions = np.clip(actions, -1.0, 1.0)

        # 2) 环境步进与存转移
        next_obs, rew, terminated, truncated, info = env.step(actions)
        done = terminated or truncated
        if step > args.warmup:
            for i in range(env.n_agents):
                agent.buffer.append({
                    "obs": obs[i].copy(), "act": actions[i].copy(),
                    "rew": np.float32(rew[i]), "next_obs": next_obs[i].copy(),
                    "done": np.float32(terminated),
                })
            if len(agent.buffer) > args.buffer_capacity:
                del agent.buffer[: len(agent.buffer) // 2]
            if len(agent.buffer) >= args.batch_size:
                losses.append(agent.update())

        if done:
            obs = env.reset(rng)
        else:
            obs = next_obs

        # 3) 日志与评估
        if step % args.eval_every == 0:
            e, k, c, r = evaluate(agent, args.eval_episodes, device,
                                  args.seed + 777, mode="learned")
            m_loss = float(np.mean(losses[-500:])) if losses else float("nan")
            print(f"[步 {step:6d}] 编队误差 {e:.3f} | 保持率 {k:.2f} | "
                  f"碰撞/步 {c:.3f} | 回报 {r:7.2f} | critic损失 {m_loss:.4f}")

    elapsed = time.time() - t_start
    e, k, c, r = evaluate(agent, args.eval_episodes, device, args.seed + 999)
    print("-" * 74)
    print(f"训练结束 | 用时 {elapsed:.1f}s | 环境步 {args.max_steps} | "
          f"梯度更新 {agent.updates} 次 | 回放池 {len(agent.buffer)}")
    print(f"评估（{args.eval_episodes} 回合，确定性策略）：平均编队误差 {e:.3f} | "
          f"保持率 {k:.2f} | 碰撞/步 {c:.3f} | 平均回报 {r:.2f}")
    print(f"对照 P 控制器：误差 {e_p:.3f} → {e:.3f} | 保持率 {k_p:.2f} → {k:.2f}")

    os.makedirs(args.save_dir, exist_ok=True)
    path = os.path.join(args.save_dir, "iddpg_drone_formation.pt")
    torch.save({"actor": agent.actor.state_dict(), "args": vars(args)}, path)
    print(f"模型已保存到 {path}")


if __name__ == "__main__":
    main()
