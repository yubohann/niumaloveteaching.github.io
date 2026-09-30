"""
第087章 多智能体环境：粒子环境

实验内容（3 个粒子去覆盖 3 个地标，环境为脚本内联的纯 numpy 实现）：
  - 参考经典"粒子环境"（MPE 的 simple spread 类任务）的最小复现：
    粒子是速度控制的二维质点，地标每回合随机旋转摆放，粒子之间会碰撞
  - 奖励 = 每个地标到最近粒子的距离负均值（团队共享）+ 地标被占据的加成
           + 碰撞惩罚
  - 算法：独立 PPO + 参数共享（一个网络服务 3 个粒子）+ GAE
  - 指标：地标覆盖率（每个地标至少有一个粒子在 0.1 半径内）、平均回报、
          最近地标距离图

运行：
    python code/ch087.py           # 完整训练（CPU 约 3-6 分钟）
    python code/ch087.py --quick   # 快速跑通（约 30-60 秒）

预期：覆盖率从随机初始化的 0.3 左右提升到 0.8～1.0（以实跑为准），
碰撞惩罚让粒子学会在覆盖地标时彼此错开。
"""

import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_STD_MIN, LOG_STD_MAX = -20.0, 2.0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# 环境：粒子覆盖（MPE simple spread 的轻量版）
# ---------------------------------------------------------------------------
class ParticleCoverEnv:
    """3 个速度控制粒子，需要在有限步数内分别覆盖 3 个地标。

    观测（14 维/粒子）：自己的位置+速度(4)、另外两个粒子的相对位置(4)、
    三个地标的相对位置(6)。
    动作：2 维速度指令 [-1,1]，经轻度惯性后作用于位置。
    奖励（团队共享）：-mean_j min_i d(i,j) + 0.2×被覆盖地标数 − 0.1×碰撞对数。
    """

    obs_dim = 14
    act_dim = 2
    n_agents = 3
    n_landmarks = 3

    def __init__(self, cover_radius: float = 0.1, collision_radius: float = 0.08,
                 max_steps: int = 50, accel: float = 0.12, friction: float = 0.6,
                 v_max: float = 1.0, bound: float = 1.0):
        self.cover_radius = cover_radius
        self.collision_radius = collision_radius
        self.max_steps = max_steps
        self.accel = accel
        self.friction = friction
        self.v_max = v_max
        self.bound = bound

    def reset(self, rng: np.random.Generator):
        # 地标：随机旋转 + 随机半径的三边形
        theta = rng.uniform(0.0, 2.0 * np.pi)
        radius = rng.uniform(0.5, 0.7)
        angles = theta + np.linspace(0.0, 2.0 * np.pi, self.n_landmarks, endpoint=False)
        self.landmarks = np.stack([radius * np.cos(angles),
                                   radius * np.sin(angles)], axis=1).astype(np.float32)
        # 粒子：从中心附近随机出发
        self.pos = rng.uniform(-0.2, 0.2, size=(self.n_agents, 2)).astype(np.float32)
        self.vel = np.zeros((self.n_agents, 2), dtype=np.float32)
        self.steps = 0
        return self._obs()

    def _obs(self) -> np.ndarray:
        obs = []
        for i in range(self.n_agents):
            others = [self.pos[j] - self.pos[i]
                      for j in range(self.n_agents) if j != i]
            landmark_rel = [lm - self.pos[i] for lm in self.landmarks]
            o = np.concatenate([self.pos[i], self.vel[i],
                                np.concatenate(others),
                                np.concatenate(landmark_rel)])
            obs.append(o)
        return np.stack(obs).astype(np.float32)

    def step(self, actions: np.ndarray):
        a = np.clip(actions, -1.0, 1.0).astype(np.float32)
        self.vel = np.clip(self.friction * self.vel + self.accel * a,
                           -self.v_max, self.v_max)
        self.pos = np.clip(self.pos + self.vel, -self.bound, self.bound)
        self.steps += 1

        dists = np.linalg.norm(self.pos[None, :, :] - self.landmarks[:, None, :],
                               axis=-1)              # (地标, 粒子)
        min_dists = dists.min(axis=1)
        reward = float(-min_dists.mean())
        covered = int(np.sum(min_dists < self.cover_radius))
        reward += 0.2 * covered                       # 覆盖率加成

        n_collisions = 0
        for i in range(self.n_agents):
            for j in range(i + 1, self.n_agents):
                if np.linalg.norm(self.pos[i] - self.pos[j]) < self.collision_radius:
                    n_collisions += 1
        reward -= 0.1 * n_collisions

        coverage = covered / self.n_landmarks
        truncated = self.steps >= self.max_steps
        info = {"coverage": coverage, "min_dists": min_dists.copy(),
                "collisions": n_collisions}
        return self._obs(), reward, False, truncated, info


# ---------------------------------------------------------------------------
# 网络：共享躯干的 Actor-Critic（与第 084/085 章同族）
# ---------------------------------------------------------------------------
class PPOActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.mu_head = nn.Linear(hidden, act_dim)
        self.v_head = nn.Linear(hidden, 1)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs: torch.Tensor):
        h = self.body(obs)
        mu = self.mu_head(h)
        std = torch.clamp(self.log_std, LOG_STD_MIN, LOG_STD_MAX).exp()
        v = self.v_head(h).squeeze(-1)
        return mu, std, v

    @torch.no_grad()
    def act_batch(self, obs_np: np.ndarray, device):
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
        mu, std, v = self.forward(obs)
        dist = Normal(mu, std)
        x = dist.sample()
        action = torch.tanh(x)
        logp = dist.log_prob(x) - torch.log(1.0 - action.pow(2) + 1e-6)
        return (action.cpu().numpy().astype(np.float32),
                logp.sum(dim=-1).cpu().numpy().astype(np.float32),
                v.cpu().numpy().astype(np.float32))

    @torch.no_grad()
    def act_deterministic(self, obs_np: np.ndarray, device) -> np.ndarray:
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
        mu, _, _ = self.forward(obs)
        return torch.tanh(mu).cpu().numpy().astype(np.float32)

    def log_prob_and_value(self, obs, act):
        mu, std, v = self.forward(obs)
        dist = Normal(mu, std)
        a = torch.clamp(act, -0.999999, 0.999999)
        x = torch.atanh(a)
        logp = dist.log_prob(x) - torch.log(1.0 - a.pow(2) + 1e-6)
        return logp.sum(dim=-1), v, dist.entropy().sum(dim=-1)


def compute_gae(rewards, values, dones, last_values, gamma, lam):
    """共享奖励下的 GAE：(T,) 奖励、(T,N) 价值 -> (T,N) 优势与回报。"""
    T, N = values.shape
    adv = np.zeros((T, N), dtype=np.float32)
    last = np.zeros(N, dtype=np.float32)
    for t in reversed(range(T)):
        next_v = last_values if t == T - 1 else values[t + 1]
        non_term = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_v * non_term - values[t]
        last = delta + gamma * lam * non_term * last
        adv[t] = last
    return adv, adv + values


# ---------------------------------------------------------------------------
# 采样与更新
# ---------------------------------------------------------------------------
def collect_rollout(env, net, rng, steps, obs, device, args):
    obs_buf, act_buf, logp_buf, val_buf = [], [], [], []
    rew_buf, done_buf, cov_buf = [], [], []
    ep_stats = []
    ep_ret, ep_cov, ep_col = 0.0, 0.0, 0
    ep_len = 0

    for _ in range(steps):
        a, logp, v = net.act_batch(obs, device)
        next_obs, rew, terminated, truncated, info = env.step(a)
        obs_buf.append(obs)
        act_buf.append(a)
        logp_buf.append(logp)
        val_buf.append(v)
        rew_buf.append(rew)
        done_buf.append(float(terminated))          # 只有截断，恒为 0
        ep_ret += rew
        ep_cov += info["coverage"]
        ep_len += 1
        ep_col += info["collisions"]
        if terminated or truncated:
            ep_stats.append({"return": ep_ret, "coverage": ep_cov / max(ep_len, 1),
                             "final_coverage": info["coverage"],
                             "collisions": ep_col / max(ep_len, 1)})
            obs = env.reset(rng)
            ep_ret, ep_cov, ep_len, ep_col = 0.0, 0.0, 0, 0
        else:
            obs = next_obs

    with torch.no_grad():
        _, _, last_v = net.forward(
            torch.as_tensor(obs, dtype=torch.float32, device=device))
        last_values = last_v.cpu().numpy().astype(np.float32)
    values = np.stack(val_buf)
    adv, ret = compute_gae(np.asarray(rew_buf, dtype=np.float32), values,
                           np.asarray(done_buf, dtype=np.float32),
                           last_values, args.gamma, args.lam)
    T, N = adv.shape
    batch = {
        "obs": np.stack(obs_buf).reshape(T * N, -1),
        "act": np.stack(act_buf).reshape(T * N, -1),
        "logp": np.stack(logp_buf).reshape(T * N),
        "adv": adv.reshape(T * N),
        "ret": ret.reshape(T * N),
    }
    return batch, ep_stats, obs


def ppo_update(net, optimizer, batch, args, device):
    obs = torch.as_tensor(batch["obs"], dtype=torch.float32, device=device)
    act = torch.as_tensor(batch["act"], dtype=torch.float32, device=device)
    logp_old = torch.as_tensor(batch["logp"], dtype=torch.float32, device=device)
    adv = torch.as_tensor(batch["adv"], dtype=torch.float32, device=device)
    ret = torch.as_tensor(batch["ret"], dtype=torch.float32, device=device)
    N = obs.shape[0]
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    stats = {"kl": 0.0, "clip_frac": 0.0, "n": 0}
    for _ in range(args.epochs):
        idx = torch.randperm(N, device=device)
        for start in range(0, N, args.batch_size):
            mb = idx[start:start + args.batch_size]
            logp, v, entropy = net.log_prob_and_value(obs[mb], act[mb])
            ratio = torch.exp(logp - logp_old[mb])
            surr1 = ratio * adv[mb]
            surr2 = torch.clamp(ratio, 1.0 - args.clip, 1.0 + args.clip) * adv[mb]
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(v, ret[mb])
            loss = policy_loss + 0.5 * value_loss - args.ent_coef * entropy.mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            optimizer.step()
            with torch.no_grad():
                stats["kl"] += (logp_old[mb] - logp).mean().item()
                stats["clip_frac"] += ((ratio - 1.0).abs() > args.clip).float().mean().item()
                stats["n"] += 1
    stats["kl"] /= max(stats["n"], 1)
    stats["clip_frac"] /= max(stats["n"], 1)
    return stats


# ---------------------------------------------------------------------------
# 评估
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(env, net, episodes, device, seed):
    rng = np.random.default_rng(seed)
    rets, covs, final_covs, colls = [], [], [], []
    for _ in range(episodes):
        obs = env.reset(rng)
        done = False
        ep_ret, cov_sum, col_sum, n = 0.0, 0.0, 0.0, 0
        info = {"coverage": 0.0, "collisions": 0}
        while not done:
            a = net.act_deterministic(obs, device)
            obs, rew, terminated, truncated, info = env.step(a)
            ep_ret += rew
            cov_sum += info["coverage"]
            col_sum += info["collisions"]
            n += 1
            done = terminated or truncated
        rets.append(ep_ret)
        covs.append(cov_sum / max(n, 1))
        final_covs.append(info["coverage"])
        colls.append(col_sum / max(n, 1))
    return (float(np.mean(rets)), float(np.mean(covs)),
            float(np.mean(final_covs)), float(np.mean(colls)))


def parse_args():
    p = argparse.ArgumentParser(description="第087章：多智能体粒子环境（覆盖任务）")
    p.add_argument("--max-steps", type=int, default=60000, help="环境步数上限")
    p.add_argument("--rollout", type=int, default=2048, help="每次更新的采样步数")
    p.add_argument("--epochs", type=int, default=4, help="PPO 更新轮数")
    p.add_argument("--batch-size", type=int, default=256, help="小批量大小")
    p.add_argument("--lr", type=float, default=3e-4, help="学习率")
    p.add_argument("--gamma", type=float, default=0.95, help="折扣因子")
    p.add_argument("--lam", type=float, default=0.95, help="GAE lambda")
    p.add_argument("--clip", type=float, default=0.2, help="裁剪范围")
    p.add_argument("--ent-coef", type=float, default=0.01, help="熵正则系数")
    p.add_argument("--hidden", type=int, default=128, help="隐藏层宽度")
    p.add_argument("--eval-every", type=int, default=10000, help="评估间隔（步）")
    p.add_argument("--eval-episodes", type=int, default=10, help="评估回合数")
    p.add_argument("--seed", type=int, default=0, help="随机种子")
    p.add_argument("--save-dir", type=str, default="runs/ch087", help="模型保存目录")
    p.add_argument("--quick", action="store_true", help="快速模式：缩小训练规模")
    return p.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.max_steps = min(args.max_steps, 12000)
        args.rollout = 1024
        args.eval_every = 4000
        args.eval_episodes = 5
    set_seed(args.seed)

    device = torch.device("cpu")
    env = ParticleCoverEnv()
    rng = np.random.default_rng(args.seed)
    net = PPOActorCritic(env.obs_dim, env.act_dim, args.hidden).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    print(f"设备 {device} | 粒子数 {env.n_agents} | 地标数 {env.n_landmarks} | "
          f"种子 {args.seed}")
    print(f"配置：max_steps={args.max_steps}, rollout={args.rollout}, "
          f"epochs={args.epochs}, batch={args.batch_size}, lr={args.lr}")

    obs = env.reset(rng)
    total_steps, update_idx = 0, 0
    all_stats = []
    t_start = time.time()

    while total_steps < args.max_steps:
        batch, ep_stats, obs = collect_rollout(
            env, net, rng, args.rollout, obs, device, args)
        total_steps += args.rollout
        all_stats.extend(ep_stats)
        stats = ppo_update(net, optimizer, batch, args, device)
        update_idx += 1

        if total_steps % args.eval_every < args.rollout:
            m_ret, m_cov, m_final, m_col = evaluate(
                env, net, args.eval_episodes, device, args.seed + 321)
            print(f"[更新 {update_idx:3d}] 步 {total_steps:6d} | 回报 {m_ret:7.2f} | "
                  f"平均覆盖率 {m_cov:.2f} | 末态覆盖率 {m_final:.2f} | "
                  f"碰撞/步 {m_col:.3f} | KL {stats['kl']:.4f}")

    elapsed = time.time() - t_start

    m_ret, m_cov, m_final, m_col = evaluate(env, net, args.eval_episodes,
                                            device, args.seed + 999)
    print("-" * 74)
    print(f"训练结束 | 用时 {elapsed:.1f}s | 环境步 {total_steps} | 更新 {update_idx} 次")
    print(f"评估（{args.eval_episodes} 回合，确定性策略）：平均回报 {m_ret:.2f} | "
          f"平均覆盖率 {m_cov:.2f} | 末态覆盖率 {m_final:.2f} | 碰撞/步 {m_col:.3f}")
    if all_stats:
        print(f"训练尾段（最后20回合）：平均回报 "
              f"{np.mean([s['return'] for s in all_stats[-20:]]):.2f}")

    os.makedirs(args.save_dir, exist_ok=True)
    path = os.path.join(args.save_dir, "ippo_particle_cover.pt")
    torch.save({"policy": net.state_dict(), "args": vars(args)}, path)
    print(f"模型已保存到 {path}")


if __name__ == "__main__":
    main()
