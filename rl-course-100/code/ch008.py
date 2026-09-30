"""
第008章 PPO的观测归一化与向量化环境

MountainCar-v0 的观测两维尺度差异巨大：
    position ∈ [−1.2, 0.6]，velocity ∈ [−0.07, 0.07]
普通 MLP 的输入层对这两个维度一视同仁，速度维度会被位置维度"淹没"。

本章把两个工程手段放在一起验证（同种子、同步数预算）：
  1) 观测归一化：RunningStats 在线统计均值/标准差，对观测做 (x−μ)/σ 并裁剪
  2) 向量化环境：单进程内同时运行 n 个环境，每个采样周期各走一步
     （教学简化：不做多进程，只模拟"多环境轮转采样"的数据分布效果）

三个对比组：
    single   ：1 个环境、原始观测
    vec      ：4 个环境、原始观测
    vecnorm  ：4 个环境、观测归一化

记录：首达 −135 分步数、最佳 100 回合均值、最终评估、墙钟时间、观测统计量。

运行：
    python code/ch008.py            # 3 组，各 25 万步，CPU 约 5-12 分钟
    python code/ch008.py --quick    # 快速跑通（约 30-60 秒）
    python code/ch008.py --variants single vecnorm

预期：vecnorm 明显快于 single；观测归一化让价值目标更均匀；
单独向量化在小批量环境下提升有限，但它是第 024 章分布式采样的基础。
"""

import argparse
import os
import random
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    """固定随机种子，保证三组对比受控。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# 观测归一化：在线均值/标准差 + 裁剪
# ---------------------------------------------------------------------------
class RunningStats:
    """对多元素张量做逐维 Welford 在线统计。"""

    def __init__(self, dim: int, clip: float = 10.0):
        self.dim = dim
        self.n = 0.0
        self.mean = np.zeros(dim, dtype=np.float64)
        self.m2 = np.zeros(dim, dtype=np.float64)
        self.clip = clip

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64).reshape(-1, self.dim)
        if x.shape[0] == 0:
            return
        self.n += x.shape[0]
        delta = x.mean(axis=0) - self.mean
        self.mean += delta * x.shape[0] / self.n
        self.m2 += ((x - self.mean) ** 2).sum(axis=0)

    @property
    def std(self) -> np.ndarray:
        if self.n < 2:
            return np.ones(self.dim)
        return np.sqrt(self.m2 / (self.n - 1))

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return np.clip((x - self.mean) / (self.std + 1e-8), -self.clip, self.clip)


# ---------------------------------------------------------------------------
# 网络：新增批量动作采样，向量化环境一次前向服务 n 个环境
# ---------------------------------------------------------------------------
class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 64):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.actor = nn.Linear(hidden, act_dim)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor):
        h = self.body(obs)
        return self.actor(h), self.critic(h).squeeze(-1)

    @torch.no_grad()
    def act_batch(self, obs: torch.Tensor):
        """对形状 (n, obs_dim) 的批量观测采样，返回 numpy 数组（动作, logp, 价值）。"""
        logits, values = self.forward(obs)
        dist = Categorical(logits=logits)
        actions = dist.sample()
        return (actions.cpu().numpy(),
                dist.log_prob(actions).cpu().numpy(),
                values.cpu().numpy())

    @torch.no_grad()
    def act(self, obs: torch.Tensor):
        a, logp, v = self.act_batch(obs)
        return int(a[0]), float(logp[0]), float(v[0])

    @torch.no_grad()
    def value(self, obs: torch.Tensor) -> float:
        _, v = self.forward(obs)
        return float(v.item())


# ---------------------------------------------------------------------------
# GAE
# ---------------------------------------------------------------------------
def compute_gae(rewards, values, terminateds, last_value, gamma: float, lam: float):
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(T)):
        next_value = last_value if t == T - 1 else values[t + 1]
        non_terminal = 1.0 - terminateds[t]
        delta = rewards[t] + gamma * next_value * non_terminal - values[t]
        last_gae = delta + gamma * lam * non_terminal * last_gae
        advantages[t] = last_gae
    returns = advantages + np.asarray(values, dtype=np.float32)
    return advantages, returns


# ---------------------------------------------------------------------------
# 向量化采样：n 个环境轮转，每个周期各走一步；每个环境单独算 GAE
# ---------------------------------------------------------------------------
def collect_rollout_vec(envs, net, steps_per_env, obs_raw, device, gamma, lam, stats):
    """采样 n * steps_per_env 步。

    参数：
        envs        : 环境列表（长度 n）
        steps_per_env: 每个环境在本轮采样中走多少步
        obs_raw     : 当前各环境的原始观测列表
        stats       : RunningStats 或 None（None 表示不做观测归一化）
    返回：batch、本轮完成的回合回报、更新后的 obs_raw
    """
    n = len(envs)
    obs_bufs = [[] for _ in range(n)]
    act_bufs = [[] for _ in range(n)]
    logp_bufs = [[] for _ in range(n)]
    rew_bufs = [[] for _ in range(n)]
    val_bufs = [[] for _ in range(n)]
    term_bufs = [[] for _ in range(n)]
    ep_returns = []
    ep_ret = [0.0] * n

    for _ in range(steps_per_env):
        raw = np.stack(obs_raw).astype(np.float32)          # (n, obs_dim)
        if stats is not None:
            stats.update(raw)                                # 只统计原始观测
            net_in = stats.normalize(raw).astype(np.float32)
        else:
            net_in = raw
        obs_t = torch.as_tensor(net_in, dtype=torch.float32, device=device)
        acts, logps, vals = net.act_batch(obs_t)

        for i in range(n):
            next_raw, r, terminated, truncated, _ = envs[i].step(int(acts[i]))
            obs_bufs[i].append(net_in[i])
            act_bufs[i].append(int(acts[i]))
            logp_bufs[i].append(float(logps[i]))
            rew_bufs[i].append(float(r))
            val_bufs[i].append(float(vals[i]))
            term_bufs[i].append(float(terminated))

            ep_ret[i] += r
            obs_raw[i] = next_raw
            if terminated or truncated:
                ep_returns.append(ep_ret[i])
                ep_ret[i] = 0.0
                reset_obs, _ = envs[i].reset()
                obs_raw[i] = reset_obs

    # 每个环境单独计算 GAE，再按环境顺序拼接
    all_obs, all_act, all_logp, all_adv, all_ret = [], [], [], [], []
    for i in range(n):
        raw_last = np.asarray(obs_raw[i], dtype=np.float32).reshape(1, -1)
        if stats is not None:
            net_last = stats.normalize(raw_last).astype(np.float32)
        else:
            net_last = raw_last
        last_value = net.value(torch.as_tensor(net_last, dtype=torch.float32, device=device))
        adv_i, ret_i = compute_gae(rew_bufs[i], val_bufs[i], term_bufs[i], last_value, gamma, lam)
        all_obs.extend(obs_bufs[i])
        all_act.extend(act_bufs[i])
        all_logp.extend(logp_bufs[i])
        all_adv.append(adv_i)
        all_ret.append(ret_i)

    batch = {
        "obs": torch.as_tensor(np.asarray(all_obs), dtype=torch.float32, device=device),
        "act": torch.as_tensor(np.asarray(all_act), dtype=torch.long, device=device),
        "logp_old": torch.as_tensor(np.asarray(all_logp), dtype=torch.float32, device=device),
        "adv": torch.as_tensor(np.concatenate(all_adv), dtype=torch.float32, device=device),
        "ret": torch.as_tensor(np.concatenate(all_ret), dtype=torch.float32, device=device),
    }
    return batch, ep_returns, obs_raw


# ---------------------------------------------------------------------------
# PPO 更新
# ---------------------------------------------------------------------------
def ppo_update(net, optimizer, batch, args):
    obs, act = batch["obs"], batch["act"]
    logp_old, adv, ret = batch["logp_old"], batch["adv"], batch["ret"]
    N = obs.shape[0]
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    stats = {"approx_kl": 0.0, "clip_frac": 0.0, "entropy": 0.0, "n_updates": 0}
    for _ in range(args.epochs):
        idx = torch.randperm(N, device=obs.device)
        for start in range(0, N, args.batch_size):
            mb = idx[start:start + args.batch_size]
            logits, value = net(obs[mb])
            dist = Categorical(logits=logits)
            logp = dist.log_prob(act[mb])

            ratio = torch.exp(logp - logp_old[mb])
            surr1 = ratio * adv[mb]
            surr2 = torch.clamp(ratio, 1.0 - args.clip, 1.0 + args.clip) * adv[mb]
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(value, ret[mb])
            entropy = dist.entropy().mean()
            loss = policy_loss + args.vf_coef * value_loss - args.ent_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            optimizer.step()

            with torch.no_grad():
                stats["approx_kl"] += (logp_old[mb] - logp).mean().item()
                stats["clip_frac"] += ((ratio - 1.0).abs() > args.clip).float().mean().item()
                stats["entropy"] += entropy.item()
                stats["n_updates"] += 1

    for k in stats:
        if k != "n_updates":
            stats[k] /= max(stats["n_updates"], 1)
    return stats


# ---------------------------------------------------------------------------
# 评估：贪婪策略，观测使用训练期统计量（或原样）
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(env, net, episodes: int, device, stats=None):
    returns = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            raw = np.asarray(obs, dtype=np.float32).reshape(1, -1)
            net_in = stats.normalize(raw).astype(np.float32) if stats is not None else raw
            logits, _ = net(torch.as_tensor(net_in, dtype=torch.float32, device=device))
            action = int(torch.argmax(logits, dim=-1).item())
            obs, r, terminated, truncated, _ = env.step(action)
            ep_ret += r
            done = terminated or truncated
        returns.append(ep_ret)
    return returns


# ---------------------------------------------------------------------------
# 训练一个变体
# ---------------------------------------------------------------------------
VARIANT_SPECS = {
    "single": (1, False),
    "vec": (4, False),
    "vecnorm": (4, True),
}


def train_one(variant: str, args):
    n_envs, use_norm = VARIANT_SPECS[variant]
    set_seed(args.seed)
    device = torch.device("cpu")

    envs = [gym.make("MountainCar-v0") for _ in range(n_envs)]
    obs_raw = []
    for i, env in enumerate(envs):
        obs_i, _ = env.reset(seed=args.seed + i)
        obs_raw.append(obs_i)

    obs_dim = envs[0].observation_space.shape[0]
    net = ActorCritic(obs_dim, envs[0].action_space.n)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    stats = RunningStats(obs_dim) if use_norm else None

    steps_per_env = max(args.steps_per_update // n_envs, 1)
    all_returns, kls = [], []
    total_steps, n_updates = 0, 0
    best100 = -np.inf
    reached_at = None
    t0 = time.time()

    print(f"\n[开始] 变体 {variant} | 环境数 {n_envs} | 观测归一化 {use_norm} | "
          f"每轮 {steps_per_env}*{n_envs}={steps_per_env * n_envs} 步")
    while total_steps < args.max_steps:
        batch, ep_returns, obs_raw = collect_rollout_vec(
            envs, net, steps_per_env, obs_raw, device, args.gamma, args.lam, stats)
        total_steps += steps_per_env * n_envs
        n_updates += 1
        all_returns.extend(ep_returns)
        if len(all_returns) >= 100:
            best100 = max(best100, float(np.mean(all_returns[-100:])))

        pstats = ppo_update(net, optimizer, batch, args)
        kls.append(pstats["approx_kl"])

        if reached_at is None and len(all_returns) >= 100 and np.mean(all_returns[-100:]) >= args.stage_reward:
            reached_at = total_steps

        if n_updates % 5 == 0 and len(all_returns) >= 20:
            print(f"  [{variant:7}] 步数 {total_steps:7d} | 回合 {len(all_returns):4d} | "
                  f"近20均 {np.mean(all_returns[-20:]):7.1f} | 最佳100 {best100:7.1f} | "
                  f"KL {pstats['approx_kl']:.4f} | 熵 {pstats['entropy']:.3f}")

    eval_returns = evaluate(gym.make("MountainCar-v0"), net, args.eval_episodes, device, stats)
    tail = all_returns[-100:] if len(all_returns) >= 100 else all_returns
    return {
        "variant": variant,
        "n_envs": n_envs,
        "use_norm": use_norm,
        "steps": total_steps,
        "episodes": len(all_returns),
        "reached_at": reached_at,
        "best100": float(best100) if best100 > -np.inf else float(np.mean(tail)),
        "last100": float(np.mean(tail)),
        "eval": float(np.mean(eval_returns)),
        "obs_mean": stats.mean.round(3).tolist() if stats is not None else "raw",
        "obs_std": stats.std.round(3).tolist() if stats is not None else "raw",
        "seconds": time.time() - t0,
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="第008章：PPO 观测归一化与向量化环境")
    p.add_argument("--variants", type=str, nargs="+", default=["single", "vec", "vecnorm"],
                   choices=list(VARIANT_SPECS.keys()), help="要对比的变体")
    p.add_argument("--max-steps", type=int, default=250000, help="每个变体的环境步数预算")
    p.add_argument("--steps-per-update", type=int, default=2048, help="每次更新的总采样步数")
    p.add_argument("--epochs", type=int, default=8, help="每批数据的更新轮数")
    p.add_argument("--batch-size", type=int, default=64, help="minibatch 大小")
    p.add_argument("--lr", type=float, default=3e-4, help="学习率")
    p.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
    p.add_argument("--lam", type=float, default=0.95, help="GAE 的 lambda")
    p.add_argument("--clip", type=float, default=0.2, help="PPO 裁剪范围")
    p.add_argument("--ent-coef", type=float, default=0.01, help="熵正则系数")
    p.add_argument("--vf-coef", type=float, default=0.5, help="价值损失系数")
    p.add_argument("--stage-reward", type=float, default=-135.0, help="阶段目标（100 回合均值）")
    p.add_argument("--eval-episodes", type=int, default=10, help="评估回合数")
    p.add_argument("--seed", type=int, default=0, help="随机种子")
    p.add_argument("--save-dir", type=str, default="runs/ch008", help="结果保存目录")
    p.add_argument("--quick", action="store_true", help="快速模式：3 组、各 1.5 万步")
    return p.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.variants = ["single", "vecnorm"]
        args.max_steps = 15000
        args.steps_per_update = 512
        args.epochs = 4
        args.stage_reward = -180.0

    print("=" * 96)
    print(f"观测归一化 × 向量化环境 | 变体 {args.variants} | 每个 {args.max_steps} 步 | 种子 {args.seed}")
    print("环境：MountainCar-v0（观测 [位置, 速度]，两维尺度相差约 17 倍）")
    print("=" * 96)

    results = []
    for v in args.variants:
        res = train_one(v, args)
        results.append(res)
        print(f"[完成] {v}: 最佳100 {res['best100']:.1f} | 近100 {res['last100']:.1f} | "
              f"评估 {res['eval']:.1f} | 用时 {res['seconds']:.1f}s")

    print("\n" + "=" * 96)
    print("对比总结（同种子、同步数预算）")
    print("-" * 96)
    print(f"{'变体':>8} | {'环境数':>5} | {'归一化':>6} | {'首达-135步数':>11} | {'最佳100':>8} | "
          f"{'近100均':>8} | {'评估(贪婪)':>9} | {'用时(s)':>8} | {'观测均值/标准差':>22}")
    for r in results:
        reached = str(r["reached_at"]) if r["reached_at"] is not None else "未达到"
        norm = "是" if r["use_norm"] else "否"
        stat = (f"{r['obs_mean']} / {r['obs_std']}" if r["use_norm"] else "raw")
        print(f"{r['variant']:>8} | {r['n_envs']:>5d} | {norm:>6} | {reached:>11} | "
              f"{r['best100']:>8.1f} | {r['last100']:>8.1f} | {r['eval']:>9.1f} | "
              f"{r['seconds']:>8.1f} | {stat:>22}")

    print("\n观察提示：")
    print("  - 观测原始统计量：位置 ≈ [-1.2, 0.6]，速度 ≈ [-0.07, 0.07]，相差约 17 倍；")
    print("  - 归一化后网络输入两维同尺度，价值目标更均匀；")
    print("  - 向量化用一次前向服务 n 个环境，且数据相关性更低，是分布式采样的最小原型。")

    os.makedirs(args.save_dir, exist_ok=True)
    path = os.path.join(args.save_dir, "obs_vec.txt")
    with open(path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(f"variant={r['variant']} n_envs={r['n_envs']} norm={r['use_norm']} "
                    f"reached={r['reached_at']} best100={r['best100']:.2f} last100={r['last100']:.2f} "
                    f"eval={r['eval']:.2f} seconds={r['seconds']:.1f}\n")
    print(f"\n结果已写入 {path}")


if __name__ == "__main__":
    main()
