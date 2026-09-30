"""
第004章 PPO的批量大小与学习率调优

PPO 的样本效率和稳定性由两个"规模"旋钮共同决定：
  - 每次更新前采样的批量（steps_per_update）：批量越大梯度越稳，但同一步数下更新次数越少
  - 学习率（lr）：步子越大越快，也越容易越界

本章在 CartPole-v1 上做 2×2 受控网格实验：
    lr ∈ {3e-4, 1e-3} × 批量 ∈ {1024, 4096}
每个组合跑相同步数预算、相同随机种子、相同 minibatch 大小，比较：
  - 首次达到 300 分（100 回合滑窗）的步数
  - 结束时的近 20 / 近 100 回合均值
  - KL、裁剪比例、熵三个健康指标
  - 墙钟时间与更新次数

运行：
    python code/ch004.py            # 4 个组合，各 6 万步，CPU 约 4-7 分钟
    python code/ch004.py --quick    # 快速跑通（约 40-70 秒）
    python code/ch004.py --lrs 3e-4 --batches 1024   # 只跑一个组合

预期：lr=3e-4 的两个批量都能稳步上升，批量大的 KL 更小；
lr=1e-3 配 1024 批量时裁剪比例与 KL 明显偏高、波动大；
lr=1e-3 配 4096 批量把噪声摊薄后，稳定性显著改善——大学习率需要大批量"兜底"。
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
    """固定随机种子：每个配置从同一条随机流出发，保证网格对比公平。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_tensor(obs, device):
    return torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)


# ---------------------------------------------------------------------------
# 网络（与前几章相同）
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
    def act(self, obs: torch.Tensor):
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action).item(), value.item()

    @torch.no_grad()
    def value(self, obs: torch.Tensor) -> float:
        _, v = self.forward(obs)
        return v.item()


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
# 采样
# ---------------------------------------------------------------------------
def collect_rollout(env, net, steps, obs_t, device, gamma, lam):
    obs_buf, act_buf, logp_buf = [], [], []
    rew_buf, val_buf, term_buf = [], [], []
    ep_returns = []

    ep_ret = 0.0
    obs_after = obs_t
    for _ in range(steps):
        a, logp, v = net.act(obs_t)
        next_obs, r, terminated, truncated, _ = env.step(a)

        obs_buf.append(obs_t.squeeze(0).cpu().numpy())
        act_buf.append(a)
        logp_buf.append(logp)
        rew_buf.append(r)
        val_buf.append(v)
        term_buf.append(float(terminated))

        ep_ret += r
        obs_t = to_tensor(next_obs, device)
        obs_after = obs_t

        if terminated or truncated:
            ep_returns.append(ep_ret)
            ep_ret = 0.0
            reset_obs, _ = env.reset()
            obs_t = to_tensor(reset_obs, device)

    last_value = net.value(obs_after)
    advantages, returns = compute_gae(rew_buf, val_buf, term_buf, last_value, gamma, lam)
    batch = {
        "obs": torch.as_tensor(np.asarray(obs_buf), dtype=torch.float32, device=device),
        "act": torch.as_tensor(np.asarray(act_buf), dtype=torch.long, device=device),
        "logp_old": torch.as_tensor(np.asarray(logp_buf), dtype=torch.float32, device=device),
        "adv": torch.as_tensor(advantages, dtype=torch.float32, device=device),
        "ret": torch.as_tensor(returns, dtype=torch.float32, device=device),
    }
    return batch, ep_returns, obs_t


# ---------------------------------------------------------------------------
# PPO 更新：额外记录梯度范数，用于观察学习率与批量的尺度关系
# ---------------------------------------------------------------------------
def ppo_update(net, optimizer, batch, args):
    obs, act = batch["obs"], batch["act"]
    logp_old, adv, ret = batch["logp_old"], batch["adv"], batch["ret"]
    N = obs.shape[0]
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    stats = {"approx_kl": 0.0, "clip_frac": 0.0, "entropy": 0.0,
             "grad_norm": 0.0, "n_updates": 0}
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
            # 记录裁剪前的梯度范数（诊断学习率是否过大）
            gn = nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            optimizer.step()

            with torch.no_grad():
                stats["approx_kl"] += (logp_old[mb] - logp).mean().item()
                stats["clip_frac"] += ((ratio - 1.0).abs() > args.clip).float().mean().item()
                stats["entropy"] += entropy.item()
                stats["grad_norm"] += float(gn)
                stats["n_updates"] += 1

    for k in stats:
        if k != "n_updates":
            stats[k] /= max(stats["n_updates"], 1)
    return stats


# ---------------------------------------------------------------------------
# 评估
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(env, net, episodes: int, device):
    returns = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            logits, _ = net(to_tensor(obs, device))
            action = int(torch.argmax(logits, dim=-1).item())
            obs, r, terminated, truncated, _ = env.step(action)
            ep_ret += r
            done = terminated or truncated
        returns.append(ep_ret)
    return returns


# ---------------------------------------------------------------------------
# 训练一个配置（lr, steps_per_update）
# ---------------------------------------------------------------------------
def train_one(lr: float, batch_steps: int, args):
    """在固定预算下训练一次，返回统计结果。"""
    set_seed(args.seed)
    device = torch.device("cpu")
    env = gym.make("CartPole-v1")
    obs, _ = env.reset(seed=args.seed)

    net = ActorCritic(env.observation_space.shape[0], env.action_space.n)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    all_returns, kls, clips, grads = [], [], [], []
    total_steps, n_updates, obs_t = 0, 0, to_tensor(obs, device)
    reached_at = None
    t0 = time.time()

    while total_steps < args.max_steps:
        batch, ep_returns, obs_t = collect_rollout(
            env, net, batch_steps, obs_t, device, args.gamma, args.lam)
        total_steps += batch_steps
        n_updates += 1
        all_returns.extend(ep_returns)

        stats = ppo_update(net, optimizer, batch, args)
        kls.append(stats["approx_kl"])
        clips.append(stats["clip_frac"])
        grads.append(stats["grad_norm"])

        if reached_at is None and len(all_returns) >= 100 and np.mean(all_returns[-100:]) >= args.stage_reward:
            reached_at = total_steps

        if len(all_returns) >= 20 and n_updates % 5 == 0:
            print(f"  lr={lr:<6} 批量={batch_steps:<5} 更新{n_updates:3d} | 步数 {total_steps:7d} | "
                  f"回合 {len(all_returns):4d} | 近20均 {np.mean(all_returns[-20:]):7.1f} | "
                  f"KL {stats['approx_kl']:.4f} | 裁剪 {stats['clip_frac']:.3f} | 熵 {stats['entropy']:.3f}")

    eval_returns = evaluate(gym.make("CartPole-v1"), net, args.eval_episodes, device)
    tail = all_returns[-100:] if len(all_returns) >= 100 else all_returns
    return {
        "lr": lr,
        "batch_steps": batch_steps,
        "steps": total_steps,
        "n_updates": n_updates,
        "reached_at": reached_at,
        "last20": float(np.mean(all_returns[-20:])),
        "last100": float(np.mean(tail)),
        "eval": float(np.mean(eval_returns)),
        "mean_kl": float(np.mean(kls)),
        "mean_clip": float(np.mean(clips)),
        "mean_grad": float(np.mean(grads)),
        "seconds": time.time() - t0,
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="第004章：PPO 批量大小与学习率调优")
    p.add_argument("--lrs", type=float, nargs="+", default=[3e-4, 1e-3], help="学习率网格")
    p.add_argument("--batches", type=int, nargs="+", default=[1024, 4096],
                   help="每次更新前采样步数网格")
    p.add_argument("--max-steps", type=int, default=60000, help="每个组合的环境步数预算")
    p.add_argument("--epochs", type=int, default=10, help="每批数据的更新轮数")
    p.add_argument("--batch-size", type=int, default=64, help="minibatch 大小（固定不变）")
    p.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
    p.add_argument("--lam", type=float, default=0.95, help="GAE 的 lambda")
    p.add_argument("--clip", type=float, default=0.2, help="PPO 裁剪范围")
    p.add_argument("--ent-coef", type=float, default=0.01, help="熵正则系数")
    p.add_argument("--vf-coef", type=float, default=0.5, help="价值损失系数")
    p.add_argument("--stage-reward", type=float, default=300.0, help="阶段目标（记录首达步数）")
    p.add_argument("--eval-episodes", type=int, default=5, help="评估回合数")
    p.add_argument("--seed", type=int, default=0, help="随机种子")
    p.add_argument("--save-dir", type=str, default="runs/ch004", help="结果保存目录")
    p.add_argument("--quick", action="store_true", help="快速模式：单组合 8 千步")
    return p.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.lrs = [3e-4]
        args.batches = [1024]
        args.max_steps = 8000
        args.epochs = 4
        args.stage_reward = 200.0

    print("=" * 92)
    print(f"批量 × 学习率网格 | lr={args.lrs} | 批量={args.batches} | "
          f"每个组合 {args.max_steps} 步 | 种子 {args.seed}")
    print("=" * 92)

    results = []
    for lr in args.lrs:
        for bs in args.batches:
            print(f"\n[开始] lr={lr}, 采样批量={bs}")
            res = train_one(lr, bs, args)
            results.append(res)
            print(f"[完成] lr={lr}, 批量={bs}: 近100均 {res['last100']:.1f} | "
                  f"评估 {res['eval']:.1f} | 平均KL {res['mean_kl']:.4f} | 用时 {res['seconds']:.1f}s")

    print("\n" + "=" * 92)
    print("网格对比总结（同种子、同步数预算、minibatch=64）")
    print("-" * 92)
    print(f"{'lr':>8} | {'批量':>6} | {'更新次数':>8} | {'首达300步数':>11} | {'近20均':>8} | "
          f"{'近100均':>8} | {'评估':>7} | {'平均KL':>8} | {'裁剪比例':>8} | {'梯度范数':>8} | {'用时(s)':>8}")
    for r in results:
        reached = str(r["reached_at"]) if r["reached_at"] is not None else "未达到"
        print(f"{r['lr']:>8.1e} | {r['batch_steps']:>6d} | {r['n_updates']:>8d} | {reached:>11} | "
              f"{r['last20']:>8.1f} | {r['last100']:>8.1f} | {r['eval']:>7.1f} | "
              f"{r['mean_kl']:>8.4f} | {r['mean_clip']:>8.3f} | {r['mean_grad']:>8.3f} | "
              f"{r['seconds']:>8.1f}")

    print("\n观察提示：")
    print("  - 同 lr 下对比两个批量：批量大的 KL / 裁剪比例通常更低，但更新次数更少；")
    print("  - 同批量下对比两个 lr：lr=1e-3 的梯度范数与裁剪比例通常明显高于 3e-4；")
    print("  - 若 lr=1e-3 + 小批量出现 KL 频繁 > 0.03，说明噪声没被批量摊薄。")

    os.makedirs(args.save_dir, exist_ok=True)
    path = os.path.join(args.save_dir, "batch_lr_grid.txt")
    with open(path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(f"lr={r['lr']:.1e} batch={r['batch_steps']} updates={r['n_updates']} "
                    f"reached300={r['reached_at']} last100={r['last100']:.2f} eval={r['eval']:.2f} "
                    f"kl={r['mean_kl']:.5f} clip={r['mean_clip']:.4f} grad={r['mean_grad']:.4f}\n")
    print(f"\n结果已写入 {path}")


if __name__ == "__main__":
    main()
