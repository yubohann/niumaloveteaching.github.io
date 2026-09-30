"""
第005章 PPO的熵正则化系数影响

PPO 的总损失里有一项熵奖励：loss = 策略损失 + c_vf·价值损失 − c_ent·熵。
c_ent 控制"保留多少随机性"：
  - c_ent = 0    ：不鼓励探索，策略可能过早变确定
  - c_ent = 0.01 ：PPO 默认值，温和地维持探索
  - c_ent = 0.05 ：强探索，策略长期保持高熵，代价是收敛慢

本章在 CartPole-v1 上固定其他所有超参数，只扫描 c_ent ∈ {0.0, 0.01, 0.05}：
  - 记录策略熵从 0.693（均匀分布）开始的下降轨迹
  - 比较首达 400 步数、最终回报与 KL 波动
  - 观察"熵崩溃"（entropy collapse）发生或没有发生时训练的表现

运行：
    python code/ch005.py            # 3 个系数，各 8 万步，CPU 约 3-6 分钟
    python code/ch005.py --quick    # 快速跑通（约 30-60 秒）
    python code/ch005.py --ent-coefs 0.0 0.01 0.05 0.1

预期：c_ent=0.0 最早冲高但熵下降最快，部分种子会卡在局部最优；
c_ent=0.01 熵平滑下降、综合最优；c_ent=0.05 熵长期在 0.6 上方，学习最慢。
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
    """固定随机种子：每个系数从同一条随机流出发，保证受控对比。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_tensor(obs, device):
    return torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)


# ---------------------------------------------------------------------------
# 网络
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

    @torch.no_grad()
    def policy_entropy(self, obs: torch.Tensor) -> float:
        """当前策略在给定观测上的熵，用于评估阶段的"纯策略熵"诊断。"""
        logits, _ = self.forward(obs)
        return float(Categorical(logits=logits).entropy().item())


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
# PPO 更新：熵项系数改为入参，并单独统计"每步熵梯度与策略梯度之比"
# ---------------------------------------------------------------------------
def ppo_update(net, optimizer, batch, args, ent_coef: float):
    """更新一次，返回诊断统计。

    额外统计 ent_ratio = |c_ent · H| / (|policy_loss| + eps)：
    它近似衡量熵项在总损失中的相对权重，用于观察熵项何时开始主导。
    """
    obs, act = batch["obs"], batch["act"]
    logp_old, adv, ret = batch["logp_old"], batch["adv"], batch["ret"]
    N = obs.shape[0]
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    stats = {"approx_kl": 0.0, "clip_frac": 0.0, "entropy": 0.0,
             "policy_loss": 0.0, "ent_ratio": 0.0, "n_updates": 0}
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
            loss = policy_loss + args.vf_coef * value_loss - ent_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            optimizer.step()

            with torch.no_grad():
                stats["approx_kl"] += (logp_old[mb] - logp).mean().item()
                stats["clip_frac"] += ((ratio - 1.0).abs() > args.clip).float().mean().item()
                stats["entropy"] += entropy.item()
                stats["policy_loss"] += abs(policy_loss.item())
                stats["ent_ratio"] += (ent_coef * entropy.item()) / (abs(policy_loss.item()) + 1e-8)
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
    returns, entropies = [], []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            entropies.append(net.policy_entropy(to_tensor(obs, device)))
            logits, _ = net(to_tensor(obs, device))
            action = int(torch.argmax(logits, dim=-1).item())
            obs, r, terminated, truncated, _ = env.step(action)
            ep_ret += r
            done = terminated or truncated
        returns.append(ep_ret)
    return returns, float(np.mean(entropies))


# ---------------------------------------------------------------------------
# 训练单个熵系数
# ---------------------------------------------------------------------------
def train_one(ent_coef: float, args):
    set_seed(args.seed)
    device = torch.device("cpu")
    env = gym.make("CartPole-v1")
    obs, _ = env.reset(seed=args.seed)

    net = ActorCritic(env.observation_space.shape[0], env.action_space.n)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    all_returns = []
    entropies, kl_list, ent_ratios = [], [], []
    total_steps, n_updates, obs_t = 0, 0, to_tensor(obs, device)
    reached_at = None
    t0 = time.time()

    print(f"\n[开始] c_ent = {ent_coef}"
          + ("（无熵正则）" if ent_coef == 0.0 else ""))
    while total_steps < args.max_steps:
        batch, ep_returns, obs_t = collect_rollout(
            env, net, args.steps_per_update, obs_t, device, args.gamma, args.lam)
        total_steps += args.steps_per_update
        n_updates += 1
        all_returns.extend(ep_returns)

        stats = ppo_update(net, optimizer, batch, args, ent_coef)
        entropies.append(stats["entropy"])
        kl_list.append(stats["approx_kl"])
        ent_ratios.append(stats["ent_ratio"])

        if reached_at is None and len(all_returns) >= 100 and np.mean(all_returns[-100:]) >= args.stage_reward:
            reached_at = total_steps

        if n_updates % 5 == 0:
            print(f"  c_ent={ent_coef:<5} 步数 {total_steps:7d} | 回合 {len(all_returns):4d} | "
                  f"近20均 {np.mean(all_returns[-20:]):7.1f} | 熵 {stats['entropy']:.3f} | "
                  f"熵项占比 {stats['ent_ratio']:.3f} | KL {stats['approx_kl']:.4f}")

    eval_returns, eval_entropy = evaluate(gym.make("CartPole-v1"), net, args.eval_episodes, device)
    tail = all_returns[-100:] if len(all_returns) >= 100 else all_returns
    first5 = float(np.mean(entropies[:5]))
    last5 = float(np.mean(entropies[-5:]))
    return {
        "ent_coef": ent_coef,
        "steps": total_steps,
        "reached_at": reached_at,
        "last100": float(np.mean(tail)),
        "eval": float(np.mean(eval_returns)),
        "entropy_first5": first5,
        "entropy_last5": last5,
        "eval_entropy": eval_entropy,
        "mean_kl": float(np.mean(kl_list)),
        "mean_ent_ratio": float(np.mean(ent_ratios)),
        "seconds": time.time() - t0,
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="第005章：PPO 熵正则化系数影响")
    p.add_argument("--ent-coefs", type=float, nargs="+", default=[0.0, 0.01, 0.05],
                   help="要对比的熵正则系数列表")
    p.add_argument("--max-steps", type=int, default=80000, help="每个系数的环境步数预算")
    p.add_argument("--steps-per-update", type=int, default=2048, help="每次更新前采样步数")
    p.add_argument("--epochs", type=int, default=10, help="每批数据的更新轮数")
    p.add_argument("--batch-size", type=int, default=64, help="minibatch 大小")
    p.add_argument("--lr", type=float, default=3e-4, help="学习率")
    p.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
    p.add_argument("--lam", type=float, default=0.95, help="GAE 的 lambda")
    p.add_argument("--clip", type=float, default=0.2, help="PPO 裁剪范围")
    p.add_argument("--vf-coef", type=float, default=0.5, help="价值损失系数")
    p.add_argument("--stage-reward", type=float, default=400.0, help="阶段目标（记录首达步数）")
    p.add_argument("--eval-episodes", type=int, default=5, help="评估回合数")
    p.add_argument("--seed", type=int, default=0, help="随机种子")
    p.add_argument("--save-dir", type=str, default="runs/ch005", help="结果保存目录")
    p.add_argument("--quick", action="store_true", help="快速模式：3 个系数、每个 1 万步")
    return p.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.max_steps = 10000
        args.steps_per_update = 512
        args.epochs = 4
        args.stage_reward = 250.0

    print("=" * 92)
    print(f"熵正则系数扫描 | c_ent={args.ent_coefs} | 每个 {args.max_steps} 步 | 种子 {args.seed}")
    print("=" * 92)

    results = []
    for c in args.ent_coefs:
        res = train_one(c, args)
        results.append(res)
        print(f"[完成] c_ent={c}: 近100均 {res['last100']:.1f} | 评估 {res['eval']:.1f} | "
              f"熵 {res['entropy_first5']:.3f} → {res['entropy_last5']:.3f} | "
              f"用时 {res['seconds']:.1f}s")

    print("\n" + "=" * 92)
    print("对比总结（同种子、同步数预算）")
    print("-" * 92)
    print(f"{'c_ent':>7} | {'首达400步数':>11} | {'近100均':>8} | {'评估':>7} | "
          f"{'熵(前5轮)':>9} | {'熵(后5轮)':>9} | {'评估熵':>8} | {'平均KL':>8} | {'熵项占比':>8} | {'用时(s)':>8}")
    for r in results:
        reached = str(r["reached_at"]) if r["reached_at"] is not None else "未达到"
        print(f"{r['ent_coef']:>7.3f} | {reached:>11} | {r['last100']:>8.1f} | {r['eval']:>7.1f} | "
              f"{r['entropy_first5']:>9.3f} | {r['entropy_last5']:>9.3f} | {r['eval_entropy']:>8.3f} | "
              f"{r['mean_kl']:>8.4f} | {r['mean_ent_ratio']:>8.3f} | {r['seconds']:>8.1f}")

    print("\n观察提示：")
    print("  - 均匀二元策略的熵上限是 ln2 ≈ 0.693；后 5 轮熵 < 0.15 视为熵崩溃倾向；")
    print("  - 若 c_ent=0.0 的评估熵接近 0 而回报不高，说明策略过早锁定在次优动作上；")
    print("  - 熵项占比持续 > 1 时，梯度主要由熵项驱动，学习必然变慢。")

    os.makedirs(args.save_dir, exist_ok=True)
    path = os.path.join(args.save_dir, "entropy_sweep.txt")
    with open(path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(f"ent_coef={r['ent_coef']} reached400={r['reached_at']} last100={r['last100']:.2f} "
                    f"eval={r['eval']:.2f} entropy={r['entropy_first5']:.3f}->{r['entropy_last5']:.3f} "
                    f"eval_entropy={r['eval_entropy']:.3f} kl={r['mean_kl']:.5f} "
                    f"ent_ratio={r['mean_ent_ratio']:.3f}\n")
    print(f"\n结果已写入 {path}")


if __name__ == "__main__":
    main()
