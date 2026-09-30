"""
第003章 PPO的裁剪目标与KL惩罚对比

PPO 有两种实现"信任域"的路线：
  - 裁剪目标（clip）：比率超出 [1-ε, 1+ε] 后梯度归零
  - KL 惩罚（kl）：策略损失用未裁剪的 ratio·Â，另加自适应系数 β 的 KL 惩罚

本章在完全相同的网络、数据、预算与随机种子下分别用两种目标训练 CartPole：
  - 记录样本效率（首达 475 的回合数）、KL 水平与波动、熵、评估回报
  - KL 模式下额外记录 β 的漂移轨迹
  - 两种模式都对同一个种子做受控对比

运行：
    python code/ch003.py              # 两种模式各 12 万步，CPU 约 4-7 分钟
    python code/ch003.py --quick      # 快速跑通（约 40-70 秒）
    python code/ch003.py --modes clip # 只跑一种模式
    python code/ch003.py --target-kl 0.005   # 收紧 KL 目标的实验

预期：两种模式通常都能学会 CartPole；clip 模式的 KL 被间接压住，
kl 模式的 β 会先快速抬升再回落。KL 目标设得过小（如 0.002）时 kl 模式明显变慢。
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
    """固定随机种子：两种模式从同一条随机流出发，保证受控对比。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_tensor(obs, device):
    return torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)


# ---------------------------------------------------------------------------
# 网络（与第 001 / 002 章相同）
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
# GAE（与第 002 章相同）
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
# 两种更新目标
# ---------------------------------------------------------------------------
def ppo_update(net, optimizer, batch, args, mode: str, beta: float):
    """统一的更新函数，按 mode 选择 clip 或 kl 目标。

    返回 (stats, beta)：
      - clip 模式：beta 原样返回；stats 含裁剪比例与 KL
      - kl   模式：beta 按"KL 超过目标 1.5 倍则加倍、低于目标 1.5 分之一则减半"自适应；
                  策略损失使用未裁剪的 ratio·Â，另加 β·KL 惩罚
    """
    obs, act = batch["obs"], batch["act"]
    logp_old, adv, ret = batch["logp_old"], batch["adv"], batch["ret"]
    N = obs.shape[0]
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
             "approx_kl": 0.0, "max_kl": 0.0, "clip_frac": 0.0, "n_updates": 0}
    epoch_kl = 0.0
    epoch_cnt = 0

    for _ in range(args.epochs):
        idx = torch.randperm(N, device=obs.device)
        for start in range(0, N, args.batch_size):
            mb = idx[start:start + args.batch_size]
            logits, value = net(obs[mb])
            dist = Categorical(logits=logits)
            logp = dist.log_prob(act[mb])
            ratio = torch.exp(logp - logp_old[mb])

            # KL 估计：采样估计 E[logp_old - logp]，用于两种模式的诊断
            kl = (logp_old[mb] - logp).mean()

            if mode == "clip":
                # 裁剪目标：越界样本梯度归零
                surr1 = ratio * adv[mb]
                surr2 = torch.clamp(ratio, 1.0 - args.clip, 1.0 + args.clip) * adv[mb]
                policy_loss = -torch.min(surr1, surr2).mean()
            else:
                # KL 惩罚目标：不做裁剪，交给 β·KL 约束步长
                policy_loss = -(ratio * adv[mb]).mean() + beta * kl

            value_loss = F.mse_loss(value, ret[mb])
            entropy = dist.entropy().mean()
            loss = policy_loss + args.vf_coef * value_loss - args.ent_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            optimizer.step()

            with torch.no_grad():
                stats["policy_loss"] += policy_loss.item()
                stats["value_loss"] += value_loss.item()
                stats["entropy"] += entropy.item()
                stats["approx_kl"] += kl.item()
                stats["max_kl"] = max(stats["max_kl"], abs(kl.item()))
                stats["clip_frac"] += ((ratio - 1.0).abs() > args.clip).float().mean().item()
                stats["n_updates"] += 1
            epoch_kl += kl.item()
            epoch_cnt += 1

        # kl 模式：每个 epoch 结束按当前 KL 水平调整 β
        if mode == "kl":
            mean_epoch_kl = epoch_kl / max(epoch_cnt, 1)
            hi = args.target_kl * 1.5
            lo = args.target_kl / 1.5
            if mean_epoch_kl > hi:
                beta *= 2.0
            elif mean_epoch_kl < lo:
                beta /= 2.0
            beta = float(np.clip(beta, 1e-6, 1e6))
            epoch_kl = 0.0
            epoch_cnt = 0

    for k in stats:
        if k not in ("n_updates", "max_kl"):
            stats[k] /= max(stats["n_updates"], 1)
    stats["beta_final"] = beta
    return stats, beta


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
# 训练单个模式
# ---------------------------------------------------------------------------
def train_one(mode: str, args):
    """用指定目标模式训练一次，返回统计结果。"""
    set_seed(args.seed)
    device = torch.device("cpu")
    env = gym.make("CartPole-v1")
    obs, _ = env.reset(seed=args.seed)

    net = ActorCritic(env.observation_space.shape[0], env.action_space.n)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    beta = args.kl_init
    all_returns = []
    kls, betas = [], []
    total_steps, obs_t = 0, to_tensor(obs, device)
    solved_at = None
    t0 = time.time()

    print(f"\n[开始] 模式 = {mode}"
          + (f" | 初始 β = {beta}" if mode == "kl" else ""))
    while total_steps < args.max_steps:
        batch, ep_returns, obs_t = collect_rollout(
            env, net, args.steps_per_update, obs_t, device, args.gamma, args.lam)
        total_steps += args.steps_per_update
        all_returns.extend(ep_returns)

        stats, beta = ppo_update(net, optimizer, batch, args, mode, beta)
        kls.append(stats["approx_kl"])
        betas.append(beta)

        if len(all_returns) >= 20 and total_steps % (args.steps_per_update * 5) == 0:
            extra = (f"β {beta:8.4f}" if mode == "kl"
                     else f"裁剪比例 {stats['clip_frac']:.3f}")
            print(f"  [{mode:4}] 步数 {total_steps:7d} | 回合 {len(all_returns):4d} | "
                  f"近20均 {np.mean(all_returns[-20:]):7.1f} | "
                  f"KL {stats['approx_kl']:.4f} | 最大KL {stats['max_kl']:.4f} | {extra}")

        if solved_at is None and len(all_returns) >= 100 and np.mean(all_returns[-100:]) >= args.target_reward:
            solved_at = len(all_returns)

        if solved_at is not None and args.stop_at_solved:
            break

    eval_returns = evaluate(gym.make("CartPole-v1"), net, args.eval_episodes, device)
    tail = all_returns[-100:] if len(all_returns) >= 100 else all_returns
    return {
        "mode": mode,
        "steps": total_steps,
        "episodes": len(all_returns),
        "last100": float(np.mean(tail)),
        "eval": float(np.mean(eval_returns)),
        "solved_at": solved_at,
        "mean_kl": float(np.mean(kls)),
        "max_kl": float(np.max(kls)),
        "beta_final": float(betas[-1]),
        "beta_max": float(np.max(betas)) if betas else 0.0,
        "seconds": time.time() - t0,
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="第003章：PPO 裁剪目标与 KL 惩罚对比")
    p.add_argument("--modes", type=str, nargs="+", default=["clip", "kl"], choices=["clip", "kl"],
                   help="要对比的目标模式")
    p.add_argument("--max-steps", type=int, default=120000, help="每种模式的环境步数预算")
    p.add_argument("--steps-per-update", type=int, default=2048, help="每次更新前采样步数")
    p.add_argument("--epochs", type=int, default=10, help="每批数据的更新轮数")
    p.add_argument("--batch-size", type=int, default=64, help="小批量大小")
    p.add_argument("--lr", type=float, default=3e-4, help="学习率")
    p.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
    p.add_argument("--lam", type=float, default=0.95, help="GAE 的 lambda")
    p.add_argument("--clip", type=float, default=0.2, help="裁剪模式的范围 ε")
    p.add_argument("--target-kl", type=float, default=0.01, help="KL 模式的目标 KL")
    p.add_argument("--kl-init", type=float, default=1.0, help="KL 模式的初始 β")
    p.add_argument("--ent-coef", type=float, default=0.01, help="熵正则系数")
    p.add_argument("--vf-coef", type=float, default=0.5, help="价值损失系数")
    p.add_argument("--target-reward", type=float, default=475.0, help="达标阈值")
    p.add_argument("--stop-at-solved", action="store_true", default=True,
                   help="首次达标后提前结束该模式（默认开）")
    p.add_argument("--no-early-stop", dest="stop_at_solved", action="store_false",
                   help="关闭达标提前结束，跑满预算")
    p.add_argument("--eval-episodes", type=int, default=5, help="评估回合数")
    p.add_argument("--seed", type=int, default=0, help="随机种子")
    p.add_argument("--save-dir", type=str, default="runs/ch003", help="结果保存目录")
    p.add_argument("--quick", action="store_true", help="快速模式：8 千步、4 轮、无早停")
    return p.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.max_steps = 8000
        args.steps_per_update = 512
        args.epochs = 4
        args.target_reward = 300.0
        args.stop_at_solved = False

    print("=" * 84)
    print(f"裁剪目标 vs KL 惩罚 | 模式 {args.modes} | 预算 {args.max_steps} 步/模式 | 种子 {args.seed}")
    print(f"clip: ε={args.clip} | kl: target_kl={args.target_kl}, β0={args.kl_init}")
    print("=" * 84)

    results = []
    for mode in args.modes:
        res = train_one(mode, args)
        results.append(res)
        print(f"[完成] {mode}: 近100均 {res['last100']:.1f} | 评估 {res['eval']:.1f} | "
              f"平均KL {res['mean_kl']:.4f} 最大KL {res['max_kl']:.4f} | 用时 {res['seconds']:.1f}s")

    print("\n" + "=" * 84)
    print("对比总结（同种子、同数据规模）")
    print("-" * 84)
    header = (f"{'模式':>6} | {'总步数':>7} | {'首达475(回合)':>12} | {'近100均':>8} | "
              f"{'评估':>7} | {'平均KL':>8} | {'最大KL':>8} | {'β终值/裁剪':>12} | {'用时(s)':>8}")
    print(header)
    for r in results:
        solved = str(r["solved_at"]) if r["solved_at"] is not None else "未达标"
        last_col = (f"β={r['beta_final']:.3g}" if r["mode"] == "kl"
                    else f"ε={args.clip}")
        print(f"{r['mode']:>6} | {r['steps']:>7d} | {solved:>12} | {r['last100']:>8.1f} | "
              f"{r['eval']:>7.1f} | {r['mean_kl']:>8.4f} | {r['max_kl']:>8.4f} | "
              f"{last_col:>12} | {r['seconds']:>8.1f}")

    if any(r["mode"] == "kl" for r in results):
        r = next(r for r in results if r["mode"] == "kl")
        print(f"\nkl 模式 β 漂移：初值 {args.kl_init} → 终值 {r['beta_final']:.4g}，"
              f"过程中最大 {r['beta_max']:.4g}（β 被 KL 反馈推着走，这就是它的调参代价）")

    os.makedirs(args.save_dir, exist_ok=True)
    path = os.path.join(args.save_dir, "clip_vs_kl.txt")
    with open(path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(f"mode={r['mode']} steps={r['steps']} solved_at={r['solved_at']} "
                    f"last100={r['last100']:.2f} eval={r['eval']:.2f} mean_kl={r['mean_kl']:.5f} "
                    f"max_kl={r['max_kl']:.5f} beta_final={r['beta_final']:.4g}\n")
    print(f"结果已写入 {path}")


if __name__ == "__main__":
    main()
