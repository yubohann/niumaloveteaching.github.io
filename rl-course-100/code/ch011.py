"""
第011章 PPO的连续动作空间的高斯策略

离散策略用 Categorical（logits），连续策略要用高斯：π(a|s) = N(μ(s), σ)。
σ 怎么参数化是本章的对比点：
  - shared：整个网络共享一组可学习的 log σ（与状态无关，参数最少、最常用）
  - state ：log σ 也由状态决定（表达能力更强，但方差可能随状态失控）

在 Pendulum-v1（力矩 ∈ [−2, 2]）上做受控对比，动作边界处理支持两种方式：
  - clamp ：对采样动作直接裁剪（简单，边界处概率质量被折叠）
  - tanh  ：a = limit·tanh(u)，并在对数概率里加雅可比修正 log(limit·(1−tanh²u))
     （严格的无界高斯变换，边界处不会产生无穷密度）

记录：回合回报（Pendulum 全负，越接近 0 越好）、策略标准差轨迹、
高斯微分熵、KL 与达标（−250 分）步数。

运行：
    python code/ch011.py                   # 2 种 σ 参数化 × 15 万步，CPU 约 4-8 分钟
    python code/ch011.py --quick           # 快速跑通（约 30-60 秒）
    python code/ch011.py --arms shared
    python code/ch011.py --squash tanh     # 比较 tanh 变换

预期：Pendulum 上两种参数化都能学，state 版前期探索更大、后期更容易压小 σ；
tanh 变体在接近边界时训练更平滑，但会牺牲一部分探索幅度。
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
from torch.distributions import Normal


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_tensor(obs, device):
    return torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)


# ---------------------------------------------------------------------------
# 高斯 Actor-Critic
# ---------------------------------------------------------------------------
class GaussianActorCritic(nn.Module):
    """共享躯干；actor 输出均值，标准差按 arm 决定是否与状态相关。

    squash="clamp"：a = clip(u, ±limit)，logp = log N(u)
    squash="tanh" ：a = limit·tanh(u)，logp = log N(u) − log(limit·(1−tanh²u))
    """

    def __init__(self, obs_dim: int, act_dim: int, act_limit: float,
                 state_std: bool, hidden: int = 64, squash: str = "clamp"):
        super().__init__()
        self.act_limit = act_limit
        self.state_std = state_std
        self.squash = squash
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.actor_mean = nn.Linear(hidden, act_dim)
        if state_std:
            self.actor_log_std = nn.Linear(hidden, act_dim)
            # 小增益初始化：初始 log σ ≈ 0（σ≈1），避免一上来就过度探索
            nn.init.orthogonal_(self.actor_log_std.weight, gain=0.01)
            nn.init.zeros_(self.actor_log_std.bias)
        else:
            self.log_std = nn.Parameter(torch.zeros(act_dim))
        self.critic = nn.Linear(hidden, 1)

    def distribution(self, obs: torch.Tensor) -> Normal:
        h = self.body(obs)
        mean = self.actor_mean(h)
        if self.state_std:
            log_std = self.actor_log_std(h)
        else:
            log_std = self.log_std.expand_as(mean)
        log_std = torch.clamp(log_std, -5.0, 2.0)
        return Normal(mean, log_std.exp())

    def value_of(self, obs: torch.Tensor):
        h = self.body(obs)
        return self.critic(h).squeeze(-1)

    def log_prob_from_u(self, u: torch.Tensor, dist: Normal) -> torch.Tensor:
        """由预变换样本 u 计算（可能带 tanh 修正的）对数概率。"""
        logp = dist.log_prob(u)
        if self.squash == "tanh":
            corr = torch.log(self.act_limit * (1.0 - torch.tanh(u) ** 2) + 1e-6)
            logp = logp - corr
        return logp

    def transform(self, u: torch.Tensor) -> torch.Tensor:
        if self.squash == "tanh":
            return self.act_limit * torch.tanh(u)
        return torch.clamp(u, -self.act_limit, self.act_limit)

    def forward(self, obs: torch.Tensor):
        return self.distribution(obs), self.value_of(obs)

    @torch.no_grad()
    def act(self, obs: torch.Tensor):
        """返回 (环境动作, 对 logp, 价值, 预变换样本 u)。"""
        dist, value = self.forward(obs)
        u = dist.sample()
        logp = self.log_prob_from_u(u, dist)
        action = self.transform(u)
        return (action.squeeze(0).cpu().numpy().astype(np.float32),
                float(logp.item()), float(value.item()),
                u.squeeze(0).cpu().numpy().astype(np.float32))

    @torch.no_grad()
    def value(self, obs: torch.Tensor) -> float:
        return float(self.value_of(obs).item())

    @torch.no_grad()
    def mean_std(self, obs: torch.Tensor):
        dist = self.distribution(obs)
        return float(dist.mean.item()), float(dist.stddev.mean().item())


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
    obs_buf, act_u_buf, logp_buf = [], [], []
    rew_buf, val_buf, term_buf = [], [], []
    ep_returns = []

    ep_ret = 0.0
    obs_after = obs_t
    for _ in range(steps):
        a, logp, v, u = net.act(obs_t)
        next_obs, r, terminated, truncated, _ = env.step(a)

        obs_buf.append(obs_t.squeeze(0).cpu().numpy())
        act_u_buf.append(u)
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
        "u": torch.as_tensor(np.asarray(act_u_buf), dtype=torch.float32, device=device),
        "logp_old": torch.as_tensor(np.asarray(logp_buf), dtype=torch.float32, device=device),
        "adv": torch.as_tensor(advantages, dtype=torch.float32, device=device),
        "ret": torch.as_tensor(returns, dtype=torch.float32, device=device),
    }
    return batch, ep_returns, obs_t


# ---------------------------------------------------------------------------
# PPO 更新：用存储的预变换样本 u 重新计算 logp
# ---------------------------------------------------------------------------
def ppo_update(net, optimizer, batch, args):
    obs, u = batch["obs"], batch["u"]
    logp_old, adv, ret = batch["logp_old"], batch["adv"], batch["ret"]
    N = obs.shape[0]
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    stats = {"approx_kl": 0.0, "clip_frac": 0.0, "entropy": 0.0,
             "std_mean": 0.0, "n_updates": 0}
    for _ in range(args.epochs):
        idx = torch.randperm(N, device=obs.device)
        for start in range(0, N, args.batch_size):
            mb = idx[start:start + args.batch_size]
            dist, value = net.forward(obs[mb])
            logp = net.log_prob_from_u(u[mb], dist)

            ratio = torch.exp(logp - logp_old[mb])
            surr1 = ratio * adv[mb]
            surr2 = torch.clamp(ratio, 1.0 - args.clip, 1.0 + args.clip) * adv[mb]
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(value, ret[mb])
            entropy = dist.entropy().mean()      # 高斯微分熵（数值量纲：nats）
            loss = policy_loss + args.vf_coef * value_loss - args.ent_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            optimizer.step()

            with torch.no_grad():
                stats["approx_kl"] += (logp_old[mb] - logp).mean().item()
                stats["clip_frac"] += ((ratio - 1.0).abs() > args.clip).float().mean().item()
                stats["entropy"] += entropy.item()
                stats["std_mean"] += dist.stddev.mean().item()
                stats["n_updates"] += 1

    for k in stats:
        if k != "n_updates":
            stats[k] /= max(stats["n_updates"], 1)
    return stats


# ---------------------------------------------------------------------------
# 评估：确定性策略（直接取分布均值，再裁剪到动作范围）
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(env, net, episodes: int, device):
    returns = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            dist = net.distribution(to_tensor(obs, device))
            action = torch.clamp(dist.mean, -net.act_limit, net.act_limit)
            obs, r, terminated, truncated, _ = env.step(action.squeeze(0).cpu().numpy())
            ep_ret += r
            done = terminated or truncated
        returns.append(ep_ret)
    return returns


# ---------------------------------------------------------------------------
# 训练一个 arm
# ---------------------------------------------------------------------------
def train_one(arm: str, args):
    set_seed(args.seed)
    device = torch.device("cpu")
    env = gym.make("Pendulum-v1")
    obs, _ = env.reset(seed=args.seed)
    act_limit = float(env.action_space.high[0])

    net = GaussianActorCritic(env.observation_space.shape[0], env.action_space.shape[0],
                              act_limit, state_std=(arm == "state"), squash=args.squash)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    all_returns, stds, kls = [], [], []
    total_steps, n_updates, obs_t = 0, 0, to_tensor(obs, device)
    reached_at = None
    t0 = time.time()

    print(f"\n[开始] arm={arm} | squash={args.squash} | 动作上限 {act_limit}")
    while total_steps < args.max_steps:
        batch, ep_returns, obs_t = collect_rollout(
            env, net, args.steps_per_update, obs_t, device, args.gamma, args.lam)
        total_steps += args.steps_per_update
        n_updates += 1
        all_returns.extend(ep_returns)

        stats = ppo_update(net, optimizer, batch, args)
        stds.append(stats["std_mean"])
        kls.append(stats["approx_kl"])

        if reached_at is None and len(all_returns) >= 20 and np.mean(all_returns[-20:]) >= args.stage_reward:
            reached_at = total_steps

        if n_updates % 5 == 0 and len(all_returns) >= 5:
            print(f"  [{arm:6}] 步数 {total_steps:7d} | 回合 {len(all_returns):4d} | "
                  f"近20均 {np.mean(all_returns[-20:]):8.1f} | σ {stats['std_mean']:.3f} | "
                  f"KL {stats['approx_kl']:.4f} | 裁剪 {stats['clip_frac']:.3f}")

    eval_returns = evaluate(gym.make("Pendulum-v1"), net, args.eval_episodes, device)
    tail = all_returns[-20:]
    return {
        "arm": arm,
        "steps": total_steps,
        "reached_at": reached_at,
        "last20": float(np.mean(tail)),
        "best20": float(max(np.mean(all_returns[i:i + 20]) for i in range(0, max(len(all_returns) - 20, 1), 20)))
        if len(all_returns) >= 20 else float(np.mean(tail)),
        "eval": float(np.mean(eval_returns)),
        "std_first": float(np.mean(stds[:5])),
        "std_last": float(np.mean(stds[-5:])),
        "mean_kl": float(np.mean(kls)),
        "seconds": time.time() - t0,
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="第011章：PPO 连续动作空间的高斯策略")
    p.add_argument("--arms", type=str, nargs="+", default=["shared", "state"],
                   choices=["shared", "state"], help="σ 参数化方式")
    p.add_argument("--squash", type=str, default="clamp", choices=["clamp", "tanh"],
                   help="动作边界处理方式")
    p.add_argument("--max-steps", type=int, default=150000, help="每个 arm 的环境步数预算")
    p.add_argument("--steps-per-update", type=int, default=2048, help="每次更新前采样步数")
    p.add_argument("--epochs", type=int, default=10, help="每批数据的更新轮数")
    p.add_argument("--batch-size", type=int, default=64, help="minibatch 大小")
    p.add_argument("--lr", type=float, default=3e-4, help="学习率")
    p.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
    p.add_argument("--lam", type=float, default=0.95, help="GAE 的 lambda")
    p.add_argument("--clip", type=float, default=0.2, help="PPO 裁剪范围")
    p.add_argument("--ent-coef", type=float, default=0.0, help="熵正则系数（连续控制常用 0）")
    p.add_argument("--vf-coef", type=float, default=0.5, help="价值损失系数")
    p.add_argument("--stage-reward", type=float, default=-250.0, help="阶段目标（20 回合均值）")
    p.add_argument("--eval-episodes", type=int, default=10, help="评估回合数")
    p.add_argument("--seed", type=int, default=0, help="随机种子")
    p.add_argument("--save-dir", type=str, default="runs/ch011", help="模型保存目录")
    p.add_argument("--quick", action="store_true", help="快速模式：2 个 arm、各 1.2 万步")
    return p.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.max_steps = 12000
        args.steps_per_update = 512
        args.epochs = 4
        args.stage_reward = -900.0

    print("=" * 96)
    print(f"高斯策略参数化对比 | arms {args.arms} | squash {args.squash} | "
          f"每个 {args.max_steps} 步 | 种子 {args.seed}")
    print("环境：Pendulum-v1（回合 200 步，回报越接近 0 越好；随机策略约 −1200）")
    print("=" * 96)

    results = []
    for arm in args.arms:
        res = train_one(arm, args)
        results.append(res)
        print(f"[完成] {arm}: 近20均 {res['last20']:.1f} | 贪婪评估 {res['eval']:.1f} | "
              f"σ {res['std_first']:.3f}→{res['std_last']:.3f} | 用时 {res['seconds']:.1f}s")

    print("\n" + "=" * 96)
    print("对比总结（同种子、同步数预算）")
    print("-" * 96)
    print(f"{'arm':>7} | {'首达-250步数':>11} | {'近20均':>9} | {'最佳20均':>9} | {'贪婪评估':>8} | "
          f"{'σ(前5轮)':>9} | {'σ(后5轮)':>9} | {'平均KL':>8} | {'用时(s)':>8}")
    for r in results:
        reached = str(r["reached_at"]) if r["reached_at"] is not None else "未达到"
        print(f"{r['arm']:>7} | {reached:>11} | {r['last20']:>9.1f} | {r['best20']:>9.1f} | "
              f"{r['eval']:>8.1f} | {r['std_first']:>9.3f} | {r['std_last']:>9.3f} | "
              f"{r['mean_kl']:>8.4f} | {r['seconds']:>8.1f}")

    print("\n观察提示：")
    print("  - shared 的 σ 是全网络唯一的一组参数，变化慢但稳；")
    print("  - state 的 σ 随状态变化：在困难状态保持探索、在熟悉状态收紧；")
    print("  - 熵项系数为 0 时 σ 完全由回报驱动收缩，观察 σ 轨迹更直观；")
    print("  - tanh 变换通过夹断前变换把动作挤进边界，需要雅可比修正确保 logp 正确。")

    os.makedirs(args.save_dir, exist_ok=True)
    path = os.path.join(args.save_dir, f"gauss_{args.squash}.txt")
    with open(path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(f"arm={r['arm']} squash={args.squash} reached={r['reached_at']} "
                    f"last20={r['last20']:.2f} best20={r['best20']:.2f} eval={r['eval']:.2f} "
                    f"std={r['std_first']:.3f}->{r['std_last']:.3f} kl={r['mean_kl']:.5f}\n")
    print(f"\n结果已写入 {path}")


if __name__ == "__main__":
    main()
