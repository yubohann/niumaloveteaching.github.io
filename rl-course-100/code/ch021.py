"""
第021章 PPO的探索策略：参数空间噪声

两种探索机制在同一任务（CartPole-v1）上对比（同种子、同预算）：
  - action：动作空间噪声——标准随机策略每步独立采样（PPO 默认）
  - param ：参数空间噪声（Plappert 等 2018）——采样前给权重加高斯扰动
            θ̃ = θ + σ·N(0, I)，用 θ̃ 采集整个批次；采完恢复 θ 并重算 logp_old；
            σ 用"扰动策略与原始策略的平均 KL"自适应调整到目标 δ 附近

记录：首达 400 的步数、评估回报、状态覆盖度（离散桶的独立访问数）、
扰动 KL 与 σ 轨迹、策略熵与 KL。

运行：
    python code/ch021.py            # 2 种探索，各 8 万步，CPU 约 3-6 分钟
    python code/ch021.py --quick    # 快速跑通（约 30-60 秒）
    python code/ch021.py --arms param --delta 0.02

预期：param 臂在早期访问到更"连贯"的行为（σ 自适应到接近目标 KL），
状态覆盖度分布不同；两种机制最终都能学会 CartPole，样本效率互有胜负。
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
    def logits_of(self, obs: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward(obs)
        return logits

    @torch.no_grad()
    def value(self, obs: torch.Tensor) -> float:
        _, v = self.forward(obs)
        return v.item()


# ---------------------------------------------------------------------------
# 参数噪声工具
# ---------------------------------------------------------------------------
def perturb_parameters(net, sigma: float):
    """给所有权重加 σ·N(0,1) 噪声，返回原始参数的克隆。"""
    originals = []
    with torch.no_grad():
        for p in net.parameters():
            originals.append(p.data.clone())
            p.data.add_(sigma * torch.randn_like(p))
    return originals


def restore_parameters(net, originals):
    with torch.no_grad():
        for p, o in zip(net.parameters(), originals):
            p.data.copy_(o)


def categorical_kl(logits_a: torch.Tensor, logits_b: torch.Tensor) -> float:
    """KL(softmax(a) ‖ softmax(b)) 的批均值。"""
    logp_a = F.log_softmax(logits_a, dim=-1)
    logp_b = F.log_softmax(logits_b, dim=-1)
    return float((logp_a.exp() * (logp_a - logp_b)).sum(dim=-1).mean().item())


# ---------------------------------------------------------------------------
# gated 状态覆盖度：把观测离散到 4 维桶，统计独立桶数
# ---------------------------------------------------------------------------
STATE_BINS = [(-2.4, 2.4), (-3.0, 3.0), (-0.21, 0.21), (-3.0, 3.0)]


def state_coverage(obs_array: np.ndarray, n_bins: int = 6) -> int:
    keys = set()
    for obs in obs_array:
        key = []
        for x, (lo, hi) in zip(obs, STATE_BINS):
            b = int(np.clip((x - lo) / (hi - lo) * n_bins, 0, n_bins - 1))
            key.append(b)
        keys.add(tuple(key))
    return len(keys)


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
# 采样：param 臂在整批采样前扰动、采样后恢复
# ---------------------------------------------------------------------------
def collect_rollout(env, net, steps, obs_t, device, gamma, lam, perturb_sigma):
    originals = None
    if perturb_sigma > 0.0:
        originals = perturb_parameters(net, perturb_sigma)

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

    if originals is not None:
        # 用扰动权重评估价值 bootstrap 会引入偏差：先恢复再算
        restore_parameters(net, originals)

    last_value = net.value(obs_after)
    advantages, returns = compute_gae(rew_buf, val_buf, term_buf, last_value, gamma, lam)

    obs_tensor = torch.as_tensor(np.asarray(obs_buf), dtype=torch.float32, device=device)
    act_tensor = torch.as_tensor(np.asarray(act_buf), dtype=torch.long, device=device)

    if originals is not None:
        # 关键：用恢复后的策略重算 logp_old（Plappert 等 2018 的标准做法）
        with torch.no_grad():
            logits = net.logits_of(obs_tensor)
            dist = Categorical(logits=logits)
            logp_old = dist.log_prob(act_tensor).cpu().numpy()
    else:
        logp_old = np.asarray(logp_buf, dtype=np.float32)

    # 注：扰动策略与原始策略的 KL 在 train_one 里用"当前参数 + 重新扰动"估计，
    # 采样函数只负责把合法的 logp_old 交给更新。
    batch = {
        "obs": obs_tensor,
        "act": act_tensor,
        "logp_old": torch.as_tensor(logp_old, dtype=torch.float32, device=device),
        "adv": torch.as_tensor(advantages, dtype=torch.float32, device=device),
        "ret": torch.as_tensor(returns, dtype=torch.float32, device=device),
    }
    return batch, ep_returns, obs_t


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
# 评估：贪婪策略（不扰动）
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
# 训练一个 arm
# ---------------------------------------------------------------------------
def train_one(arm: str, args):
    use_param = (arm == "param")
    set_seed(args.seed)
    device = torch.device("cpu")
    env = gym.make("CartPole-v1")
    obs, _ = env.reset(seed=args.seed)
    obs_t = to_tensor(obs, device)

    net = ActorCritic(env.observation_space.shape[0], env.action_space.n)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    sigma = args.sigma_init
    all_returns, kls, sigmas, kls_pert, coverages = [], [], [], [], []
    total_steps, n_updates = 0, 0
    reached_at = None
    t0 = time.time()

    print(f"\n[开始] arm={arm} | 参数噪声 {use_param} | σ0={sigma} | 目标 KL δ={args.delta}")
    while total_steps < args.max_steps:
        batch, ep_returns, obs_t = collect_rollout(
            env, net, args.steps_per_update, obs_t, device, args.gamma, args.lam,
            sigma if use_param else 0.0)
        total_steps += args.steps_per_update
        n_updates += 1
        all_returns.extend(ep_returns)
        coverages.append(state_coverage(batch["obs"].cpu().numpy()))

        stats = ppo_update(net, optimizer, batch, args)
        kls.append(stats["approx_kl"])

        if use_param:
            # 在同一批观测上重新扰动一次，估计 KL(π_perturbed ‖ π_base)，用于 σ 自适应
            originals = perturb_parameters(net, sigma)
            with torch.no_grad():
                logits_pert = net.logits_of(batch["obs"])
            restore_parameters(net, originals)
            with torch.no_grad():
                logits_base = net.logits_of(batch["obs"])
            kl_pert_now = categorical_kl(logits_pert, logits_base)
            kls_pert.append(kl_pert_now)
            if kl_pert_now < args.delta / 1.5:
                sigma = min(sigma * 1.01, args.sigma_max)
            elif kl_pert_now > args.delta * 1.5:
                sigma = max(sigma / 1.01, args.sigma_min)
            sigmas.append(sigma)

        if reached_at is None and len(all_returns) >= 100 and np.mean(all_returns[-100:]) >= args.target_reward:
            reached_at = total_steps
            print(f"  [{arm}] 达到 {args.target_reward}：步数 {reached_at}")

        if n_updates % 5 == 0 and len(all_returns) >= 20:
            extra = (f"σ {sigma:.4f} | 扰动KL {kls_pert[-1]:.4f}" if use_param else "")
            print(f"  [{arm:6}] 步数 {total_steps:6d} | 回合 {len(all_returns):4d} | "
                  f"近20均 {np.mean(all_returns[-20:]):7.1f} | 覆盖桶 {coverages[-1]:3d} | "
                  f"KL {stats['approx_kl']:.4f} | {extra}")

    eval_returns = evaluate(gym.make("CartPole-v1"), net, args.eval_episodes, device)
    tail = all_returns[-100:] if len(all_returns) >= 100 else all_returns
    return {
        "arm": arm,
        "steps": total_steps,
        "reached_at": reached_at,
        "last100": float(np.mean(tail)),
        "eval": float(np.mean(eval_returns)),
        "coverage": float(np.mean(coverages)),
        "mean_kl": float(np.mean(kls)),
        "sigma_first": float(np.mean(sigmas[:5])) if sigmas else 0.0,
        "sigma_last": float(np.mean(sigmas[-5:])) if sigmas else 0.0,
        "kl_pert_mean": float(np.mean(kls_pert)) if kls_pert else 0.0,
        "seconds": time.time() - t0,
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="第021章：PPO 参数空间噪声探索")
    p.add_argument("--arms", type=str, nargs="+", default=["action", "param"],
                   choices=["action", "param"], help="探索机制")
    p.add_argument("--delta", type=float, default=0.01, help="参数噪声的目标 KL")
    p.add_argument("--sigma-init", type=float, default=0.01, help="参数噪声初始 σ")
    p.add_argument("--sigma-min", type=float, default=1e-4, help="σ 下限")
    p.add_argument("--sigma-max", type=float, default=0.1, help="σ 上限")
    p.add_argument("--max-steps", type=int, default=80000, help="每个 arm 的环境步数预算")
    p.add_argument("--steps-per-update", type=int, default=2048, help="每次更新前采样步数")
    p.add_argument("--epochs", type=int, default=10, help="每批数据的更新轮数")
    p.add_argument("--batch-size", type=int, default=64, help="minibatch 大小")
    p.add_argument("--lr", type=float, default=3e-4, help="学习率")
    p.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
    p.add_argument("--lam", type=float, default=0.95, help="GAE 的 lambda")
    p.add_argument("--clip", type=float, default=0.2, help="PPO 裁剪范围")
    p.add_argument("--ent-coef", type=float, default=0.01, help="熵正则系数")
    p.add_argument("--vf-coef", type=float, default=0.5, help="价值损失系数")
    p.add_argument("--target-reward", type=float, default=400.0, help="阶段目标（100 回合滑窗）")
    p.add_argument("--eval-episodes", type=int, default=10, help="评估回合数")
    p.add_argument("--seed", type=int, default=0, help="随机种子")
    p.add_argument("--save-dir", type=str, default="runs/ch021", help="模型保存目录")
    p.add_argument("--quick", action="store_true", help="快速模式：2 个 arm、各 8 千步")
    return p.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.max_steps = 8000
        args.steps_per_update = 512
        args.epochs = 4
        args.target_reward = 300.0
        args.eval_episodes = 5

    print("=" * 96)
    print(f"探索机制对比 | arms {args.arms} | 每个 {args.max_steps} 步 | 种子 {args.seed}")
    print("param：θ̃ = θ + σ·N(0,I) 采整批 → 恢复 θ → 重算 logp_old → 正常更新；")
    print(f"       σ 自适应到目标 KL δ={args.delta}，σ ∈ [{args.sigma_min}, {args.sigma_max}]")
    print("=" * 96)

    results = []
    for arm in args.arms:
        res = train_one(arm, args)
        results.append(res)
        print(f"[完成] {arm}: 近100均 {res['last100']:.1f} | 评估 {res['eval']:.1f} | "
              f"覆盖桶均 {res['coverage']:.1f} | 用时 {res['seconds']:.1f}s")

    print("\n" + "=" * 96)
    print("对比总结（同种子、同步数预算、同超参数）")
    print("-" * 96)
    print(f"{'arm':>7} | {'首达400步数':>11} | {'近100均':>8} | {'评估':>7} | "
          f"{'覆盖桶(均)':>10} | {'平均KL':>8} | {'σ初':>7} | {'σ终':>7} | "
          f"{'扰动KL(均)':>10} | {'用时(s)':>8}")
    for r in results:
        reached = str(r["reached_at"]) if r["reached_at"] is not None else "未达到"
        print(f"{r['arm']:>7} | {reached:>11} | {r['last100']:>8.1f} | {r['eval']:>7.1f} | "
              f"{r['coverage']:>10.1f} | {r['mean_kl']:>8.4f} | {r['sigma_first']:>7.4f} | "
              f"{r['sigma_last']:>7.4f} | {r['kl_pert_mean']:>10.4f} | {r['seconds']:>8.1f}")

    print("\n观察提示：")
    print("  - 覆盖桶 = 每批观测离散到 6×6×6×6 的空间中被访问过的桶数，衡量探索多样性；")
    print("  - param 臂 σ 会自适应到使扰动 KL 接近 δ：太大则 σ 收缩，太小则 σ 增大；")
    print("  - action 臂每步独立采样，扰动 KL 列没有意义（显示 0）。")

    os.makedirs(args.save_dir, exist_ok=True)
    path = os.path.join(args.save_dir, "exploration.txt")
    with open(path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(f"arm={r['arm']} reached={r['reached_at']} last100={r['last100']:.2f} "
                    f"eval={r['eval']:.2f} coverage={r['coverage']:.2f} kl={r['mean_kl']:.5f} "
                    f"sigma_first={r['sigma_first']:.5f} sigma_last={r['sigma_last']:.5f} "
                    f"kl_pert={r['kl_pert_mean']:.5f} seconds={r['seconds']:.1f}\n")
    print(f"\n结果已写入 {path}")


if __name__ == "__main__":
    main()
