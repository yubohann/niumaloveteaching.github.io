"""
第002章 PPO的广义优势估计（GAE）实现

本章把 PPO 里的优势估计单独拿出来做实验：
  - 用"定义式（穷举未来项的显式求和）"对递推实现做数值自检，确保 GAE 写对
  - 在完全相同的训练流程下扫描 GAE 的 λ ∈ {0.0, 0.8, 0.95, 1.0}
    （λ=0 退化为单步 TD 残差，λ=1 退化为蒙特卡洛回报减基线）
  - 每个 λ 记录：优势的原始标准差（方差代理量）、样本效率、最终回报

运行：
    python code/ch002.py             # 完整实验：4 个 λ，CPU 约 3-6 分钟
    python code/ch002.py --quick     # 快速跑通（约 30-60 秒）
    python code/ch002.py --lams 0.95 # 只跑单个 λ
    python code/ch002.py --no-check  # 跳过数值自检（一般不需要）

预期：所有 λ 都能学会 CartPole，但 λ=0 前期慢、优势方差最小；
λ=1 早期涨得快、后期滑窗波动大；λ∈[0.9, 0.95] 通常最稳。
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
    """固定随机种子，保证不同 λ 的对比是受控实验。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_tensor(obs, device):
    """把 gym 返回的 numpy 观测转成 (1, obs_dim) 的 float32 张量。"""
    return torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)


# ---------------------------------------------------------------------------
# 网络：共享躯干 + 策略头 + 价值头（与第 001 章相同）
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
        """返回 (动作, 采样时刻的 log 概率, 状态价值)。"""
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action).item(), value.item()

    @torch.no_grad()
    def value(self, obs: torch.Tensor) -> float:
        _, v = self.forward(obs)
        return v.item()


# ---------------------------------------------------------------------------
# GAE：递推实现 + 定义式参考实现（用于自检）
# ---------------------------------------------------------------------------
def compute_gae(rewards, values, terminateds, last_value, gamma: float, lam: float):
    """从后向前递推 GAE(γ, λ)。

    递推式：
        δ_t = r_t + γ (1-d_t) V(s_{t+1}) - V(s_t)
        Â_t = δ_t + γ λ (1-d_t) Â_{t+1}

    其中 d_t 为真实终止标志（截断不算终止，仍需 bootstrap）。
    """
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


def gae_reference(rewards, values, terminateds, last_value, gamma: float, lam: float):
    """GAE 的"定义式"参考实现：对每个 t 显式累加未来所有 δ。

    Â_t = Σ_l (γλ)^l δ_{t+l}，遇到真实终止后所有后续项为 0。
    复杂度 O(T²)，只用于离线自检，不进训练循环。
    """
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float64)
    for t in range(T):
        acc = 0.0
        coeff = 1.0
        for l in range(T - t):
            i = t + l
            next_value = last_value if i == T - 1 else values[i + 1]
            non_terminal = 1.0 - terminateds[i]
            delta = rewards[i] + gamma * next_value * non_terminal - values[i]
            acc += coeff * delta
            coeff *= gamma * lam * non_terminal
        adv[t] = acc
    return adv


def self_test_gae(verbose: bool = True) -> bool:
    """在固定合成轨迹上比较递推实现与定义式实现，并检查两个退化情形。

    检查点：
      1) 随机轨迹上，递推结果与定义式结果最大误差 < 1e-6；
      2) λ=1 时 Â_t 应等于蒙特卡洛回报 G_t 减去 V(s_t)；
      3) λ=0 时 Â_t 应等于单步 TD 残差 δ_t。
    """
    rng = np.random.default_rng(123)
    T = 8
    rewards = rng.normal(0.0, 0.5, T).astype(np.float32)
    values = rng.normal(0.0, 1.0, T).astype(np.float32)
    terminateds = np.zeros(T, dtype=np.float32)
    terminateds[5] = 1.0          # 第 5 步发生真实终止
    gamma, lam, last_v = 0.99, 0.9, 0.7

    adv_rec, _ = compute_gae(rewards, values, terminateds, last_v, gamma, lam)
    adv_ref = gae_reference(rewards, values, terminateds, last_v, gamma, lam)
    err1 = float(np.max(np.abs(adv_rec - adv_ref)))

    # λ=1 → 蒙特卡洛：手工计算每个时刻的折扣回报
    adv_mc, _ = compute_gae(rewards, values, terminateds, last_v, 1.0, 1.0)
    mc_target = np.zeros(T)
    running = last_v
    for t in reversed(range(T)):
        running = rewards[t] + gamma * running * (1.0 - terminateds[t])
        mc_target[t] = running - values[t]
    err2 = float(np.max(np.abs(adv_mc - mc_target)))

    # λ=0 → 单步 TD 残差
    adv_td, _ = compute_gae(rewards, values, terminateds, last_v, gamma, 0.0)
    td_target = np.zeros(T)
    for t in range(T):
        nv = last_v if t == T - 1 else values[t + 1]
        td_target[t] = rewards[t] + gamma * nv * (1.0 - terminateds[t]) - values[t]
    err3 = float(np.max(np.abs(adv_td - td_target)))

    ok = err1 < 1e-6 and err2 < 1e-5 and err3 < 1e-6
    if verbose:
        print("GAE 数值自检：")
        print(f"  [{'PASS' if err1 < 1e-6 else 'FAIL'}] 递推 vs 定义式   最大误差 {err1:.2e}")
        print(f"  [{'PASS' if err2 < 1e-5 else 'FAIL'}] λ=1  vs 蒙特卡洛 最大误差 {err2:.2e}")
        print(f"  [{'PASS' if err3 < 1e-6 else 'FAIL'}] λ=0  vs 单步TD   最大误差 {err3:.2e}")
    return ok


# ---------------------------------------------------------------------------
# 采样一条 rollout
# ---------------------------------------------------------------------------
def collect_rollout(env, net, steps, obs_t, device, gamma, lam):
    """采样 steps 步，返回 batch（含优势与回报目标）、完成的回合回报、最新观测。"""
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
        term_buf.append(float(terminated))   # 截断不计入终止

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
        # 标准化之前的优势尺度：作为 GAE 方差的可观测代理量
        "adv_raw_std": float(np.std(advantages)),
        "adv_raw_mean": float(np.mean(advantages)),
    }
    return batch, ep_returns, obs_t


# ---------------------------------------------------------------------------
# PPO 更新（与第 001 章一致）
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
# 评估：确定性（贪婪）策略
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
# 训练单个 λ
# ---------------------------------------------------------------------------
def train_one(lam: float, args):
    """在固定预算下训练一次 GAE(λ)，返回统计结果。"""
    set_seed(args.seed)                    # 所有 λ 用同一随机流，受控对比
    env = gym.make("CartPole-v1")
    obs, _ = env.reset(seed=args.seed)

    net = ActorCritic(env.observation_space.shape[0], env.action_space.n)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    device = torch.device("cpu")

    all_returns, adv_stds, kl_list = [], [], []
    total_steps, obs_t = 0, to_tensor(obs, device)
    solved_at = None
    t0 = time.time()

    snapshots = []   # 每 10 次更新记录一次滑窗回报，供曲线对比
    while total_steps < args.max_steps:
        batch, ep_returns, obs_t = collect_rollout(
            env, net, args.steps_per_update, obs_t, device, args.gamma, lam)
        total_steps += args.steps_per_update
        all_returns.extend(ep_returns)

        stats = ppo_update(net, optimizer, batch, args)
        adv_stds.append(batch["adv_raw_std"])
        kl_list.append(stats["approx_kl"])

        if len(all_returns) >= 20:
            snapshots.append((total_steps, float(np.mean(all_returns[-20:]))))

        if solved_at is None and len(all_returns) >= 100 and np.mean(all_returns[-100:]) >= args.target_reward:
            solved_at = len(all_returns)

        if len(all_returns) >= 20 and total_steps % (args.steps_per_update * 10) == 0:
            print(f"  λ={lam:<4} 步数 {total_steps:7d} | 回合 {len(all_returns):4d} | "
                  f"近20均 {np.mean(all_returns[-20:]):7.1f} | 优势std {batch['adv_raw_std']:6.2f} | "
                  f"KL {stats['approx_kl']:.4f}")

    eval_returns = evaluate(gym.make("CartPole-v1"), net, args.eval_episodes, device)
    tail = all_returns[-100:] if len(all_returns) >= 100 else all_returns
    return {
        "lam": lam,
        "steps": total_steps,
        "episodes": len(all_returns),
        "last100": float(np.mean(tail)),
        "eval": float(np.mean(eval_returns)),
        "solved_at": solved_at,
        "adv_std": float(np.mean(adv_stds)),
        "kl": float(np.mean(kl_list)),
        "seconds": time.time() - t0,
        "snapshots": snapshots,
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="第002章：PPO 的广义优势估计（GAE）实现")
    p.add_argument("--lams", type=float, nargs="+", default=[0.0, 0.8, 0.95, 1.0],
                   help="要对比的 GAE λ 列表")
    p.add_argument("--max-steps", type=int, default=60000, help="每个 λ 的环境步数预算")
    p.add_argument("--steps-per-update", type=int, default=1024, help="每次更新前采样步数")
    p.add_argument("--epochs", type=int, default=8, help="每批数据的更新轮数")
    p.add_argument("--batch-size", type=int, default=64, help="小批量大小")
    p.add_argument("--lr", type=float, default=3e-4, help="学习率")
    p.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
    p.add_argument("--clip", type=float, default=0.2, help="PPO 裁剪范围")
    p.add_argument("--ent-coef", type=float, default=0.01, help="熵正则系数")
    p.add_argument("--vf-coef", type=float, default=0.5, help="价值损失系数")
    p.add_argument("--target-reward", type=float, default=475.0, help="达标阈值（仅记录，提前不终止）")
    p.add_argument("--eval-episodes", type=int, default=5, help="评估回合数")
    p.add_argument("--seed", type=int, default=0, help="随机种子")
    p.add_argument("--save-dir", type=str, default="runs/ch002", help="模型保存目录")
    p.add_argument("--no-check", action="store_true", help="跳过启动时的 GAE 数值自检")
    p.add_argument("--quick", action="store_true", help="快速模式：2 个 λ、每个 8000 步")
    return p.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.lams = [0.0, 0.95]
        args.max_steps = 8000
        args.steps_per_update = 512
        args.epochs = 4
        args.target_reward = 300.0

    if not args.no_check:
        ok = self_test_gae(verbose=True)
        if not ok:
            raise RuntimeError("GAE 自检失败，请先修正 compute_gae 再训练")
        print()

    print("=" * 78)
    print(f"GAE λ 扫描实验 | λ={args.lams} | 每个 λ 预算 {args.max_steps} 步 | 种子 {args.seed}")
    print("=" * 78)

    results = []
    for lam in args.lams:
        print(f"\n[开始] λ = {lam}")
        res = train_one(lam, args)
        results.append(res)
        print(f"[完成] λ={lam}: 近100均 {res['last100']:.1f} | 贪婪评估 {res['eval']:.1f} | "
              f"平均优势std {res['adv_std']:.2f} | 用时 {res['seconds']:.1f}s")

    print("\n" + "=" * 78)
    print("对比总结（同一随机种子，同一预算）")
    print("-" * 78)
    print(f"{'λ':>6} | {'总步数':>7} | {'回合数':>6} | {'近100均':>8} | {'贪婪评估':>8} | "
          f"{'首达100回合':>10} | {'平均优势std':>10} | {'用时(s)':>8}")
    for r in results:
        solved = f"{r['solved_at']:>10d}" if r["solved_at"] is not None else f"{'未达标':>10}"
        print(f"{r['lam']:>6} | {r['steps']:>7d} | {r['episodes']:>6d} | {r['last100']:>8.1f} | "
              f"{r['eval']:>8.1f} | {solved} | {r['adv_std']:>10.2f} | {r['seconds']:>8.1f}")

    print("\n每 10 次更新的近20回合滑窗均值（步数: 回报）")
    for r in results:
        pts = r["snapshots"][::10][:6]
        curve = " ".join(f"{s}:{v:.0f}" for s, v in pts)
        print(f"  λ={r['lam']:<4} {curve}")

    os.makedirs(args.save_dir, exist_ok=True)
    summary_path = os.path.join(args.save_dir, "lambda_sweep.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(f"lam={r['lam']} steps={r['steps']} last100={r['last100']:.2f} "
                    f"eval={r['eval']:.2f} adv_std={r['adv_std']:.3f} solved_at={r['solved_at']}\n")
    print(f"\n结果已写入 {summary_path}（模型未保存：本实验只做 λ 对比）")


if __name__ == "__main__":
    main()
