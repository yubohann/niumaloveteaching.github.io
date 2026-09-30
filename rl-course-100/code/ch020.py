"""
第020章 PPO的稀疏奖励处理

MountainCar-v0 是稀疏奖励的经典代表：每步固定 −1、只有把车开上山坡（位置≥0.5）
才会终止并停止扣分。策略必须学会"先反向积累动能再冲坡"的迂回解法，
纯 −1 的奖励几乎不给出"哪一步更接近目标"的信号。

对比两种奖励设置（同种子、同步数预算）：
  - baseline：原始奖励（每步 −1）
  - shaping ：势能塑形（Potential-Based Reward Shaping）：
        Φ(s) = position(s)
        r' = r + η · ( γ·Φ(s') − Φ(s) )
      势能差保证最优策略不变（Ng 等 1999），η 控制塑形强度。

记录：成功率（近100回合，terminated 视为成功）、100 回合滑窗回报、
首达 90% 成功率步数、评估成功率与用时。

运行：
    python code/ch020.py            # 2 种设置，各 20 万步，CPU 约 4-8 分钟
    python code/ch020.py --quick    # 快速跑通（约 30-60 秒）
    python code/ch020.py --arms shaping --eta 200

预期：baseline 在 20 万步内很少稳定成功；shaping 通常在 8 万～15 万步内把成功率推过 90%，
但塑形回报的数值与原始回报不可直接比较（见正文说明）。
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
# 势能塑形
# ---------------------------------------------------------------------------
def potential(obs: np.ndarray, kind: str) -> float:
    """Φ(s)：position 用位置；height 用 sin(3·position)（坡的高度）。"""
    if kind == "position":
        return float(obs[0])
    if kind == "height":
        return float(np.sin(3.0 * obs[0]))
    raise ValueError(f"未知的势能类型: {kind}")


# ---------------------------------------------------------------------------
# 采样：可选势能塑形
# ---------------------------------------------------------------------------
def collect_rollout(env, net, steps, obs_t, device, gamma, lam, use_shaping, eta, phi_kind):
    obs_buf, act_buf, logp_buf = [], [], []
    rew_buf, val_buf, term_buf = [], [], []
    ep_returns = []
    ep_success = []

    ep_ret = 0.0
    success_flag = False
    prev_obs = obs_t.squeeze(0).cpu().numpy()
    obs_after = obs_t
    for _ in range(steps):
        a, logp, v = net.act(obs_t)
        next_obs, r, terminated, truncated, _ = env.step(a)

        r_used = r
        if use_shaping:
            r_used = r + eta * (gamma * potential(next_obs, phi_kind) - potential(prev_obs, phi_kind))

        obs_buf.append(obs_t.squeeze(0).cpu().numpy())
        act_buf.append(a)
        logp_buf.append(logp)
        rew_buf.append(r_used)
        val_buf.append(v)
        term_buf.append(float(terminated))

        ep_ret += r
        success_flag = success_flag or bool(terminated)
        prev_obs = np.asarray(next_obs, dtype=np.float64)
        obs_t = to_tensor(next_obs, device)
        obs_after = obs_t

        if terminated or truncated:
            ep_returns.append(ep_ret)
            ep_success.append(1.0 if success_flag else 0.0)
            ep_ret, success_flag = 0.0, False
            reset_obs, _ = env.reset()
            prev_obs = np.asarray(reset_obs, dtype=np.float64)
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
    return batch, ep_returns, ep_success, obs_t


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
# 评估：贪婪策略，统计成功率与原始回报
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(env, net, episodes: int, device):
    returns, success = [], []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_ret = 0.0
        ok = False
        while not done:
            logits, _ = net(to_tensor(obs, device))
            action = int(torch.argmax(logits, dim=-1).item())
            obs, r, terminated, truncated, _ = env.step(action)
            ep_ret += r
            ok = ok or bool(terminated)
            done = terminated or truncated
        returns.append(ep_ret)
        success.append(1.0 if ok else 0.0)
    return float(np.mean(returns)), float(np.mean(success))


# ---------------------------------------------------------------------------
# 训练一个 arm
# ---------------------------------------------------------------------------
def train_one(arm: str, args):
    use_shaping = (arm == "shaping")
    set_seed(args.seed)
    device = torch.device("cpu")
    env = gym.make("MountainCar-v0")
    obs, _ = env.reset(seed=args.seed)
    obs_t = to_tensor(obs, device)

    net = ActorCritic(env.observation_space.shape[0], env.action_space.n)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    all_returns, all_success, kls = [], [], []
    total_steps, n_updates = 0, 0
    reached_at = None
    t0 = time.time()

    print(f"\n[开始] arm={arm} | 塑形 {use_shaping} | η={args.eta} | 势能 {args.phi}")
    while total_steps < args.max_steps:
        batch, ep_returns, ep_success, obs_t = collect_rollout(
            env, net, args.steps_per_update, obs_t, device, args.gamma, args.lam,
            use_shaping, args.eta, args.phi)
        total_steps += args.steps_per_update
        n_updates += 1
        all_returns.extend(ep_returns)
        all_success.extend(ep_success)

        stats = ppo_update(net, optimizer, batch, args)
        kls.append(stats["approx_kl"])

        if reached_at is None and len(all_success) >= 100 and np.mean(all_success[-100:]) >= 0.9:
            reached_at = total_steps
            print(f"  [{arm:8}] 近100成功率首次 ≥90%（步数 {reached_at}）")

        if n_updates % 5 == 0 and len(all_success) >= 20:
            print(f"  [{arm:8}] 步数 {total_steps:7d} | 回合 {len(all_success):5d} | "
                  f"近20成功 {np.mean(all_success[-20:]):.2f} | "
                  f"回报 {np.mean(all_returns[-20:]):8.1f} | KL {stats['approx_kl']:.4f}")

    eval_returns, eval_success = evaluate(gym.make("MountainCar-v0"), net, args.eval_episodes, device)
    tail_s = all_success[-100:] if len(all_success) >= 100 else all_success
    return {
        "arm": arm,
        "steps": total_steps,
        "reached_at": reached_at,
        "success100": float(np.mean(tail_s)),
        "train_return20": float(np.mean(all_returns[-20:])),
        "eval_return": eval_returns,
        "eval_success": eval_success,
        "mean_kl": float(np.mean(kls)),
        "seconds": time.time() - t0,
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="第020章：PPO 稀疏奖励处理（势能塑形）")
    p.add_argument("--arms", type=str, nargs="+", default=["baseline", "shaping"],
                   choices=["baseline", "shaping"], help="奖励设置")
    p.add_argument("--eta", type=float, default=100.0, help="势能塑形强度 η")
    p.add_argument("--phi", type=str, default="position", choices=["position", "height"],
                   help="势能函数：位置或坡高")
    p.add_argument("--max-steps", type=int, default=200000, help="每个 arm 的环境步数预算")
    p.add_argument("--steps-per-update", type=int, default=2048, help="每次更新前采样步数")
    p.add_argument("--epochs", type=int, default=10, help="每批数据的更新轮数")
    p.add_argument("--batch-size", type=int, default=64, help="minibatch 大小")
    p.add_argument("--lr", type=float, default=3e-4, help="学习率")
    p.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
    p.add_argument("--lam", type=float, default=0.95, help="GAE 的 lambda")
    p.add_argument("--clip", type=float, default=0.2, help="PPO 裁剪范围")
    p.add_argument("--ent-coef", type=float, default=0.01, help="熵正则系数")
    p.add_argument("--vf-coef", type=float, default=0.5, help="价值损失系数")
    p.add_argument("--eval-episodes", type=int, default=20, help="评估回合数")
    p.add_argument("--seed", type=int, default=0, help="随机种子")
    p.add_argument("--save-dir", type=str, default="runs/ch020", help="模型保存目录")
    p.add_argument("--quick", action="store_true", help="快速模式：2 个 arm、各 2 万步")
    return p.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.max_steps = 20000
        args.steps_per_update = 512
        args.epochs = 4
        args.eval_episodes = 10

    print("=" * 96)
    print(f"稀疏奖励处理 | arms {args.arms} | 每个 {args.max_steps} 步 | 种子 {args.seed}")
    print(f"势能塑形：Φ={args.phi}，η={args.eta}，r' = r + η(γΦ(s') − Φ(s))")
    print("=" * 96)

    results = []
    for arm in args.arms:
        res = train_one(arm, args)
        results.append(res)
        print(f"[完成] {arm}: 近100成功率 {res['success100']:.2f} | 评估成功率 {res['eval_success']:.2f} | "
              f"评估回报 {res['eval_return']:.1f} | 用时 {res['seconds']:.1f}s")

    print("\n" + "=" * 96)
    print("对比总结（同种子、同步数预算）")
    print("-" * 96)
    print(f"{'arm':>9} | {'首达90%成功率步数':>15} | {'近100成功率':>11} | "
          f"{'训练近20回报':>12} | {'评估回报':>8} | {'评估成功率':>10} | {'用时(s)':>8}")
    for r in results:
        reached = str(r["reached_at"]) if r["reached_at"] is not None else "未达到"
        print(f"{r['arm']:>9} | {reached:>15} | {r['success100']:>11.2f} | "
              f"{r['train_return20']:>12.1f} | {r['eval_return']:>8.1f} | "
              f"{r['eval_success']:>10.2f} | {r['seconds']:>8.1f}")

    print("\n观察提示：")
    print("  - 成功率是唯一公平指标：塑形回报与原始回报的数值不可直接比较；")
    print("  - baseline 需要靠自己撞出成功轨迹，经常 20 万步都学不稳；")
    print("  - 势能塑形只改变过程奖励，最优策略不变（Ng 等 1999 的定理）；")
    print("  - η 太大时价值尺度失衡、拟合更吃力；η 太小则塑形信号不足。")

    os.makedirs(args.save_dir, exist_ok=True)
    path = os.path.join(args.save_dir, "sparse.txt")
    with open(path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(f"arm={r['arm']} reached90={r['reached_at']} success100={r['success100']:.3f} "
                    f"eval_success={r['eval_success']:.3f} eval_return={r['eval_return']:.2f} "
                    f"seconds={r['seconds']:.1f}\n")
    print(f"\n结果已写入 {path}")


if __name__ == "__main__":
    main()
