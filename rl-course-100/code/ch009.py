"""
第009章 PPO的时序差分与蒙特卡洛混合

优势估计是整个谱系，GAE 只是其中一条曲线。本章把谱系上的几个代表点拆开：
  - TD(0)      ：单步 TD 残差（偏差最大、方差最小）
  - n-step     ：固定 n 步回报 + V(s_{t+n}) bootstrap（默认 n=4）
  - GAE(0.95)  ：指数加权的混合（第 002 章的主角）
  - 蒙特卡洛    ：整条轨迹的折扣回报减基线（无偏、方差最大）

实验分两段：
  1) 诊断：用同一个策略采一批固定轨迹，比较各估计器的优势标准差、
     与蒙特卡洛优势的相关系数、平均绝对偏差（不训练，秒级完成）
  2) 训练：用四种优势估计器分别训练 CartPole，比较首达 400 步数与最终回报

运行：
    python code/ch009.py            # 诊断 + 4 组训练，各 6 万步，CPU 约 3-6 分钟
    python code/ch009.py --quick    # 快速跑通（约 30-60 秒）
    python code/ch009.py --modes td nstep gae
    python code/ch009.py --nstep 8  # 改变 n-step 的 n

预期：诊断表上，优势 std 按 TD(0) < n-step < GAE < MC 递增，
与 MC 的相关系数也递增；训练上 TD(0) 前期最慢，GAE 综合最稳。
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
# 优势估计器集合
# ---------------------------------------------------------------------------
def compute_nstep(rewards, values, terminateds, last_value, n: int, gamma: float):
    """固定 n 步优势：Â_t = Σ_{l<n} γ^l r_{t+l} + γ^k V(s_{t+k}) − V(s_t)。

    回合在 n 步内终止时不 bootstrap；数据不够 n 步时用 last_value 收尾。
    """
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    for t in range(T):
        g = 0.0
        discount = 1.0
        k = 0
        bootstrap = True
        while k < n and t + k < T:
            g += discount * rewards[t + k]
            discount *= gamma
            k += 1
            if terminateds[t + k - 1]:     # 本条轨迹在此终止：不 bootstrap
                bootstrap = False
                break
        if bootstrap:
            next_v = values[t + k] if t + k < T else last_value
            g += discount * next_v
        adv[t] = g - values[t]
    return adv


def compute_gae(rewards, values, terminateds, last_value, gamma: float, lam: float):
    """GAE(γ, λ)：λ=0 退化为 TD(0)，λ=1 退化为蒙特卡洛。"""
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(T)):
        next_value = last_value if t == T - 1 else values[t + 1]
        non_terminal = 1.0 - terminateds[t]
        delta = rewards[t] + gamma * next_value * non_terminal - values[t]
        last_gae = delta + gamma * lam * non_terminal * last_gae
        advantages[t] = last_gae
    return advantages


def compute_advantages(mode: str, rewards, values, terminateds, last_value,
                       gamma: float, lam: float, nstep: int):
    """按模式计算优势。mc 等价于 GAE(λ=1)。"""
    if mode == "td":
        return compute_nstep(rewards, values, terminateds, last_value, 1, gamma)
    if mode == "nstep":
        return compute_nstep(rewards, values, terminateds, last_value, nstep, gamma)
    if mode == "gae":
        return compute_gae(rewards, values, terminateds, last_value, gamma, lam)
    if mode == "mc":
        return compute_gae(rewards, values, terminateds, last_value, gamma, 1.0)
    raise ValueError(f"未知优势估计模式: {mode}")


# ---------------------------------------------------------------------------
# 采样（先采集原始数据，再按模式算优势）
# ---------------------------------------------------------------------------
def collect_raw(env, net, steps, obs_t, device):
    """只采集原始轨迹数据，不做优势计算。返回 raw 字典与完成的回合回报。"""
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

    raw = {
        "obs": np.asarray(obs_buf),
        "act": np.asarray(act_buf),
        "logp": np.asarray(logp_buf),
        "rew": rew_buf,
        "val": val_buf,
        "term": term_buf,
        "last_value": net.value(obs_after),
    }
    return raw, ep_returns, obs_t


def raw_to_batch(raw, mode, device, gamma, lam, nstep):
    adv = compute_advantages(mode, raw["rew"], raw["val"], raw["term"],
                             raw["last_value"], gamma, lam, nstep)
    ret = adv + np.asarray(raw["val"], dtype=np.float32)
    return {
        "obs": torch.as_tensor(raw["obs"], dtype=torch.float32, device=device),
        "act": torch.as_tensor(raw["act"], dtype=torch.long, device=device),
        "logp_old": torch.as_tensor(raw["logp"], dtype=torch.float32, device=device),
        "adv": torch.as_tensor(adv, dtype=torch.float32, device=device),
        "ret": torch.as_tensor(ret, dtype=torch.float32, device=device),
        "adv_raw_std": float(np.std(adv)),
    }


# ---------------------------------------------------------------------------
# 诊断：同一批数据上比较四个估计器
# ---------------------------------------------------------------------------
def estimator_diagnostics(args, steps: int = 4096):
    """用固定种子的初始策略采一批数据，比较四个估计器的统计特性。"""
    set_seed(args.seed + 777)
    device = torch.device("cpu")
    env = gym.make("CartPole-v1")
    obs, _ = env.reset(seed=args.seed + 777)
    net = ActorCritic(env.observation_space.shape[0], env.action_space.n)
    obs_t = to_tensor(obs, device)

    raw, _, _ = collect_raw(env, net, steps, obs_t, device)
    mc = compute_advantages("mc", raw["rew"], raw["val"], raw["term"],
                            raw["last_value"], args.gamma, args.lam, args.nstep)

    print("同一批数据上的估计器诊断（策略为随机初始化的网络，数据量 4096 步）")
    print("-" * 84)
    print(f"{'估计器':>10} | {'优势均值':>10} | {'优势std':>10} | {'与MC相关':>10} | "
          f"{'与MC平均绝对差':>14} | {'|Â|>5 占比':>10}")
    rows = []
    for mode, name in [("td", "TD(0)"), ("nstep", f"n-step({args.nstep})"),
                       ("gae", f"GAE({args.lam})"), ("mc", "MC")]:
        adv = compute_advantages(mode, raw["rew"], raw["val"], raw["term"],
                                 raw["last_value"], args.gamma, args.lam, args.nstep)
        corr = float(np.corrcoef(adv, mc)[0, 1]) if np.std(adv) > 1e-8 else float("nan")
        mad = float(np.mean(np.abs(adv - mc)))
        big = float(np.mean(np.abs(adv) > 5.0))
        rows.append((name, float(np.mean(adv)), float(np.std(adv)), corr, mad, big))
        print(f"{name:>10} | {rows[-1][1]:>10.3f} | {rows[-1][2]:>10.3f} | {corr:>10.3f} | "
              f"{mad:>14.3f} | {big:>10.3f}")
    print("说明：MC 作为参照不是“真值”，但它是无偏的（在折扣回报意义下）；")
    print("      相关系数越高说明估计器越贴近无偏目标，std 越大说明梯度越噪。")
    return rows


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
# 训练一个估计器
# ---------------------------------------------------------------------------
def train_one(mode: str, args):
    set_seed(args.seed)
    device = torch.device("cpu")
    env = gym.make("CartPole-v1")
    obs, _ = env.reset(seed=args.seed)
    net = ActorCritic(env.observation_space.shape[0], env.action_space.n)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    all_returns, adv_stds, kls = [], [], []
    total_steps, n_updates, obs_t = 0, 0, to_tensor(obs, device)
    reached_at = None
    t0 = time.time()

    print(f"\n[开始] 优势估计 = {mode}")
    while total_steps < args.max_steps:
        raw, ep_returns, obs_t = collect_raw(env, net, args.steps_per_update, obs_t, device)
        total_steps += args.steps_per_update
        n_updates += 1
        all_returns.extend(ep_returns)

        batch = raw_to_batch(raw, mode, device, args.gamma, args.lam, args.nstep)
        stats = ppo_update(net, optimizer, batch, args)
        adv_stds.append(batch["adv_raw_std"])
        kls.append(stats["approx_kl"])

        if reached_at is None and len(all_returns) >= 100 and np.mean(all_returns[-100:]) >= args.stage_reward:
            reached_at = total_steps

        if n_updates % 5 == 0 and len(all_returns) >= 20:
            print(f"  [{mode:5}] 步数 {total_steps:7d} | 回合 {len(all_returns):4d} | "
                  f"近20均 {np.mean(all_returns[-20:]):7.1f} | 优势std {batch['adv_raw_std']:6.2f} | "
                  f"KL {stats['approx_kl']:.4f}")

    eval_returns = evaluate(gym.make("CartPole-v1"), net, args.eval_episodes, device)
    tail = all_returns[-100:] if len(all_returns) >= 100 else all_returns
    return {
        "mode": mode,
        "steps": total_steps,
        "reached_at": reached_at,
        "last100": float(np.mean(tail)),
        "eval": float(np.mean(eval_returns)),
        "adv_std": float(np.mean(adv_stds)),
        "mean_kl": float(np.mean(kls)),
        "seconds": time.time() - t0,
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="第009章：PPO 时序差分与蒙特卡洛混合")
    p.add_argument("--modes", type=str, nargs="+", default=["td", "nstep", "gae", "mc"],
                   choices=["td", "nstep", "gae", "mc"], help="要对比的优势估计器")
    p.add_argument("--nstep", type=int, default=4, help="n-step 估计器的 n")
    p.add_argument("--max-steps", type=int, default=60000, help="每个估计器的环境步数预算")
    p.add_argument("--steps-per-update", type=int, default=2048, help="每次更新前采样步数")
    p.add_argument("--epochs", type=int, default=10, help="每批数据的更新轮数")
    p.add_argument("--batch-size", type=int, default=64, help="minibatch 大小")
    p.add_argument("--lr", type=float, default=3e-4, help="学习率")
    p.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
    p.add_argument("--lam", type=float, default=0.95, help="GAE 的 lambda")
    p.add_argument("--clip", type=float, default=0.2, help="PPO 裁剪范围")
    p.add_argument("--ent-coef", type=float, default=0.01, help="熵正则系数")
    p.add_argument("--vf-coef", type=float, default=0.5, help="价值损失系数")
    p.add_argument("--stage-reward", type=float, default=400.0, help="阶段目标（记录首达步数）")
    p.add_argument("--eval-episodes", type=int, default=5, help="评估回合数")
    p.add_argument("--seed", type=int, default=0, help="随机种子")
    p.add_argument("--save-dir", type=str, default="runs/ch009", help="结果保存目录")
    p.add_argument("--no-diagnostics", action="store_true", help="跳过开头的估计器诊断")
    p.add_argument("--quick", action="store_true", help="快速模式：2 个估计器、各 8 千步")
    return p.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.modes = ["td", "gae"]
        args.max_steps = 8000
        args.steps_per_update = 512
        args.epochs = 4
        args.stage_reward = 250.0

    print("=" * 96)
    print(f"优势估计器对比 | 估计器 {args.modes} | 每个 {args.max_steps} 步 | 种子 {args.seed}")
    print("=" * 96)

    diag_rows = None
    if not args.no_diagnostics:
        diag_rows = estimator_diagnostics(args)
        print()

    results = []
    for mode in args.modes:
        res = train_one(mode, args)
        results.append(res)
        print(f"[完成] {mode}: 近100均 {res['last100']:.1f} | 评估 {res['eval']:.1f} | "
              f"平均优势std {res['adv_std']:.2f} | 用时 {res['seconds']:.1f}s")

    print("\n" + "=" * 96)
    print("训练对比总结（同种子、同步数预算）")
    print("-" * 96)
    print(f"{'估计器':>8} | {'首达400步数':>11} | {'近100均':>8} | {'评估':>7} | "
          f"{'平均优势std':>11} | {'平均KL':>8} | {'用时(s)':>8}")
    for r in results:
        reached = str(r["reached_at"]) if r["reached_at"] is not None else "未达到"
        print(f"{r['mode']:>8} | {reached:>11} | {r['last100']:>8.1f} | {r['eval']:>7.1f} | "
              f"{r['adv_std']:>11.2f} | {r['mean_kl']:>8.4f} | {r['seconds']:>8.1f}")

    print("\n观察提示：")
    print("  - TD(0) 优势方差最小但偏差最大，前 1-2 万步通常最慢；")
    print("  - MC 优势方差最大，早期波动明显、偶发快速冲高；")
    print("  - GAE(0.95) 处在两者中间，是默认推荐的折中；")
    print("  - n-step 把偏差-方差的选择离散化：n 越小越像 TD，越大越像 MC。")

    os.makedirs(args.save_dir, exist_ok=True)
    path = os.path.join(args.save_dir, "adv_estimators.txt")
    with open(path, "w", encoding="utf-8") as f:
        if diag_rows is not None:
            for name, mean, std, corr, mad, big in diag_rows:
                f.write(f"diag {name}: mean={mean:.4f} std={std:.4f} corr_mc={corr:.4f} "
                        f"mad_mc={mad:.4f} big_frac={big:.4f}\n")
        for r in results:
            f.write(f"train {r['mode']}: reached400={r['reached_at']} last100={r['last100']:.2f} "
                    f"eval={r['eval']:.2f} adv_std={r['adv_std']:.3f} kl={r['mean_kl']:.5f}\n")
    print(f"\n结果已写入 {path}")


if __name__ == "__main__":
    main()
