"""
第006章 PPO的神经网络架构设计

PPO 的 Actor-Critic 网络可以被设计成不同的宽度、深度和激活函数。
本章在 CartPole-v1 上对四种架构做受控对比（同种子、同步数预算）：
    "32-32-tanh"        ：轻量，参数少、前向快
    "64-64-tanh"        ：课程默认基线
    "128-128-128-tanh"  ：深且宽，容量最大
    "64-64-relu"        ：换激活函数，观察非线性选择的影响

每个架构记录：参数量、首达400步数、最终回报、贪婪评估、KL 与墙钟时间，
并计算"每万步墙钟时间"来量化容量带来的计算成本。

可选的 --init ortho 会启用 PPO 常用的小增益正交初始化（策略头 gain=0.01），
用来观察初始化对早期稳定性的影响。

运行：
    python code/ch006.py                  # 4 种架构，各 6 万步，CPU 约 3-6 分钟
    python code/ch006.py --quick          # 快速跑通（约 30-60 秒）
    python code/ch006.py --archs 32-32-tanh 64-64-tanh
    python code/ch006.py --init ortho     # 正交初始化版对比

预期：CartPole 上 32-32 与 64-64 的表现接近，128x3 前向更慢但步数效率没有明显提升；
ReLU 版本通常可用但在小网络上略逊于 tanh；ortho 初始化能降低早期 KL 波动。
"""

import argparse
import math
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
    """固定随机种子：不同架构从同一条随机流出发，保证受控对比。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_tensor(obs, device):
    return torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)


def parse_arch(spec: str):
    """把 "128-128-128-tanh" 解析成 ([128,128,128], nn.Tanh)。"""
    parts = spec.split("-")
    act_name = parts[-1].lower()
    sizes = [int(x) for x in parts[:-1]]
    act_cls = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU}.get(act_name)
    if act_cls is None or len(sizes) == 0:
        raise ValueError(f"无法解析架构: {spec}（示例：64-64-tanh）")
    return sizes, act_cls


# ---------------------------------------------------------------------------
# 网络：隐藏层结构与激活由外部指定
# ---------------------------------------------------------------------------
class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes, act_cls, init: str = "default"):
        super().__init__()
        layers, last = [], obs_dim
        for h in hidden_sizes:
            layers += [nn.Linear(last, h), act_cls()]
            last = h
        self.body = nn.Sequential(*layers)
        self.actor = nn.Linear(last, act_dim)
        self.critic = nn.Linear(last, 1)
        if init == "ortho":
            self._orthogonal_init()
        self.hidden_sizes = tuple(hidden_sizes)

    def _orthogonal_init(self) -> None:
        """PPO 常用初始化：躯干 gain=√2，价值头 gain=1，策略头 gain=0.01。

        策略头小增益让初始动作分布接近均匀，减少训练初期的越界更新。
        """
        for m in self.body:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2.0))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.zeros_(self.actor.bias)

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
# 训练一个架构
# ---------------------------------------------------------------------------
def train_one(spec: str, args):
    set_seed(args.seed)
    device = torch.device("cpu")
    env = gym.make("CartPole-v1")
    obs, _ = env.reset(seed=args.seed)

    hidden_sizes, act_cls = parse_arch(spec)
    net = ActorCritic(env.observation_space.shape[0], env.action_space.n,
                      hidden_sizes, act_cls, init=args.init)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    n_params = sum(p.numel() for p in net.parameters())

    all_returns, kls = [], []
    total_steps, n_updates, obs_t = 0, 0, to_tensor(obs, device)
    reached_at = None
    t0 = time.time()

    print(f"\n[开始] 架构 {spec} | 参数量 {n_params} | 初始化 {args.init}")
    while total_steps < args.max_steps:
        batch, ep_returns, obs_t = collect_rollout(
            env, net, args.steps_per_update, obs_t, device, args.gamma, args.lam)
        total_steps += args.steps_per_update
        n_updates += 1
        all_returns.extend(ep_returns)

        stats = ppo_update(net, optimizer, batch, args)
        kls.append(stats["approx_kl"])

        if reached_at is None and len(all_returns) >= 100 and np.mean(all_returns[-100:]) >= args.stage_reward:
            reached_at = total_steps

        if n_updates % 5 == 0:
            print(f"  {spec:<18} 步数 {total_steps:7d} | 回合 {len(all_returns):4d} | "
                  f"近20均 {np.mean(all_returns[-20:]):7.1f} | KL {stats['approx_kl']:.4f} | "
                  f"熵 {stats['entropy']:.3f}")

    eval_returns = evaluate(gym.make("CartPole-v1"), net, args.eval_episodes, device)
    tail = all_returns[-100:] if len(all_returns) >= 100 else all_returns
    sec = time.time() - t0
    return {
        "spec": spec,
        "params": n_params,
        "reached_at": reached_at,
        "last100": float(np.mean(tail)),
        "eval": float(np.mean(eval_returns)),
        "mean_kl": float(np.mean(kls)),
        "seconds": sec,
        "seconds_per_10k": sec / (total_steps / 10000.0),
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="第006章：PPO 神经网络架构设计")
    p.add_argument("--archs", type=str, nargs="+",
                   default=["32-32-tanh", "64-64-tanh", "128-128-128-tanh", "64-64-relu"],
                   help="架构列表，格式如 64-64-tanh")
    p.add_argument("--init", type=str, default="default", choices=["default", "ortho"],
                   help="参数初始化：default 为 PyTorch 默认，ortho 为 PPO 常用正交初始化")
    p.add_argument("--max-steps", type=int, default=60000, help="每个架构的环境步数预算")
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
    p.add_argument("--save-dir", type=str, default="runs/ch006", help="结果保存目录")
    p.add_argument("--quick", action="store_true", help="快速模式：3 种架构、各 8 千步")
    return p.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.archs = ["32-32-tanh", "64-64-tanh", "64-64-relu"]
        args.max_steps = 8000
        args.steps_per_update = 512
        args.epochs = 4
        args.stage_reward = 250.0

    print("=" * 96)
    print(f"网络架构对比 | 架构 {args.archs} | 初始化 {args.init} | "
          f"每个 {args.max_steps} 步 | 种子 {args.seed}")
    print("=" * 96)

    results = []
    for spec in args.archs:
        res = train_one(spec, args)
        results.append(res)
        print(f"[完成] {spec}: 参数 {res['params']} | 近100均 {res['last100']:.1f} | "
              f"评估 {res['eval']:.1f} | 用时 {res['seconds']:.1f}s")

    print("\n" + "=" * 96)
    print("架构对比总结（同种子、同步数预算）")
    print("-" * 96)
    print(f"{'架构':>18} | {'参数量':>8} | {'首达400步数':>11} | {'近100均':>8} | {'评估':>7} | "
          f"{'平均KL':>8} | {'用时(s)':>8} | {'秒/万步':>8}")
    for r in results:
        reached = str(r["reached_at"]) if r["reached_at"] is not None else "未达到"
        print(f"{r['spec']:>18} | {r['params']:>8d} | {reached:>11} | {r['last100']:>8.1f} | "
              f"{r['eval']:>7.1f} | {r['mean_kl']:>8.4f} | {r['seconds']:>8.1f} | "
              f"{r['seconds_per_10k']:>8.2f}")

    print("\n观察提示：")
    print("  - 步数效率（首达400）看学习能力，秒/万步看计算成本，二者一起才是一次完整权衡；")
    print("  - CartPole 状态只有 4 维，容量过剩（128x3）通常只增加成本、不改善步数效率；")
    print("  - 换激活函数（relu）在小网络上可能变慢，可结合 KL 与评估回报判断。")

    os.makedirs(args.save_dir, exist_ok=True)
    path = os.path.join(args.save_dir, f"arch_compare_{args.init}.txt")
    with open(path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(f"arch={r['spec']} init={args.init} params={r['params']} "
                    f"reached400={r['reached_at']} last100={r['last100']:.2f} eval={r['eval']:.2f} "
                    f"kl={r['mean_kl']:.5f} sec_per_10k={r['seconds_per_10k']:.2f}\n")
    print(f"\n结果已写入 {path}")


if __name__ == "__main__":
    main()
