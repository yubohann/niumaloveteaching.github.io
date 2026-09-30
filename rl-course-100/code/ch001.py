"""
第001章 PPO在CartPole上的首次实现

本章用一个文件实现完整的 PPO（Proximal Policy Optimization）训练流程：
  - Actor-Critic 共享躯干的 MLP 网络
  - 采样一批 rollout（steps_per_update 步）
  - 广义优势估计（GAE）
  - PPO 裁剪目标（clip objective）
  - 多轮（epochs）小批量（minibatch）更新

运行：
    python code/ch001.py           # 完整训练（CPU 约 2-5 分钟）
    python code/ch001.py --quick   # 快速跑通（约 20-40 秒，只验证流程不保证达标）

预期：CartPole-v1 通常在 10 万～30 万环境步内，100 回合滑窗平均回报达到 475 以上。
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
    """固定随机种子，保证实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_tensor(obs, device):
    """把 gym 返回的 numpy 观测转成 (1, obs_dim) 的 float32 张量。"""
    return torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)


# ---------------------------------------------------------------------------
# 网络：共享躯干 + 策略头 + 价值头
# ---------------------------------------------------------------------------
class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 64):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.actor = nn.Linear(hidden, act_dim)   # 输出离散动作的 logits
        self.critic = nn.Linear(hidden, 1)        # 输出状态价值 V(s)

    def forward(self, obs: torch.Tensor):
        h = self.body(obs)
        return self.actor(h), self.critic(h).squeeze(-1)

    @torch.no_grad()
    def act(self, obs: torch.Tensor):
        """输入 (1, obs_dim) 的观测，返回 (动作, log 概率, 状态价值)。"""
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action).item(), value.item()

    @torch.no_grad()
    def value(self, obs: torch.Tensor) -> float:
        """只计算状态价值，用于 rollout 结尾的 bootstrap。"""
        _, v = self.forward(obs)
        return v.item()


# ---------------------------------------------------------------------------
# GAE：广义优势估计
# ---------------------------------------------------------------------------
def compute_gae(rewards, values, terminateds, last_value, gamma: float, lam: float):
    """从后向前递推 GAE。

    参数：
        rewards      : 长度 T 的奖励列表
        values       : 长度 T 的状态价值 V(s_t)
        terminateds  : 长度 T 的终止标志（1.0 表示真实终止，截断为 0.0）
        last_value   : s_T 的 bootstrap 价值（对截断回合有效）
    返回：
        advantages, returns（均为 float32 数组）
    """
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(T)):
        next_value = last_value if t == T - 1 else values[t + 1]
        non_terminal = 1.0 - terminateds[t]       # 真实终止时价值为 0
        delta = rewards[t] + gamma * next_value * non_terminal - values[t]
        last_gae = delta + gamma * lam * non_terminal * last_gae
        advantages[t] = last_gae
    returns = advantages + np.asarray(values, dtype=np.float32)
    return advantages, returns


# ---------------------------------------------------------------------------
# 采样一批 rollout
# ---------------------------------------------------------------------------
def collect_rollout(env, net, steps, obs_t, device, gamma, lam):
    """与环境交互 steps 步，返回训练 batch、本批次完成的回合回报、最新观测张量。"""
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
        # 只有真实终止才把价值切断；时间截断（truncated）仍用 bootstrap
        term_buf.append(float(terminated))

        ep_ret += r
        obs_t = to_tensor(next_obs, device)
        obs_after = obs_t

        if terminated or truncated:
            ep_returns.append(ep_ret)
            ep_ret = 0.0
            reset_obs, _ = env.reset()
            obs_t = to_tensor(reset_obs, device)

    # rollout 结尾的 bootstrap 价值
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
    """在收集到的 batch 上做 epochs 轮小批量更新，返回平均诊断指标。"""
    obs, act = batch["obs"], batch["act"]
    logp_old, adv, ret = batch["logp_old"], batch["adv"], batch["ret"]
    N = obs.shape[0]

    # 优势函数标准化：降低梯度方差
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
             "approx_kl": 0.0, "clip_frac": 0.0, "n_updates": 0}

    for _ in range(args.epochs):
        idx = torch.randperm(N, device=obs.device)
        for start in range(0, N, args.batch_size):
            mb = idx[start:start + args.batch_size]
            logits, value = net(obs[mb])
            dist = Categorical(logits=logits)
            logp = dist.log_prob(act[mb])

            # 新旧策略的概率比
            ratio = torch.exp(logp - logp_old[mb])
            # PPO 裁剪目标：取未裁剪与裁剪后目标的较小值
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
                approx_kl = (logp_old[mb] - logp).mean().item()
                clip_frac = ((ratio - 1.0).abs() > args.clip).float().mean().item()
            stats["policy_loss"] += policy_loss.item()
            stats["value_loss"] += value_loss.item()
            stats["entropy"] += entropy.item()
            stats["approx_kl"] += approx_kl
            stats["clip_frac"] += clip_frac
            stats["n_updates"] += 1

    for k in stats:
        if k != "n_updates":
            stats[k] /= max(stats["n_updates"], 1)
    return stats


# ---------------------------------------------------------------------------
# 评估：用确定性（贪婪）策略跑若干回合
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
            action = int(torch.argmax(logits, dim=-1).item())  # 贪婪动作
            obs, r, terminated, truncated, _ = env.step(action)
            ep_ret += r
            done = terminated or truncated
        returns.append(ep_ret)
    return returns


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="第001章：PPO 在 CartPole 上的首次实现")
    p.add_argument("--episodes", type=int, default=1500, help="最多训练多少个回合")
    p.add_argument("--max-steps", type=int, default=400000, help="环境步数上限（触发即停）")
    p.add_argument("--steps-per-update", type=int, default=2048, help="每次更新前采样的步数")
    p.add_argument("--epochs", type=int, default=10, help="每批数据的更新轮数")
    p.add_argument("--batch-size", type=int, default=64, help="小批量大小")
    p.add_argument("--lr", type=float, default=3e-4, help="学习率")
    p.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
    p.add_argument("--lam", type=float, default=0.95, help="GAE 的 lambda")
    p.add_argument("--clip", type=float, default=0.2, help="PPO 裁剪范围")
    p.add_argument("--ent-coef", type=float, default=0.01, help="熵正则系数")
    p.add_argument("--vf-coef", type=float, default=0.5, help="价值损失系数")
    p.add_argument("--target-reward", type=float, default=475.0, help="视为解决任务的滑窗平均回报")
    p.add_argument("--eval-episodes", type=int, default=5, help="训练结束后的评估回合数")
    p.add_argument("--seed", type=int, default=0, help="随机种子")
    p.add_argument("--save-dir", type=str, default="runs/ch001", help="模型保存目录")
    p.add_argument("--quick", action="store_true", help="快速模式：缩小训练规模，几十秒跑完")
    return p.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.episodes = min(args.episodes, 120)
        args.steps_per_update = 768
        args.epochs = 6
        args.max_steps = min(args.max_steps, 20000)

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make("CartPole-v1")
    obs, _ = env.reset(seed=args.seed)

    net = ActorCritic(env.observation_space.shape[0], env.action_space.n).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    print(f"设备: {device} | 环境: CartPole-v1 | 种子: {args.seed}")
    print(f"配置: episodes={args.episodes}, steps_per_update={args.steps_per_update}, "
          f"epochs={args.epochs}, batch={args.batch_size}, lr={args.lr}")

    all_returns = []
    total_steps = 0
    update_idx = 0
    solved_at = None
    obs_t = to_tensor(obs, device)
    t_start = time.time()

    while len(all_returns) < args.episodes and total_steps < args.max_steps:
        # 1) 采样
        batch, ep_returns, obs_t = collect_rollout(
            env, net, args.steps_per_update, obs_t, device, args.gamma, args.lam)
        total_steps += args.steps_per_update
        all_returns.extend(ep_returns)

        # 2) 更新
        stats = ppo_update(net, optimizer, batch, args)
        update_idx += 1

        # 3) 日志：滑窗回报 + 关键诊断量
        recent = all_returns[-20:]
        recent100 = all_returns[-100:]
        print(f"[更新 {update_idx:3d}] 步数 {total_steps:7d} | 回合 {len(all_returns):4d} | "
              f"近20均 {np.mean(recent):7.1f} | KL {stats['approx_kl']:.4f} | "
              f"裁剪比例 {stats['clip_frac']:.3f} | 熵 {stats['entropy']:.3f}")

        # 4) 早停：滑窗平均达到目标即视为解决
        if solved_at is None and len(all_returns) >= 100 and np.mean(recent100) >= args.target_reward:
            solved_at = len(all_returns)
            print(f"达到目标：第 {solved_at} 回合时近100回合平均回报 "
                  f"{np.mean(recent100):.1f} >= {args.target_reward}")

    elapsed = time.time() - t_start

    # 评估：确定性策略
    eval_env = gym.make("CartPole-v1")
    eval_returns = evaluate(eval_env, net, args.eval_episodes, device)

    print("-" * 70)
    print(f"训练结束 | 用时 {elapsed:.1f}s | 总步数 {total_steps}")
    print(f"训练回合数 {len(all_returns)} | 最后100回合平均 {np.mean(all_returns[-100:]):.1f}")
    print(f"评估（{args.eval_episodes} 回合，贪婪策略）: {[round(r) for r in eval_returns]} "
          f"平均 {np.mean(eval_returns):.1f}")

    os.makedirs(args.save_dir, exist_ok=True)
    ckpt = os.path.join(args.save_dir, "actor_critic.pt")
    torch.save(net.state_dict(), ckpt)
    print(f"模型已保存到 {ckpt}")


if __name__ == "__main__":
    main()
