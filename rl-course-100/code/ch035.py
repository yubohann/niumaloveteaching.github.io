"""
第035章 SAC的奖励尺度敏感性

把同一个 Pendulum-v1 的奖励整体乘上一个常数 scale，是很多人调环境时的
第一步操作；但奖励尺度一变，SAC 里所有"和奖励同量纲"的量（Q 值、TD 目标、
策略损失里的 Q 项）都会整体缩放，而熵项 alpha*logp 只受 alpha 影响。
本章固定自动温度调节与网络结构，只改训练奖励的尺度：
    scale = 0.1 : 奖励信号被压小，熵项相对"变贵"
    scale = 1.0 : 原始尺度（对照组）
    scale = 10  : 奖励信号被放大，|Q| 同比增大

评估始终在原始（未缩放）环境上进行，保证三组的回报可以直接比较；
同时记录 |Q| 幅度与 alpha 轨迹，观察自动温度调节如何应对尺度变化。

运行：
    python code/ch035.py           # 三种尺度依次训练（CPU 约 6-10 分钟）
    python code/ch035.py --quick   # 快速跑通（约 1 分钟，只验证流程）
    python code/ch035.py --scales 0.1,1.0 --max-steps 30000

预期：scale=1 为基准；scale=0.1 早期学习明显变慢、最终回报偏低；
      scale=10 的 |Q| 约为基准的 10 倍，回报接近但训练更"脆"。
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

LOG_STD_MIN, LOG_STD_MAX = -20.0, 2.0


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_env(env_id: str, seed: int):
    env = gym.make(env_id)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env


# ---------------------------------------------------------------------------
# 经验回放池（存的是缩放后的训练奖励）
# ---------------------------------------------------------------------------
class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, act_dim: int):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rew = np.zeros(capacity, dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(self, obs, act, rew, next_obs, done) -> None:
        i = self.ptr
        self.obs[i] = obs
        self.act[i] = act
        self.rew[i] = rew            # 传入时已经乘过 scale
        self.next_obs[i] = next_obs
        self.done[i] = done
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict:
        idx = np.random.randint(0, self.size, size=batch_size)
        return {
            "obs": torch.as_tensor(self.obs[idx]),
            "act": torch.as_tensor(self.act[idx]),
            "rew": torch.as_tensor(self.rew[idx]),
            "next_obs": torch.as_tensor(self.next_obs[idx]),
            "done": torch.as_tensor(self.done[idx]),
        }


# ---------------------------------------------------------------------------
# 网络
# ---------------------------------------------------------------------------
class QNet(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, act], dim=-1)).squeeze(-1)


class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256,
                 act_scale: float = 1.0):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mean_layer = nn.Linear(hidden, act_dim)
        self.log_std_layer = nn.Linear(hidden, act_dim)
        self.register_buffer("act_scale", torch.tensor(float(act_scale)))

    def forward(self, obs: torch.Tensor):
        h = self.body(obs)
        mean = self.mean_layer(h)
        log_std = torch.clamp(self.log_std_layer(h), LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, obs: torch.Tensor, deterministic: bool = False):
        mean, log_std = self.forward(obs)
        if deterministic:
            return torch.tanh(mean) * self.act_scale, None
        dist = torch.distributions.Normal(mean, log_std.exp())
        x = dist.rsample()
        y = torch.tanh(x)
        action = y * self.act_scale
        log_prob = dist.log_prob(x) - torch.log(1.0 - y.pow(2) + 1e-6)
        return action, log_prob.sum(dim=-1)


# ---------------------------------------------------------------------------
# SAC 智能体：结构与第 034 章一致（双 Q + 目标网络 + 自动温度）
# ---------------------------------------------------------------------------
class SACAgent:
    def __init__(self, obs_dim: int, act_dim: int, act_scale: float,
                 alpha: float = 0.2, target_entropy: float = -1.0,
                 lr: float = 3e-4, gamma: float = 0.99, tau: float = 0.005,
                 hidden: int = 256):
        self.alpha = float(alpha)
        self.target_entropy = float(target_entropy)
        self.gamma = gamma
        self.tau = tau

        self.q1 = QNet(obs_dim, act_dim, hidden)
        self.q2 = QNet(obs_dim, act_dim, hidden)
        self.q1_target = QNet(obs_dim, act_dim, hidden)
        self.q2_target = QNet(obs_dim, act_dim, hidden)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        for p in self.q1_target.parameters():
            p.requires_grad_(False)
        for p in self.q2_target.parameters():
            p.requires_grad_(False)
        self.policy = GaussianPolicy(obs_dim, act_dim, hidden, act_scale)

        self.q_optimizer = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.log_alpha = torch.tensor(math.log(alpha), requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)

    @property
    def current_alpha(self) -> float:
        return float(self.log_alpha.exp().item())

    @torch.no_grad()
    def act(self, obs, deterministic: bool = False) -> np.ndarray:
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        action, _ = self.policy.sample(obs_t, deterministic=deterministic)
        return action.squeeze(0).numpy()

    def update(self, batch: dict) -> dict:
        obs, act, rew = batch["obs"], batch["act"], batch["rew"]
        next_obs, done = batch["next_obs"], batch["done"]

        # 评论家：目标值里的奖励已经是缩放后的数值
        with torch.no_grad():
            next_act, next_logp = self.policy.sample(next_obs)
            q_next = torch.min(self.q1_target(next_obs, next_act),
                               self.q2_target(next_obs, next_act))
            target = rew + self.gamma * (1.0 - done) * (
                q_next - self.current_alpha * next_logp)

        q1_pred = self.q1(obs, act)
        q2_pred = self.q2(obs, act)
        q_loss = F.mse_loss(q1_pred, target) + F.mse_loss(q2_pred, target)

        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # 演员：Q 项随奖励尺度缩放，熵项的"相对价格"因此改变
        new_act, logp = self.policy.sample(obs)
        q_new = torch.min(self.q1(obs, new_act), self.q2(obs, new_act))
        policy_loss = (self.current_alpha * logp - q_new).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # 温度系数自动调节
        alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        with torch.no_grad():
            for p, tp in zip(self.q1.parameters(), self.q1_target.parameters()):
                tp.mul_(1.0 - self.tau).add_(self.tau * p)
            for p, tp in zip(self.q2.parameters(), self.q2_target.parameters()):
                tp.mul_(1.0 - self.tau).add_(self.tau * p)

        return {
            "q_loss": float(q_loss.item()),
            "policy_loss": float(policy_loss.item()),
            "mean_abs_q": float(q1_pred.abs().mean().item()),
            "entropy": float(-logp.mean().item()),
            "alpha": self.current_alpha,
        }


# ---------------------------------------------------------------------------
# 评估：始终在原始（未缩放）奖励上进行
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(env, agent: SACAgent, episodes: int) -> float:
    returns = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            action = agent.act(obs, deterministic=True)
            obs, r, terminated, truncated, _ = env.step(action)
            ep_ret += r                     # 原始奖励，不乘 scale
            done = terminated or truncated
        returns.append(ep_ret)
    return float(np.mean(returns))


# ---------------------------------------------------------------------------
# 单个奖励尺度的训练
# ---------------------------------------------------------------------------
def train_scale(scale: float, args) -> dict:
    set_seed(args.seed)
    env = make_env("Pendulum-v1", args.seed)
    eval_env = make_env("Pendulum-v1", args.seed + 500)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_scale = float(env.action_space.high[0])

    agent = SACAgent(obs_dim, act_dim, act_scale, alpha=args.alpha,
                     target_entropy=args.target_entropy, lr=args.lr,
                     gamma=args.gamma, tau=args.tau, hidden=args.hidden)
    buf = ReplayBuffer(args.buffer_size, obs_dim, act_dim)

    history = {"steps": [], "eval_return": [], "mean_abs_q": [], "alpha": []}
    ep_returns_raw = []                    # 原始奖励的回合回报（供对比）
    ep_returns_scaled = []                 # 训练奖励的回合回报
    obs, _ = env.reset(seed=args.seed)
    ep_ret_raw, ep_ret_scaled = 0.0, 0.0
    last_stats = {"mean_abs_q": 0.0, "q_loss": 0.0, "entropy": 0.0}
    eval_ret = float("nan")
    t_start = time.time()

    print(f"[scale={scale}] 开始训练 | Pendulum-v1 | 种子 {args.seed}")

    for step in range(1, args.max_steps + 1):
        if step <= args.start_steps:
            action = env.action_space.sample()
        else:
            action = agent.act(obs, deterministic=False)

        next_obs, rew_raw, terminated, truncated, _ = env.step(action)
        rew = rew_raw * scale              # 训练奖励按尺度缩放
        buf.add(obs, action, rew, next_obs, float(terminated))
        obs = next_obs
        ep_ret_raw += rew_raw
        ep_ret_scaled += rew

        if terminated or truncated:
            ep_returns_raw.append(ep_ret_raw)
            ep_returns_scaled.append(ep_ret_scaled)
            ep_ret_raw, ep_ret_scaled = 0.0, 0.0
            obs, _ = env.reset()

        if buf.size >= args.batch_size:
            last_stats = agent.update(buf.sample(args.batch_size))

        if step % args.eval_every == 0 or step == args.max_steps:
            eval_ret = evaluate(eval_env, agent, args.eval_episodes)
            recent_scaled = float(np.mean(ep_returns_scaled[-20:])) if ep_returns_scaled else float("nan")
            history["steps"].append(step)
            history["eval_return"].append(eval_ret)
            history["mean_abs_q"].append(last_stats["mean_abs_q"])
            history["alpha"].append(agent.current_alpha)
            print(f"[scale={scale}] 步数 {step:6d} | 近20训练回报 {recent_scaled:9.1f} | "
                  f"评估(原始) {eval_ret:8.1f} | |Q| {last_stats['mean_abs_q']:8.1f} | "
                  f"alpha {agent.current_alpha:.3f} | 熵 {last_stats['entropy']:.3f}")

    elapsed = time.time() - t_start
    print(f"[scale={scale}] 训练结束 | 用时 {elapsed:.1f}s | 最终评估(原始) {eval_ret:.1f} | "
          f"|Q| {last_stats['mean_abs_q']:.1f}")

    os.makedirs(args.save_dir, exist_ok=True)
    ckpt = os.path.join(args.save_dir, f"sac_scale{scale}.pt")
    torch.save({"policy": agent.policy.state_dict(),
                "q1": agent.q1.state_dict(),
                "q2": agent.q2.state_dict(),
                "scale": scale}, ckpt)
    print(f"[scale={scale}] 模型已保存到 {ckpt}")

    return {"scale": scale, "final_eval": eval_ret, "elapsed": elapsed,
            "history": history}


# ---------------------------------------------------------------------------
# 可选绘图
# ---------------------------------------------------------------------------
def maybe_plot(results, save_dir: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("未安装 matplotlib，跳过绘图")
        return
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for res in results:
        hist = res["history"]
        axes[0].plot(hist["steps"], hist["eval_return"], marker="o",
                     label=f"scale={res['scale']}")
        axes[1].plot(hist["steps"], hist["mean_abs_q"], marker="o",
                     label=f"scale={res['scale']}")
    axes[0].set_xlabel("环境步数")
    axes[0].set_ylabel("原始奖励评估回报")
    axes[0].set_title("评估回报（未缩放口径）")
    axes[0].legend()
    axes[1].set_xlabel("环境步数")
    axes[1].set_ylabel("|Q| 平均值")
    axes[1].set_title("Q 值幅度随奖励尺度缩放")
    axes[1].legend()
    fig.tight_layout()
    out = os.path.join(save_dir, "ch035_reward_scale.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"曲线已保存到 {out}")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="第035章：SAC 的奖励尺度敏感性")
    p.add_argument("--scales", type=str, default="0.1,1.0,10.0",
                   help="训练奖励的缩放系数列表")
    p.add_argument("--alpha", type=float, default=0.2, help="alpha 初值")
    p.add_argument("--target-entropy", type=float, default=-1.0, help="目标熵")
    p.add_argument("--max-steps", type=int, default=15000, help="每种尺度的步数上限")
    p.add_argument("--start-steps", type=int, default=1000, help="纯随机探索步数")
    p.add_argument("--buffer-size", type=int, default=100000, help="回放池容量")
    p.add_argument("--batch-size", type=int, default=256, help="批量大小")
    p.add_argument("--hidden", type=int, default=256, help="隐藏层宽度")
    p.add_argument("--lr", type=float, default=3e-4, help="学习率")
    p.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
    p.add_argument("--tau", type=float, default=0.005, help="目标网络软更新系数")
    p.add_argument("--eval-every", type=int, default=3000, help="评估与日志间隔")
    p.add_argument("--eval-episodes", type=int, default=3, help="每次评估的回合数")
    p.add_argument("--seed", type=int, default=0, help="随机种子")
    p.add_argument("--save-dir", type=str, default="runs/ch035", help="模型保存目录")
    p.add_argument("--quick", action="store_true", help="快速模式：几十秒跑完")
    return p.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.max_steps = min(args.max_steps, 2500)
        args.start_steps = min(args.start_steps, 500)
        args.eval_every = 1200
        args.eval_episodes = 2

    scales = [float(x) for x in args.scales.split(",") if x.strip()]
    for s in scales:
        if s <= 0:
            raise SystemExit("奖励尺度必须为正数")

    results = [train_scale(s, args) for s in scales]

    print("=" * 74)
    print("奖励尺度对比汇总（Pendulum-v1，评估一律用原始奖励，以实跑为准）")
    for res in results:
        hist = res["history"]
        print(f"  scale={res['scale']:<5g}: 最终评估 {res['final_eval']:8.1f} | "
              f"|Q| 轨迹 {[round(q, 1) for q in hist['mean_abs_q'][-3:]]}")

    maybe_plot(results, args.save_dir)


if __name__ == "__main__":
    main()
