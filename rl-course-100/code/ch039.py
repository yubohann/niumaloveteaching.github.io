"""
第039章 SAC的观测归一化

Pendulum-v1 的观测三个维度量纲差异很大：
    cos(theta) ∈ [-1, 1]、sin(theta) ∈ [-1, 1]、theta_dot ∈ [-8, 8]
网络的第一层同时看到这三种尺度，梯度会被大尺度维度主导，学习变慢。
本章实现一个滑动统计（Welford 递推）的观测归一化器，在训练过程中
累计均值与方差，把观测逐维标准化并裁剪到 [-clip, clip]。
回放池里存的是原始观测，采样时才用"当前统计量"归一化，避免旧数据
被早期的旧统计量污染。

实验：在 Pendulum-v1 上对比
    none : 原始观测直接送进网络
    run  : 训练中在线统计并归一化
观察评估回报曲线、|Q| 幅度与 theta_dot 维度的估计标准差。

运行：
    python code/ch039.py           # 两种设置依次训练（CPU 约 5-8 分钟）
    python code/ch039.py --quick   # 快速跑通（约 1 分钟）
    python code/ch039.py --obs-norms run --max-steps 30000

预期：run 组在前 1/3 训练里明显领先，后期两组收敛到接近水平；
      归一化主要买的是"早期学习速度"，不是最终上限。
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
ALL_NORMS = ["none", "run"]


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
# 观测归一化：Welford 在线均值/方差
# ---------------------------------------------------------------------------
class RunningObsNormalizer:
    """在线统计观测的均值与方差，逐维标准化后裁剪。"""

    def __init__(self, dim: int, clip: float = 10.0, enabled: bool = True):
        self.dim = dim
        self.clip = clip
        self.enabled = enabled
        self.mean = np.zeros(dim, dtype=np.float64)
        self.var = np.ones(dim, dtype=np.float64)
        self.count = 1e-4          # 防止除零

    def observe(self, obs) -> None:
        """用一条新观测更新统计量（Welford 递推，数值稳定）。"""
        if not self.enabled:
            return
        x = np.asarray(obs, dtype=np.float64)
        self.count += 1.0
        delta = x - self.mean
        self.mean += delta / self.count
        self.var += delta * (x - self.mean)

    def std(self) -> np.ndarray:
        return np.sqrt(self.var / self.count + 1e-8)

    def normalize(self, obs) -> np.ndarray:
        """标准化并裁剪；enabled=False 时原样返回。"""
        if not self.enabled:
            return np.asarray(obs, dtype=np.float32)
        z = (np.asarray(obs, dtype=np.float64) - self.mean) / self.std()
        return np.clip(z, -self.clip, self.clip).astype(np.float32)

    def normalize_tensor(self, obs: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return obs
        mean = torch.as_tensor(self.mean, dtype=torch.float32)
        std = torch.as_tensor(self.std(), dtype=torch.float32)
        z = (obs - mean) / std
        return torch.clamp(z, -self.clip, self.clip)


# ---------------------------------------------------------------------------
# 经验回放池：存原始观测，归一化在采样后做
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
        self.rew[i] = rew
        self.next_obs[i] = next_obs
        self.done[i] = done
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample_raw(self, batch_size: int) -> dict:
        """返回未经归一化的张量；归一化交给智能体按当前统计量处理。"""
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
# SAC 智能体：带观测归一化器
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
    def act(self, obs_norm: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """注意：入参必须是已经归一化好的观测。"""
        obs_t = torch.as_tensor(obs_norm, dtype=torch.float32).unsqueeze(0)
        action, _ = self.policy.sample(obs_t, deterministic=deterministic)
        return action.squeeze(0).numpy()

    def update(self, batch: dict, normalizer: RunningObsNormalizer) -> dict:
        # 用当前统计量归一化观测（回放池里的原始观测不受影响）
        obs = normalizer.normalize_tensor(batch["obs"])
        next_obs = normalizer.normalize_tensor(batch["next_obs"])
        act, rew, done = batch["act"], batch["rew"], batch["done"]

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

        new_act, logp = self.policy.sample(obs)
        q_new = torch.min(self.q1(obs, new_act), self.q2(obs, new_act))
        policy_loss = (self.current_alpha * logp - q_new).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

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
# 评估：使用与训练相同的归一化统计量
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(env, agent: SACAgent, episodes: int,
             normalizer: RunningObsNormalizer) -> float:
    returns = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            obs_norm = normalizer.normalize(obs)
            action = agent.act(obs_norm, deterministic=True)
            obs, r, terminated, truncated, _ = env.step(action)
            ep_ret += r
            done = terminated or truncated
        returns.append(ep_ret)
    return float(np.mean(returns))


# ---------------------------------------------------------------------------
# 单种归一化设置的训练
# ---------------------------------------------------------------------------
def train_norm(mode: str, args) -> dict:
    set_seed(args.seed)
    env = make_env("Pendulum-v1", args.seed)
    eval_env = make_env("Pendulum-v1", args.seed + 500)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_scale = float(env.action_space.high[0])

    normalizer = RunningObsNormalizer(obs_dim, clip=args.clip, enabled=(mode == "run"))
    agent = SACAgent(obs_dim, act_dim, act_scale, alpha=args.alpha,
                     target_entropy=args.target_entropy, lr=args.lr,
                     gamma=args.gamma, tau=args.tau, hidden=args.hidden)
    buf = ReplayBuffer(args.buffer_size, obs_dim, act_dim)

    history = {"steps": [], "eval_return": [], "mean_abs_q": [], "std_tail": []}
    ep_returns = []
    obs, _ = env.reset(seed=args.seed)
    ep_ret = 0.0
    last_stats = {"mean_abs_q": 0.0}
    eval_ret = float("nan")
    t_start = time.time()

    print(f"[{mode}] 开始训练 | Pendulum-v1 | 种子 {args.seed} | "
          f"归一化={'开' if normalizer.enabled else '关'}")

    for step in range(1, args.max_steps + 1):
        if step <= args.start_steps:
            action = env.action_space.sample()
        else:
            obs_norm = normalizer.normalize(obs)
            action = agent.act(obs_norm, deterministic=False)

        next_obs, rew, terminated, truncated, _ = env.step(action)
        # 先更新统计量再存原始观测；采样时再按最新统计量归一化
        normalizer.observe(next_obs)
        buf.add(obs, action, rew, next_obs, float(terminated))
        obs = next_obs
        ep_ret += rew

        if terminated or truncated:
            ep_returns.append(ep_ret)
            ep_ret = 0.0
            obs, _ = env.reset()

        if buf.size >= args.batch_size:
            last_stats = agent.update(buf.sample_raw(args.batch_size), normalizer)

        if step % args.eval_every == 0 or step == args.max_steps:
            eval_ret = evaluate(eval_env, agent, args.eval_episodes, normalizer)
            recent = float(np.mean(ep_returns[-20:])) if ep_returns else float("nan")
            history["steps"].append(step)
            history["eval_return"].append(eval_ret)
            history["mean_abs_q"].append(last_stats["mean_abs_q"])
            history["std_tail"].append(float(normalizer.std()[-1]))
            print(f"[{mode}] 步数 {step:6d} | 近20均值 {recent:9.1f} | "
                  f"评估 {eval_ret:9.1f} | |Q| {last_stats['mean_abs_q']:7.1f} | "
                  f"theta_dot 估计std {normalizer.std()[-1]:.3f}")

    elapsed = time.time() - t_start
    print(f"[{mode}] 训练结束 | 用时 {elapsed:.1f}s | 最终评估 {eval_ret:.1f}")

    os.makedirs(args.save_dir, exist_ok=True)
    ckpt = os.path.join(args.save_dir, f"sac_obsnorm_{mode}.pt")
    torch.save({"policy": agent.policy.state_dict(),
                "q1": agent.q1.state_dict(),
                "obs_mean": normalizer.mean,
                "obs_std": normalizer.std(),
                "mode": mode}, ckpt)
    print(f"[{mode}] 模型与归一化统计量已保存到 {ckpt}")

    return {"mode": mode, "final_eval": eval_ret, "elapsed": elapsed,
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
                     label=res["mode"])
        axes[1].plot(hist["steps"], hist["std_tail"], marker="o",
                     label=res["mode"])
    axes[0].set_xlabel("环境步数")
    axes[0].set_ylabel("确定性评估回报")
    axes[0].set_title("评估回报")
    axes[0].legend()
    axes[1].set_xlabel("环境步数")
    axes[1].set_ylabel("theta_dot 维度的估计标准差")
    axes[1].set_title("在线统计量表")
    axes[1].legend()
    fig.tight_layout()
    out = os.path.join(save_dir, "ch039_obs_norm.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"曲线已保存到 {out}")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="第039章：SAC 的观测归一化")
    p.add_argument("--obs-norms", type=str, default="none,run",
                   help="要对比的观测处理：none=原始，run=在线归一化")
    p.add_argument("--clip", type=float, default=10.0, help="归一化后的裁剪范围")
    p.add_argument("--alpha", type=float, default=0.2, help="alpha 初值")
    p.add_argument("--target-entropy", type=float, default=-1.0, help="目标熵")
    p.add_argument("--max-steps", type=int, default=15000, help="每种设置的步数上限")
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
    p.add_argument("--save-dir", type=str, default="runs/ch039", help="模型保存目录")
    p.add_argument("--quick", action="store_true", help="快速模式：几十秒跑完")
    return p.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.max_steps = min(args.max_steps, 2500)
        args.start_steps = min(args.start_steps, 500)
        args.eval_every = 1200
        args.eval_episodes = 2

    modes = [m.strip() for m in args.obs_norms.split(",") if m.strip()]
    for m in modes:
        if m not in ALL_NORMS:
            raise SystemExit(f"未知设置 {m}，可选：{ALL_NORMS}")

    results = [train_norm(m, args) for m in modes]

    print("=" * 74)
    print("观测归一化对比汇总（Pendulum-v1，以实跑为准）")
    for res in results:
        hist = res["history"]
        print(f"  {res['mode']:>4s}: 最终评估 {res['final_eval']:8.1f} | "
              f"评估轨迹 {[round(r, 0) for r in hist['eval_return']]}")

    maybe_plot(results, args.save_dir)


if __name__ == "__main__":
    main()
