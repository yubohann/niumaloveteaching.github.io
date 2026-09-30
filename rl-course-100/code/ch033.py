"""
第033章 SAC的双Q网络与目标网络

SAC 的稳定性靠两根柱子撑着：
  1) 双 Q 网络：训练两个独立的 Q 网络，自举目标取两者较小值（clipped
     double-Q），抑制"最大化操作 + 估计误差"带来的过估计；
  2) 目标网络：用一份滞后参数计算自举目标（软更新），避免"追着自己
     的尾巴更新"导致的发散。

本章在 Pendulum-v1 上做三组消融实验，其余代码完全一致：
    double_target  : 双 Q + 目标网络（SAC 默认配置）
    single_target  : 只用 Q1 + 目标网络（验证双 Q 的作用）
    double_notarget: 双 Q，但自举直接用在线网络（验证目标网络的作用）

对比指标：确定性评估回报、|Q| 的平均幅度（发散时会爆掉）、
          Q 损失的大小。

运行：
    python code/ch033.py           # 三组消融依次训练（CPU 约 6-10 分钟）
    python code/ch033.py --quick   # 快速跑通（约 1 分钟，只验证流程）
    python code/ch033.py --configs double_target --max-steps 20000   # 只跑一组

预期：double_target 稳定收敛；single_target 也能学会但波动更大、Q 幅度偏高；
      double_notarget 的 Q 幅度很快发散，回报崩坏。
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
ALL_CONFIGS = ["double_target", "single_target", "double_notarget"]


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    """固定随机种子，保证三组消融从同一起点出发。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_env(env_id: str, seed: int):
    env = gym.make(env_id)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env


# ---------------------------------------------------------------------------
# 经验回放池
# ---------------------------------------------------------------------------
class ReplayBuffer:
    """定长回放池：预分配数组，均匀采样。"""

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
        # 只记真实终止；Pendulum 200 步截断仍然要 bootstrap
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
    """动作价值网络 Q(s,a)。"""

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
    """tanh 压缩高斯策略。"""

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
# SAC 智能体：自举方式由 config 决定
# ---------------------------------------------------------------------------
class SACAgent:
    def __init__(self, obs_dim: int, act_dim: int, act_scale: float,
                 config: str, alpha: float = 0.2, target_entropy: float = -1.0,
                 lr: float = 3e-4, gamma: float = 0.99, tau: float = 0.005,
                 hidden: int = 256):
        assert config in ALL_CONFIGS, f"未知配置: {config}"
        self.config = config
        self.use_double = config != "single_target"   # 是否有第二个 Q
        self.use_target = config != "double_notarget"  # 是否使用目标网络
        self.alpha = float(alpha)
        self.target_entropy = float(target_entropy)
        self.gamma = gamma
        self.tau = tau

        self.q1 = QNet(obs_dim, act_dim, hidden)
        self.q2 = QNet(obs_dim, act_dim, hidden) if self.use_double else None
        if self.use_target:
            self.q1_target = QNet(obs_dim, act_dim, hidden)
            self.q1_target.load_state_dict(self.q1.state_dict())
            for p in self.q1_target.parameters():
                p.requires_grad_(False)
            if self.use_double:
                self.q2_target = QNet(obs_dim, act_dim, hidden)
                self.q2_target.load_state_dict(self.q2.state_dict())
                for p in self.q2_target.parameters():
                    p.requires_grad_(False)
        self.policy = GaussianPolicy(obs_dim, act_dim, hidden, act_scale)

        q_params = list(self.q1.parameters())
        if self.use_double:
            q_params += list(self.q2.parameters())
        self.q_optimizer = torch.optim.Adam(q_params, lr=lr)
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

    def _q_values_online(self, obs: torch.Tensor, act: torch.Tensor):
        """返回 (min 或唯一) 在线 Q 值，用于策略损失。"""
        q1 = self.q1(obs, act)
        if self.use_double:
            q2 = self.q2(obs, act)
            return torch.min(q1, q2), q1, q2
        return q1, q1, None

    def update(self, batch: dict) -> dict:
        obs, act, rew = batch["obs"], batch["act"], batch["rew"]
        next_obs, done = batch["next_obs"], batch["done"]

        # ---------------- 1) 评论家 ----------------
        with torch.no_grad():
            next_act, next_logp = self.policy.sample(next_obs)
            if self.use_target:
                if self.use_double:
                    q_next = torch.min(self.q1_target(next_obs, next_act),
                                       self.q2_target(next_obs, next_act))
                else:
                    q_next = self.q1_target(next_obs, next_act)
            else:
                # 消融：直接用在线网络做自举
                if self.use_double:
                    q_next = torch.min(self.q1(next_obs, next_act),
                                       self.q2(next_obs, next_act))
                else:
                    q_next = self.q1(next_obs, next_act)
            target = rew + self.gamma * (1.0 - done) * (
                q_next - self.current_alpha * next_logp)

        q1_pred = self.q1(obs, act)
        if self.use_double:
            q2_pred = self.q2(obs, act)
            q_loss = F.mse_loss(q1_pred, target) + F.mse_loss(q2_pred, target)
            q_abs_mean = 0.5 * (q1_pred.abs().mean() + q2_pred.abs().mean())
        else:
            q_loss = F.mse_loss(q1_pred, target)
            q_abs_mean = q1_pred.abs().mean()

        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # ---------------- 2) 演员 ----------------
        new_act, logp = self.policy.sample(obs)
        q_new, _, _ = self._q_values_online(obs, new_act)
        policy_loss = (self.current_alpha * logp - q_new).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # ---------------- 3) 温度系数（自动调节） ----------------
        alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # ---------------- 4) 目标网络软更新 ----------------
        if self.use_target:
            with torch.no_grad():
                for p, tp in zip(self.q1.parameters(), self.q1_target.parameters()):
                    tp.mul_(1.0 - self.tau).add_(self.tau * p)
                if self.use_double:
                    for p, tp in zip(self.q2.parameters(), self.q2_target.parameters()):
                        tp.mul_(1.0 - self.tau).add_(self.tau * p)

        # 诊断：目标滞后量 = 在线 Q 与目标 Q 的平均绝对差（无目标网络时为 0）
        target_gap = 0.0
        if self.use_target:
            with torch.no_grad():
                gap = (self.q1(obs, act) - self.q1_target(obs, act)).abs().mean()
                target_gap = float(gap.item())

        return {
            "q_loss": float(q_loss.item()),
            "policy_loss": float(policy_loss.item()),
            "alpha_loss": float(alpha_loss.item()),
            "mean_abs_q": float(q_abs_mean.item()),
            "target_gap": target_gap,
            "alpha": self.current_alpha,
        }


# ---------------------------------------------------------------------------
# 评估
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
            ep_ret += r
            done = terminated or truncated
        returns.append(ep_ret)
    return float(np.mean(returns))


# ---------------------------------------------------------------------------
# 单组配置的训练
# ---------------------------------------------------------------------------
def train_config(config: str, args) -> dict:
    set_seed(args.seed)                       # 三组消融共享同一种子
    env = make_env("Pendulum-v1", args.seed)
    eval_env = make_env("Pendulum-v1", args.seed + 500)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_scale = float(env.action_space.high[0])

    agent = SACAgent(obs_dim, act_dim, act_scale, config=config,
                     alpha=args.alpha, target_entropy=args.target_entropy,
                     lr=args.lr, gamma=args.gamma, tau=args.tau,
                     hidden=args.hidden)
    buf = ReplayBuffer(args.buffer_size, obs_dim, act_dim)

    history = {"steps": [], "eval_return": [], "mean_abs_q": [], "q_loss": []}
    ep_returns = []
    obs, _ = env.reset(seed=args.seed)
    ep_ret = 0.0
    last_stats = {"mean_abs_q": 0.0, "q_loss": 0.0, "target_gap": 0.0}
    eval_ret = float("nan")
    t_start = time.time()

    print(f"[{config}] 开始训练 | Pendulum-v1 | 种子 {args.seed} | "
          f"双Q={agent.use_double} 目标网络={agent.use_target}")

    for step in range(1, args.max_steps + 1):
        if step <= args.start_steps:
            action = env.action_space.sample()
        else:
            action = agent.act(obs, deterministic=False)

        next_obs, rew, terminated, truncated, _ = env.step(action)
        buf.add(obs, action, rew, next_obs, float(terminated))
        obs = next_obs
        ep_ret += rew

        if terminated or truncated:
            ep_returns.append(ep_ret)
            ep_ret = 0.0
            obs, _ = env.reset()

        if buf.size >= args.batch_size:
            last_stats = agent.update(buf.sample(args.batch_size))

        if step % args.eval_every == 0 or step == args.max_steps:
            eval_ret = evaluate(eval_env, agent, args.eval_episodes)
            recent = float(np.mean(ep_returns[-20:])) if ep_returns else float("nan")
            history["steps"].append(step)
            history["eval_return"].append(eval_ret)
            history["mean_abs_q"].append(last_stats["mean_abs_q"])
            history["q_loss"].append(last_stats["q_loss"])
            print(f"[{config}] 步数 {step:6d} | 近20均值 {recent:8.1f} | "
                  f"评估 {eval_ret:9.1f} | |Q| {last_stats['mean_abs_q']:8.1f} | "
                  f"Q损失 {last_stats['q_loss']:9.2f} | 目标滞后 {last_stats['target_gap']:.3f}")

    elapsed = time.time() - t_start
    print(f"[{config}] 训练结束 | 用时 {elapsed:.1f}s | 最终评估 {eval_ret:.1f} | "
          f"|Q| {last_stats['mean_abs_q']:.1f}")

    os.makedirs(args.save_dir, exist_ok=True)
    ckpt = os.path.join(args.save_dir, f"sac_{config}.pt")
    torch.save({"policy": agent.policy.state_dict(),
                "q1": agent.q1.state_dict(),
                "config": config}, ckpt)
    print(f"[{config}] 模型已保存到 {ckpt}")

    return {"config": config, "final_eval": eval_ret, "elapsed": elapsed,
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
                     label=res["config"])
        axes[1].plot(hist["steps"], hist["mean_abs_q"], marker="o",
                     label=res["config"])
    axes[0].set_xlabel("环境步数")
    axes[0].set_ylabel("确定性评估回报")
    axes[0].set_title("评估回报")
    axes[0].legend()
    axes[1].set_xlabel("环境步数")
    axes[1].set_ylabel("|Q| 平均值")
    axes[1].set_title("Q 值幅度（发散诊断）")
    axes[1].legend()
    fig.tight_layout()
    out = os.path.join(save_dir, "ch033_ablation.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"曲线已保存到 {out}")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="第033章：SAC 的双 Q 网络与目标网络")
    p.add_argument("--configs", type=str, default="double_target,single_target,double_notarget",
                   help="要运行的消融配置，逗号分隔")
    p.add_argument("--alpha", type=float, default=0.2, help="alpha 初值")
    p.add_argument("--target-entropy", type=float, default=-1.0, help="目标熵")
    p.add_argument("--max-steps", type=int, default=15000, help="每组配置的步数上限")
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
    p.add_argument("--save-dir", type=str, default="runs/ch033", help="模型保存目录")
    p.add_argument("--quick", action="store_true", help="快速模式：几十秒跑完")
    return p.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.max_steps = min(args.max_steps, 2500)
        args.start_steps = min(args.start_steps, 500)
        args.eval_every = 1200
        args.eval_episodes = 2

    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    for c in configs:
        if c not in ALL_CONFIGS:
            raise SystemExit(f"未知配置 {c}，可选：{ALL_CONFIGS}")

    results = [train_config(c, args) for c in configs]

    print("=" * 74)
    print("消融汇总（Pendulum-v1，数值随机种子波动，以实跑为准）")
    for res in results:
        hist = res["history"]
        print(f"  {res['config']:>16s}: 最终评估 {res['final_eval']:9.1f} | "
              f"|Q| 轨迹 {[round(q, 1) for q in hist['mean_abs_q']]}")

    maybe_plot(results, args.save_dir)


if __name__ == "__main__":
    main()
