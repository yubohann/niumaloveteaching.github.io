"""
第032章 SAC的自动温度系数调整

SAC（Soft Actor-Critic）在策略目标里用温度系数 alpha 加权熵项：
    J(pi) = E[ Q(s,a) - alpha * log pi(a|s) ]
alpha 越大越鼓励探索。手工调 alpha 对环境敏感，本章实现 SAC 原文中的
自动温度调节：把 log_alpha 当作可学习参数，用
    L(alpha) = -E[ log_alpha * (log pi(a|s) + target_entropy) ]
（其中 log pi 要 detach，梯度只流向 alpha）让 alpha 自动追一个目标熵。
target_entropy 取 -动作维度，即连续动作下常见的启发式目标。

实验：在 Pendulum-v1 上依次运行
    fixed: 固定 alpha = 0.2
    auto : 自动调节（target_entropy = -1）
并对比评估回报、alpha 轨迹、平均熵与 |Q| 量级。

运行：
    python code/ch032.py           # 两种模式依次训练（CPU 约 4-8 分钟）
    python code/ch032.py --quick   # 快速跑通（约 1 分钟，只验证流程）
    python code/ch032.py --mode auto --max-steps 40000   # 只跑一种模式

预期：auto 模式下 alpha 从 0.2 附近自动先升后降；两种模式最终评估回报
      接近（Pendulum 上典型为 -300 ~ -150），但 auto 模式省去人工调参。
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
    """固定随机种子，保证实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_env(env_id: str, seed: int):
    """创建环境并锁定动作空间随机种子。"""
    env = gym.make(env_id)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env


# ---------------------------------------------------------------------------
# 经验回放池
# ---------------------------------------------------------------------------
class ReplayBuffer:
    """定长回放池：预分配 numpy 数组，add 为 O(1)，sample 均匀采样。"""

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
        # done 只记录真实终止（terminated）；时间截断（truncated）继续 bootstrap
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
    """动作价值网络 Q(s,a)：把状态与动作拼接后过 MLP，输出标量。"""

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
    """tanh 压缩高斯策略：输出均值与 log 标准差，重参数化采样。"""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256,
                 act_scale: float = 1.0):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mean_layer = nn.Linear(hidden, act_dim)
        self.log_std_layer = nn.Linear(hidden, act_dim)
        # tanh 输出在 [-1,1]，乘 act_scale 映射到环境动作范围（Pendulum 为 [-2,2]）
        self.register_buffer("act_scale", torch.tensor(float(act_scale)))

    def forward(self, obs: torch.Tensor):
        h = self.body(obs)
        mean = self.mean_layer(h)
        log_std = torch.clamp(self.log_std_layer(h), LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, obs: torch.Tensor, deterministic: bool = False):
        """返回 (动作, log 概率)。确定性模式下只返回动作与 None。"""
        mean, log_std = self.forward(obs)
        if deterministic:
            return torch.tanh(mean) * self.act_scale, None
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        x = dist.rsample()                      # 重参数化：梯度可穿过采样
        y = torch.tanh(x)
        action = y * self.act_scale
        # tanh 变换后的对数概率，减去雅可比修正项 log(1 - y^2)
        log_prob = dist.log_prob(x) - torch.log(1.0 - y.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1)         # 各动作维度求和 -> (B,)
        return action, log_prob


# ---------------------------------------------------------------------------
# SAC 智能体：本章的重点是 alpha 的处理方式
# ---------------------------------------------------------------------------
class SACAgent:
    def __init__(self, obs_dim: int, act_dim: int, act_scale: float,
                 mode: str, alpha: float, target_entropy: float,
                 lr: float = 3e-4, alpha_lr: float = 3e-4,
                 gamma: float = 0.99, tau: float = 0.005, hidden: int = 256):
        self.mode = mode                    # "fixed" 或 "auto"
        self.alpha = float(alpha)           # 固定模式下用这个值
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

        # 自动模式下：log_alpha 是可学习参数，用独立优化器更新
        if self.mode == "auto":
            self.log_alpha = torch.tensor(math.log(alpha), requires_grad=True)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

    @property
    def current_alpha(self) -> float:
        if self.mode == "auto":
            return float(self.log_alpha.exp().item())
        return self.alpha

    @torch.no_grad()
    def act(self, obs, deterministic: bool = False) -> np.ndarray:
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        action, _ = self.policy.sample(obs_t, deterministic=deterministic)
        return action.squeeze(0).numpy()

    def update(self, batch: dict) -> dict:
        obs, act, rew = batch["obs"], batch["act"], batch["rew"]
        next_obs, done = batch["next_obs"], batch["done"]

        # ---------------- 1) 评论家：双 Q 的 Bellman 残差 ----------------
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

        # ---------------- 2) 演员：最大化 Q - alpha * log pi ----------------
        new_act, logp = self.policy.sample(obs)
        q_new = torch.min(self.q1(obs, new_act), self.q2(obs, new_act))
        policy_loss = (self.current_alpha * logp - q_new).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # ---------------- 3) 温度系数：本章的核心差异 ----------------
        alpha_loss_val = 0.0
        if self.mode == "auto":
            # logp 视为常数（detach），只把 log_alpha 往"熵缺口"方向推
            alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha_loss_val = float(alpha_loss.item())

        # ---------------- 4) 目标网络软更新 ----------------
        with torch.no_grad():
            for p, tp in zip(self.q1.parameters(), self.q1_target.parameters()):
                tp.mul_(1.0 - self.tau).add_(self.tau * p)
            for p, tp in zip(self.q2.parameters(), self.q2_target.parameters()):
                tp.mul_(1.0 - self.tau).add_(self.tau * p)

        return {
            "q_loss": float(q_loss.item()),
            "policy_loss": float(policy_loss.item()),
            "alpha_loss": alpha_loss_val,
            "mean_q": float(q1_pred.mean().item()),
            "mean_abs_q": float(q1_pred.abs().mean().item()),
            "entropy": float(-logp.mean().item()),
            "alpha": self.current_alpha,
        }


# ---------------------------------------------------------------------------
# 评估
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(env, agent: SACAgent, episodes: int) -> float:
    """用确定性策略（tanh(mean) * scale）跑若干回合，返回平均原始回报。"""
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
# 单种模式的训练
# ---------------------------------------------------------------------------
def train_mode(mode: str, args) -> dict:
    """训练一种 alpha 模式，返回历史记录与最终评估回报。"""
    set_seed(args.seed)
    env = make_env("Pendulum-v1", args.seed)
    eval_env = make_env("Pendulum-v1", args.seed + 500)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_scale = float(env.action_space.high[0])      # Pendulum 动作上限 2.0

    agent = SACAgent(obs_dim, act_dim, act_scale, mode=mode,
                     alpha=args.alpha, target_entropy=args.target_entropy,
                     lr=args.lr, alpha_lr=args.alpha_lr, gamma=args.gamma,
                     tau=args.tau, hidden=args.hidden)
    buf = ReplayBuffer(args.buffer_size, obs_dim, act_dim)

    history = {"steps": [], "eval_return": [], "alpha": [], "entropy": []}
    ep_returns = []
    obs, _ = env.reset(seed=args.seed)
    ep_ret = 0.0
    last_stats = {"mean_abs_q": 0.0, "entropy": 0.0}
    eval_ret = float("nan")
    t_start = time.time()

    print(f"[{mode}] 开始训练 | 环境 Pendulum-v1 | 种子 {args.seed} | "
          f"alpha 初值 {args.alpha} | target_entropy {args.target_entropy}")

    for step in range(1, args.max_steps + 1):
        # 前 start_steps 步纯随机探索，先把回放池垫起来
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
            history["alpha"].append(agent.current_alpha)
            history["entropy"].append(last_stats["entropy"])
            print(f"[{mode}] 步数 {step:6d} | 近20回合均值 {recent:8.1f} | "
                  f"评估 {eval_ret:8.1f} | alpha {agent.current_alpha:.3f} | "
                  f"熵 {last_stats['entropy']:.3f} | |Q| {last_stats['mean_abs_q']:.1f}")

    elapsed = time.time() - t_start
    print(f"[{mode}] 训练结束 | 用时 {elapsed:.1f}s | 最终评估回报 {eval_ret:.1f} | "
          f"最终 alpha {agent.current_alpha:.3f}")

    os.makedirs(args.save_dir, exist_ok=True)
    ckpt = os.path.join(args.save_dir, f"sac_{mode}.pt")
    torch.save({"policy": agent.policy.state_dict(),
                "q1": agent.q1.state_dict(),
                "q2": agent.q2.state_dict(),
                "alpha": agent.current_alpha}, ckpt)
    print(f"[{mode}] 模型已保存到 {ckpt}")

    return {"mode": mode, "final_eval": eval_ret, "elapsed": elapsed,
            "history": history}


# ---------------------------------------------------------------------------
# 可选绘图（惰性导入 matplotlib，Agg 后端，只写文件不弹窗）
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
        axes[1].plot(hist["steps"], hist["alpha"], marker="o",
                     label=res["mode"])
    axes[0].set_xlabel("环境步数")
    axes[0].set_ylabel("确定性评估回报")
    axes[0].set_title("评估回报对比")
    axes[0].legend()
    axes[1].set_xlabel("环境步数")
    axes[1].set_ylabel("alpha")
    axes[1].set_title("温度系数轨迹")
    axes[1].legend()
    fig.tight_layout()
    out = os.path.join(save_dir, "ch032_alpha.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"曲线已保存到 {out}")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="第032章：SAC 的自动温度系数调整")
    p.add_argument("--mode", type=str, default="both",
                   choices=["both", "fixed", "auto"],
                   help="both=依次训练两种模式；fixed=固定 alpha；auto=自动调节")
    p.add_argument("--alpha", type=float, default=0.2, help="alpha 初值（auto 模式的起点）")
    p.add_argument("--alpha-lr", type=float, default=3e-4, help="log_alpha 的学习率")
    p.add_argument("--target-entropy", type=float, default=-1.0,
                   help="目标熵，默认 -动作维度")
    p.add_argument("--max-steps", type=int, default=20000,
                   help="每种模式的环境步数上限")
    p.add_argument("--start-steps", type=int, default=1000,
                   help="开始训练前的纯随机探索步数")
    p.add_argument("--buffer-size", type=int, default=100000, help="回放池容量")
    p.add_argument("--batch-size", type=int, default=256, help="批量大小")
    p.add_argument("--hidden", type=int, default=256, help="隐藏层宽度")
    p.add_argument("--lr", type=float, default=3e-4, help="网络学习率")
    p.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
    p.add_argument("--tau", type=float, default=0.005, help="目标网络软更新系数")
    p.add_argument("--eval-every", type=int, default=4000, help="评估与日志间隔（步）")
    p.add_argument("--eval-episodes", type=int, default=3, help="每次评估的回合数")
    p.add_argument("--seed", type=int, default=0, help="随机种子")
    p.add_argument("--save-dir", type=str, default="runs/ch032", help="模型保存目录")
    p.add_argument("--quick", action="store_true", help="快速模式：几十秒跑完")
    return p.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.max_steps = min(args.max_steps, 3000)
        args.start_steps = min(args.start_steps, 500)
        args.eval_every = 1500
        args.eval_episodes = 2

    modes = ["fixed", "auto"] if args.mode == "both" else [args.mode]
    results = []
    for mode in modes:
        results.append(train_mode(mode, args))

    print("=" * 74)
    print("对比汇总（Pendulum-v1，数值随机种子波动，以实跑为准）")
    for res in results:
        print(f"  {res['mode']:>5s}: 最终评估回报 {res['final_eval']:8.1f} | "
              f"用时 {res['elapsed']:5.1f}s | alpha 轨迹 "
              f"{[round(a, 3) for a in res['history']['alpha']]}")
    if len(results) == 2:
        diff = results[1]["final_eval"] - results[0]["final_eval"]
        print(f"  auto - fixed 的评估回报差: {diff:+.1f}")

    maybe_plot(results, args.save_dir)


if __name__ == "__main__":
    main()
