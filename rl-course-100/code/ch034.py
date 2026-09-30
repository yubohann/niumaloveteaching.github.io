"""
第034章 SAC的延迟策略更新

标准 SAC 每个环境步都做一次"评论家 + 演员 + 温度系数"更新。TD3 的经验
是：策略网络不要跟着评论家每一步都动，而是延迟若干步再更新一次，等
评论家的价值估计更可信之后再让策略去追。本章把 SAC 的更新拆成两部分：
    update_critic(): 每步都调用（含目标网络软更新）
    update_actor() : 每 policy_delay 步调用一次（策略损失 + alpha 损失）
然后在 Pendulum-v1 上对比 policy_delay = 1 / 2 / 4 三种节奏：
    delay=1 是标准 SAC；delay=2 是常见的 TD3 式设定；delay=4 更激进。

对比指标：确定性评估回报、策略更新次数、|Q| 幅度、每次策略更新的损失。

运行：
    python code/ch034.py           # 三种延迟依次训练（CPU 约 6-10 分钟）
    python code/ch034.py --quick   # 快速跑通（约 1 分钟，只验证流程）
    python code/ch034.py --delays 1,2 --max-steps 30000   # 只对比 1 和 2

预期：Pendulum 上三种节奏差距不大（±50 以内），delay=4 学习速度略慢；
      延迟更新的真正价值在高估计噪声的环境里更明显。
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
# SAC 智能体：评论家与演员可以分步调用（本章核心改动）
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
        self.actor_updates = 0            # 策略被更新的次数（诊断用）

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

    def update_critic(self, batch: dict) -> dict:
        """评论家更新：每步都调用；同时做目标网络软更新。"""
        obs, act, rew = batch["obs"], batch["act"], batch["rew"]
        next_obs, done = batch["next_obs"], batch["done"]

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

        # 目标网络软更新跟着评论家走
        with torch.no_grad():
            for p, tp in zip(self.q1.parameters(), self.q1_target.parameters()):
                tp.mul_(1.0 - self.tau).add_(self.tau * p)
            for p, tp in zip(self.q2.parameters(), self.q2_target.parameters()):
                tp.mul_(1.0 - self.tau).add_(self.tau * p)

        return {"q_loss": float(q_loss.item()),
                "mean_abs_q": float(q1_pred.abs().mean().item())}

    def update_actor(self, batch: dict) -> dict:
        """演员 + 温度系数更新：由调用方按 policy_delay 控制频率。"""
        obs = batch["obs"]
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

        self.actor_updates += 1
        return {"policy_loss": float(policy_loss.item()),
                "alpha_loss": float(alpha_loss.item()),
                "mean_logp": float(logp.mean().item())}


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
# 单种延迟的训练
# ---------------------------------------------------------------------------
def train_delay(delay: int, args) -> dict:
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

    history = {"steps": [], "eval_return": [], "mean_abs_q": [],
               "actor_updates": [], "alpha": []}
    ep_returns = []
    obs, _ = env.reset(seed=args.seed)
    ep_ret = 0.0
    last_critic = {"mean_abs_q": 0.0, "q_loss": 0.0}
    last_actor = {"policy_loss": 0.0}
    critic_steps = 0
    eval_ret = float("nan")
    t_start = time.time()

    print(f"[delay={delay}] 开始训练 | Pendulum-v1 | 种子 {args.seed}")

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

        # 更新节奏：评论家每步更新；演员每 delay 次评论家更新一次
        if buf.size >= args.batch_size:
            last_critic = agent.update_critic(buf.sample(args.batch_size))
            critic_steps += 1
            if critic_steps % delay == 0:
                # 策略更新用一份新采样的 batch，避免复用评论家那批数据
                last_actor = agent.update_actor(buf.sample(args.batch_size))

        if step % args.eval_every == 0 or step == args.max_steps:
            eval_ret = evaluate(eval_env, agent, args.eval_episodes)
            recent = float(np.mean(ep_returns[-20:])) if ep_returns else float("nan")
            history["steps"].append(step)
            history["eval_return"].append(eval_ret)
            history["mean_abs_q"].append(last_critic["mean_abs_q"])
            history["actor_updates"].append(agent.actor_updates)
            history["alpha"].append(agent.current_alpha)
            print(f"[delay={delay}] 步数 {step:6d} | 近20均值 {recent:8.1f} | "
                  f"评估 {eval_ret:9.1f} | 演员更新 {agent.actor_updates:5d} | "
                  f"|Q| {last_critic['mean_abs_q']:7.1f} | "
                  f"策略损失 {last_actor['policy_loss']:8.3f} | alpha {agent.current_alpha:.3f}")

    elapsed = time.time() - t_start
    print(f"[delay={delay}] 训练结束 | 用时 {elapsed:.1f}s | 最终评估 {eval_ret:.1f} | "
          f"演员更新次数 {agent.actor_updates}")

    os.makedirs(args.save_dir, exist_ok=True)
    ckpt = os.path.join(args.save_dir, f"sac_delay{delay}.pt")
    torch.save({"policy": agent.policy.state_dict(),
                "q1": agent.q1.state_dict(),
                "q2": agent.q2.state_dict(),
                "delay": delay}, ckpt)
    print(f"[delay={delay}] 模型已保存到 {ckpt}")

    return {"delay": delay, "final_eval": eval_ret, "elapsed": elapsed,
            "actor_updates": agent.actor_updates, "history": history}


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
                     label=f"delay={res['delay']}")
        axes[1].plot(hist["steps"], hist["mean_abs_q"], marker="o",
                     label=f"delay={res['delay']}")
    axes[0].set_xlabel("环境步数")
    axes[0].set_ylabel("确定性评估回报")
    axes[0].set_title("评估回报")
    axes[0].legend()
    axes[1].set_xlabel("环境步数")
    axes[1].set_ylabel("|Q| 平均值")
    axes[1].set_title("价值估计幅度")
    axes[1].legend()
    fig.tight_layout()
    out = os.path.join(save_dir, "ch034_delay.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"曲线已保存到 {out}")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="第034章：SAC 的延迟策略更新")
    p.add_argument("--delays", type=str, default="1,2,4",
                   help="要对比的策略延迟（评论家每更新 k 次，演员更新 1 次）")
    p.add_argument("--alpha", type=float, default=0.2, help="alpha 初值")
    p.add_argument("--target-entropy", type=float, default=-1.0, help="目标熵")
    p.add_argument("--max-steps", type=int, default=15000, help="每种延迟的步数上限")
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
    p.add_argument("--save-dir", type=str, default="runs/ch034", help="模型保存目录")
    p.add_argument("--quick", action="store_true", help="快速模式：几十秒跑完")
    return p.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.max_steps = min(args.max_steps, 2500)
        args.start_steps = min(args.start_steps, 500)
        args.eval_every = 1200
        args.eval_episodes = 2

    delays = [int(x) for x in args.delays.split(",") if x.strip()]
    for d in delays:
        if d < 1:
            raise SystemExit("delay 必须是不小于 1 的整数")

    results = [train_delay(d, args) for d in delays]

    print("=" * 74)
    print("延迟更新对比汇总（Pendulum-v1，数值随机种子波动，以实跑为准）")
    for res in results:
        print(f"  delay={res['delay']}: 最终评估 {res['final_eval']:9.1f} | "
              f"演员更新 {res['actor_updates']:5d} 次 | 用时 {res['elapsed']:5.1f}s")

    maybe_plot(results, args.save_dir)


if __name__ == "__main__":
    main()
