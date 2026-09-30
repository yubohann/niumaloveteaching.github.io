"""
第040章 SAC的奖励归一化

第 035 章证明"奖励乘固定常数"会改变 SAC 的有效温度（等价于 alpha 除以 c）。
那么让缩放系数自己动起来呢？本章实现两种在线奖励归一化：
    none      : 不缩放（对照组）
    step_std  : 用"每步奖励"的运行标准差做缩放，scale = 1 / std(r)
    return_std: 用"折扣回合回报"的运行标准差做缩放，
                scale = 1 / std(G)（PPO 常用做法，搬到 SAC 上要小心）
统计量都用 Welford 递推在线维护，只观察原始奖励/原始回报，避免反馈回路。
训练奖励乘 scale 后进回放池；评估始终在原始奖励上进行。

实验在 Pendulum-v1 上对比三种设置，重点观察：
    - scale 轨迹：step_std 稳定在 0.3 附近，return_std 会掉到 0.01 以下；
    - 评估回报：return_std 因为把奖励压得过小而明显落后；
    - auto alpha 的联动：奖励被压小，alpha 会被自动压低来"补偿"。

运行：
    python code/ch040.py           # 三种设置依次训练（CPU 约 6-10 分钟）
    python code/ch040.py --quick   # 快速跑通（约 1 分钟）
    python code/ch040.py --reward-norms step_std --max-steps 30000

预期：none 与 step_std 接近（step_std 可能略慢），return_std 早期明显
      落后；这正是"把 PPO 的回报归一化直接搬给 SAC"时的经典坑。
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
ALL_NORMS = ["none", "step_std", "return_std"]


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
# 奖励归一化器：Welford 在线统计
# ---------------------------------------------------------------------------
class RewardNormalizer:
    """按模式维护运行统计，输出奖励缩放系数 scale。

    none      : scale 恒为 1
    step_std  : 统计每步奖励 r，scale = clamp(1/std(r), 0.05, 5)
    return_std: 统计折扣回合回报 G，scale = clamp(1/std(G), 1e-3, 100)
    """

    def __init__(self, mode: str, clip_low: float = 0.0, clip_high: float = 0.0):
        assert mode in ALL_NORMS
        self.mode = mode
        self.mean = 0.0
        self.m2 = 0.0
        self.count = 1e-4
        self.clip_low = clip_low
        self.clip_high = clip_high
        if mode == "step_std":
            self.clip_low, self.clip_high = 0.05, 5.0
        elif mode == "return_std":
            self.clip_low, self.clip_high = 1e-3, 100.0

    def _update(self, x: float) -> None:
        self.count += 1.0
        delta = x - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (x - self.mean)

    def observe_step(self, rew: float) -> None:
        if self.mode == "step_std":
            self._update(rew)

    def observe_return(self, ret: float) -> None:
        if self.mode == "return_std":
            self._update(ret)

    def std(self) -> float:
        return float(np.sqrt(self.m2 / self.count + 1e-12))

    def scale(self) -> float:
        if self.mode == "none":
            return 1.0
        s = 1.0 / (self.std() + 1e-8)
        return float(np.clip(s, self.clip_low, self.clip_high))


# ---------------------------------------------------------------------------
# 经验回放池：存缩放后的训练奖励
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
        self.rew[i] = rew          # 传入时已经乘过归一化 scale
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
# SAC 智能体：与第 039 章一致（本章不改网络，只改奖励数据流）
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
# 评估：原始奖励，不缩放
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
# 单个归一化模式的训练
# ---------------------------------------------------------------------------
def train_norm(mode: str, args) -> dict:
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
    normalizer = RewardNormalizer(mode)

    history = {"steps": [], "eval_return": [], "scale": [], "alpha": []}
    ep_returns_raw = []
    obs, _ = env.reset(seed=args.seed)
    ep_ret_raw = 0.0
    disc_return = 0.0              # 折扣回合回报（用于 return_std 统计）
    discount = 1.0
    last_stats = {"mean_abs_q": 0.0}
    eval_ret = float("nan")
    t_start = time.time()

    print(f"[{mode}] 开始训练 | Pendulum-v1 | 种子 {args.seed} | "
          f"reward_norm={mode}")

    for step in range(1, args.max_steps + 1):
        if step <= args.start_steps:
            action = env.action_space.sample()
        else:
            action = agent.act(obs, deterministic=False)

        next_obs, rew_raw, terminated, truncated, _ = env.step(action)
        # 统计量只观察原始奖励/回报，避免"缩放反馈回路"
        normalizer.observe_step(rew_raw)
        scale = normalizer.scale()
        rew_train = rew_raw * scale
        buf.add(obs, action, rew_train, next_obs, float(terminated))
        obs = next_obs
        ep_ret_raw += rew_raw
        disc_return += discount * rew_raw
        discount *= args.gamma

        if terminated or truncated:
            normalizer.observe_return(disc_return)
            ep_returns_raw.append(ep_ret_raw)
            ep_ret_raw = 0.0
            disc_return = 0.0
            discount = 1.0
            obs, _ = env.reset()

        if buf.size >= args.batch_size:
            last_stats = agent.update(buf.sample(args.batch_size))

        if step % args.eval_every == 0 or step == args.max_steps:
            eval_ret = evaluate(eval_env, agent, args.eval_episodes)
            recent = float(np.mean(ep_returns_raw[-20:])) if ep_returns_raw else float("nan")
            history["steps"].append(step)
            history["eval_return"].append(eval_ret)
            history["scale"].append(scale)
            history["alpha"].append(agent.current_alpha)
            print(f"[{mode}] 步数 {step:6d} | 近20原始回报 {recent:9.1f} | "
                  f"评估 {eval_ret:9.1f} | scale {scale:.5f} | "
                  f"alpha {agent.current_alpha:.4f} | |Q| {last_stats['mean_abs_q']:7.1f}")

    elapsed = time.time() - t_start
    print(f"[{mode}] 训练结束 | 用时 {elapsed:.1f}s | 最终评估 {eval_ret:.1f} | "
          f"最终 scale {scale:.5f}")

    os.makedirs(args.save_dir, exist_ok=True)
    ckpt = os.path.join(args.save_dir, f"sac_rewnorm_{mode}.pt")
    torch.save({"policy": agent.policy.state_dict(),
                "q1": agent.q1.state_dict(),
                "reward_norm": mode}, ckpt)
    print(f"[{mode}] 模型已保存到 {ckpt}")

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
        axes[1].plot(hist["steps"], hist["scale"], marker="o",
                     label=res["mode"])
    axes[0].set_xlabel("环境步数")
    axes[0].set_ylabel("原始奖励评估回报")
    axes[0].set_title("评估回报")
    axes[0].legend()
    axes[1].set_xlabel("环境步数")
    axes[1].set_ylabel("奖励缩放系数 scale")
    axes[1].set_title("归一化系数轨迹")
    axes[1].legend()
    fig.tight_layout()
    out = os.path.join(save_dir, "ch040_reward_norm.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"曲线已保存到 {out}")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="第040章：SAC 的奖励归一化")
    p.add_argument("--reward-norms", type=str, default="none,step_std,return_std",
                   help="要对比的奖励归一化方式")
    p.add_argument("--alpha", type=float, default=0.2, help="alpha 初值")
    p.add_argument("--target-entropy", type=float, default=-1.0, help="目标熵")
    p.add_argument("--max-steps", type=int, default=12000, help="每种模式的步数上限")
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
    p.add_argument("--save-dir", type=str, default="runs/ch040", help="模型保存目录")
    p.add_argument("--quick", action="store_true", help="快速模式：几十秒跑完")
    return p.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.max_steps = min(args.max_steps, 2000)
        args.start_steps = min(args.start_steps, 500)
        args.eval_every = 1000
        args.eval_episodes = 2

    modes = [m.strip() for m in args.reward_norms.split(",") if m.strip()]
    for m in modes:
        if m not in ALL_NORMS:
            raise SystemExit(f"未知模式 {m}，可选：{ALL_NORMS}")

    results = [train_norm(m, args) for m in modes]

    print("=" * 74)
    print("奖励归一化对比汇总（Pendulum-v1，评估用原始奖励，以实跑为准）")
    for res in results:
        hist = res["history"]
        print(f"  {res['mode']:>10s}: 最终评估 {res['final_eval']:8.1f} | "
              f"scale 末值 {hist['scale'][-1]:.5f} | "
              f"alpha 末值 {hist['alpha'][-1]:.4f}")

    maybe_plot(results, args.save_dir)


if __name__ == "__main__":
    main()
