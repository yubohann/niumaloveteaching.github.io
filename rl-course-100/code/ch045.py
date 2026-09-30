"""
第045章 SAC的探索与利用平衡

SAC 的探索水平由温度系数 alpha 直接控制：alpha 越大，策略越随机。本章在
Pendulum-v1 上对比三种配置，量化"探索"与"利用"的取舍：

  - exploit : 目标熵设为 -0.1（远低于标准 -1），alpha 自动缩小，几乎只做利用
  - balanced: 目标熵 -1.0，SAC 论文的标准配置
  - explore : 固定 alpha=0.5 不学习，保持强随机探索

对每种配置记录：评估回报、策略动作标准差、回合内摆角覆盖的熵（12 个扇区
上的分布熵，越大说明探索到的状态越丰富）以及最终 alpha。

运行：
    python code/ch045.py           # 三种配置依次训练（CPU 约 4-6 分钟）
    python code/ch045.py --quick   # 快速跑通（约 1 分钟）

预期：explore 覆盖熵最高但评估回报最差（噪声大）；exploit 快速收敛却可能
卡在局部策略；balanced 兼顾两者，评估回报最好。
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

LOG_STD_MIN, LOG_STD_MAX = -20.0, 2.0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_env(env_id: str, seed: int):
    env = gym.make(env_id)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env


class ReplayBuffer:
    """标准均匀回放池（继承第 043 章实现）。"""

    def __init__(self, capacity: int, obs_dim: int, act_dim: int):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rew = np.zeros((capacity, 1), dtype=np.float32)
        self.done = np.zeros((capacity, 1), dtype=np.float32)
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

    def sample(self, obs: torch.Tensor):
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        eps = torch.randn_like(mean)
        u = mean + std * eps
        a = torch.tanh(u)
        log_prob = -0.5 * (eps ** 2 + 2.0 * log_std + np.log(2.0 * np.pi))
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        log_prob = log_prob - torch.log(1.0 - a ** 2 + 1e-6).sum(dim=-1, keepdim=True)
        action = a * self.act_scale
        log_prob = log_prob - torch.log(self.act_scale).sum()
        return action, log_prob

    def mean_action(self, obs: torch.Tensor) -> torch.Tensor:
        mean, _ = self.forward(obs)
        return torch.tanh(mean) * self.act_scale


def soft_update(net: nn.Module, target: nn.Module, tau: float) -> None:
    for tp, p in zip(target.parameters(), net.parameters()):
        tp.data.mul_(1.0 - tau).add_(tau * p.data)


# ---------------------------------------------------------------------------
# SAC：本章按"探索模式"配置温度系数
# ---------------------------------------------------------------------------
MODE_CONFIG = {
    "exploit": {"target_entropy": -0.1, "fixed_alpha": None},
    "balanced": {"target_entropy": -1.0, "fixed_alpha": None},
    "explore": {"target_entropy": -1.0, "fixed_alpha": 0.5},
}


class SAC:
    def __init__(self, obs_dim: int, act_dim: int, action_scale: float,
                 mode: str, args):
        cfg = MODE_CONFIG[mode]
        self.mode = mode
        self.policy = GaussianPolicy(obs_dim, act_dim, args.hidden, action_scale)
        self.q1 = QNet(obs_dim, act_dim, args.hidden)
        self.q2 = QNet(obs_dim, act_dim, args.hidden)
        self.q1_target = QNet(obs_dim, act_dim, args.hidden)
        self.q2_target = QNet(obs_dim, act_dim, args.hidden)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        for p in self.q1_target.parameters():
            p.requires_grad_(False)
        for p in self.q2_target.parameters():
            p.requires_grad_(False)

        self.target_entropy = cfg["target_entropy"]
        self.fixed_alpha = cfg["fixed_alpha"]
        self.log_alpha = torch.tensor(np.log(args.alpha_init), requires_grad=True)
        self.opt_q = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=args.lr)
        self.opt_policy = torch.optim.Adam(self.policy.parameters(), lr=args.lr)
        self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=args.lr)

    @property
    def alpha(self) -> torch.Tensor:
        """explore 模式返回固定的 0.5；其余模式返回自动调节值。"""
        if self.fixed_alpha is not None:
            return torch.tensor(self.fixed_alpha)
        return self.log_alpha.exp().clamp(1e-4, 10.0)

    @torch.no_grad()
    def act(self, obs, deterministic: bool = False) -> np.ndarray:
        obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
        if deterministic:
            return self.policy.mean_action(obs_t).squeeze(0).numpy()
        action, _ = self.policy.sample(obs_t)
        return action.squeeze(0).numpy()

    def update(self, batch: dict, args) -> dict:
        obs, act = batch["obs"], batch["act"]
        rew, next_obs, done = batch["rew"], batch["next_obs"], batch["done"]

        with torch.no_grad():
            next_act, next_logp = self.policy.sample(next_obs)
            q_next = torch.minimum(self.q1_target(next_obs, next_act),
                                   self.q2_target(next_obs, next_act))
            backup = rew + args.gamma * (1.0 - done) * (q_next - self.alpha * next_logp)

        q1 = self.q1(obs, act)
        q2 = self.q2(obs, act)
        q_loss = F.mse_loss(q1, backup) + F.mse_loss(q2, backup)
        self.opt_q.zero_grad()
        q_loss.backward()
        self.opt_q.step()

        new_act, logp = self.policy.sample(obs)
        q_pi = torch.minimum(self.q1(obs, new_act), self.q2(obs, new_act))
        policy_loss = (self.alpha * logp - q_pi).mean()
        self.opt_policy.zero_grad()
        policy_loss.backward()
        self.opt_policy.step()

        # explore 模式固定温度，不更新 log_alpha
        if self.fixed_alpha is None:
            alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()
            self.opt_alpha.zero_grad()
            alpha_loss.backward()
            self.opt_alpha.step()

        with torch.no_grad():
            soft_update(self.q1, self.q1_target, args.tau)
            soft_update(self.q2, self.q2_target, args.tau)

        return {"q_loss": q_loss.item(), "policy_loss": policy_loss.item(),
                "alpha": float(self.alpha.item()), "entropy": float(-logp.mean().item())}


@torch.no_grad()
def evaluate(agent: SAC, env, episodes: int) -> float:
    returns = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            action = agent.act(obs, deterministic=True)
            obs, rew, terminated, truncated, _ = env.step(action)
            ep_ret += rew
            done = terminated or truncated
        returns.append(ep_ret)
    return float(np.mean(returns))


@torch.no_grad()
def measure_exploration(agent: SAC, env, episodes: int, n_bins: int = 12):
    """用随机策略跑若干回合，量化探索行为。

    返回：
        action_std : 策略输出分布的平均标准差（探索噪声大小）
        act_abs    : 平均 |动作|（实际动作幅度）
        coverage_H : 摆角落在 12 个扇区上的分布熵（状态覆盖广度）
        mean_ret   : 随机策略的平均回报
    """
    stds, act_abs, bins, returns = [], [], [], []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
            _, log_std = agent.policy.forward(obs_t)
            stds.append(float(log_std.exp().mean().item()))
            action = agent.act(obs, deterministic=False)
            act_abs.append(float(np.abs(action).mean()))
            # Pendulum 观测为 [cos θ, sin θ, θ̇]，由前两维反推摆角
            angle = np.arctan2(float(obs[1]), float(obs[0]))
            bins.append(int((angle + np.pi) / (2 * np.pi) * n_bins) % n_bins)
            next_obs, rew, terminated, truncated, _ = env.step(action)
            ep_ret += rew
            obs = next_obs
            done = terminated or truncated
        returns.append(ep_ret)
    counts = np.bincount(np.asarray(bins), minlength=n_bins).astype(np.float64)
    probs = counts / max(counts.sum(), 1.0)
    coverage_H = float(-np.sum(probs[probs > 0] * np.log(probs[probs > 0])))
    return {
        "action_std": float(np.mean(stds)),
        "act_abs": float(np.mean(act_abs)),
        "coverage_H": coverage_H,
        "mean_ret": float(np.mean(returns)),
    }


# ---------------------------------------------------------------------------
# 训练一种探索配置
# ---------------------------------------------------------------------------
def train_mode(mode: str, args):
    set_seed(args.seed)  # 三种配置同种子，控制变量
    env = make_env(args.env, args.seed)
    eval_env = make_env(args.env, args.seed + 100)
    probe_env = make_env(args.env, args.seed + 200)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    action_scale = float(env.action_space.high[0])
    agent = SAC(obs_dim, act_dim, action_scale, mode, args)
    buffer = ReplayBuffer(args.buffer_size, obs_dim, act_dim)

    obs, _ = env.reset(seed=args.seed)
    ep_ret = 0.0
    ep_returns = []
    history = []
    solved_at = None
    t_start = time.time()
    print(f"\n===== 配置 {mode}：目标熵 {agent.target_entropy}, "
          f"固定温度 {agent.fixed_alpha} =====")

    for step in range(1, args.max_steps + 1):
        if step <= args.start_steps:
            action = env.action_space.sample()
        else:
            action = agent.act(obs, deterministic=False)
        next_obs, rew, terminated, truncated, _ = env.step(action)
        buffer.add(obs, action, rew, next_obs, float(terminated))
        obs = next_obs
        ep_ret += rew

        if terminated or truncated:
            ep_returns.append(ep_ret)
            ep_ret = 0.0
            obs, _ = env.reset()

        if step % args.update_every == 0 and buffer.size >= args.start_steps:
            stats = agent.update(buffer.sample(args.batch_size), args)

        if step % args.eval_every == 0:
            eval_ret = evaluate(agent, eval_env, args.eval_episodes)
            expl = measure_exploration(agent, probe_env, args.probe_episodes)
            history.append((step, eval_ret, expl))
            recent20 = np.mean(ep_returns[-20:]) if ep_returns else float("nan")
            print(f"[{mode}] 步数 {step:6d} | 近20回合回报 {recent20:8.1f} | "
                  f"评估 {eval_ret:8.1f} | 动作std {expl['action_std']:.3f} | "
                  f"覆盖熵 {expl['coverage_H']:.2f} | alpha {stats['alpha']:.3f}")
            if solved_at is None and eval_ret >= args.target_return:
                solved_at = step
                print(f"[{mode}] 首次达到评估回报 {args.target_return}（步数 {step}）")

    elapsed = time.time() - t_start
    final_eval = evaluate(agent, eval_env, args.eval_episodes)
    final_expl = measure_exploration(agent, probe_env, args.probe_episodes)
    best_eval = max([h[1] for h in history] + [final_eval])
    print(f"[{mode}] 结束：最好评估 {best_eval:.1f} | 最终评估 {final_eval:.1f} | "
          f"覆盖熵 {final_expl['coverage_H']:.2f} | 动作std {final_expl['action_std']:.3f} | "
          f"用时 {elapsed:.1f}s")
    return {"mode": mode, "best_eval": best_eval, "final_eval": final_eval,
            "action_std": final_expl["action_std"], "coverage_H": final_expl["coverage_H"],
            "solved_at": solved_at, "elapsed": elapsed}


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="第045章：SAC 的探索与利用平衡")
    p.add_argument("--env", type=str, default="Pendulum-v1", help="环境 ID")
    p.add_argument("--modes", type=str, default="exploit,balanced,explore",
                   help="逗号分隔：exploit / balanced / explore")
    p.add_argument("--max-steps", type=int, default=18000, help="每种配置的环境步数上限")
    p.add_argument("--start-steps", type=int, default=1000, help="训练前纯随机探索步数")
    p.add_argument("--update-every", type=int, default=1, help="每多少环境步做一次更新")
    p.add_argument("--eval-every", type=int, default=3000, help="评估间隔")
    p.add_argument("--eval-episodes", type=int, default=5, help="评估回合数")
    p.add_argument("--probe-episodes", type=int, default=5, help="探索诊断使用的回合数")
    p.add_argument("--buffer-size", type=int, default=100000, help="回放池容量")
    p.add_argument("--batch-size", type=int, default=256, help="批大小")
    p.add_argument("--hidden", type=int, default=256, help="隐藏层宽度")
    p.add_argument("--lr", type=float, default=3e-4, help="学习率")
    p.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
    p.add_argument("--tau", type=float, default=0.005, help="目标网络软更新系数")
    p.add_argument("--alpha-init", type=float, default=0.2, help="温度系数初值")
    p.add_argument("--target-return", type=float, default=-400.0, help="达标阈值")
    p.add_argument("--seed", type=int, default=0, help="随机种子")
    p.add_argument("--save-dir", type=str, default="runs/ch045", help="输出目录")
    p.add_argument("--quick", action="store_true", help="快速模式：缩短训练")
    return p.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.max_steps = min(args.max_steps, 4000)
        args.eval_every = 1000
        args.start_steps = min(args.start_steps, 400)
        args.probe_episodes = 3

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    print(f"环境: {args.env} | 种子: {args.seed} | 配置: {modes}")
    print(f"配置: max_steps={args.max_steps}, batch={args.batch_size}, alpha_init={args.alpha_init}")

    results = [train_mode(m, args) for m in modes]

    print("\n" + "=" * 88)
    print("探索/利用配置对比（覆盖熵越大探索越广，动作 std 越大策略越随机）")
    print(f"{'配置':<10}{'最好评估':>10}{'最终评估':>10}{'达标步数':>10}"
          f"{'动作std':>10}{'覆盖熵':>10}")
    for r in results:
        solved = r["solved_at"] if r["solved_at"] is not None else -1
        print(f"{r['mode']:<10}{r['best_eval']:>10.1f}{r['final_eval']:>10.1f}"
              f"{solved:>10}{r['action_std']:>10.3f}{r['coverage_H']:>10.2f}")
    print("=" * 88)
    print("结论提示：explore 的覆盖熵与动作 std 最大但评估回报最差；exploit 收敛")
    print("快、回报可能更好也可能卡在局部；balanced（目标熵 -1）通常在 Pendulum")
    print("这类连续控制任务上给出最好的评估回报。")

    os.makedirs(args.save_dir, exist_ok=True)
    print(f"输出目录：{args.save_dir}")


if __name__ == "__main__":
    main()
