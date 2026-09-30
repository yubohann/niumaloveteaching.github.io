"""
第051章 SAC的连续控制基准测试

用同一套 SAC 超参数在三个连续控制任务上做基准测试：

  - Pendulum-v1             摆杆起摆稳定（一维动作，200 步上限）
  - MountainCarContinuous-v0 小车爬山（一维动作，999 步上限）
  - Reach2D                 本章内联实现的轻量环境：二维点质量到达随机目标

每个任务先测随机策略的基线回报，再训练 20000 步并按阈值统计达标步数；
最后把回报归一化到 [0,1] 区间（（得分 − 随机基线）/（专家参考 − 随机基线）），
横向比较四类指标：随机基线、最好/最终评估、达标步数、归一化得分。

运行：
    python code/ch051.py           # 三个环境依次训练（CPU 约 5-8 分钟）
    python code/ch051.py --quick   # 快速跑通（约 1 分钟）

预期：Pendulum 与 Reach2D 稳定达标，归一化得分较高；MountainCarContinuous
需要更长的探索，两万步内常只能接近阈值——这正是基准测试要暴露的差异。
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

# 每个任务的"专家参考回报"（用于归一化），数值为经验量级
EXPERT_REF = {
    "Pendulum-v1": -150.0,
    "MountainCarContinuous-v0": 95.0,
    "Reach2D": 24.0,
}
# 每个任务的达标阈值（评估回报 >= 阈值算达标）
SOLVE_THRESHOLD = {
    "Pendulum-v1": -400.0,
    "MountainCarContinuous-v0": 90.0,
    "Reach2D": 12.0,
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# 内联轻量环境：二维点质量到达随机目标（纯 numpy，无额外依赖）
# ---------------------------------------------------------------------------
class Reach2DEnv(gym.Env):
    """状态 = [x, y, vx, vy, gx, gy]，动作 = 二维加速度。

    每步奖励 1.5·exp(-2·dist) - 0.05 - 0.02·|a|²，到达目标（dist < 0.1）
    额外奖励 +10 并终止；超过 max_steps 截断。
    """

    metadata = {"render_modes": []}

    def __init__(self, dt: float = 0.1, max_steps: int = 80):
        super().__init__()
        self.dt = dt
        self.max_steps = max_steps
        high = np.array([3.0, 3.0, 3.0, 3.0, 3.0, 3.0], dtype=np.float32)
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.rng = np.random.default_rng(0)
        self.state = np.zeros(6, dtype=np.float32)
        self.steps = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        goal = self.rng.uniform(-1.2, 1.2, size=2)
        self.state = np.zeros(6, dtype=np.float32)
        self.state[4:] = goal
        self.steps = 0
        return self.state.copy(), {}

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        self.steps += 1
        # 一阶动力学：速度带 0.1 阻尼，位置积分速度
        self.state[2:4] += (a - 0.1 * self.state[2:4]) * self.dt
        self.state[0:2] += self.state[2:4] * self.dt
        dist = float(np.linalg.norm(self.state[0:2] - self.state[4:6]))
        reward = 1.5 * math.exp(-2.0 * dist) - 0.05 - 0.02 * float(np.sum(a ** 2))
        terminated = dist < 0.1
        truncated = self.steps >= self.max_steps
        if terminated:
            reward += 10.0
        return self.state.copy(), float(reward), bool(terminated), bool(truncated), {}


def make_env(name: str, seed: int):
    """按名称创建环境：内置自定义环境或 gymnasium 环境。"""
    if name == "Reach2D":
        env = Reach2DEnv()
    else:
        env = gym.make(name)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env


class ReplayBuffer:
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


class SAC:
    def __init__(self, obs_dim: int, act_dim: int, action_scale: float, args):
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
        self.log_alpha = torch.tensor(np.log(args.alpha_init), requires_grad=True)
        self.target_entropy = -float(act_dim)
        self.opt_q = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=args.lr)
        self.opt_policy = torch.optim.Adam(self.policy.parameters(), lr=args.lr)
        self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=args.lr)

    @property
    def alpha(self) -> torch.Tensor:
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
            backup = rew + args.gamma * (1.0 - done) * (q_next - self.alpha.detach() * next_logp)

        q1 = self.q1(obs, act)
        q2 = self.q2(obs, act)
        q_loss = F.mse_loss(q1, backup) + F.mse_loss(q2, backup)
        self.opt_q.zero_grad()
        q_loss.backward()
        self.opt_q.step()

        new_act, logp = self.policy.sample(obs)
        q_pi = torch.minimum(self.q1(obs, new_act), self.q2(obs, new_act))
        policy_loss = (self.alpha.detach() * logp - q_pi).mean()
        self.opt_policy.zero_grad()
        policy_loss.backward()
        self.opt_policy.step()

        alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()
        self.opt_alpha.zero_grad()
        alpha_loss.backward()
        self.opt_alpha.step()

        with torch.no_grad():
            soft_update(self.q1, self.q1_target, args.tau)
            soft_update(self.q2, self.q2_target, args.tau)

        return {"q_loss": q_loss.item(), "alpha": float(self.alpha.item())}


# ---------------------------------------------------------------------------
# 评估
# ---------------------------------------------------------------------------
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
def evaluate_random(env, episodes: int) -> float:
    """随机策略基线：归一化得分的下界参考。"""
    returns = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            action = env.action_space.sample()
            obs, rew, terminated, truncated, _ = env.step(action)
            ep_ret += rew
            done = terminated or truncated
        returns.append(ep_ret)
    return float(np.mean(returns))


# ---------------------------------------------------------------------------
# 单个环境的完整基准
# ---------------------------------------------------------------------------
def benchmark_env(name: str, args):
    set_seed(args.seed)  # 每个环境同种子，控制变量
    env = make_env(name, args.seed)
    eval_env = make_env(name, args.seed + 100)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    action_scale = float(env.action_space.high[0])
    agent = SAC(obs_dim, act_dim, action_scale, args)
    buffer = ReplayBuffer(args.buffer_size, obs_dim, act_dim)

    random_baseline = evaluate_random(eval_env, args.eval_episodes)
    threshold = SOLVE_THRESHOLD[name]
    expert_ref = EXPERT_REF[name]
    obs, _ = env.reset(seed=args.seed)
    ep_ret = 0.0
    ep_returns = []
    eval_returns = []
    solved_at = None
    t_start = time.time()
    print(f"\n===== 环境 {name}：obs_dim={obs_dim}, act_dim={act_dim}, "
          f"动作范围 ±{action_scale:.1f}，随机基线 {random_baseline:.1f} =====")

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
            agent.update(buffer.sample(args.batch_size), args)

        if step % args.eval_every == 0:
            eval_ret = evaluate(agent, eval_env, args.eval_episodes)
            eval_returns.append(eval_ret)
            recent20 = np.mean(ep_returns[-20:]) if ep_returns else float("nan")
            print(f"[{name}] 步数 {step:6d} | 近20回合回报 {recent20:9.1f} | "
                  f"评估 {eval_ret:9.1f} | 阈值 {threshold:.0f}")
            if solved_at is None and eval_ret >= threshold:
                solved_at = step
                print(f"[{name}] 首次达到阈值 {threshold}（步数 {step}）")

    elapsed = time.time() - t_start
    final_eval = evaluate(agent, eval_env, args.eval_episodes)
    all_evals = eval_returns + [final_eval]
    best_eval = max(all_evals)
    norm = (final_eval - random_baseline) / max(expert_ref - random_baseline, 1e-9)
    norm = float(np.clip(norm, -1.0, 1.0))
    print(f"[{name}] 结束：最好评估 {best_eval:.1f} | 最终评估 {final_eval:.1f} | "
          f"归一化得分 {norm:.3f} | 用时 {elapsed:.1f}s")
    return {"name": name, "random": random_baseline, "best_eval": best_eval,
            "final_eval": final_eval, "solved_at": solved_at, "norm": norm,
            "elapsed": elapsed, "evals": all_evals}


# ---------------------------------------------------------------------------
# 可选绘图
# ---------------------------------------------------------------------------
def maybe_plot(results, save_dir: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("未安装 matplotlib，跳过绘图。")
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    names = [r["name"] for r in results]
    axes[0].bar(names, [r["norm"] for r in results], color="#2563eb")
    axes[0].set_ylim(-0.2, 1.05)
    axes[0].set_ylabel("normalized score")
    axes[0].set_title("normalized final score")
    for r in results:
        axes[1].plot(np.arange(len(r["evals"])) * 2000, r["evals"], marker="o",
                     label=r["name"])
    axes[1].set_xlabel("env steps")
    axes[1].set_ylabel("eval return")
    axes[1].set_title("eval curves")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "benchmark.png")
    fig.savefig(path, dpi=120)
    print(f"图表已保存到 {path}")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="第051章：SAC 的连续控制基准测试")
    p.add_argument("--envs", type=str,
                   default="Pendulum-v1,MountainCarContinuous-v0,Reach2D",
                   help="逗号分隔的环境列表（支持内联环境 Reach2D）")
    p.add_argument("--max-steps", type=int, default=20000, help="每个环境的环境步数上限")
    p.add_argument("--start-steps", type=int, default=1000, help="训练前纯随机探索步数")
    p.add_argument("--update-every", type=int, default=1, help="每多少环境步做一次更新")
    p.add_argument("--eval-every", type=int, default=2000, help="评估间隔")
    p.add_argument("--eval-episodes", type=int, default=10, help="评估回合数（基准测试取 10 更稳）")
    p.add_argument("--buffer-size", type=int, default=100000, help="回放池容量")
    p.add_argument("--batch-size", type=int, default=256, help="批大小")
    p.add_argument("--hidden", type=int, default=256, help="隐藏层宽度")
    p.add_argument("--lr", type=float, default=3e-4, help="学习率")
    p.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
    p.add_argument("--tau", type=float, default=0.005, help="目标网络软更新系数")
    p.add_argument("--alpha-init", type=float, default=0.2, help="温度系数初值")
    p.add_argument("--seed", type=int, default=0, help="随机种子")
    p.add_argument("--save-dir", type=str, default="runs/ch051", help="输出目录")
    p.add_argument("--plot", action="store_true", help="保存基准测试图表（需 matplotlib）")
    p.add_argument("--quick", action="store_true", help="快速模式：缩短训练")
    return p.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.max_steps = min(args.max_steps, 4000)
        args.eval_every = 1000
        args.start_steps = min(args.start_steps, 400)
        args.eval_episodes = 3

    envs = [x.strip() for x in args.envs.split(",") if x.strip()]
    print(f"种子: {args.seed} | 环境: {envs}")
    print(f"统一配置: max_steps={args.max_steps}, batch={args.batch_size}, "
          f"lr={args.lr}, tau={args.tau}")

    results = [benchmark_env(name, args) for name in envs]

    print("\n" + "=" * 100)
    print("连续控制基准测试汇总（归一化得分 =（最终评估 − 随机基线）/（专家参考 − 随机基线））")
    print(f"{'环境':<26}{'随机基线':>10}{'最好评估':>10}{'最终评估':>10}"
          f"{'达标步数':>10}{'归一化':>9}{'用时(s)':>9}")
    for r in results:
        solved = r["solved_at"] if r["solved_at"] is not None else -1
        print(f"{r['name']:<26}{r['random']:>10.1f}{r['best_eval']:>10.1f}"
              f"{r['final_eval']:>10.1f}{solved:>10}{r['norm']:>9.3f}{r['elapsed']:>9.1f}")
    print("=" * 100)
    print("结论提示：同一套超参数在不同任务上的样本效率差异很大；Pendulum 与")
    print("Reach2D 通常在 2 万步内达标，MountainCarContinuous 的稀疏成功信号需要")
    print("更长的探索预算。基准测试的价值就在于把这种差异量化出来。")

    if args.plot:
        maybe_plot(results, args.save_dir)

    os.makedirs(args.save_dir, exist_ok=True)
    print(f"输出目录：{args.save_dir}")


if __name__ == "__main__":
    main()
