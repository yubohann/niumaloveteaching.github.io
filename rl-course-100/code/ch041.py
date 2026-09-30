"""
第041章 SAC的熵系数调度

第 032 章让 alpha 通过梯度自动适应"熵缺口"；本章换一条路：人为给 alpha
安排一条时间表，按训练进度从大到小退火——
    const : 常数 alpha（0.2），对照
    linear: alpha 从 alpha_start 线性降到 alpha_end
    cosine: alpha 从 alpha_start 按余弦曲线降到 alpha_end
    auto  : 重新交给自动温度调节（可选，用来对照两种思路）
调度型 alpha 不经过任何优化器，每个环境步由训练循环直接设定。

实验在 Pendulum-v1 上进行，观察评估回报、alpha 轨迹、熵轨迹与 |Q|。
调度与自动温度的对比回答一个很实际的问题：如果我已经知道"早期多探索、
后期少探索"，那还需要让算法自己学 alpha 吗？

运行：
    python code/ch041.py           # const/linear/cosine 依次训练（CPU 约 6-10 分钟）
    python code/ch041.py --quick   # 快速跑通（约 1 分钟）
    python code/ch041.py --schedules const,linear,cosine,auto --max-steps 20000

预期：linear/cosine 的熵轨迹跟随 alpha 一起下降，评估回报通常与
      常数基线相当或略好；auto 模式省去设定起止值，但需要选目标熵。
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
ALL_SCHEDULES = ["const", "linear", "cosine", "auto"]


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


def schedule_alpha(name: str, step: int, total: int, start: float, end: float,
                   const: float) -> float:
    """按调度类型计算当前步的 alpha。"""
    progress = min(1.0, step / max(total, 1))
    if name == "const":
        return const
    if name == "linear":
        return start + (end - start) * progress
    if name == "cosine":
        # 从 start 平滑降到 end：0.5*(1+cos(pi*progress))
        return end + 0.5 * (start - end) * (1.0 + math.cos(math.pi * progress))
    raise ValueError(f"未知调度: {name}")


# ---------------------------------------------------------------------------
# 经验回放池
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
# SAC 智能体：alpha 由外部设定（调度）或内部学习（auto）
# ---------------------------------------------------------------------------
class SACAgent:
    def __init__(self, obs_dim: int, act_dim: int, act_scale: float,
                 schedule: str, alpha: float = 0.2, target_entropy: float = -1.0,
                 lr: float = 3e-4, gamma: float = 0.99, tau: float = 0.005,
                 hidden: int = 256):
        self.schedule = schedule
        self.alpha_value = float(alpha)      # 调度模式下由训练循环覆写
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

        # 只有 auto 模式需要 log_alpha 与独立优化器
        if self.schedule == "auto":
            self.log_alpha = torch.tensor(math.log(alpha), requires_grad=True)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)

    @property
    def current_alpha(self) -> float:
        if self.schedule == "auto":
            return float(self.log_alpha.exp().item())
        return self.alpha_value

    def set_alpha(self, value: float) -> None:
        """训练循环为调度模式设定 alpha。"""
        if self.schedule != "auto":
            self.alpha_value = float(value)

    @torch.no_grad()
    def act(self, obs, deterministic: bool = False) -> np.ndarray:
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        action, _ = self.policy.sample(obs_t, deterministic=deterministic)
        return action.squeeze(0).numpy()

    def update(self, batch: dict) -> dict:
        obs, act, rew = batch["obs"], batch["act"], batch["rew"]
        next_obs, done = batch["next_obs"], batch["done"]
        alpha = self.current_alpha

        with torch.no_grad():
            next_act, next_logp = self.policy.sample(next_obs)
            q_next = torch.min(self.q1_target(next_obs, next_act),
                               self.q2_target(next_obs, next_act))
            target = rew + self.gamma * (1.0 - done) * (q_next - alpha * next_logp)

        q1_pred = self.q1(obs, act)
        q2_pred = self.q2(obs, act)
        q_loss = F.mse_loss(q1_pred, target) + F.mse_loss(q2_pred, target)

        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        new_act, logp = self.policy.sample(obs)
        q_new = torch.min(self.q1(obs, new_act), self.q2(obs, new_act))
        policy_loss = (alpha * logp - q_new).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # auto 模式才更新 alpha；调度模式的 alpha 由外部时间表决定
        if self.schedule == "auto":
            alpha_loss = -(self.log_alpha *
                           (logp + self.target_entropy).detach()).mean()
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
# 单种调度的训练
# ---------------------------------------------------------------------------
def train_schedule(name: str, args) -> dict:
    set_seed(args.seed)
    env = make_env("Pendulum-v1", args.seed)
    eval_env = make_env("Pendulum-v1", args.seed + 500)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_scale = float(env.action_space.high[0])

    # auto 模式从 alpha_start 出发学习；其余模式由调度函数控制
    init_alpha = args.alpha_start if name != "const" else args.alpha_const
    agent = SACAgent(obs_dim, act_dim, act_scale, schedule=name,
                     alpha=init_alpha, target_entropy=args.target_entropy,
                     lr=args.lr, gamma=args.gamma, tau=args.tau,
                     hidden=args.hidden)
    buf = ReplayBuffer(args.buffer_size, obs_dim, act_dim)

    history = {"steps": [], "eval_return": [], "alpha": [], "entropy": []}
    ep_returns = []
    obs, _ = env.reset(seed=args.seed)
    ep_ret = 0.0
    last_stats = {"mean_abs_q": 0.0, "entropy": 0.0}
    eval_ret = float("nan")
    t_start = time.time()

    print(f"[{name}] 开始训练 | Pendulum-v1 | 种子 {args.seed} | "
          f"alpha 起点 {init_alpha}")

    for step in range(1, args.max_steps + 1):
        # 调度模式：每个环境步先设定当前 alpha
        if name != "auto":
            agent.set_alpha(schedule_alpha(name, step, args.max_steps,
                                           args.alpha_start, args.alpha_end,
                                           args.alpha_const))

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
            print(f"[{name}] 步数 {step:6d} | 近20均值 {recent:9.1f} | "
                  f"评估 {eval_ret:9.1f} | alpha {agent.current_alpha:.4f} | "
                  f"熵 {last_stats['entropy']:.3f} | |Q| {last_stats['mean_abs_q']:7.1f}")

    elapsed = time.time() - t_start
    print(f"[{name}] 训练结束 | 用时 {elapsed:.1f}s | 最终评估 {eval_ret:.1f} | "
          f"最终 alpha {agent.current_alpha:.4f}")

    os.makedirs(args.save_dir, exist_ok=True)
    ckpt = os.path.join(args.save_dir, f"sac_sched_{name}.pt")
    torch.save({"policy": agent.policy.state_dict(),
                "q1": agent.q1.state_dict(),
                "schedule": name}, ckpt)
    print(f"[{name}] 模型已保存到 {ckpt}")

    return {"schedule": name, "final_eval": eval_ret, "elapsed": elapsed,
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
                     label=res["schedule"])
        axes[1].plot(hist["steps"], hist["alpha"], marker="o",
                     label=res["schedule"])
    axes[0].set_xlabel("环境步数")
    axes[0].set_ylabel("确定性评估回报")
    axes[0].set_title("评估回报")
    axes[0].legend()
    axes[1].set_xlabel("环境步数")
    axes[1].set_ylabel("alpha")
    axes[1].set_title("熵系数轨迹")
    axes[1].legend()
    fig.tight_layout()
    out = os.path.join(save_dir, "ch041_alpha_schedule.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"曲线已保存到 {out}")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="第041章：SAC 的熵系数调度")
    p.add_argument("--schedules", type=str, default="const,linear,cosine",
                   help="要对比的调度：const / linear / cosine / auto")
    p.add_argument("--alpha-const", type=float, default=0.2, help="const 模式的常数 alpha")
    p.add_argument("--alpha-start", type=float, default=0.4, help="退火起点（linear/cosine/auto）")
    p.add_argument("--alpha-end", type=float, default=0.02, help="退火终点（linear/cosine）")
    p.add_argument("--target-entropy", type=float, default=-1.0, help="auto 模式的目标熵")
    p.add_argument("--max-steps", type=int, default=12000, help="每种调度的步数上限")
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
    p.add_argument("--save-dir", type=str, default="runs/ch041", help="模型保存目录")
    p.add_argument("--quick", action="store_true", help="快速模式：几十秒跑完")
    return p.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.max_steps = min(args.max_steps, 2000)
        args.start_steps = min(args.start_steps, 500)
        args.eval_every = 1000
        args.eval_episodes = 2

    names = [s.strip() for s in args.schedules.split(",") if s.strip()]
    for s in names:
        if s not in ALL_SCHEDULES:
            raise SystemExit(f"未知调度 {s}，可选：{ALL_SCHEDULES}")

    results = [train_schedule(s, args) for s in names]

    print("=" * 74)
    print("熵系数调度对比汇总（Pendulum-v1，以实跑为准）")
    for res in results:
        hist = res["history"]
        print(f"  {res['schedule']:>6s}: 最终评估 {res['final_eval']:8.1f} | "
              f"alpha 轨迹 {[round(a, 3) for a in hist['alpha']]} | "
              f"熵轨迹 {[round(e, 2) for e in hist['entropy']]}")

    maybe_plot(results, args.save_dir)


if __name__ == "__main__":
    main()
