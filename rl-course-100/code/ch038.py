"""
第038章 SAC的循环策略网络

给 Pendulum-v1 戴上一只"眼罩"：把观测里的角速度维度强制置零，环境即刻
变成部分可观测（POMDP）——只看一眼角度，你无法知道摆杆正在往哪边转。
挡板策略（MLP）只吃当前这一帧，缺少的信息不可能凭空补出来；
循环策略（GRU）把过去 window 帧组织成序列，从角度的变化率里"推断"速度。

本章在同一套 SAC 框架里对比：
    mlp : 策略只吃当前（被遮挡的）观测
    gru : 策略吃最近 window 帧，用 GRU 汇总历史后输出动作分布
两者的评论家都吃完整 window（评论家可以用更多信息降低方差），
因此差异只来自策略能否利用历史。

运行：
    python code/ch038.py           # 两种策略依次训练（CPU 约 6-10 分钟）
    python code/ch038.py --quick   # 快速跑通（约 1-2 分钟）
    python code/ch038.py --policies gru --max-steps 20000

预期：mlp 大量回合拿不到好控制，评估回报长期在 -1000 附近；
      gru 能从角度序列里推出速度，明显好于 mlp，但仍不如
      第 031 章能看到完整观测的 SAC。
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
ALL_POLICIES = ["mlp", "gru"]


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class MaskVelocity(gym.ObservationWrapper):
    """把观测的角速度维度置零：制造部分可观测的 Pendulum。"""

    def __init__(self, env):
        super().__init__(env)

    def observation(self, obs):
        masked = np.asarray(obs, dtype=np.float32).copy()
        masked[2] = 0.0            # 丢掉 theta_dot
        return masked


def make_env(env_id: str, seed: int):
    env = MaskVelocity(gym.make(env_id))
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env


# ---------------------------------------------------------------------------
# 观测窗口：维护最近 window 帧观测，reset 时用初始帧填满
# ---------------------------------------------------------------------------
class ObsWindow:
    def __init__(self, size: int, obs_dim: int):
        self.size = size
        self.obs_dim = obs_dim
        self.buf = np.zeros((size, obs_dim), dtype=np.float32)

    def reset(self, obs) -> np.ndarray:
        """新回合开始：用同一帧填满整个窗口。"""
        obs = np.asarray(obs, dtype=np.float32)
        for i in range(self.size):
            self.buf[i] = obs
        return self.buf.copy()

    def push(self, obs) -> np.ndarray:
        """把新观测塞进窗口尾部，返回新的对齐副本。"""
        self.buf = np.roll(self.buf, shift=-1, axis=0)
        self.buf[-1] = np.asarray(obs, dtype=np.float32)
        return self.buf.copy()


# ---------------------------------------------------------------------------
# 经验回放池：存的是窗口而不是单帧观测
# ---------------------------------------------------------------------------
class ReplayBuffer:
    def __init__(self, capacity: int, window: int, obs_dim: int):
        self.capacity = capacity
        self.obs = np.zeros((capacity, window, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, window, obs_dim), dtype=np.float32)
        self.act = np.zeros((capacity, 1), dtype=np.float32)
        self.rew = np.zeros(capacity, dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(self, obs_win, act, rew, next_win, done) -> None:
        i = self.ptr
        self.obs[i] = obs_win
        self.next_obs[i] = next_win
        self.act[i] = act
        self.rew[i] = rew
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
class QNetWindow(nn.Module):
    """Q(窗口, a)：把整个窗口展平后与动作拼接，输出标量。"""

    def __init__(self, window: int, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        in_dim = window * obs_dim + act_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, win: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        flat = win.reshape(win.shape[0], -1)
        return self.net(torch.cat([flat, act], dim=-1)).squeeze(-1)


class GaussianPolicyBase(nn.Module):
    """两种策略共享的采样/对数概率逻辑，特征提取交给子类。"""

    def __init__(self, act_dim: int, hidden: int = 256, act_scale: float = 1.0):
        super().__init__()
        self.mean_layer = nn.Linear(hidden, act_dim)
        self.log_std_layer = nn.Linear(hidden, act_dim)
        self.register_buffer("act_scale", torch.tensor(float(act_scale)))

    def features(self, win: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def distribution(self, win: torch.Tensor):
        h = self.features(win)
        mean = self.mean_layer(h)
        log_std = torch.clamp(self.log_std_layer(h), LOG_STD_MIN, LOG_STD_MAX)
        return torch.distributions.Normal(mean, log_std.exp())

    def sample(self, win: torch.Tensor, deterministic: bool = False):
        dist = self.distribution(win)
        if deterministic:
            return torch.tanh(dist.mean) * self.act_scale, None
        x = dist.rsample()
        y = torch.tanh(x)
        action = y * self.act_scale
        log_prob = dist.log_prob(x) - torch.log(1.0 - y.pow(2) + 1e-6)
        return action, log_prob.sum(dim=-1)


class MLPPolicy(GaussianPolicyBase):
    """挡板策略：只看窗口的最后一帧。"""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256,
                 act_scale: float = 1.0):
        super().__init__(act_dim, hidden, act_scale)
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )

    def features(self, win: torch.Tensor) -> torch.Tensor:
        return self.body(win[:, -1, :])


class GRUPolicy(GaussianPolicyBase):
    """循环策略：用 GRU 汇总整个窗口，取最后时刻的隐状态。"""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256,
                 act_scale: float = 1.0):
        super().__init__(act_dim, hidden, act_scale)
        self.gru = nn.GRU(obs_dim, hidden, batch_first=True)
        self.post = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU())

    def features(self, win: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(win)          # (B, L, H)
        return self.post(out[:, -1, :])


def build_policy(policy_type: str, obs_dim: int, act_dim: int, hidden: int,
                 act_scale: float):
    if policy_type == "mlp":
        return MLPPolicy(obs_dim, act_dim, hidden, act_scale)
    return GRUPolicy(obs_dim, act_dim, hidden, act_scale)


# ---------------------------------------------------------------------------
# SAC 智能体
# ---------------------------------------------------------------------------
class SACAgent:
    def __init__(self, window: int, obs_dim: int, act_dim: int, act_scale: float,
                 policy_type: str, alpha: float = 0.2, target_entropy: float = -1.0,
                 lr: float = 3e-4, gamma: float = 0.99, tau: float = 0.005,
                 hidden: int = 256):
        assert policy_type in ALL_POLICIES
        self.policy_type = policy_type
        self.alpha = float(alpha)
        self.target_entropy = float(target_entropy)
        self.gamma = gamma
        self.tau = tau

        self.q1 = QNetWindow(window, obs_dim, act_dim, hidden)
        self.q2 = QNetWindow(window, obs_dim, act_dim, hidden)
        self.q1_target = QNetWindow(window, obs_dim, act_dim, hidden)
        self.q2_target = QNetWindow(window, obs_dim, act_dim, hidden)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        for p in self.q1_target.parameters():
            p.requires_grad_(False)
        for p in self.q2_target.parameters():
            p.requires_grad_(False)
        self.policy = build_policy(policy_type, obs_dim, act_dim, hidden, act_scale)

        self.q_optimizer = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.log_alpha = torch.tensor(math.log(alpha), requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)

    @property
    def current_alpha(self) -> float:
        return float(self.log_alpha.exp().item())

    @torch.no_grad()
    def act(self, win: np.ndarray, deterministic: bool = False) -> np.ndarray:
        win_t = torch.as_tensor(win, dtype=torch.float32).unsqueeze(0)
        action, _ = self.policy.sample(win_t, deterministic=deterministic)
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
# 评估
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(env, agent: SACAgent, episodes: int, window: int) -> float:
    obs_dim = env.observation_space.shape[0]
    win = ObsWindow(window, obs_dim)
    returns = []
    for _ in range(episodes):
        obs, _ = env.reset()
        cur = win.reset(obs)
        done = False
        ep_ret = 0.0
        while not done:
            action = agent.act(cur, deterministic=True)
            obs, r, terminated, truncated, _ = env.step(action)
            ep_ret += r
            done = terminated or truncated
            if not done:
                cur = win.push(obs)
        returns.append(ep_ret)
    return float(np.mean(returns))


# ---------------------------------------------------------------------------
# 单个策略的训练
# ---------------------------------------------------------------------------
def train_policy(policy_type: str, args) -> dict:
    set_seed(args.seed)
    env = make_env("Pendulum-v1", args.seed)
    eval_env = make_env("Pendulum-v1", args.seed + 500)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_scale = float(env.action_space.high[0])

    agent = SACAgent(args.window, obs_dim, act_dim, act_scale,
                     policy_type=policy_type, alpha=args.alpha,
                     target_entropy=args.target_entropy, lr=args.lr,
                     gamma=args.gamma, tau=args.tau, hidden=args.hidden)
    buf = ReplayBuffer(args.buffer_size, args.window, obs_dim)

    history = {"steps": [], "eval_return": [], "mean_abs_q": [], "entropy": []}
    ep_returns = []
    obs, _ = env.reset(seed=args.seed)
    win = ObsWindow(args.window, obs_dim)
    cur_win = win.reset(obs)
    ep_ret = 0.0
    last_stats = {"mean_abs_q": 0.0, "entropy": 0.0}
    eval_ret = float("nan")
    t_start = time.time()

    print(f"[{policy_type}] 开始训练 | 部分可观测 Pendulum | 窗口 {args.window} | "
          f"种子 {args.seed}")

    for step in range(1, args.max_steps + 1):
        if step <= args.start_steps:
            action = env.action_space.sample()
        else:
            action = agent.act(cur_win, deterministic=False)

        next_obs, rew, terminated, truncated, _ = env.step(action)
        done_ep = terminated or truncated
        next_win = win.push(next_obs)
        buf.add(cur_win, action, rew, next_win, float(terminated))

        ep_ret += rew
        if done_ep:
            ep_returns.append(ep_ret)
            ep_ret = 0.0
            obs, _ = env.reset()
            cur_win = win.reset(obs)          # 新回合：窗口清零重填
        else:
            cur_win = next_win

        if buf.size >= args.batch_size:
            last_stats = agent.update(buf.sample(args.batch_size))

        if step % args.eval_every == 0 or step == args.max_steps:
            eval_ret = evaluate(eval_env, agent, args.eval_episodes, args.window)
            recent = float(np.mean(ep_returns[-20:])) if ep_returns else float("nan")
            history["steps"].append(step)
            history["eval_return"].append(eval_ret)
            history["mean_abs_q"].append(last_stats["mean_abs_q"])
            history["entropy"].append(last_stats["entropy"])
            print(f"[{policy_type}] 步数 {step:6d} | 近20均值 {recent:9.1f} | "
                  f"评估 {eval_ret:9.1f} | 熵 {last_stats['entropy']:.3f} | "
                  f"alpha {agent.current_alpha:.3f} | |Q| {last_stats['mean_abs_q']:7.1f}")

    elapsed = time.time() - t_start
    print(f"[{policy_type}] 训练结束 | 用时 {elapsed:.1f}s | 最终评估 {eval_ret:.1f}")

    os.makedirs(args.save_dir, exist_ok=True)
    ckpt = os.path.join(args.save_dir, f"sac_{policy_type}.pt")
    torch.save({"policy": agent.policy.state_dict(),
                "q1": agent.q1.state_dict(),
                "q2": agent.q2.state_dict(),
                "policy_type": policy_type}, ckpt)
    print(f"[{policy_type}] 模型已保存到 {ckpt}")

    return {"policy_type": policy_type, "final_eval": eval_ret,
            "elapsed": elapsed, "history": history}


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
                     label=res["policy_type"])
        axes[1].plot(hist["steps"], hist["mean_abs_q"], marker="o",
                     label=res["policy_type"])
    axes[0].set_xlabel("环境步数")
    axes[0].set_ylabel("确定性评估回报")
    axes[0].set_title("评估回报（部分可观测）")
    axes[0].legend()
    axes[1].set_xlabel("环境步数")
    axes[1].set_ylabel("|Q| 平均值")
    axes[1].set_title("价值估计幅度")
    axes[1].legend()
    fig.tight_layout()
    out = os.path.join(save_dir, "ch038_recurrent.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"曲线已保存到 {out}")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="第038章：SAC 的循环策略网络")
    p.add_argument("--policies", type=str, default="mlp,gru",
                   help="要对比的策略结构：mlp / gru")
    p.add_argument("--window", type=int, default=8, help="策略可见的历史帧数")
    p.add_argument("--alpha", type=float, default=0.2, help="alpha 初值")
    p.add_argument("--target-entropy", type=float, default=-1.0, help="目标熵")
    p.add_argument("--max-steps", type=int, default=12000, help="每种策略的步数上限")
    p.add_argument("--start-steps", type=int, default=1000, help="纯随机探索步数")
    p.add_argument("--buffer-size", type=int, default=50000, help="回放池容量（条）")
    p.add_argument("--batch-size", type=int, default=256, help="批量大小")
    p.add_argument("--hidden", type=int, default=256, help="隐藏层宽度")
    p.add_argument("--lr", type=float, default=3e-4, help="学习率")
    p.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
    p.add_argument("--tau", type=float, default=0.005, help="目标网络软更新系数")
    p.add_argument("--eval-every", type=int, default=3000, help="评估与日志间隔")
    p.add_argument("--eval-episodes", type=int, default=3, help="每次评估的回合数")
    p.add_argument("--seed", type=int, default=0, help="随机种子")
    p.add_argument("--save-dir", type=str, default="runs/ch038", help="模型保存目录")
    p.add_argument("--quick", action="store_true", help="快速模式：1-2 分钟跑完")
    return p.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.max_steps = min(args.max_steps, 2000)
        args.start_steps = min(args.start_steps, 500)
        args.eval_every = 1000
        args.eval_episodes = 2

    policies = [p.strip() for p in args.policies.split(",") if p.strip()]
    for p in policies:
        if p not in ALL_POLICIES:
            raise SystemExit(f"未知策略 {p}，可选：{ALL_POLICIES}")

    results = [train_policy(p, args) for p in policies]

    print("=" * 74)
    print("循环策略对比汇总（部分可观测 Pendulum，以实跑为准）")
    for res in results:
        print(f"  {res['policy_type']:>3s}: 最终评估 {res['final_eval']:9.1f} | "
              f"用时 {res['elapsed']:5.1f}s")

    maybe_plot(results, args.save_dir)


if __name__ == "__main__":
    main()
