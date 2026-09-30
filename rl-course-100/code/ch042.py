"""
第042章 SAC的Q网络过估计分析

本实验在 Pendulum-v1 上对比 SAC 中三种双 Q 组合方式对价值高估的影响：
  - min   : 双 Q 取最小（SAC 默认，TD3 风格，抑制过估计）
  - mean  : 双 Q 取平均
  - single: 只用 Q1（单 Q 估计器，没有第二把尺子纠偏）

诊断方法：定期让当前策略在环境中跑若干回合，记录每个时刻的 (s, a) 及
其"折扣回报 G_t"，再比较网络输出 Q(s,a) 与真实回报的均值差
    bias = mean(Q(s,a) - G_t)
bias > 0 说明网络对动作价值过于乐观（过估计），bias < 0 说明偏保守。

运行：
    python code/ch042.py           # 三种估计器依次训练（CPU 约 5-8 分钟）
    python code/ch042.py --quick   # 快速跑通（约 1 分钟，只观察趋势）

预期：single 的 bias 明显为正；min 接近 0 甚至略负；mean 居中。
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
    """固定随机种子，保证实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_env(env_id: str, seed: int):
    """创建环境并锁定随机种子。"""
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
        # done 只记录真实终止（terminated），时间截断（truncated）仍然 bootstrap
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
    """动作价值网络 Q(s,a)，输入状态与动作的拼接。"""

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
    """tanh 压缩高斯策略：输出动作均值与 log 标准差，重参数化采样。"""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256,
                 act_scale: float = 1.0):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mean_layer = nn.Linear(hidden, act_dim)
        self.log_std_layer = nn.Linear(hidden, act_dim)
        # 把 tanh 输出 [-1,1] 映射到动作空间的范围，例如 Pendulum 的 [-2,2]
        self.register_buffer("act_scale", torch.tensor(float(act_scale)))

    def forward(self, obs: torch.Tensor):
        h = self.body(obs)
        mean = self.mean_layer(h)
        log_std = torch.clamp(self.log_std_layer(h), LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, obs: torch.Tensor):
        """重参数化采样，返回动作与对应的对数概率（含 tanh 与缩放修正）。"""
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
        """确定性动作（tanh 后的均值），评估时使用。"""
        mean, _ = self.forward(obs)
        return torch.tanh(mean) * self.act_scale


def soft_update(net: nn.Module, target: nn.Module, tau: float) -> None:
    """软更新：target = (1 - tau) * target + tau * net。"""
    for tp, p in zip(target.parameters(), net.parameters()):
        tp.data.mul_(1.0 - tau).add_(tau * p.data)


# ---------------------------------------------------------------------------
# SAC 智能体（estimator 决定双 Q 的组合方式）
# ---------------------------------------------------------------------------
class SAC:
    def __init__(self, obs_dim: int, act_dim: int, action_scale: float,
                 estimator: str, args):
        self.estimator = estimator
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
        self.target_entropy = -float(act_dim)  # 标准目标熵：每个动作维度 -1
        self.opt_q = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=args.lr)
        self.opt_policy = torch.optim.Adam(self.policy.parameters(), lr=args.lr)
        self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=args.lr)

    @property
    def alpha(self) -> torch.Tensor:
        """温度系数 a = exp(log_alpha)，下限 1e-4 防止塌缩。"""
        return self.log_alpha.exp().clamp(1e-4, 10.0)

    def combine(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """按照本章的实验设置组合双 Q 输出。"""
        if self.estimator == "single":
            return q1
        if self.estimator == "mean":
            return 0.5 * (q1 + q2)
        return torch.minimum(q1, q2)  # 默认 min：SAC 的标准做法

    @torch.no_grad()
    def act(self, obs, deterministic: bool = False) -> np.ndarray:
        """与环境交互的动作；deterministic=True 时用均值动作评估。"""
        obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
        if deterministic:
            return self.policy.mean_action(obs_t).squeeze(0).numpy()
        action, _ = self.policy.sample(obs_t)
        return action.squeeze(0).numpy()

    @torch.no_grad()
    def q_values(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        """给出 (s,a) 的价值估计，用于过估计诊断。"""
        return self.combine(self.q1(obs, act), self.q2(obs, act))

    def update(self, batch: dict, args) -> dict:
        """一次 SAC 更新：双 Q 回归 -> 策略提升 -> 温度调整 -> 软更新。"""
        obs, act = batch["obs"], batch["act"]
        rew, next_obs, done = batch["rew"], batch["next_obs"], batch["done"]

        # 1) 计算目标值：r + gamma * (min/mean/single Q - alpha * log pi)
        with torch.no_grad():
            next_act, next_logp = self.policy.sample(next_obs)
            q_next = self.combine(self.q1_target(next_obs, next_act),
                                  self.q2_target(next_obs, next_act))
            backup = rew + args.gamma * (1.0 - done) * (q_next - self.alpha.detach() * next_logp)

        # 2) 双 Q 回归
        q1 = self.q1(obs, act)
        q2 = self.q2(obs, act)
        q1_loss = F.mse_loss(q1, backup)
        q2_loss = F.mse_loss(q2, backup)
        q_loss = q1_loss + q2_loss
        self.opt_q.zero_grad()
        q_loss.backward()
        self.opt_q.step()

        # 3) 策略提升：最大化 Q - alpha * log pi
        new_act, logp = self.policy.sample(obs)
        q_pi = self.combine(self.q1(obs, new_act), self.q2(obs, new_act))
        policy_loss = (self.alpha.detach() * logp - q_pi).mean()
        self.opt_policy.zero_grad()
        policy_loss.backward()
        self.opt_policy.step()

        # 4) 自动温度：让策略熵贴近目标熵
        alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()
        self.opt_alpha.zero_grad()
        alpha_loss.backward()
        self.opt_alpha.step()

        # 5) 软更新目标网络
        with torch.no_grad():
            soft_update(self.q1, self.q1_target, args.tau)
            soft_update(self.q2, self.q2_target, args.tau)

        return {
            "q1_loss": q1_loss.item(),
            "q2_loss": q2_loss.item(),
            "policy_loss": policy_loss.item(),
            "alpha": float(self.alpha.item()),
            "q_mean": float(q1.mean().item()),
        }


# ---------------------------------------------------------------------------
# 评估与过估计诊断
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(agent: SAC, env, episodes: int) -> float:
    """用确定性策略评估，返回平均回合回报。"""
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
def measure_q_bias(agent: SAC, env, episodes: int, gamma: float):
    """跑若干回合，比较 Q(s,a) 与真实折扣回报 G_t 的偏差。

    返回 (bias 均值, |Q| 均值)。bias = Q - G_t > 0 表示过估计。
    注意：回合末尾 G_t 被截断会带来轻微低估，但三种估计器口径一致，不影响对比。
    """
    biases, q_abs = [], []
    for _ in range(episodes):
        obs, _ = env.reset()
        obs_list, act_list, rew_list = [], [], []
        terminated = truncated = False
        while not (terminated or truncated):
            action = agent.act(obs, deterministic=False)
            next_obs, rew, terminated, truncated, _ = env.step(action)
            obs_list.append(np.asarray(obs, dtype=np.float32))
            act_list.append(np.asarray(action, dtype=np.float32))
            rew_list.append(float(rew))
            obs = next_obs
        # 从后向前计算每一步的折扣回报 G_t
        returns = np.zeros(len(rew_list), dtype=np.float32)
        running = 0.0
        for t in reversed(range(len(rew_list))):
            running = rew_list[t] + gamma * running
            returns[t] = running
        obs_t = torch.as_tensor(np.asarray(obs_list))
        act_t = torch.as_tensor(np.asarray(act_list))
        q = agent.q_values(obs_t, act_t)
        biases.append((q - torch.as_tensor(returns)).mean().item())
        q_abs.append(q.abs().mean().item())
    return float(np.mean(biases)), float(np.mean(q_abs))


# ---------------------------------------------------------------------------
# 单个估计器的完整训练
# ---------------------------------------------------------------------------
def train_estimator(estimator: str, args):
    """训练并评估一个估计器，返回记录与最终统计。"""
    set_seed(args.seed)  # 三种估计器使用相同种子，尽量控制变量
    env = make_env(args.env, args.seed)
    eval_env = make_env(args.env, args.seed + 100)
    probe_env = make_env(args.env, args.seed + 200)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    action_scale = float(env.action_space.high[0])
    agent = SAC(obs_dim, act_dim, action_scale, estimator, args)
    buffer = ReplayBuffer(args.buffer_size, obs_dim, act_dim)

    obs, _ = env.reset(seed=args.seed)
    ep_ret = 0.0
    ep_returns = []
    history = []  # (步数, 评估回报, Q 偏差, |Q| 均值)
    t_start = time.time()
    print(f"\n===== 估计器 {estimator}：开始训练（上限 {args.max_steps} 步） =====")

    for step in range(1, args.max_steps + 1):
        if step <= args.start_steps:
            action = env.action_space.sample()  # 前期随机探索
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

        if buffer.size >= args.start_steps and step % args.update_every == 0:
            stats = agent.update(buffer.sample(args.batch_size), args)

        if step % args.eval_every == 0:
            eval_ret = evaluate(agent, eval_env, args.eval_episodes)
            bias, q_abs = measure_q_bias(agent, probe_env, args.bias_episodes, args.gamma)
            history.append((step, eval_ret, bias, q_abs))
            recent20 = np.mean(ep_returns[-20:]) if ep_returns else float("nan")
            print(f"[{estimator}] 步数 {step:6d} | 近20回合回报 {recent20:8.1f} | "
                  f"评估 {eval_ret:8.1f} | Q偏差 {bias:+7.2f} | |Q|均值 {q_abs:6.1f} | "
                  f"alpha {stats['alpha']:.3f}")

    elapsed = time.time() - t_start
    final_eval = evaluate(agent, eval_env, args.eval_episodes)
    best_eval = max([h[1] for h in history] + [final_eval])
    final_bias, final_q_abs = history[-1][2], history[-1][3]
    print(f"[{estimator}] 结束：最好评估 {best_eval:.1f} | 最终评估 {final_eval:.1f} | "
          f"最终 Q 偏差 {final_bias:+.2f} | 用时 {elapsed:.1f}s")
    return {
        "estimator": estimator,
        "history": history,
        "best_eval": best_eval,
        "final_eval": final_eval,
        "final_bias": final_bias,
        "final_q_abs": final_q_abs,
        "elapsed": elapsed,
    }


# ---------------------------------------------------------------------------
# 可选绘图（惰性导入 matplotlib，只在 --plot 时执行）
# ---------------------------------------------------------------------------
def maybe_plot(results, save_dir: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")  # 无窗口后端，保存到文件即可
        import matplotlib.pyplot as plt
    except ImportError:
        print("未安装 matplotlib，跳过绘图。")
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for r in results:
        steps = [h[0] for h in r["history"]]
        axes[0].plot(steps, [h[1] for h in r["history"]], label=r["estimator"])
        axes[1].plot(steps, [h[2] for h in r["history"]], label=r["estimator"])
    axes[0].set_xlabel("env steps")
    axes[0].set_ylabel("eval return")
    axes[0].set_title("eval return")
    axes[1].axhline(0.0, color="gray", ls="--", lw=1)
    axes[1].set_xlabel("env steps")
    axes[1].set_ylabel("Q bias")
    axes[1].set_title("overestimation bias = Q - G_t")
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.3)
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "q_bias.png")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"曲线已保存到 {path}")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="第042章：SAC 的 Q 网络过估计分析")
    p.add_argument("--env", type=str, default="Pendulum-v1", help="环境 ID")
    p.add_argument("--estimators", type=str, default="min,mean,single",
                   help="逗号分隔的估计器列表：min / mean / single")
    p.add_argument("--max-steps", type=int, default=24000, help="每个估计器的环境步数上限")
    p.add_argument("--start-steps", type=int, default=1000, help="训练前纯随机探索步数")
    p.add_argument("--update-every", type=int, default=1, help="每多少环境步做一次梯度更新")
    p.add_argument("--eval-every", type=int, default=4000, help="每多少步评估一次")
    p.add_argument("--eval-episodes", type=int, default=5, help="评估回合数")
    p.add_argument("--bias-episodes", type=int, default=5, help="诊断 Q 偏差使用的回合数")
    p.add_argument("--buffer-size", type=int, default=100000, help="回放池容量")
    p.add_argument("--batch-size", type=int, default=256, help="批大小")
    p.add_argument("--hidden", type=int, default=256, help="隐藏层宽度")
    p.add_argument("--lr", type=float, default=3e-4, help="学习率")
    p.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
    p.add_argument("--tau", type=float, default=0.005, help="目标网络软更新系数")
    p.add_argument("--alpha-init", type=float, default=0.2, help="温度系数初值")
    p.add_argument("--seed", type=int, default=0, help="随机种子")
    p.add_argument("--save-dir", type=str, default="runs/ch042", help="输出目录")
    p.add_argument("--plot", action="store_true", help="保存 Q 偏差曲线图（需 matplotlib）")
    p.add_argument("--quick", action="store_true", help="快速模式：缩短训练")
    return p.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.max_steps = min(args.max_steps, 4800)
        args.eval_every = 1200
        args.bias_episodes = 3
        args.start_steps = min(args.start_steps, 400)

    estimators = [e.strip() for e in args.estimators.split(",") if e.strip()]
    print(f"环境: {args.env} | 种子: {args.seed} | 估计器: {estimators}")
    print(f"配置: max_steps={args.max_steps}, batch={args.batch_size}, "
          f"update_every={args.update_every}, tau={args.tau}")

    results = []
    for est in estimators:
        results.append(train_estimator(est, args))

    # 汇总对比表
    print("\n" + "=" * 78)
    print("估计器对比（Q 偏差为正表示过估计，越接近 0 越好）")
    print(f"{'估计器':<10}{'最好评估回报':>14}{'最终评估回报':>14}{'最终 Q 偏差':>14}{'最终|Q|均值':>14}")
    for r in results:
        print(f"{r['estimator']:<10}{r['best_eval']:>14.1f}{r['final_eval']:>14.1f}"
              f"{r['final_bias']:>+14.2f}{r['final_q_abs']:>14.1f}")
    print("=" * 78)
    print("结论提示：single 往往学到最大的正偏差；min 用双网络的较小值做目标，")
    print("目标是系统性偏保守，长期能把 bias 拉回到 0 附近；mean 介于两者之间。")

    if args.plot:
        maybe_plot(results, args.save_dir)

    os.makedirs(args.save_dir, exist_ok=True)
    print(f"输出目录：{args.save_dir}")


if __name__ == "__main__":
    main()
