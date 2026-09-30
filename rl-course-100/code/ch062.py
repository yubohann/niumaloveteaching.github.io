"""
第062章 SAC的在线与离线对比

同一份 SAC 实现，两种数据来源的对照实验：
  - online ：边与环境交互边训练，数据随策略不断刷新
  - offline：先用固定行为策略收集数据集，训练阶段不再与环境交互
除数据来源外，两种模式的网络结构、学习率、批大小、梯度更新次数完全一致，
并额外记录"平均 Q 估计"与"真实评估回报"的差距，用来观察离线训练的分布偏移。

运行：
    python code/ch062.py          # 完整实验：在线 + 离线（CPU 约 6-12 分钟）
    python code/ch062.py --quick  # 快速跑通（约 1-2 分钟）
    python code/ch062.py --mode online

预期（以实跑为准）：在线 SAC 在 Pendulum-v1 上评估回报约 -250 ~ -150；
离线 SAC 受分布偏移影响停在行为策略附近（约 -700 ~ -400），
且平均 Q 估计明显高于它的真实评估回报。
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
from torch.distributions import Normal


# ---------------------------------------------------------------------------
# 通用工具
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    """固定随机种子，保证实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# 经验回放池
# ---------------------------------------------------------------------------
class ReplayBuffer:
    """固定容量的经验回放池：存 numpy 数组，采样时批量转张量。"""

    def __init__(self, capacity: int, obs_dim: int, act_dim: int):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rew = np.zeros((capacity, 1), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.done = np.zeros((capacity, 1), dtype=np.float32)

    def add(self, obs, act, rew, next_obs, done) -> None:
        i = self.ptr
        self.obs[i] = obs
        self.act[i] = act
        self.rew[i] = rew
        self.next_obs[i] = next_obs
        self.done[i] = done
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device):
        """随机采样一个批次，返回张量字典。"""
        idx = np.random.randint(0, self.size, size=batch_size)
        return {
            "obs": torch.as_tensor(self.obs[idx], device=device),
            "act": torch.as_tensor(self.act[idx], device=device),
            "rew": torch.as_tensor(self.rew[idx], device=device),
            "next_obs": torch.as_tensor(self.next_obs[idx], device=device),
            "done": torch.as_tensor(self.done[idx], device=device),
        }


# ---------------------------------------------------------------------------
# 网络：高斯策略 + 双 Q
# ---------------------------------------------------------------------------
class GaussianPolicy(nn.Module):
    """对角高斯策略 + tanh 压缩，动作范围 [-act_limit, act_limit]。"""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128, act_limit: float = 1.0):
        super().__init__()
        self.act_limit = act_limit
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mean = nn.Linear(hidden, act_dim)
        self.log_std = nn.Linear(hidden, act_dim)

    def forward(self, obs: torch.Tensor):
        h = self.body(obs)
        mean = self.mean(h)
        log_std = torch.clamp(self.log_std(h), -4.0, 0.5)  # 限制探索噪声范围
        return mean, log_std

    def sample(self, obs: torch.Tensor):
        """重参数化采样，返回（压缩后的动作, 对数概率）。"""
        mean, log_std = self.forward(obs)
        dist = Normal(mean, log_std.exp())
        x = dist.rsample()
        action = torch.tanh(x)
        # tanh 换元的对数概率修正
        logp = dist.log_prob(x) - torch.log(1.0 - action.pow(2) + 1e-6)
        return action * self.act_limit, logp.sum(-1, keepdim=True)

    @torch.no_grad()
    def mean_action(self, obs: torch.Tensor) -> torch.Tensor:
        """确定性动作：取分布的均值再压缩，用于评估。"""
        mean, _ = self.forward(obs)
        return torch.tanh(mean) * self.act_limit


class TwinQ(nn.Module):
    """双 Q 网络：两个独立 MLP，取较小值抑制过估计。"""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128):
        super().__init__()
        self.q1 = self._mlp(obs_dim + act_dim, hidden)
        self.q2 = self._mlp(obs_dim + act_dim, hidden)

    @staticmethod
    def _mlp(in_dim: int, hidden: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs: torch.Tensor, act: torch.Tensor):
        x = torch.cat([obs, act], dim=-1)
        return self.q1(x), self.q2(x)


# ---------------------------------------------------------------------------
# SAC 智能体：双 Q + 目标网络 + 自动温度
# ---------------------------------------------------------------------------
class SACAgent:
    def __init__(self, obs_dim: int, act_dim: int, act_limit: float, args, device):
        self.device = device
        self.policy = GaussianPolicy(obs_dim, act_dim, args.hidden, act_limit).to(device)
        self.q = TwinQ(obs_dim, act_dim, args.hidden).to(device)
        self.q_target = TwinQ(obs_dim, act_dim, args.hidden).to(device)
        self.q_target.load_state_dict(self.q.state_dict())
        self.policy_opt = torch.optim.Adam(self.policy.parameters(), lr=args.lr)
        self.q_opt = torch.optim.Adam(self.q.parameters(), lr=args.lr)
        self.log_alpha = torch.tensor(np.log(args.alpha), dtype=torch.float32,
                                      requires_grad=True, device=device)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=args.lr)
        self.target_entropy = -float(act_dim)  # 目标熵取动作维度的负数
        self.gamma = args.gamma
        self.tau = args.tau

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def update(self, batch):
        """一次完整更新：批评家 -> 演员 -> 温度 -> 软更新目标网络。"""
        obs, act = batch["obs"], batch["act"]
        rew, next_obs, done = batch["rew"], batch["next_obs"], batch["done"]

        # 1) 双 Q 的 Bellman 回归；下一动作与熵项都由当前策略给出
        with torch.no_grad():
            next_act, next_logp = self.policy.sample(next_obs)
            q1_t, q2_t = self.q_target(next_obs, next_act)
            target = rew + self.gamma * (1.0 - done) * (
                torch.min(q1_t, q2_t) - self.alpha * next_logp)
        q1, q2 = self.q(obs, act)
        q_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.q_opt.zero_grad()
        q_loss.backward()
        self.q_opt.step()

        # 2) 策略：最大化 Q 减熵，重参数化梯度穿过采样
        new_act, logp = self.policy.sample(obs)
        q1_n, q2_n = self.q(obs, new_act)
        policy_loss = (self.alpha.detach() * logp - torch.min(q1_n, q2_n)).mean()
        self.policy_opt.zero_grad()
        policy_loss.backward()
        self.policy_opt.step()

        # 3) 自动温度：把平均熵拉向目标熵
        alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        # 4) 软更新目标网络
        with torch.no_grad():
            for p, p_t in zip(self.q.parameters(), self.q_target.parameters()):
                p_t.data.mul_(1.0 - self.tau).add_(self.tau * p.data)

        return {"q_loss": q_loss.item(), "policy_loss": policy_loss.item(),
                "alpha": self.alpha.item(), "logp": logp.mean().item()}


# ---------------------------------------------------------------------------
# 行为策略：给离线数据集用（粗启发式 + 随机探索的混合）
# ---------------------------------------------------------------------------
def behavior_action(obs, rng) -> np.ndarray:
    """固定行为策略 pi_beta：55% 粗能量泵浦启发式，45% 均匀随机力矩。"""
    cos_t, sin_t, thdot = float(obs[0]), float(obs[1]), float(obs[2])
    if rng.random() < 0.55:
        # 摆动阶段满力泵浦；能量够了就滑行等 PD 捕捉直立
        energy = 0.5 * thdot * thdot + 15.0 * cos_t   # 系数 15 与 3g/2 项对应
        if cos_t < 0.9:
            u = 2.0 * (1.0 if thdot >= 0 else -1.0) if energy < 15.0 else 0.0
        else:
            u = -6.0 * sin_t - 1.5 * thdot   # PD 平衡：增益 6 高于静态所需的 5
        u += rng.normal(0.0, 0.5)
    else:
        u = rng.uniform(-2.0, 2.0)
    return np.array([np.clip(u, -2.0, 2.0)], dtype=np.float32)


# ---------------------------------------------------------------------------
# 评估与 Q 诊断
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(env, agent, episodes: int, device) -> float:
    """用确定性（均值）策略评估若干回合，返回平均回报。"""
    returns = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            action = agent.policy.mean_action(obs_t).squeeze(0).cpu().numpy()
            obs, r, terminated, truncated, _ = env.step(action)
            ep_ret += r
            done = terminated or truncated
        returns.append(ep_ret)
    return float(np.mean(returns))


def make_probe_idx(buffer: ReplayBuffer, n: int, seed: int) -> np.ndarray:
    """在数据集上固定一批探测样本，用于追踪 Q 估计的漂移。"""
    rng = np.random.RandomState(seed)
    return rng.randint(0, buffer.size, size=n)


@torch.no_grad()
def probe_q(agent, buffer: ReplayBuffer, idx: np.ndarray, device) -> float:
    """计算固定探测批次上的平均 min(Q1, Q2)。"""
    obs = torch.as_tensor(buffer.obs[idx], device=device)
    act = torch.as_tensor(buffer.act[idx], device=device)
    q1, q2 = agent.q(obs, act)
    return float(torch.min(q1, q2).mean().item())


# ---------------------------------------------------------------------------
# 在线训练：边交互边更新
# ---------------------------------------------------------------------------
def train_online(args, device):
    env = gym.make("Pendulum-v1")
    eval_env = gym.make("Pendulum-v1")
    obs, _ = env.reset(seed=args.seed)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = float(env.action_space.high[0])

    agent = SACAgent(obs_dim, act_dim, act_limit, args, device)
    buffer = ReplayBuffer(args.buffer_size, obs_dim, act_dim)
    probe_idx = None
    curve = []
    t0 = time.time()

    for step in range(1, args.steps + 1):
        # 前 start_steps 步用行为策略预热，保证缓冲池里有足够多样的数据
        if step <= args.start_steps:
            action = behavior_action(obs, np.random)
        else:
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                action = agent.policy.sample(obs_t)[0].squeeze(0).cpu().numpy()
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        buffer.add(obs, action, (reward,), next_obs, (float(done),))
        obs = next_obs
        if done:
            obs, _ = env.reset()

        if step > args.start_steps:
            stats = agent.update(buffer.sample(args.batch_size, device))
            grad_steps = step - args.start_steps
            if grad_steps % args.eval_every == 0:
                if probe_idx is None:
                    probe_idx = make_probe_idx(buffer, 2048, args.seed + 1)
                ret = evaluate(eval_env, agent, args.eval_episodes, device)
                q_mean = probe_q(agent, buffer, probe_idx, device)
                curve.append((grad_steps, ret, q_mean))
                print(f"[在线 更新 {grad_steps:5d}] 评估回报 {ret:8.1f} | "
                      f"平均Q {q_mean:8.1f} | α {stats['alpha']:.3f} | "
                      f"用时 {time.time() - t0:5.1f}s")

    return agent, curve, time.time() - t0


# ---------------------------------------------------------------------------
# 离线训练：固定数据集 + 零环境交互（评估用的交互不算训练数据）
# ---------------------------------------------------------------------------
def train_offline(args, device):
    env = gym.make("Pendulum-v1")       # 只用来收集数据集与评估
    eval_env = gym.make("Pendulum-v1")
    obs, _ = env.reset(seed=args.seed)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = float(env.action_space.high[0])

    # 阶段一：用行为策略收集固定数据集
    buffer = ReplayBuffer(args.dataset_size, obs_dim, act_dim)
    t0 = time.time()
    while buffer.size < args.dataset_size:
        action = behavior_action(obs, np.random)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        buffer.add(obs, action, (reward,), next_obs, (float(done),))
        obs = next_obs
        if done:
            obs, _ = env.reset()
    print(f"[离线] 数据集收集完成：{buffer.size} 条 transition，"
          f"耗时 {time.time() - t0:.1f}s")

    # 阶段二：只做梯度更新，不再产生任何新的训练数据
    agent = SACAgent(obs_dim, act_dim, act_limit, args, device)
    probe_idx = make_probe_idx(buffer, 2048, args.seed + 1)
    curve = []
    for grad_steps in range(1, args.grad_steps + 1):
        stats = agent.update(buffer.sample(args.batch_size, device))
        if grad_steps % args.eval_every == 0:
            ret = evaluate(eval_env, agent, args.eval_episodes, device)
            q_mean = probe_q(agent, buffer, probe_idx, device)
            curve.append((grad_steps, ret, q_mean))
            print(f"[离线 更新 {grad_steps:5d}] 评估回报 {ret:8.1f} | "
                  f"平均Q {q_mean:8.1f} | α {stats['alpha']:.3f} | "
                  f"用时 {time.time() - t0:5.1f}s")

    return agent, curve, time.time() - t0


# ---------------------------------------------------------------------------
# 绘图（惰性导入 matplotlib，只保存文件不弹窗）
# ---------------------------------------------------------------------------
def save_curve_plot(save_dir, online_curve, offline_curve) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("未安装 matplotlib，跳过绘图（不影响实验结论）")
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    if online_curve:
        xs = [c[0] for c in online_curve]
        ys = [c[1] for c in online_curve]
        ax.plot(xs, ys, marker="o", label="online")
    if offline_curve:
        xs = [c[0] for c in offline_curve]
        ys = [c[1] for c in offline_curve]
        ax.plot(xs, ys, marker="s", label="offline")
    ax.set_xlabel("gradient updates")
    ax.set_ylabel("eval return")
    ax.legend()
    ax.grid(alpha=0.3)
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "online_vs_offline.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"评估曲线已保存到 {path}")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="第062章：SAC 的在线与离线对比")
    p.add_argument("--mode", type=str, default="both", choices=["online", "offline", "both"],
                   help="运行哪些模式")
    p.add_argument("--steps", type=int, default=30000, help="在线模式的环境步数")
    p.add_argument("--dataset-size", type=int, default=30000, help="离线数据集大小（transition 数）")
    p.add_argument("--grad-steps", type=int, default=None,
                   help="离线模式梯度更新次数，默认与在线模式一致")
    p.add_argument("--start-steps", type=int, default=3000, help="在线模式的纯随机/行为策略预热步数")
    p.add_argument("--buffer-size", type=int, default=100000, help="在线回放池容量")
    p.add_argument("--batch-size", type=int, default=256, help="小批量大小")
    p.add_argument("--hidden", type=int, default=128, help="隐藏层宽度")
    p.add_argument("--lr", type=float, default=3e-4, help="学习率")
    p.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
    p.add_argument("--tau", type=float, default=0.005, help="目标网络软更新系数")
    p.add_argument("--alpha", type=float, default=0.2, help="温度系数初值（自动调整）")
    p.add_argument("--eval-every", type=int, default=2500, help="每多少次梯度更新评估一次")
    p.add_argument("--eval-episodes", type=int, default=10, help="每次评估的回合数")
    p.add_argument("--seed", type=int, default=0, help="随机种子")
    p.add_argument("--save-dir", type=str, default="runs/ch062", help="模型与曲线保存目录")
    p.add_argument("--quick", action="store_true", help="快速模式：缩小规模，几十秒到两分钟跑完")
    return p.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.steps = min(args.steps, 6000)
        args.dataset_size = min(args.dataset_size, 6000)
        args.start_steps = min(args.start_steps, 500)
        args.eval_every = 1000
        args.eval_episodes = 3
        args.hidden = 64
    if args.grad_steps is None:
        args.grad_steps = max(1, args.steps - args.start_steps)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"设备: {device} | 环境: Pendulum-v1 | 种子: {args.seed}")
    print(f"配置: mode={args.mode}, steps={args.steps}, dataset={args.dataset_size}, "
          f"grad_steps={args.grad_steps}, batch={args.batch_size}, lr={args.lr}")

    results = {}
    if args.mode in ("online", "both"):
        agent_on, curve_on, dt_on = train_online(args, device)
        results["online"] = (curve_on, dt_on)
    else:
        curve_on = None
    if args.mode in ("offline", "both"):
        agent_off, curve_off, dt_off = train_offline(args, device)
        results["offline"] = (curve_off, dt_off)
    else:
        curve_off = None

    print("=" * 72)
    for name, (curve, dt) in results.items():
        if not curve:
            continue
        last3 = np.mean([c[1] for c in curve[-3:]])
        best = max(c[1] for c in curve)
        final_q = curve[-1][2]
        print(f"[{name:7s}] 用时 {dt:6.1f}s | 最后3次评估均值 {last3:8.1f} | "
              f"最佳评估 {best:8.1f} | 最终平均Q估计 {final_q:8.1f}")
    if "online" in results and "offline" in results:
        gap_on = results["online"][0][-1][1] - results["online"][0][-1][2]
        gap_off = results["offline"][0][-1][1] - results["offline"][0][-1][2]
        print(f"真实回报 − 平均Q：在线 {gap_on:8.1f} | 离线 {gap_off:8.1f} "
              f"（差距越大，过估计越严重）")

    os.makedirs(args.save_dir, exist_ok=True)
    if curve_on is not None:
        ckpt = os.path.join(args.save_dir, "online_agent.pt")
        torch.save(agent_on.policy.state_dict(), ckpt)
        print(f"在线策略已保存到 {ckpt}")
    if curve_off is not None:
        ckpt = os.path.join(args.save_dir, "offline_agent.pt")
        torch.save(agent_off.policy.state_dict(), ckpt)
        print(f"离线策略已保存到 {ckpt}")
    save_curve_plot(args.save_dir, curve_on, curve_off)


if __name__ == "__main__":
    main()
