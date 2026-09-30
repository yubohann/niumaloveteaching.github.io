"""
第031章 SAC在Pendulum上的首次实现

第二篇开场：从 on-policy 的 PPO 切换到 off-policy 的 SAC。
在 Pendulum-v1（连续动作 [-2, 2]，第 030 章里 PPO 磨蹭 6 万步也上不去的
那块试验田）上，用不到 3 万步实现一个完整的 SAC：

  - 经验回放：收集到的 (s, a, r, s', done) 全部留在缓冲区，反复采样训练
  - 双 Q 网络 + 目标网络：clipped double-Q 抑制过估计，Polyak 软更新
  - tanh 压缩高斯策略：重参数化采样，log 概率带雅可比修正
  - 自动温度：把熵系数 alpha 当作可学习参数，目标熵 = -动作维度

与 PPO 的结构差异（本章的核心看点）：
  1) 采样与更新解耦：每次环境步（warmup 之后）都做一次梯度更新，
     而 PPO 必须等一整批新数据；旧数据永远有效
  2) 不需要优势函数、不需要 GAE：Q 网络直接学动作价值
  3) 探索强度由 alpha 自动调节：奖励尺度变了也不用重调探索参数

运行：
    python code/ch031.py            # 2.5 万步（CPU 约 3-6 分钟）
    python code/ch031.py --quick    # 快速跑通（约 30-60 秒）
    python code/ch031.py --max-steps 50000 --eval-interval 2000
    python code/ch031.py --init-alpha 0.05     # 从更低的初始温度出发

预期：2.5 万步内评估通常能到 -300 ~ -150（随机策略约 -1200），
对比第 030 章同预算的 PPO（约 -260 ~ -200 且仍在下坡），SAC 用更少的
数据走得更远。数值区间以实跑为准。
"""

import argparse
import math
import os
import random
import sys
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_STD_MIN, LOG_STD_MAX = -20.0, 2.0   # log_std 夹紧范围（SAC 常用取值）


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    """固定随机种子，保证实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_tensor(obs, device):
    """单条观测 -> (1, obs_dim) 张量。"""
    return torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)


# ---------------------------------------------------------------------------
# 经验回放缓冲区：SAC 的"燃料仓"
# ---------------------------------------------------------------------------
class ReplayBuffer:
    """定长环形缓冲区，存储 (obs, act, rew, next_obs, done)。"""

    def __init__(self, capacity: int, obs_dim: int, act_dim: int):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rew = np.zeros((capacity, 1), dtype=np.float32)
        self.done = np.zeros((capacity, 1), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(self, obs, act, rew, next_obs, done):
        i = self.ptr
        self.obs[i] = obs
        self.act[i] = act
        self.rew[i] = rew
        self.next_obs[i] = next_obs
        self.done[i] = done              # 只标记真实终止；时间截断仍要 bootstrap
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.RandomState):
        idx = rng.randint(0, self.size, size=batch_size)
        return (self.obs[idx], self.act[idx], self.rew[idx],
                self.next_obs[idx], self.done[idx])


# ---------------------------------------------------------------------------
# 网络
# ---------------------------------------------------------------------------
class SquashedGaussianPolicy(nn.Module):
    """tanh 压缩高斯策略：输出 [shift-scale, shift+scale] 区间内的动作。"""

    def __init__(self, obs_dim: int, act_dim: int, act_low, act_high, hidden: int = 128):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mu = nn.Linear(hidden, act_dim)
        self.log_std = nn.Linear(hidden, act_dim)
        # 动作仿射：a = shift + scale * tanh(u)
        self.register_buffer("act_shift",
                             torch.as_tensor((act_high + act_low) / 2.0, dtype=torch.float32))
        self.register_buffer("act_scale",
                             torch.as_tensor((act_high - act_low) / 2.0, dtype=torch.float32))

    def forward(self, obs: torch.Tensor):
        """返回 (动作, 对数概率)；用重参数化采样，两项都可导。"""
        h = self.body(obs)
        mu = self.mu(h)
        log_std = self.log_std(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()
        dist = torch.distributions.Normal(mu, std)

        u = dist.rsample()                       # 重参数化：梯度可穿过采样
        tanh_u = torch.tanh(u)
        action = self.act_shift + self.act_scale * tanh_u

        # log π(a) = log N(u) - Σ log|da/du|，雅可比含 scale 与 tanh 导数
        logp = dist.log_prob(u)
        logp -= torch.log(self.act_scale * (1.0 - tanh_u.pow(2)) + 1e-6)
        return action, logp.sum(dim=-1)

    @torch.no_grad()
    def act(self, obs_t):
        """推理用（无梯度）：采一个动作并返回 numpy。"""
        action, _ = self.forward(obs_t)
        return action.squeeze(0).cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def deterministic_action(self, obs_t):
        """评估用（无梯度）：取 tanh(mu) 的确定性动作。"""
        h = self.body(obs_t)
        mu = self.mu(h)
        action = self.act_shift + self.act_scale * torch.tanh(mu)
        return action.squeeze(0).cpu().numpy().astype(np.float32)


class QNet(nn.Module):
    """动作价值网络 Q(s, a) -> 标量。"""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs: torch.Tensor, act: torch.Tensor):
        return self.body(torch.cat([obs, act], dim=-1)).squeeze(-1)


# ---------------------------------------------------------------------------
# SAC 核心更新
# ---------------------------------------------------------------------------
def soft_update(target: nn.Module, source: nn.Module, tau: float):
    """Polyak 软更新：θ_target ← τθ + (1-τ)θ_target。"""
    with torch.no_grad():
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.copy_(tau * sp.data + (1.0 - tau) * tp.data)


def update_agent(agent, batch, args, device):
    """对一批回放数据做一次 SAC 更新，返回诊断量。"""
    obs, act, rew, next_obs, done = batch
    obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
    act = torch.as_tensor(act, dtype=torch.float32, device=device)
    rew = torch.as_tensor(rew, dtype=torch.float32, device=device).squeeze(-1)
    next_obs = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
    done = torch.as_tensor(done, dtype=torch.float32, device=device).squeeze(-1)

    # ---- 1) 计算 TD 目标：双 Q 取小 + 目标策略熵 ----
    with torch.no_grad():
        next_action, next_logp = agent.policy(next_obs)
        q1_next = agent.q1_target(next_obs, next_action)
        q2_next = agent.q2_target(next_obs, next_action)
        q_next = torch.min(q1_next, q2_next) - agent.alpha * next_logp
        target = rew + args.gamma * (1.0 - done) * q_next

    # ---- 2) 双 Q 损失：两个网络都向同一个目标回归 ----
    q1 = agent.q1(obs, act)
    q2 = agent.q2(obs, act)
    q_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
    agent.q_opt.zero_grad()
    q_loss.backward()
    agent.q_opt.step()

    # ---- 3) 策略损失：最大化 Q - alpha * log π（alpha 不参与策略梯度）----
    pi_action, pi_logp = agent.policy(obs)
    q_pi = torch.min(agent.q1(obs, pi_action), agent.q2(obs, pi_action))
    pi_loss = (agent.alpha.detach() * pi_logp - q_pi).mean()
    agent.pi_opt.zero_grad()
    pi_loss.backward()
    agent.pi_opt.step()

    # ---- 4) 自动温度：让平均 log π 靠近目标熵（-动作维度）----
    alpha_loss = -(agent.log_alpha * (pi_logp + agent.target_entropy).detach()).mean()
    agent.alpha_opt.zero_grad()
    alpha_loss.backward()
    agent.alpha_opt.step()

    # ---- 5) 目标网络软更新 ----
    soft_update(agent.q1_target, agent.q1, args.tau)
    soft_update(agent.q2_target, agent.q2, args.tau)

    return {
        "q_loss": float(q_loss.item()),
        "pi_loss": float(pi_loss.item()),
        "alpha": float(agent.alpha.item()),
        "q_mean": float(q1.mean().item()),
        "logp_mean": float(pi_logp.mean().item()),
    }


# ---------------------------------------------------------------------------
# 评估
# ---------------------------------------------------------------------------
def evaluate(env, policy, episodes: int, device):
    returns = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            action = policy.deterministic_action(to_tensor(obs, device))
            obs, r, terminated, truncated, _ = env.step(action)
            ep_ret += r
            done = terminated or truncated
        returns.append(ep_ret)
    return returns


def sparkline(values, width=56) -> str:
    """把评估曲线画成文本火花线（终端编码不支持方块字符时降级为 ASCII）。"""
    if not values:
        return "(无评估数据)"
    blocks = "▁▂▃▄▅▆▇█"
    try:
        blocks.encode(sys.stdout.encoding or "utf-8")
    except (UnicodeEncodeError, LookupError):
        blocks = ".,-~+=*#"
    v = np.asarray(values, dtype=np.float64)
    if len(v) > width:
        idx = np.linspace(0, len(v) - 1, width).astype(np.int64)
        v = v[idx]
    lo, hi = float(v.min()), float(v.max())
    span = max(hi - lo, 1e-9)
    line = "".join(blocks[min(len(blocks) - 1,
                              int((x - lo) / span * (len(blocks) - 1) + 0.5))] for x in v)
    return f"min={lo:.0f} max={hi:.0f}\n    |{line}|"


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="第031章：SAC 在 Pendulum 上的首次实现")
    p.add_argument("--max-steps", type=int, default=25000, help="总环境步数（含 warmup）")
    p.add_argument("--warmup", type=int, default=1000, help="前多少步用随机动作填充缓冲区")
    p.add_argument("--batch-size", type=int, default=128, help="每次更新的采样批量")
    p.add_argument("--buffer-size", type=int, default=100000, help="经验回放容量")
    p.add_argument("--hidden", type=int, default=128, help="网络隐藏层宽度")
    p.add_argument("--lr", type=float, default=3e-4, help="Q 与策略网络学习率")
    p.add_argument("--alpha-lr", type=float, default=3e-4, help="温度参数学习率")
    p.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
    p.add_argument("--tau", type=float, default=0.005, help="目标网络软更新系数")
    p.add_argument("--init-alpha", type=float, default=0.2, help="温度参数初始值")
    p.add_argument("--update-every", type=int, default=1, help="每个环境步做几次梯度更新")
    p.add_argument("--eval-interval", type=int, default=2000, help="每隔多少步做一次评估")
    p.add_argument("--eval-episodes", type=int, default=5, help="每次评估的回合数")
    p.add_argument("--seed", type=int, default=0, help="随机种子")
    p.add_argument("--save-dir", type=str, default="runs/ch031", help="模型保存目录")
    p.add_argument("--quick", action="store_true", help="快速模式：缩小训练规模，几十秒跑完")
    return p.parse_args()


class Agent:
    """把四个网络、三个优化器与温度参数组装在一起。"""

    def __init__(self, obs_dim, act_dim, act_low, act_high, args, device):
        self.device = device
        self.policy = SquashedGaussianPolicy(obs_dim, act_dim, act_low, act_high,
                                             args.hidden).to(device)
        self.q1 = QNet(obs_dim, act_dim, args.hidden).to(device)
        self.q2 = QNet(obs_dim, act_dim, args.hidden).to(device)
        self.q1_target = QNet(obs_dim, act_dim, args.hidden).to(device)
        self.q2_target = QNet(obs_dim, act_dim, args.hidden).to(device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.pi_opt = torch.optim.Adam(self.policy.parameters(), lr=args.lr)
        self.q_opt = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=args.lr)

        # 温度参数用 log 形式存储，保证 alpha > 0
        self.log_alpha = torch.tensor(math.log(args.init_alpha),
                                      dtype=torch.float32, requires_grad=True)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=args.alpha_lr)
        self.alpha = self.log_alpha.exp().detach()

        # 目标熵：连续控制的常用选择是 -dim(A)（Pendulum 即 -1）
        self.target_entropy = -float(act_dim)

    def refresh_alpha(self):
        """优化器 step 之后刷新当前 alpha 标量（供损失与日志使用）。"""
        self.alpha = self.log_alpha.exp().detach()


def main():
    args = parse_args()
    if args.quick:
        args.max_steps = min(args.max_steps, 4000)
        args.warmup = min(args.warmup, 500)
        args.hidden = 64
        args.batch_size = 64
        args.eval_interval = 500
        args.eval_episodes = 3

    set_seed(args.seed)
    device = torch.device("cpu")

    env = gym.make("Pendulum-v1")
    obs, _ = env.reset(seed=args.seed)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_low = env.action_space.low.astype(np.float32)
    act_high = env.action_space.high.astype(np.float32)

    buffer = ReplayBuffer(args.buffer_size, obs_dim, act_dim)
    agent = Agent(obs_dim, act_dim, act_low, act_high, args, device)
    rng = np.random.RandomState(args.seed)

    print(f"设备: {device} | 环境: Pendulum-v1 | 种子: {args.seed}")
    print(f"配置: max_steps={args.max_steps}, warmup={args.warmup}, "
          f"batch={args.batch_size}, hidden={args.hidden}, "
          f"alpha0={args.init_alpha}, target_entropy={agent.target_entropy}")

    ep_ret = 0.0
    ep_returns = []
    eval_history = []
    last_stats = None
    t_start = time.time()

    for step in range(1, args.max_steps + 1):
        # ---- 采样：warmup 用随机动作，之后用当前策略 ----
        if step <= args.warmup or buffer.size < args.batch_size:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                action = agent.policy.act(to_tensor(obs, device))

        next_obs, r, terminated, truncated, _ = env.step(action)
        buffer.add(obs, np.asarray(action, dtype=np.float32), r, next_obs,
                   float(terminated))
        ep_ret += r
        obs = next_obs
        if terminated or truncated:
            ep_returns.append(ep_ret)
            ep_ret = 0.0
            obs, _ = env.reset()

        # ---- 更新：off-policy，每次环境步都能学习 ----
        if buffer.size >= max(args.warmup, args.batch_size):
            for _ in range(args.update_every):
                batch = buffer.sample(args.batch_size, rng)
                last_stats = update_agent(agent, batch, args, device)
                agent.refresh_alpha()

        # ---- 评估与日志 ----
        if step % args.eval_interval == 0:
            eval_returns = evaluate(env, agent.policy, args.eval_episodes, device)
            eval_mean = float(np.mean(eval_returns))
            eval_history.append(eval_mean)
            recent = ep_returns[-10:] if ep_returns else [0.0]
            stats_txt = ""
            if last_stats is not None:
                stats_txt = (f" | α {last_stats['alpha']:.3f} | "
                             f"Q {last_stats['q_mean']:7.2f} | "
                             f"logπ {last_stats['logp_mean']:+.2f}")
            print(f"步数 {step:6d} | 回合 {len(ep_returns):3d} | "
                  f"训练近10均 {np.mean(recent):8.1f} | 评估 {eval_mean:8.1f}"
                  f"{stats_txt}")

    elapsed = time.time() - t_start
    final_eval = evaluate(env, agent.policy, max(args.eval_episodes, 10), device)
    final_mean = float(np.mean(final_eval))

    print("-" * 78)
    print(f"训练结束 | 用时 {elapsed:.1f}s | 总步数 {args.max_steps} | "
          f"缓冲区 {buffer.size} 条")
    print(f"评估曲线（间隔 {args.eval_interval} 步）：")
    print("  " + sparkline(eval_history))
    print(f"最终评估（{max(args.eval_episodes, 10)} 回合，确定性策略）："
          f"{[round(v) for v in final_eval]} 平均 {final_mean:.1f}")
    print(f"最好一次评估 {max(eval_history) if eval_history else float('nan'):.1f} | "
          f"参考：随机策略约 -1200，优秀策略约 -200 ~ -150")

    os.makedirs(args.save_dir, exist_ok=True)
    ckpt = {
        "policy": agent.policy.state_dict(),
        "q1": agent.q1.state_dict(),
        "q2": agent.q2.state_dict(),
        "log_alpha": agent.log_alpha.detach(),
    }
    path = os.path.join(args.save_dir, "sac_pendulum.pt")
    torch.save(ckpt, path)
    print(f"模型已保存到 {path}")
    print("提示：把这条评估曲线与第 030 章 PPO 的曲线对比——SAC 用大约一半的"
          "交互步数达到更好的水平。下一章专门研究温度参数 alpha 的自动调整行为。")


if __name__ == "__main__":
    main()
