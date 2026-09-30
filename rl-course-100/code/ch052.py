"""
第052章 SAC的稀疏奖励处理

把 MountainCarContinuous-v0 改造成"纯稀疏奖励"版本（到达山顶 +100，其余
每步 -1），然后对比两种训练方式：

  - sparse : 直接学习纯稀疏奖励。成功信号出现概率极低，SAC 主要靠随机
             探索偶尔发现"先退后进"的轨迹
  - shaped : 叠加势函数塑形（potential-based shaping）
             r' = r + gamma * phi(s') - phi(s)，phi(s) = 50 * position
             塑形不改变最优策略，但给每一步提供指向山顶的稠密梯度

诊断指标：成功率达到 50% 所需的环境步数、首个成功回合、评估成功率和
评估回合内的最大位置（进度指标）。

运行：
    python code/ch052.py           # 两种模式依次训练（CPU 约 5-8 分钟）
    python code/ch052.py --quick   # 快速跑通（约 1 分钟）

预期：shaped 通常在一万步内出现首个成功、两三万步内稳定爬山；sparse 的
首个成功要晚得多，个别种子在两万步内完全找不到成功轨迹。
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


def make_env(env_id: str, seed: int, shaping: float = 0.0, gamma: float = 0.99):
    """创建稀疏奖励版 MountainCarContinuous。

    shaping > 0 时启用势函数塑形：phi(s) = shaping * position，
    塑形项 F = gamma * phi(s') - phi(s) 加到稀疏奖励上。
    """
    env = gym.make(env_id)
    env = SparseShapingWrapper(env, shaping=shaping, gamma=gamma)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env


class SparseShapingWrapper(gym.Wrapper):
    """把稠密奖励改成"到山顶 +100、其余 -1"，可选势函数塑形。

    塑形使用势函数 phi(s) = shaping * position，步内奖励修正为
    F = gamma * phi(s') - phi(s)。势存储在 wrapper 上，随环境推进更新。
    """

    def __init__(self, env, shaping: float = 0.0, gamma: float = 0.99):
        super().__init__(env)
        self.shaping = shaping
        self.gamma = gamma
        self._phi_prev = 0.0

    def _phi(self, obs) -> float:
        return self.shaping * float(obs[0])  # 位置越高势能越大

    def step(self, action):
        obs, _orig_rew, terminated, truncated, info = self.env.step(action)
        reward = 100.0 if terminated else -1.0
        if self.shaping > 0.0:
            phi_next = self._phi(obs)
            reward += self.gamma * phi_next - self._phi_prev
            self._phi_prev = phi_next
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._phi_prev = self._phi(obs)
        return obs, info


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
        rew, done = rew.squeeze(-1), done.squeeze(-1)      # 与 Q 网络的 (B,) 输出对齐

        with torch.no_grad():
            next_act, next_logp = self.policy.sample(next_obs)
            next_logp = next_logp.squeeze(-1)
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
        logp = logp.squeeze(-1)                            # 同上：统一为 (B,)
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


@torch.no_grad()
def evaluate(agent: SAC, env, episodes: int):
    """返回 (平均回报, 成功率, 平均最大位置)。"""
    returns, successes, max_positions = [], [], []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_ret = 0.0
        max_pos = -1.2
        success = False
        while not done:
            action = agent.act(obs, deterministic=True)
            obs, rew, terminated, truncated, _ = env.step(action)
            ep_ret += rew
            max_pos = max(max_pos, float(env.unwrapped.state[0]))
            success = success or terminated
            done = terminated or truncated
        returns.append(ep_ret)
        successes.append(float(success))
        max_positions.append(max_pos)
    return float(np.mean(returns)), float(np.mean(successes)), float(np.mean(max_positions))


# ---------------------------------------------------------------------------
# 训练一种模式
# ---------------------------------------------------------------------------
def train_mode(mode: str, args):
    shaping = 0.0 if mode == "sparse" else args.shaping
    set_seed(args.seed)  # 两种模式同种子，控制变量
    env = make_env(args.env, args.seed, shaping=shaping, gamma=args.gamma)
    eval_env = make_env(args.env, args.seed + 100, shaping=shaping, gamma=args.gamma)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    action_scale = float(env.action_space.high[0])
    agent = SAC(obs_dim, act_dim, action_scale, args)
    buffer = ReplayBuffer(args.buffer_size, obs_dim, act_dim)

    obs, _ = env.reset(seed=args.seed)
    ep_ret = 0.0
    ep_returns = []
    successes = []           # 每个回合是否到山顶
    first_success_ep = None
    solved_at = None         # 滑窗成功率首次 >= 50% 的步数
    window = args.success_window
    eval_history = []
    t_start = time.time()
    print(f"\n===== 模式 {mode}（shaping={shaping}）：开始训练 =====")

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
            successes.append(float(terminated))
            if terminated and first_success_ep is None:
                first_success_ep = len(successes)
                print(f"[{mode}] 第 {first_success_ep} 个回合首次到达山顶（步数 {step}）")
            ep_ret = 0.0
            obs, _ = env.reset()

        if step % args.update_every == 0 and buffer.size >= min(args.start_steps, 1000):
            agent.update(buffer.sample(args.batch_size), args)

        if step % args.eval_every == 0:
            eval_ret, eval_succ, eval_pos = evaluate(agent, eval_env, args.eval_episodes)
            eval_history.append((step, eval_ret, eval_succ, eval_pos))
            recent = successes[-window:] if len(successes) >= window else successes
            train_succ = float(np.mean(recent)) if recent else 0.0
            print(f"[{mode}] 步数 {step:6d} | 近{window}回合成功率 {train_succ:5.2f} | "
                  f"评估 {eval_ret:8.1f} | 评估成功率 {eval_succ:.2f} | "
                  f"最大位置 {eval_pos:+.3f}")
            if solved_at is None and len(successes) >= window and train_succ >= 0.5:
                solved_at = step
                print(f"[{mode}] 滑窗成功率首次达到 50%（步数 {step}）")

    elapsed = time.time() - t_start
    eval_ret, eval_succ, eval_pos = evaluate(agent, eval_env, args.eval_episodes)
    best_eval = max([h[1] for h in eval_history] + [eval_ret])
    final_succ = float(np.mean(successes[-100:])) if successes else 0.0
    print(f"[{mode}] 结束：最好评估 {best_eval:.1f} | 最终评估 {eval_ret:.1f} | "
          f"最后100回合成功率 {final_succ:.2f} | 评估成功率 {eval_succ:.2f} | "
          f"用时 {elapsed:.1f}s")
    return {"mode": mode, "best_eval": best_eval, "final_eval": eval_ret,
            "train_succ": final_succ, "eval_succ": eval_succ, "eval_pos": eval_pos,
            "first_success_ep": first_success_ep, "solved_at": solved_at,
            "elapsed": elapsed}


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="第052章：SAC 的稀疏奖励处理")
    p.add_argument("--env", type=str, default="MountainCarContinuous-v0",
                   help="基础环境（会被改造成稀疏奖励）")
    p.add_argument("--modes", type=str, default="sparse,shaped",
                   help="逗号分隔：sparse / shaped")
    p.add_argument("--shaping", type=float, default=50.0,
                   help="势函数系数 phi(s) = shaping * position")
    p.add_argument("--max-steps", type=int, default=30000, help="每种模式的环境步数上限")
    p.add_argument("--start-steps", type=int, default=2000, help="训练前纯随机探索步数")
    p.add_argument("--update-every", type=int, default=1, help="每多少环境步做一次更新")
    p.add_argument("--eval-every", type=int, default=5000, help="评估间隔")
    p.add_argument("--eval-episodes", type=int, default=5, help="评估回合数")
    p.add_argument("--success-window", type=int, default=20, help="滑窗成功率使用的回合数")
    p.add_argument("--buffer-size", type=int, default=100000, help="回放池容量")
    p.add_argument("--batch-size", type=int, default=256, help="批大小")
    p.add_argument("--hidden", type=int, default=256, help="隐藏层宽度")
    p.add_argument("--lr", type=float, default=3e-4, help="学习率")
    p.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
    p.add_argument("--tau", type=float, default=0.005, help="目标网络软更新系数")
    p.add_argument("--alpha-init", type=float, default=0.2, help="温度系数初值")
    p.add_argument("--seed", type=int, default=0, help="随机种子")
    p.add_argument("--save-dir", type=str, default="runs/ch052", help="输出目录")
    p.add_argument("--quick", action="store_true", help="快速模式：缩短训练")
    return p.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.max_steps = min(args.max_steps, 4000)
        args.eval_every = 1000
        args.start_steps = min(args.start_steps, 400)
        args.eval_episodes = 3
        args.success_window = min(args.success_window, 10)

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    print(f"环境: {args.env}（稀疏奖励改造）| 种子: {args.seed} | 模式: {modes}")
    print(f"配置: max_steps={args.max_steps}, shaping={args.shaping}, "
          f"batch={args.batch_size}, lr={args.lr}")

    results = [train_mode(m, args) for m in modes]

    print("\n" + "=" * 100)
    print("稀疏奖励处理对比（成功率 = 到达山顶的回合占比）")
    print(f"{'模式':<10}{'最好评估':>10}{'最终评估':>10}{'训练成功率':>12}"
          f"{'评估成功率':>12}{'首次成功回合':>14}{'50%达标步数':>13}")
    for r in results:
        first = r["first_success_ep"] if r["first_success_ep"] is not None else -1
        solved = r["solved_at"] if r["solved_at"] is not None else -1
        print(f"{r['mode']:<10}{r['best_eval']:>10.1f}{r['final_eval']:>10.1f}"
              f"{r['train_succ']:>12.2f}{r['eval_succ']:>12.2f}{first:>14}{solved:>13}")
    print("=" * 100)
    print("结论提示：纯稀疏奖励下策略只能靠随机探索偶遇成功轨迹；势函数塑形把")
    print("'往高处走'变成逐步可学的信号，首个成功与 50% 达标来得更早。塑形是")
    print("势函数形式（F = gamma*phi(s') - phi(s)），不改变最优策略。")

    os.makedirs(args.save_dir, exist_ok=True)
    print(f"输出目录：{args.save_dir}")


if __name__ == "__main__":
    main()
