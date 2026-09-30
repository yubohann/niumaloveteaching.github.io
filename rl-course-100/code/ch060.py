"""
第060章 SAC的模型预测控制结合

把 SAC 从"纯反应式控制器"升级为"预测式控制器"：先用 SAC 训练策略与双 Q，
再用回放数据训练一个动力学模型集成（3 个 MLP 预测状态增量），最后在评估时
做随机打靶 MPC（model predictive control）：

    候选动作序列 K 条 × 预测视界 H 步
    每步用集成模型预测下一状态，奖励用真实奖励函数计算
    序列得分 = 折扣奖励之和 + gamma^H * min(Q1, Q2)(末端状态, 末端动作)
    执行得分最高序列的第一个动作，下一步重新规划

对比：纯策略（SAC 确定性输出） vs SAC+MPC（H=2 与 H=6），记录回报、
模型预测误差（验证集 MSE）与规划耗时。

运行：
    python code/ch060.py           # SAC 训练 + 模型训练 + MPC 评估（CPU 约 3-5 分钟）
    python code/ch060.py --quick   # 快速跑通（约 40 秒）

预期：MPC 的评估回报不劣于纯策略（Pendulum 上通常略好或持平），代价是
每步毫秒级的规划开销；H 从 2 增到 6 时回报往往先升后平，耗时线性增长。
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


def make_pendulum(seed: int):
    env = gym.make("Pendulum-v1")
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


class DynamicsModel(nn.Module):
    """预测状态增量：next_obs - obs = f(obs, act)。"""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, obs_dim),
        )

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, act], dim=-1))


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


# ---------------------------------------------------------------------------
# 动力学模型训练
# ---------------------------------------------------------------------------
def train_dynamics_models(buffer: ReplayBuffer, obs_dim: int, act_dim: int,
                          n_models: int, steps: int, batch_size: int,
                          lr: float, seed: int):
    """在回放数据上训练模型集成，返回模型列表与验证 MSE。"""
    set_seed(seed)
    models = [DynamicsModel(obs_dim, act_dim) for _ in range(n_models)]
    opts = [torch.optim.Adam(m.parameters(), lr=lr) for m in models]

    # 验证集：回放池里最近的 5000 条（与训练采样有交集，作诊断用）
    n_val = min(5000, buffer.size)
    val_obs = torch.as_tensor(buffer.obs[buffer.size - n_val:buffer.size])
    val_act = torch.as_tensor(buffer.act[buffer.size - n_val:buffer.size])
    val_next = torch.as_tensor(buffer.next_obs[buffer.size - n_val:buffer.size])
    val_delta = val_next - val_obs

    for step in range(1, steps + 1):
        batch = buffer.sample(batch_size)
        delta_target = batch["next_obs"] - batch["obs"]
        for i, m in enumerate(models):
            pred = m(batch["obs"], batch["act"])
            loss = F.mse_loss(pred, delta_target)
            opts[i].zero_grad()
            loss.backward()
            opts[i].step()
        if step % max(steps // 4, 1) == 0:
            with torch.no_grad():
                mses = [F.mse_loss(m(val_obs, val_act), val_delta).item()
                        for m in models]
            print(f"[模型] 训练步 {step:5d} | 集成验证 MSE "
                  f"{np.mean(mses):.2e}（各成员 {[f'{v:.2e}' for v in mses]}）")

    with torch.no_grad():
        mses = [F.mse_loss(m(val_obs, val_act), val_delta).item() for m in models]
    return models, float(np.mean(mses))


# ---------------------------------------------------------------------------
# 随机打靶 MPC
# ---------------------------------------------------------------------------
def pendulum_reward(obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
    """Pendulum 的真实奖励函数（用于 MPC 的逐步打分）。"""
    cos_t, sin_t, thd = obs[:, 0], obs[:, 1], obs[:, 2]
    theta = torch.atan2(sin_t, cos_t)
    return -(theta ** 2 + 0.1 * thd ** 2 + 0.001 * act.squeeze(-1) ** 2)


def plan_action(models, agent: SAC, obs: np.ndarray, horizon: int,
                n_samples: int, gamma: float, noise_std: float) -> np.ndarray:
    """随机打靶：采样 K 条动作序列，用集成模型滚动预测，选得分最高者的首动作。"""
    obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32))
    with torch.no_grad():
        base = agent.policy.mean_action(obs_t.unsqueeze(0)).squeeze(0)  # 以策略为中心采样
    noise = torch.randn(n_samples, horizon, base.shape[0]) * noise_std
    actions = (base.view(1, 1, -1) + noise).clamp(-2.0, 2.0)

    states = obs_t.unsqueeze(0).repeat(n_samples, 1)
    total = torch.zeros(n_samples)
    for t in range(horizon):
        a_t = actions[:, t]
        ins = torch.cat([states, a_t], dim=-1)
        with torch.no_grad():
            deltas = torch.stack([m(ins) for m in models], dim=0).mean(dim=0)
        states = states + deltas
        total = total + (gamma ** t) * pendulum_reward(states, a_t)
    with torch.no_grad():
        q_term = torch.minimum(agent.q1(states, actions[:, -1]),
                               agent.q2(states, actions[:, -1]))
    total = total + (gamma ** horizon) * q_term
    best = int(torch.argmax(total).item())
    return actions[best, 0].numpy()


def evaluate_mpc(models, agent: SAC, env, episodes: int, horizon: int,
                 n_samples: int, gamma: float, noise_std: float):
    """MPC 评估：返回 (平均回报, 每步平均规划耗时 ms)。"""
    returns = []
    plan_times = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            t0 = time.perf_counter()
            action = plan_action(models, agent, obs, horizon, n_samples, gamma, noise_std)
            plan_times.append((time.perf_counter() - t0) * 1000.0)
            obs, rew, terminated, truncated, _ = env.step(action)
            ep_ret += rew
            done = terminated or truncated
        returns.append(ep_ret)
    return float(np.mean(returns)), float(np.mean(plan_times))


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="第060章：SAC 的模型预测控制结合")
    p.add_argument("--horizons", type=str, default="2,6", help="MPC 预测视界列表")
    p.add_argument("--mpc-samples", type=int, default=32, help="MPC 打靶候选数 K")
    p.add_argument("--mpc-noise", type=float, default=0.5, help="候选动作序列的动作噪声")
    p.add_argument("--ensemble", type=int, default=3, help="动力学模型集成的成员数")
    p.add_argument("--model-steps", type=int, default=2000, help="模型训练梯度步数")
    p.add_argument("--model-lr", type=float, default=1e-3, help="模型学习率")
    p.add_argument("--max-steps", type=int, default=15000, help="SAC 训练步数")
    p.add_argument("--start-steps", type=int, default=1000, help="训练前纯随机探索步数")
    p.add_argument("--update-every", type=int, default=1, help="每多少环境步做一次更新")
    p.add_argument("--eval-every", type=int, default=3000, help="训练期间评估间隔")
    p.add_argument("--eval-episodes", type=int, default=5, help="每种方法的评估回合数")
    p.add_argument("--buffer-size", type=int, default=100000, help="回放池容量")
    p.add_argument("--batch-size", type=int, default=256, help="批大小")
    p.add_argument("--hidden", type=int, default=256, help="SAC 隐藏层宽度")
    p.add_argument("--lr", type=float, default=3e-4, help="SAC 学习率")
    p.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
    p.add_argument("--tau", type=float, default=0.005, help="目标网络软更新系数")
    p.add_argument("--alpha-init", type=float, default=0.2, help="温度系数初值")
    p.add_argument("--seed", type=int, default=0, help="随机种子")
    p.add_argument("--save-dir", type=str, default="runs/ch060", help="输出目录")
    p.add_argument("--quick", action="store_true", help="快速模式：缩短训练")
    return p.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.max_steps = min(args.max_steps, 3000)
        args.eval_every = 1500
        args.start_steps = min(args.start_steps, 300)
        args.model_steps = min(args.model_steps, 300)
        args.eval_episodes = 2
        args.mpc_samples = min(args.mpc_samples, 16)
        args.horizons = "2,4"

    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]
    set_seed(args.seed)

    # 1) SAC 训练
    env = make_pendulum(args.seed)
    eval_env = make_pendulum(args.seed + 100)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    action_scale = float(env.action_space.high[0])
    agent = SAC(obs_dim, act_dim, action_scale, args)
    buffer = ReplayBuffer(args.buffer_size, obs_dim, act_dim)

    print(f"种子: {args.seed} | SAC 训练 {args.max_steps} 步 | "
          f"MPC: K={args.mpc_samples}, H={horizons}, 集成 {args.ensemble} 个模型")
    obs, _ = env.reset()
    ep_ret = 0.0
    ep_returns = []
    t_start = time.time()
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
        if step % args.update_every == 0 and buffer.size >= min(args.start_steps, 1000):
            agent.update(buffer.sample(args.batch_size), args)
        if step % args.eval_every == 0:
            eval_ret = evaluate(agent, eval_env, args.eval_episodes)
            recent20 = np.mean(ep_returns[-20:]) if ep_returns else float("nan")
            print(f"[SAC] 步数 {step:6d} | 近20回合回报 {recent20:8.1f} | 评估 {eval_ret:8.1f}")
    print(f"[SAC] 训练完成，用时 {time.time() - t_start:.1f}s，"
          f"回放池 {buffer.size} 条")

    # 2) 动力学模型集成
    print(f"\n===== 训练动力学模型集成（{args.ensemble} 个 MLP，"
          f"{args.model_steps} 步梯度） =====")
    models, val_mse = train_dynamics_models(
        buffer, obs_dim, act_dim, args.ensemble, args.model_steps,
        args.batch_size, args.model_lr, args.seed + 42)
    print(f"[模型] 最终验证 MSE：{val_mse:.3e}")

    # 3) 评估对比
    print("\n===== 评估：纯策略 vs SAC+MPC =====")
    policy_ret = evaluate(agent, eval_env, args.eval_episodes)
    print(f"[纯策略] 评估回报 {policy_ret:.1f}")
    rows = [("纯策略", "-", policy_ret, 0.0)]
    for h in horizons:
        ret, ms = evaluate_mpc(models, agent, eval_env, args.eval_episodes,
                               h, args.mpc_samples, args.gamma, args.mpc_noise)
        rows.append((f"SAC+MPC", h, ret, ms))
        print(f"[SAC+MPC] H={h} | 评估回报 {ret:.1f} | 规划耗时 {ms:.2f} ms/步")

    print("\n" + "=" * 80)
    print("方法对比（K = 打靶候选数；耗时只统计规划，不含环境步进）")
    print(f"{'方法':<12}{'H':>4}{'评估回报':>12}{'规划耗时(ms/步)':>18}")
    for name, h, ret, ms in rows:
        print(f"{name:<12}{str(h):>4}{ret:>12.1f}{ms:>18.2f}")
    print("=" * 80)
    print("结论提示：Pendulum 上 MPC 与纯策略通常打平或略优（任务本身对动作")
    print("连续性要求高，打靶噪声会引入抖动）；真正拉开差距的场景需要'多步")
    print("前瞻'才能避免的陷阱，或策略本身欠拟合的时期。H 增大后收益递减、")
    print("耗时线性增长，K 与 H 的取值要在控制质量与算力之间折中。")
    print(f"模型验证 MSE 可作为信任度参考：{val_mse:.3e}")

    os.makedirs(args.save_dir, exist_ok=True)
    print(f"输出目录：{args.save_dir}")


if __name__ == "__main__":
    main()
