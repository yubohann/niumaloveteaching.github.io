"""
第061章 SAC的离线强化学习

完全不用环境交互，只靠一批固定数据训练策略。数据由"随机 + 脚本专家 + 噪声"
的混合行为策略在 Pendulum 上采集 5 万条转移。对比三种离线训练方式：

  - bc          : 纯行为克隆（监督学习，只回归数据里的动作）
  - sac         : 朴素离线 SAC（标准 SAC 更新，但只从固定数据集采样）
  - cql         : 离线 SAC + 保守惩罚（对策略采样出的 OOD 动作压低 Q 值）

环境只在评估时使用（每 2000 次梯度更新评估 5 个回合）。记录评估回报与
"探针批次的 min(Q1,Q2) 均值"——朴素离线 SAC 的 Q 值常会虚假膨胀。

运行：
    python code/ch061.py           # 数据采集 + 三种离线训练（CPU 约 3-5 分钟）
    python code/ch061.py --quick   # 快速跑通（约 50 秒）

预期：bc 稳定但上限受数据质量限制；sac 早期看起来不错，Q 值持续膨胀后
评估回报崩坏；cql 的 Q 值被压住，评估回报最稳。在线 SAC 的对比留到下一章。
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_pendulum(seed: int):
    env = gym.make("Pendulum-v1")
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env


# ---------------------------------------------------------------------------
# 行为策略：70% 概率用脚本专家（带噪声），30% 概率随机
# ---------------------------------------------------------------------------
def expert_action(obs) -> np.ndarray:
    """能量泵 + PD 的脚本专家（与第 057 章同一实现）。

    E = 0.5θ̇² + 15cosθ 与 gymnasium 简化动力学的 3g/2 项对应；
    PD 比例增益 6 需高于静态平衡所需的 5。
    """
    cos_t, sin_t, thd = float(obs[0]), float(obs[1]), float(obs[2])
    theta = math.atan2(sin_t, cos_t)
    energy = 0.5 * thd * thd + 15.0 * math.cos(theta)
    if abs(theta) < 0.4:
        u = -6.0 * theta - 1.5 * thd
    else:
        u = 0.015 * thd * (15.0 - energy)
    return np.array([np.clip(u, -2.0, 2.0)], dtype=np.float32)


class ReplayBuffer:
    """用于承载固定离线数据集（容量 = 数据量，不再写入新数据）。"""

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

    def update(self, batch: dict, args, cql_coef: float = 0.0) -> dict:
        """一次离线更新；cql_coef > 0 时加入保守惩罚（CQL 风格）。"""
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

        if cql_coef > 0.0:
            # 保守惩罚：抬高数据动作的 Q、压低 OOD（策略采样+均匀采样）的 Q
            n_ood = args.cql_samples
            # 策略采样：对每个状态采 n_ood 个动作
            obs_rep = obs.repeat_interleave(n_ood, dim=0)  # (B*n, obs)
            policy_actions, _ = self.policy.sample(obs_rep)
            uniform_actions = (torch.rand_like(policy_actions) * 2.0 - 1.0) \
                * self.policy.act_scale
            ood_actions = torch.cat([policy_actions, uniform_actions], dim=0)
            obs_rep2 = obs.repeat_interleave(2 * n_ood, dim=0)
            q1_ood = self.q1(obs_rep2, ood_actions).view(obs.shape[0], 2 * n_ood)
            q2_ood = self.q2(obs_rep2, ood_actions).view(obs.shape[0], 2 * n_ood)
            cql1 = torch.logsumexp(q1_ood, dim=1) - q1
            cql2 = torch.logsumexp(q2_ood, dim=1) - q2
            q_loss = q_loss + cql_coef * (cql1 + cql2).mean()

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

    def bc_update(self, batch: dict, args) -> dict:
        """纯行为克隆：只回归数据集里的动作（不使用奖励与 Q）。"""
        mean_action = self.policy.mean_action(batch["obs"])
        loss = F.mse_loss(mean_action, batch["act"])
        self.opt_policy.zero_grad()
        loss.backward()
        self.opt_policy.step()
        return {"bc_loss": loss.item()}

    @torch.no_grad()
    def probe_q(self, probe: dict) -> float:
        """固定探针批次的 min(Q1,Q2) 均值：诊断 Q 值是否虚假膨胀。"""
        q = torch.minimum(self.q1(probe["obs"], probe["act"]),
                          self.q2(probe["obs"], probe["act"]))
        return float(q.mean().item())


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
# 数据采集
# ---------------------------------------------------------------------------
def collect_dataset(n_transitions: int, seed: int, noise: float, random_frac: float):
    """用混合行为策略采集固定数据集，返回 (ReplayBuffer, 行为策略平均回报)。"""
    env = make_pendulum(seed)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    buffer = ReplayBuffer(n_transitions, obs_dim, act_dim)
    rng = np.random.default_rng(seed)
    obs, _ = env.reset()
    ep_ret = 0.0
    ep_returns = []
    while buffer.size < n_transitions:
        if rng.random() < random_frac:
            action = env.action_space.sample()
        else:
            action = expert_action(obs) + rng.normal(0.0, noise, size=1).astype(np.float32)
            action = np.clip(action, -2.0, 2.0).astype(np.float32)
        next_obs, rew, terminated, truncated, _ = env.step(action)
        buffer.add(obs, action, rew, next_obs, float(terminated))
        obs = next_obs
        ep_ret += rew
        if terminated or truncated:
            ep_returns.append(ep_ret)
            ep_ret = 0.0
            obs, _ = env.reset()
    return buffer, float(np.mean(ep_returns[-30:]))


# ---------------------------------------------------------------------------
# 离线训练一种方法
# ---------------------------------------------------------------------------
def train_offline(mode: str, dataset: ReplayBuffer, probe: dict, args):
    set_seed(args.seed)  # 三种方法同种子，控制变量
    obs_dim = dataset.obs.shape[1]
    act_dim = dataset.act.shape[1]
    eval_env = make_pendulum(args.seed + 100)
    agent = SAC(obs_dim, act_dim, 2.0, args)

    eval_history = []
    q_history = []
    t_start = time.time()
    print(f"\n===== 离线训练 [{mode}]，共 {args.train_steps} 次梯度更新 =====")

    for step in range(1, args.train_steps + 1):
        batch = dataset.sample(args.batch_size)
        if mode == "bc":
            stats = agent.bc_update(batch, args)
        elif mode == "sac":
            stats = agent.update(batch, args, cql_coef=0.0)
        else:  # cql
            stats = agent.update(batch, args, cql_coef=args.cql_coef)

        if step % args.eval_every == 0:
            eval_ret = evaluate(agent, eval_env, args.eval_episodes)
            eval_history.append((step, eval_ret))
            if mode == "bc":
                q_mean = float("nan")
            else:
                q_mean = agent.probe_q(probe)
                q_history.append(q_mean)
            q_str = f"{q_mean:8.2f}" if mode != "bc" else "     n/a"
            print(f"[{mode}] 更新 {step:6d} | 评估 {eval_ret:8.1f} | 探针 Q 均值 {q_str}")

    elapsed = time.time() - t_start
    final_eval = evaluate(agent, eval_env, args.eval_episodes)
    best_eval = max([h[1] for h in eval_history] + [final_eval])
    q_final = agent.probe_q(probe) if mode != "bc" else float("nan")
    print(f"[{mode}] 结束：最好评估 {best_eval:.1f} | 最终评估 {final_eval:.1f} | "
          f"探针 Q 均值 {q_final if mode != 'bc' else float('nan'):.2f} | 用时 {elapsed:.1f}s")
    return {"mode": mode, "best_eval": best_eval, "final_eval": final_eval,
            "q_final": q_final, "history": eval_history, "elapsed": elapsed}


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="第061章：SAC 的离线强化学习")
    p.add_argument("--modes", type=str, default="bc,sac,cql",
                   help="逗号分隔：bc / sac / cql")
    p.add_argument("--dataset-size", type=int, default=50000, help="离线数据集大小")
    p.add_argument("--random-frac", type=float, default=0.3, help="行为策略中随机动作的比例")
    p.add_argument("--behavior-noise", type=float, default=0.5, help="行为策略中专家动作的噪声")
    p.add_argument("--train-steps", type=int, default=8000, help="每种方法的梯度更新次数")
    p.add_argument("--eval-every", type=int, default=2000, help="评估间隔（按更新次数）")
    p.add_argument("--eval-episodes", type=int, default=5, help="评估回合数")
    p.add_argument("--cql-coef", type=float, default=1.0, help="CQL 保守惩罚系数")
    p.add_argument("--cql-samples", type=int, default=8, help="CQL 每个状态的 OOD 采样数（策略+均匀各一份）")
    p.add_argument("--batch-size", type=int, default=256, help="批大小")
    p.add_argument("--hidden", type=int, default=256, help="隐藏层宽度")
    p.add_argument("--lr", type=float, default=3e-4, help="学习率")
    p.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
    p.add_argument("--tau", type=float, default=0.005, help="目标网络软更新系数")
    p.add_argument("--alpha-init", type=float, default=0.2, help="温度系数初值")
    p.add_argument("--seed", type=int, default=0, help="随机种子")
    p.add_argument("--save-dir", type=str, default="runs/ch061", help="输出目录")
    p.add_argument("--quick", action="store_true", help="快速模式：缩短训练")
    return p.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.dataset_size = min(args.dataset_size, 6000)
        args.train_steps = min(args.train_steps, 1200)
        args.eval_every = 400
        args.eval_episodes = 3

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    print(f"种子: {args.seed} | 方法: {modes} | 数据集 {args.dataset_size} 条 | "
          f"更新 {args.train_steps} 次 | cql_coef={args.cql_coef}")

    # 1) 采集固定数据集
    set_seed(args.seed)
    print("\n===== 采集离线数据集（随机 + 脚本专家混合行为策略） =====")
    t0 = time.time()
    dataset, behavior_return = collect_dataset(args.dataset_size, args.seed,
                                               args.behavior_noise, args.random_frac)
    print(f"数据集完成：{dataset.size} 条 | 行为策略平均回报 {behavior_return:.1f} | "
          f"用时 {time.time() - t0:.1f}s")

    # 探针批次（固定 512 条，追踪 Q 值尺度）
    probe_idx = np.random.randint(0, dataset.size, size=min(512, dataset.size))
    probe = {
        "obs": torch.as_tensor(dataset.obs[probe_idx]),
        "act": torch.as_tensor(dataset.act[probe_idx]),
    }

    # 2) 三种离线方法
    results = [train_offline(m, dataset, probe, args) for m in modes]

    # 3) 汇总
    print("\n" + "=" * 88)
    print("离线强化学习对比（行为策略参考回报见上；环境只用于评估）")
    print(f"{'方法':<8}{'最好评估':>10}{'最终评估':>10}{'探针 Q 均值':>14}{'用时(s)':>10}")
    for r in results:
        q = r["q_final"]
        q_str = f"{q:14.2f}" if not np.isnan(q) else f"{'n/a':>14}"
        print(f"{r['mode']:<8}{r['best_eval']:>10.1f}{r['final_eval']:>10.1f}"
              f"{q_str}{r['elapsed']:>10.1f}")
    print("=" * 88)
    print("结论提示：bc 稳定但受数据质量限制；朴素离线 sac 的 Q 值会持续膨胀，")
    print("最终评估可能崩坏；cql 用保守惩罚压住 OOD 动作的 Q，回报更稳。下一章")
    print("会用同一套评估协议对比'在线 SAC'，看看离线省下的交互代价是多少。")

    os.makedirs(args.save_dir, exist_ok=True)
    print(f"输出目录：{args.save_dir}")


if __name__ == "__main__":
    main()
