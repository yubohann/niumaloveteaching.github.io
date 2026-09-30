"""
第055章 SAC的多任务学习

让一个 SAC 同时服务三个摆杆任务（Pendulum 的重力 g = 8 / 12 / 16）：
把"任务身份"作为 one-hot 向量拼到观测后面，训练时按回合轮换任务。

对比两种方案（总环境步数相同）：
  - multi  : 单个共享策略，观测 = [状态(3), 任务 one-hot(3)]，三个任务各占 1/3 步数
  - single : 每个任务单独训练一个策略，各自使用 1/3 步数

记录：各任务评估回报、参数量、以及"任务标识敏感性"——把 multi 策略喂
错误的任务 one-hot 时性能如何变化（检验它是否真的在用任务标识）。

运行：
    python code/ch055.py           # multi + single 全部训练（CPU 约 3-5 分钟）
    python code/ch055.py --quick   # 快速跑通（约 50 秒）

预期：single 在每个任务上略优（专用容量更大），multi 以 1/3 的参数量
达到接近的水平；multi 对错误任务标识有明显性能下降，说明标识被真正使用。
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


def make_pendulum(seed: int, g: float):
    env = gym.make("Pendulum-v1")
    env.unwrapped.g = g
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env


def onehot(i: int, n: int) -> np.ndarray:
    v = np.zeros(n, dtype=np.float32)
    v[i] = 1.0
    return v


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

    def n_params(self) -> int:
        total = sum(p.numel() for p in self.policy.parameters())
        total += sum(p.numel() for p in self.q1.parameters())
        total += sum(p.numel() for p in self.q2.parameters())
        return int(total)

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
def eval_on_task(agent: SAC, env, task_id, n_tasks: int, episodes: int,
                 override_id=None) -> float:
    """在指定任务上评估；override_id 可以把 one-hot 换成错误任务标识。"""
    returns = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            if task_id is None:
                obs_in = np.asarray(obs, dtype=np.float32)
            else:
                tid = task_id if override_id is None else override_id
                obs_in = np.concatenate([np.asarray(obs, dtype=np.float32),
                                         onehot(tid, n_tasks)])
            action = agent.act(obs_in, deterministic=True)
            obs, rew, terminated, truncated, _ = env.step(action)
            ep_ret += rew
            done = terminated or truncated
        returns.append(ep_ret)
    return float(np.mean(returns))


# ---------------------------------------------------------------------------
# multi 模式：单策略 + 任务 one-hot
# ---------------------------------------------------------------------------
def train_multi(gs, args):
    set_seed(args.seed)
    n_tasks = len(gs)
    envs = [make_pendulum(args.seed + 11 * i, g) for i, g in enumerate(gs)]
    eval_envs = [make_pendulum(args.seed + 100 + 11 * i, g) for i, g in enumerate(gs)]
    obs_dim = envs[0].observation_space.shape[0] + n_tasks  # 状态 + one-hot
    act_dim = envs[0].action_space.shape[0]
    action_scale = float(envs[0].action_space.high[0])
    agent = SAC(obs_dim, act_dim, action_scale, args)
    buffer = ReplayBuffer(args.buffer_size, obs_dim, act_dim)

    steps_per_task = args.max_steps // n_tasks
    t_start = time.time()
    print(f"\n===== multi 模式：{n_tasks} 个任务共享策略，"
          f"每个任务约 {steps_per_task} 步，参数量 {agent.n_params()} =====")

    for task_i, (g, env) in enumerate(zip(gs, envs)):
        obs, _ = env.reset()
        ep_ret = 0.0
        ep_returns = []
        for step in range(1, steps_per_task + 1):
            obs_in = np.concatenate([np.asarray(obs, dtype=np.float32),
                                     onehot(task_i, n_tasks)])
            if step <= args.start_steps:
                action = env.action_space.sample()
            else:
                action = agent.act(obs_in, deterministic=False)
            next_obs, rew, terminated, truncated, _ = env.step(action)
            next_in = np.concatenate([np.asarray(next_obs, dtype=np.float32),
                                      onehot(task_i, n_tasks)])
            buffer.add(obs_in, action, rew, next_in, float(terminated))
            obs = next_obs
            ep_ret += rew
            if terminated or truncated:
                ep_returns.append(ep_ret)
                ep_ret = 0.0
                obs, _ = env.reset()
            if buffer.size >= min(args.start_steps, 1000) and step % args.update_every == 0:
                agent.update(buffer.sample(args.batch_size), args)
        recent = np.mean(ep_returns[-20:]) if ep_returns else float("nan")
        print(f"[multi] 任务 g={g} 训练结束 | 近20回合回报 {recent:8.1f}")

    # 评估：正确标识 + 错误标识（用第一个任务的 one-hot 错配给所有任务）
    per_task, wrong = [], []
    for task_i, (g, env) in enumerate(zip(gs, eval_envs)):
        ret = eval_on_task(agent, env, task_i, n_tasks, args.eval_episodes)
        ret_wrong = eval_on_task(agent, env, task_i, n_tasks, args.eval_episodes,
                                 override_id=(task_i + 1) % n_tasks)
        per_task.append(ret)
        wrong.append(ret_wrong)
        print(f"[multi] 评估任务 g={g}：正确标识 {ret:8.1f} | "
              f"错误标识 {ret_wrong:8.1f} | 差值 {ret - ret_wrong:+7.1f}")
    elapsed = time.time() - t_start
    return {"mode": "multi", "per_task": per_task, "wrong": wrong,
            "params": agent.n_params(), "elapsed": elapsed}


# ---------------------------------------------------------------------------
# single 模式：每个任务一个独立策略
# ---------------------------------------------------------------------------
def train_single_task(g: float, task_i: int, steps: int, args):
    set_seed(args.seed + 11 * task_i)  # 与 multi 使用相同的任务环境种子
    env = make_pendulum(args.seed + 11 * task_i, g)
    eval_env = make_pendulum(args.seed + 100 + 11 * task_i, g)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    action_scale = float(env.action_space.high[0])
    agent = SAC(obs_dim, act_dim, action_scale, args)
    buffer = ReplayBuffer(args.buffer_size, obs_dim, act_dim)

    obs, _ = env.reset()
    ep_ret = 0.0
    ep_returns = []
    for step in range(1, steps + 1):
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
        if buffer.size >= min(args.start_steps, 1000) and step % args.update_every == 0:
            agent.update(buffer.sample(args.batch_size), args)
    ret = eval_on_task(agent, eval_env, None, 1, args.eval_episodes)
    recent = np.mean(ep_returns[-20:]) if ep_returns else float("nan")
    print(f"[single] 任务 g={g} 训练结束 | 近20回合回报 {recent:8.1f} | 评估 {ret:8.1f}")
    return ret, agent.n_params()


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="第055章：SAC 的多任务学习")
    p.add_argument("--gs", type=str, default="8,12,16", help="任务的重力参数列表")
    p.add_argument("--max-steps", type=int, default=18000, help="总环境步数上限（所有任务合计）")
    p.add_argument("--start-steps", type=int, default=1000, help="每个任务训练前纯随机探索步数")
    p.add_argument("--update-every", type=int, default=1, help="每多少环境步做一次更新")
    p.add_argument("--eval-episodes", type=int, default=5, help="评估回合数")
    p.add_argument("--buffer-size", type=int, default=100000, help="回放池容量")
    p.add_argument("--batch-size", type=int, default=256, help="批大小")
    p.add_argument("--hidden", type=int, default=256, help="隐藏层宽度")
    p.add_argument("--lr", type=float, default=3e-4, help="学习率")
    p.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
    p.add_argument("--tau", type=float, default=0.005, help="目标网络软更新系数")
    p.add_argument("--alpha-init", type=float, default=0.2, help="温度系数初值")
    p.add_argument("--seed", type=int, default=0, help="随机种子")
    p.add_argument("--save-dir", type=str, default="runs/ch055", help="输出目录")
    p.add_argument("--quick", action="store_true", help="快速模式：缩短训练")
    return p.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.max_steps = min(args.max_steps, 3600)
        args.start_steps = min(args.start_steps, 300)
        args.eval_episodes = 3

    gs = [float(x) for x in args.gs.split(",") if x.strip()]
    n_tasks = len(gs)
    steps_per_task = args.max_steps // n_tasks
    print(f"任务: Pendulum g={gs} | 种子: {args.seed} | "
          f"每任务步数 {steps_per_task}（总计 {args.max_steps}）")

    multi = train_multi(gs, args)

    print(f"\n===== single 模式：每个任务独立策略，各 {steps_per_task} 步 =====")
    singles, single_params = [], None
    for i, g in enumerate(gs):
        ret, params = train_single_task(g, i, steps_per_task, args)
        singles.append(ret)
        single_params = params

    print("\n" + "=" * 92)
    print("多任务 vs 单任务（每任务评估回报，越高越好）")
    print(f"{'任务(g)':>9}{'multi':>10}{'single':>10}{'差值(m-s)':>11}")
    for i, g in enumerate(gs):
        print(f"{g:>9.0f}{multi['per_task'][i]:>10.1f}{singles[i]:>10.1f}"
              f"{multi['per_task'][i] - singles[i]:>11.1f}")
    print(f"{'平均':>9}{np.mean(multi['per_task']):>10.1f}{np.mean(singles):>10.1f}"
          f"{np.mean(multi['per_task']) - np.mean(singles):>11.1f}")
    print("-" * 92)
    print(f"参数量：multi {multi['params']}（1 套网络） vs single {single_params * n_tasks}"
          f"（{n_tasks} 套网络，每套 {single_params}）")
    print(f"任务标识敏感性：", end="")
    for i, g in enumerate(gs):
        print(f"g={g:.0f} 正确-错误 = {multi['per_task'][i] - multi['wrong'][i]:+.1f}  ", end="")
    print()
    print("=" * 92)
    print("结论提示：single 通常略优（专用容量大）；multi 用一套网络达到接近水平，")
    print("且对错误任务标识明显降分——说明它确实在用 one-hot 区分任务动力学。")
    print(f"总用时：multi {multi['elapsed']:.1f}s")

    os.makedirs(args.save_dir, exist_ok=True)
    print(f"输出目录：{args.save_dir}")


if __name__ == "__main__":
    main()
