"""
第043章 SAC的样本复用与经验回放

SAC 是 off-policy 算法：一条经验存进回放池后可以被反复抽样、反复训练。
本章用 Pendulum-v1 做三组对照，量化"经验回放"带来的样本效率：
  - replay : 大容量回放池（默认 10 万），全场随机采样，经验被复用数百次
  - small  : 小容量回放池（默认 2000），只看最近的数据，复用次数少但样本新鲜
  - noreplay: 伪在线模式——只保留最近一个 batch 的数据，每条经验只用一次

对每组对照记录：评估回报、到达回报阈值所需步数、数据保留率（当前驻留池内的
样本数 / 总共收集的样本数）、采样批次的平均"年龄"（存入后隔了多少步才被抽到）。

运行：
    python code/ch043.py           # 三种模式依次训练（CPU 约 4-6 分钟）
    python code/ch043.py --quick   # 快速跑通（约 1 分钟）

预期：replay 的样本效率最高；noreplay 明显最慢；small 早期进步快、后期受
遗忘拖累。年龄指标会显示大池子里的样本明显"更旧"。
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


def make_env(env_id: str, seed: int):
    env = gym.make(env_id)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env


# ---------------------------------------------------------------------------
# 经验回放池（带"存入时刻"记录，用于统计样本年龄）
# ---------------------------------------------------------------------------
class ReplayBuffer:
    """定长回放池；额外记录每条经验存入时的环境步数，便于统计数据新鲜度。"""

    def __init__(self, capacity: int, obs_dim: int, act_dim: int):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rew = np.zeros((capacity, 1), dtype=np.float32)
        self.done = np.zeros((capacity, 1), dtype=np.float32)
        self.stamp = np.zeros(capacity, dtype=np.int64)  # 存入时的环境步数
        self.ptr = 0
        self.size = 0

    def add(self, obs, act, rew, next_obs, done, step: int) -> None:
        i = self.ptr
        self.obs[i] = obs
        self.act[i] = act
        self.rew[i] = rew
        self.next_obs[i] = next_obs
        self.done[i] = done  # 只记录真实终止
        self.stamp[i] = step
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, current_step: int = 0) -> dict:
        """均匀采样；current_step 用于计算被采样数据的平均年龄。"""
        idx = np.random.randint(0, self.size, size=batch_size)
        age = float(np.mean(current_step - self.stamp[idx])) if current_step else 0.0
        return {
            "obs": torch.as_tensor(self.obs[idx]),
            "act": torch.as_tensor(self.act[idx]),
            "rew": torch.as_tensor(self.rew[idx]),
            "next_obs": torch.as_tensor(self.next_obs[idx]),
            "done": torch.as_tensor(self.done[idx]),
            "age": age,
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
    """tanh 压缩高斯策略（与第 042 章相同的实现）。"""

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

        return {"q_loss": q_loss.item(), "policy_loss": policy_loss.item(),
                "alpha": float(self.alpha.item())}


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
# 训练一种回放模式
# ---------------------------------------------------------------------------
def train_mode(mode: str, args):
    """mode: replay（大池）/ small（小池）/ noreplay（只用最新 batch）。"""
    set_seed(args.seed)  # 三种模式同种子，控制变量
    if mode == "replay":
        capacity = args.buffer_size
    elif mode == "small":
        capacity = args.small_capacity
    else:
        capacity = args.batch_size  # 伪在线：池子只装得下一个 batch

    env = make_env(args.env, args.seed)
    eval_env = make_env(args.env, args.seed + 100)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    action_scale = float(env.action_space.high[0])
    agent = SAC(obs_dim, act_dim, action_scale, args)
    buffer = ReplayBuffer(capacity, obs_dim, act_dim)

    min_size = args.start_steps if mode != "noreplay" else args.batch_size
    obs, _ = env.reset(seed=args.seed)
    ep_ret = 0.0
    ep_returns = []
    history = []  # (步数, 评估回报, 平均年龄)
    ages = []
    n_updates = 0
    collected = 0
    solved_at = None
    t_start = time.time()
    print(f"\n===== 模式 {mode}：回放池容量 {capacity}，开始训练 =====")

    for step in range(1, args.max_steps + 1):
        if step <= args.start_steps:
            action = env.action_space.sample()
        else:
            action = agent.act(obs, deterministic=False)
        next_obs, rew, terminated, truncated, _ = env.step(action)
        buffer.add(obs, action, rew, next_obs, float(terminated), step)
        collected += 1
        obs = next_obs
        ep_ret += rew

        if terminated or truncated:
            ep_returns.append(ep_ret)
            ep_ret = 0.0
            obs, _ = env.reset()

        if step % args.update_every == 0 and buffer.size >= min_size:
            batch = buffer.sample(args.batch_size, step)
            agent.update(batch, args)
            n_updates += 1
            ages.append(batch["age"])

        if step % args.eval_every == 0:
            eval_ret = evaluate(agent, eval_env, args.eval_episodes)
            mean_age = float(np.mean(ages[-200:])) if ages else 0.0
            history.append((step, eval_ret, mean_age))
            recent20 = np.mean(ep_returns[-20:]) if ep_returns else float("nan")
            retention = buffer.size / max(collected, 1)
            print(f"[{mode}] 步数 {step:6d} | 近20回合回报 {recent20:8.1f} | "
                  f"评估 {eval_ret:8.1f} | 保留率 {retention:5.3f} | 数据年龄 {mean_age:7.1f}")
            if solved_at is None and eval_ret >= args.target_return:
                solved_at = step
                print(f"[{mode}] 首次达到评估回报 {args.target_return}（步数 {step}）")

    elapsed = time.time() - t_start
    retention = buffer.size / max(collected, 1)
    final_eval = evaluate(agent, eval_env, args.eval_episodes)
    best_eval = max([h[1] for h in history] + [final_eval])
    mean_age = float(np.mean(ages)) if ages else 0.0
    print(f"[{mode}] 结束：最好评估 {best_eval:.1f} | 最终评估 {final_eval:.1f} | "
          f"保留率 {retention:.3f} | 全程平均年龄 {mean_age:.1f} | 用时 {elapsed:.1f}s")
    return {
        "mode": mode, "capacity": capacity, "best_eval": best_eval,
        "final_eval": final_eval, "solved_at": solved_at, "retention": retention,
        "mean_age": mean_age, "n_updates": n_updates, "elapsed": elapsed,
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="第043章：SAC 的样本复用与经验回放")
    p.add_argument("--env", type=str, default="Pendulum-v1", help="环境 ID")
    p.add_argument("--modes", type=str, default="replay,noreplay,small",
                   help="逗号分隔：replay / small / noreplay")
    p.add_argument("--max-steps", type=int, default=20000, help="每个模式的环境步数上限")
    p.add_argument("--start-steps", type=int, default=1000, help="训练前纯随机探索步数")
    p.add_argument("--update-every", type=int, default=1, help="每多少环境步做一次更新")
    p.add_argument("--eval-every", type=int, default=4000, help="每多少步评估一次")
    p.add_argument("--eval-episodes", type=int, default=5, help="评估回合数")
    p.add_argument("--buffer-size", type=int, default=100000, help="replay 模式回放池容量")
    p.add_argument("--small-capacity", type=int, default=2000, help="small 模式回放池容量")
    p.add_argument("--batch-size", type=int, default=256, help="批大小")
    p.add_argument("--hidden", type=int, default=256, help="隐藏层宽度")
    p.add_argument("--lr", type=float, default=3e-4, help="学习率")
    p.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
    p.add_argument("--tau", type=float, default=0.005, help="目标网络软更新系数")
    p.add_argument("--alpha-init", type=float, default=0.2, help="温度系数初值")
    p.add_argument("--target-return", type=float, default=-400.0,
                   help="评估回报达到该值视为解决（用于统计所需步数）")
    p.add_argument("--seed", type=int, default=0, help="随机种子")
    p.add_argument("--save-dir", type=str, default="runs/ch043", help="输出目录")
    p.add_argument("--quick", action="store_true", help="快速模式：缩短训练")
    return p.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.max_steps = min(args.max_steps, 4000)
        args.eval_every = 1000
        args.start_steps = min(args.start_steps, 400)

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    print(f"环境: {args.env} | 种子: {args.seed} | 对照模式: {modes}")
    print(f"配置: max_steps={args.max_steps}, batch={args.batch_size}, "
          f"大池={args.buffer_size}, 小池={args.small_capacity}")

    results = [train_mode(m, args) for m in modes]

    print("\n" + "=" * 92)
    print("样本复用对比（达到阈值的步数越小越好；保留率 = 驻留池内样本 / 收集样本）")
    print(f"{'模式':<10}{'池容量':>9}{'最好评估':>10}{'最终评估':>10}"
          f"{'达标步数':>10}{'保留率':>10}{'平均年龄':>10}")
    for r in results:
        solved = r["solved_at"] if r["solved_at"] is not None else -1
        print(f"{r['mode']:<10}{r['capacity']:>9}{r['best_eval']:>10.1f}"
              f"{r['final_eval']:>10.1f}{solved:>10}{r['retention']:>10.3f}"
              f"{r['mean_age']:>10.1f}")
    print("=" * 92)
    print("结论提示：保留率越高，早期经验越不容易被丢弃；平均年龄越大，说明样本")
    print("来自更久以前。noreplay 只保留最新一个 batch，样本高度相关、年龄最小，")
    print("但每次都把历史经验扔光，典型表现是训练更慢、曲线噪声更大。")

    os.makedirs(args.save_dir, exist_ok=True)
    print(f"输出目录：{args.save_dir}")


if __name__ == "__main__":
    main()
