"""
第058章 SAC的鲁棒性测试

训练两个 Pendulum 策略：
  - clean : 干净观测的标准训练
  - noisy : 训练时对观测加入 sigma=0.05 高斯噪声（噪声增强训练）

然后在同一套扰动电池下评估两者的表现（扰动只发生在评估时）：
  - 观测噪声 sigma ∈ {0.02, 0.05, 0.10}
  - 动作噪声 sigma ∈ {0.10, 0.30}
  - 动力学偏移：重力 g ∈ {8, 12}（训练时 g=10）
  - 动作延迟：执行前 k ∈ {2, 4} 步的旧动作

记录每种条件下的平均回报与相对干净条件的"保持率"（干净回报 ÷ 条件回报；
Pendulum 回报恒为负，用幅度比的口径：1 表示无损失，越小损失越大），
比较两种训练的鲁棒性差距。

运行：
    python code/ch058.py           # 训练 2 个策略 + 扰动电池（CPU 约 2-4 分钟）
    python code/ch058.py --quick   # 快速跑通（约 40 秒）

预期：观测噪声档位上两个策略的保持率几乎相同，噪声增强没有带来可辨的
差异；出现差异的是重力偏移 g=12 与动作延迟 4 步档位（参考实跑中 noisy
的保持率分别高出约 0.17 与 0.06）。每种条件只评估 4 回合，读数噪声很大，
方向性结论需要多种子、更多回合复核。以实跑为准。
"""
import argparse
import os
import random
import time
from collections import deque

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


def make_pendulum(seed: int, g: float = 10.0):
    env = gym.make("Pendulum-v1")
    env.unwrapped.g = g
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


def train_agent(train_obs_noise: float, args, tag: str):
    """训练一个策略；train_obs_noise > 0 时对观测做噪声增强。"""
    set_seed(args.seed)  # 两个策略同种子，控制变量
    env = make_pendulum(args.seed, 10.0)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    action_scale = float(env.action_space.high[0])
    agent = SAC(obs_dim, act_dim, action_scale, args)
    buffer = ReplayBuffer(args.buffer_size, obs_dim, act_dim)

    obs, _ = env.reset()
    ep_ret = 0.0
    ep_returns = []
    t_start = time.time()
    print(f"\n===== 训练 [{tag}]：观测噪声 sigma={train_obs_noise} =====")

    for step in range(1, args.max_steps + 1):
        if step <= args.start_steps:
            action = env.action_space.sample()
            obs_in = obs
        else:
            # 训练噪声增强：策略与回放池都看到带噪观测
            obs_in = obs + np.random.normal(0.0, train_obs_noise, size=obs.shape).astype(np.float32) \
                if train_obs_noise > 0 else obs
            action = agent.act(obs_in, deterministic=False)
        next_obs, rew, terminated, truncated, _ = env.step(action)
        next_in = next_obs + np.random.normal(0.0, train_obs_noise, size=next_obs.shape).astype(np.float32) \
            if train_obs_noise > 0 else next_obs
        buffer.add(obs_in, action, rew, next_in, float(terminated))
        obs = next_obs
        ep_ret += rew
        if terminated or truncated:
            ep_returns.append(ep_ret)
            ep_ret = 0.0
            obs, _ = env.reset()

        if step % args.update_every == 0 and buffer.size >= min(args.start_steps, 1000):
            agent.update(buffer.sample(args.batch_size), args)

        if step % args.eval_every == 0:
            eval_ret = evaluate_robust(agent, g=10.0, obs_noise=0.0, act_noise=0.0,
                                       delay=0, seed=args.seed + 100,
                                       episodes=args.eval_episodes)
            recent20 = np.mean(ep_returns[-20:]) if ep_returns else float("nan")
            print(f"[{tag}] 步数 {step:6d} | 近20回合回报 {recent20:8.1f} | 干净评估 {eval_ret:8.1f}")

    print(f"[{tag}] 训练完成，用时 {time.time() - t_start:.1f}s")
    return agent


def evaluate_robust(agent: SAC, g: float, obs_noise: float, act_noise: float,
                    delay: int, seed: int, episodes: int) -> float:
    """在指定扰动组合下评估：观测噪声、动作噪声、重力偏移、动作延迟。"""
    env = make_pendulum(seed, g)
    returns = []
    for _ in range(episodes):
        obs, _ = env.reset()
        # 延迟缓冲：保留最近 delay+1 个动作，执行最旧的那个实现 delay 步延迟
        hist_len = delay + 1 if delay > 0 else 1
        action_hist = deque([np.zeros(env.action_space.shape, dtype=np.float32)] * hist_len,
                            maxlen=hist_len)
        done = False
        ep_ret = 0.0
        while not done:
            obs_in = obs + np.random.normal(0.0, obs_noise, size=obs.shape).astype(np.float32) \
                if obs_noise > 0 else obs
            a = agent.act(obs_in, deterministic=True)
            if act_noise > 0:
                a = a + np.random.normal(0.0, act_noise, size=a.shape).astype(np.float32)
            a = np.clip(a, -2.0, 2.0)
            if delay > 0:
                action_hist.append(a)
                executed = action_hist[0]  # 执行 delay 步之前的旧动作
            else:
                executed = a
            obs, rew, terminated, truncated, _ = env.step(executed)
            ep_ret += rew
            done = terminated or truncated
        returns.append(ep_ret)
    return float(np.mean(returns))


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="第058章：SAC 的鲁棒性测试")
    p.add_argument("--max-steps", type=int, default=12000, help="每个策略的训练步数")
    p.add_argument("--start-steps", type=int, default=1000, help="训练前纯随机探索步数")
    p.add_argument("--train-noise", type=float, default=0.05, help="噪声增强训练的观测噪声 sigma")
    p.add_argument("--update-every", type=int, default=1, help="每多少环境步做一次更新")
    p.add_argument("--eval-every", type=int, default=3000, help="训练期间的干净评估间隔")
    p.add_argument("--eval-episodes", type=int, default=4, help="每种扰动的评估回合数")
    p.add_argument("--buffer-size", type=int, default=100000, help="回放池容量")
    p.add_argument("--batch-size", type=int, default=256, help="批大小")
    p.add_argument("--hidden", type=int, default=256, help="隐藏层宽度")
    p.add_argument("--lr", type=float, default=3e-4, help="学习率")
    p.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
    p.add_argument("--tau", type=float, default=0.005, help="目标网络软更新系数")
    p.add_argument("--alpha-init", type=float, default=0.2, help="温度系数初值")
    p.add_argument("--seed", type=int, default=0, help="随机种子")
    p.add_argument("--save-dir", type=str, default="runs/ch058", help="输出目录")
    p.add_argument("--quick", action="store_true", help="快速模式：缩短训练")
    return p.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.max_steps = min(args.max_steps, 3000)
        args.eval_every = 1500
        args.start_steps = min(args.start_steps, 300)
        args.eval_episodes = 2

    print(f"种子: {args.seed} | 训练步数 {args.max_steps} | "
          f"噪声增强 sigma={args.train_noise} | 评估回合数 {args.eval_episodes}")

    agents = {
        "clean": train_agent(0.0, args, "clean"),
        "noisy": train_agent(args.train_noise, args, "noisy"),
    }

    # 扰动电池：每个条件 (名称, g, obs_noise, act_noise, delay)
    conditions = [("干净（基线）", 10.0, 0.0, 0.0, 0),
                  ("观测噪声 0.02", 10.0, 0.02, 0.0, 0),
                  ("观测噪声 0.05", 10.0, 0.05, 0.0, 0),
                  ("观测噪声 0.10", 10.0, 0.10, 0.0, 0),
                  ("动作噪声 0.10", 10.0, 0.0, 0.10, 0),
                  ("动作噪声 0.30", 10.0, 0.0, 0.30, 0),
                  ("重力偏移 g=8", 8.0, 0.0, 0.0, 0),
                  ("重力偏移 g=12", 12.0, 0.0, 0.0, 0),
                  ("动作延迟 2 步", 10.0, 0.0, 0.0, 2),
                  ("动作延迟 4 步", 10.0, 0.0, 0.0, 4)]

    print("\n" + "=" * 92)
    print("鲁棒性测试（每种条件评估一次；保持率 = 干净回报 / 条件回报，回报恒为负，越小损失越大）")
    print(f"{'条件':<16}{'clean 回报':>12}{'clean 保持率':>13}"
          f"{'noisy 回报':>12}{'noisy 保持率':>13}{'差值(n-c)':>11}")
    base = {}
    results = []
    for i, (name, g, on, an, dl) in enumerate(conditions):
        row = {}
        for tag, agent in agents.items():
            ret = evaluate_robust(agent, g=g, obs_noise=on, act_noise=an, delay=dl,
                                  seed=args.seed + 200 + i, episodes=args.eval_episodes)
            row[tag] = ret
        if name.startswith("干净"):
            base = row
        results.append((name, row))
    for name, row in results:
        # 回报恒为负：用"干净 ÷ 条件"（幅度比）并在负方向做除零保护；
        # 若写成 row/max(base, 1e-9)，负分母会被顶成 1e-9，比例变成天文数字
        r_clean = base["clean"] / min(row["clean"], -1e-9) if base else float("nan")
        r_noisy = base["noisy"] / min(row["noisy"], -1e-9) if base else float("nan")
        print(f"{name:<16}{row['clean']:>12.1f}{r_clean:>13.2f}"
              f"{row['noisy']:>12.1f}{r_noisy:>13.2f}{row['noisy'] - row['clean']:>11.1f}")
    print("=" * 92)
    avg_clean = np.mean([base["clean"] / min(r["clean"], -1e-9) for _, r in results[1:]])
    avg_noisy = np.mean([base["noisy"] / min(r["noisy"], -1e-9) for _, r in results[1:]])
    print(f"平均保持率（不含干净基线）：clean {avg_clean:.2f} | noisy {avg_noisy:.2f}")
    print("结论提示：参考实跑（种子 0，默认配置）中，观测噪声档位上两个")
    print("策略的保持率几乎相同，噪声增强没有带来可辨的差异；出现差异的是重力")
    print("偏移 g=12 与动作延迟 4 步档位。单次条件评估的读数噪声很大，方向性")
    print("结论需要多种子、更多回合复核；四类扰动分开测量，才能说清强弱在哪。")

    os.makedirs(args.save_dir, exist_ok=True)
    print(f"输出目录：{args.save_dir}")


if __name__ == "__main__":
    main()
