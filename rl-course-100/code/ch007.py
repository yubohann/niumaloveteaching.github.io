"""
第007章 PPO的奖励归一化实现

本章用脚本内联实现的轻量环境 NoisyGridWorld 制造"奖励量级剧烈变化"的处境：
  - 5x5 网格，从左上角走到右下角，普通步奖励 −0.05，到达目标 +1
  - 每个回合开始时随机抽取一个奖励尺度 s ∈ [0.2, 5.0]，本回合所有奖励乘以 s
  - 于是回报目标的量级在不同回合之间相差 25 倍，价值网络很难同时拟合

对比三种奖励处理方式（其他一切相同）：
  - none  ：原始奖励直接进入 GAE 与价值目标
  - scale ：用"折扣回报的滑动标准差"归一化奖励（RND/大规模 RL 常用做法）
  - clip  ：奖励裁剪到 [−1, 1]（DQN-Atari 风格）

记录：成功率（近100回合）、原始终值回报、价值损失、梯度范数与首达 80% 成功率的步数。

运行：
    python code/ch007.py            # 3 种处理，各 8 万步，CPU 约 2-5 分钟
    python code/ch007.py --quick    # 快速跑通（约 20-40 秒）
    python code/ch007.py --modes none scale

预期：none 的价值损失被大尺度回合带偏、成功率波动大；
scale 把不同尺度拉回同一量级，成功率最高、价值损失最稳；
clip 有改善但会破坏"目标奖励与步惩罚的比值"，通常不如 scale。
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
from torch.distributions import Categorical


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    """固定随机种子，保证三种处理方式从同一条随机流出发。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_tensor(obs, device):
    return torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)


# ---------------------------------------------------------------------------
# 自定义环境：带随机奖励尺度的 GridWorld
# ---------------------------------------------------------------------------
class NoisyGridWorld(gym.Env):
    """5x5 网格，0=上 1=右 2=下 3=左；到达 (4,4) 获得 +1，步惩罚 −0.05。

    reset 时从 [0.2, 5.0] 均匀抽取 reward_scale，本回合所有奖励乘以它。
    fixed_scale 不为 None 时改用固定尺度（评估时用 1.0）。
    """

    metadata = {"render_modes": []}

    def __init__(self, size: int = 5, max_steps: int = 40,
                 fixed_scale=None, scale_low: float = 0.2, scale_high: float = 5.0):
        super().__init__()
        self.size = size
        self.max_steps = max_steps
        self.fixed_scale = fixed_scale
        self.scale_low = scale_low
        self.scale_high = scale_high
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(4)
        self._rng = np.random.default_rng(0)
        self.pos = np.zeros(2, dtype=np.int64)
        self.steps = 0
        self.reward_scale = 1.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.pos = np.array([0, 0], dtype=np.int64)
        self.steps = 0
        if self.fixed_scale is None:
            self.reward_scale = float(self._rng.uniform(self.scale_low, self.scale_high))
        else:
            self.reward_scale = float(self.fixed_scale)
        return self._obs(), {}

    def _obs(self):
        return (self.pos / (self.size - 1)).astype(np.float32)

    def step(self, action):
        # 确定性地移动；撞墙则原地不动
        deltas = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
        dy, dx = deltas[int(action)]
        self.pos[0] = int(np.clip(self.pos[0] + dy, 0, self.size - 1))
        self.pos[1] = int(np.clip(self.pos[1] + dx, 0, self.size - 1))
        self.steps += 1

        success = bool(self.pos[0] == self.size - 1 and self.pos[1] == self.size - 1)
        raw_r = 1.0 if success else -0.05
        r = raw_r * self.reward_scale

        terminated = success
        truncated = self.steps >= self.max_steps
        info = {"success": success, "raw_reward": raw_r, "reward_scale": self.reward_scale}
        return self._obs(), float(r), terminated, truncated, info


# ---------------------------------------------------------------------------
# 滑动统计量（Welford 算法）：跟踪折扣回报的标准差
# ---------------------------------------------------------------------------
class RunningStats:
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, x: float) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (x - self.mean)

    @property
    def std(self) -> float:
        if self.n < 2:
            return 1.0
        return float(np.sqrt(self.m2 / (self.n - 1)))


class RewardProcessor:
    """三种奖励处理方式的统一封装。

    none  : 原样返回
    scale : r / std(折扣回报)，std 来自最近若干回合的 Welford 统计；归一化后裁剪到 [−10, 10]
    clip  : r 裁剪到 [−1, 1]
    """

    def __init__(self, mode: str, warmup_episodes: int = 5):
        self.mode = mode
        self.warmup = warmup_episodes
        self.stats = RunningStats()

    def process(self, r: float, gamma: float = 0.99) -> float:
        if self.mode == "none":
            return r
        if self.mode == "clip":
            return float(np.clip(r, -1.0, 1.0))
        if self.mode == "scale":
            if self.stats.n < self.warmup:
                return r                       # 预热期：直接用原始奖励
            std = max(self.stats.std, 1e-3)
            return float(np.clip(r / std, -10.0, 10.0))
        raise ValueError(f"未知的奖励处理模式: {self.mode}")

    def finish_episode(self, discounted_return: float) -> None:
        if self.mode == "scale":
            self.stats.update(discounted_return)


# ---------------------------------------------------------------------------
# 网络
# ---------------------------------------------------------------------------
class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 64):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.actor = nn.Linear(hidden, act_dim)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor):
        h = self.body(obs)
        return self.actor(h), self.critic(h).squeeze(-1)

    @torch.no_grad()
    def act(self, obs: torch.Tensor):
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action).item(), value.item()

    @torch.no_grad()
    def value(self, obs: torch.Tensor) -> float:
        _, v = self.forward(obs)
        return v.item()


# ---------------------------------------------------------------------------
# GAE
# ---------------------------------------------------------------------------
def compute_gae(rewards, values, terminateds, last_value, gamma: float, lam: float):
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(T)):
        next_value = last_value if t == T - 1 else values[t + 1]
        non_terminal = 1.0 - terminateds[t]
        delta = rewards[t] + gamma * next_value * non_terminal - values[t]
        last_gae = delta + gamma * lam * non_terminal * last_gae
        advantages[t] = last_gae
    returns = advantages + np.asarray(values, dtype=np.float32)
    return advantages, returns


# ---------------------------------------------------------------------------
# 采样：奖励先经 RewardProcessor 处理，再进入 GAE
# ---------------------------------------------------------------------------
def collect_rollout(env, net, steps, obs_t, device, gamma, lam, processor):
    obs_buf, act_buf, logp_buf = [], [], []
    rew_buf, val_buf, term_buf = [], [], []
    ep_returns, ep_success = [], []

    ep_ret = 0.0
    ep_disc = 0.0      # 折扣回报（用于滑动统计）
    ep_gamma = 1.0
    success_flag = False
    obs_after = obs_t
    for _ in range(steps):
        a, logp, v = net.act(obs_t)
        next_obs, r, terminated, truncated, info = env.step(a)

        r_used = processor.process(r, gamma)

        obs_buf.append(obs_t.squeeze(0).cpu().numpy())
        act_buf.append(a)
        logp_buf.append(logp)
        rew_buf.append(r_used)
        val_buf.append(v)
        term_buf.append(float(terminated))

        ep_ret += r
        ep_disc += ep_gamma * r
        ep_gamma *= gamma
        success_flag = success_flag or bool(info.get("success", False))

        obs_t = to_tensor(next_obs, device)
        obs_after = obs_t

        if terminated or truncated:
            ep_returns.append(ep_ret)
            ep_success.append(1.0 if success_flag else 0.0)
            processor.finish_episode(ep_disc)
            ep_ret, ep_disc, ep_gamma, success_flag = 0.0, 0.0, 1.0, False
            reset_obs, _ = env.reset()
            obs_t = to_tensor(reset_obs, device)

    last_value = net.value(obs_after)
    advantages, returns = compute_gae(rew_buf, val_buf, term_buf, last_value, gamma, lam)
    batch = {
        "obs": torch.as_tensor(np.asarray(obs_buf), dtype=torch.float32, device=device),
        "act": torch.as_tensor(np.asarray(act_buf), dtype=torch.long, device=device),
        "logp_old": torch.as_tensor(np.asarray(logp_buf), dtype=torch.float32, device=device),
        "adv": torch.as_tensor(advantages, dtype=torch.float32, device=device),
        "ret": torch.as_tensor(returns, dtype=torch.float32, device=device),
    }
    return batch, ep_returns, ep_success, obs_t


# ---------------------------------------------------------------------------
# PPO 更新：额外记录价值损失与梯度范数（奖励尺度失衡会先在这两项上暴露）
# ---------------------------------------------------------------------------
def ppo_update(net, optimizer, batch, args):
    obs, act = batch["obs"], batch["act"]
    logp_old, adv, ret = batch["logp_old"], batch["adv"], batch["ret"]
    N = obs.shape[0]
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    stats = {"approx_kl": 0.0, "clip_frac": 0.0, "entropy": 0.0,
             "value_loss": 0.0, "grad_norm": 0.0, "n_updates": 0}
    for _ in range(args.epochs):
        idx = torch.randperm(N, device=obs.device)
        for start in range(0, N, args.batch_size):
            mb = idx[start:start + args.batch_size]
            logits, value = net(obs[mb])
            dist = Categorical(logits=logits)
            logp = dist.log_prob(act[mb])

            ratio = torch.exp(logp - logp_old[mb])
            surr1 = ratio * adv[mb]
            surr2 = torch.clamp(ratio, 1.0 - args.clip, 1.0 + args.clip) * adv[mb]
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(value, ret[mb])
            entropy = dist.entropy().mean()
            loss = policy_loss + args.vf_coef * value_loss - args.ent_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            gn = nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            optimizer.step()

            with torch.no_grad():
                stats["approx_kl"] += (logp_old[mb] - logp).mean().item()
                stats["clip_frac"] += ((ratio - 1.0).abs() > args.clip).float().mean().item()
                stats["entropy"] += entropy.item()
                stats["value_loss"] += value_loss.item()
                stats["grad_norm"] += float(gn)
                stats["n_updates"] += 1

    for k in stats:
        if k != "n_updates":
            stats[k] /= max(stats["n_updates"], 1)
    return stats


# ---------------------------------------------------------------------------
# 评估：固定尺度 1.0，统计贪婪策略成功率
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(env, net, episodes: int, device):
    returns, success = [], []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_ret = 0.0
        ok = False
        while not done:
            logits, _ = net(to_tensor(obs, device))
            action = int(torch.argmax(logits, dim=-1).item())
            obs, r, terminated, truncated, info = env.step(action)
            ep_ret += r
            ok = ok or bool(info.get("success", False))
            done = terminated or truncated
        returns.append(ep_ret)
        success.append(1.0 if ok else 0.0)
    return returns, float(np.mean(success))


# ---------------------------------------------------------------------------
# 训练一种奖励处理方式
# ---------------------------------------------------------------------------
def train_one(mode: str, args):
    set_seed(args.seed)
    device = torch.device("cpu")
    env = NoisyGridWorld()
    obs, _ = env.reset(seed=args.seed)

    net = ActorCritic(env.observation_space.shape[0], env.action_space.n)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    processor = RewardProcessor(mode)

    all_returns, all_success = [], []
    vlosses, grads = [], []
    total_steps, n_updates, obs_t = 0, 0, to_tensor(obs, device)
    reached_at = None
    t0 = time.time()

    print(f"\n[开始] 奖励处理 = {mode}")
    while total_steps < args.max_steps:
        batch, ep_returns, ep_success, obs_t = collect_rollout(
            env, net, args.steps_per_update, obs_t, device, args.gamma, args.lam, processor)
        total_steps += args.steps_per_update
        n_updates += 1
        all_returns.extend(ep_returns)
        all_success.extend(ep_success)

        stats = ppo_update(net, optimizer, batch, args)
        vlosses.append(stats["value_loss"])
        grads.append(stats["grad_norm"])

        if reached_at is None and len(all_success) >= 100 and np.mean(all_success[-100:]) >= 0.8:
            reached_at = total_steps

        if n_updates % 5 == 0 and len(all_success) >= 20:
            print(f"  [{mode:5}] 步数 {total_steps:7d} | 回合 {len(all_success):4d} | "
                  f"近20成功 {np.mean(all_success[-20:]):.2f} | 回报 {np.mean(all_returns[-20:]):8.2f} | "
                  f"价值损失 {stats['value_loss']:8.3f} | 梯度 {stats['grad_norm']:.3f}")

    eval_env = NoisyGridWorld(fixed_scale=1.0)
    eval_returns, eval_success = evaluate(eval_env, net, args.eval_episodes, device)
    tail_s = all_success[-100:]
    return {
        "mode": mode,
        "steps": total_steps,
        "reached_at": reached_at,
        "success100": float(np.mean(tail_s)),
        "last20_return": float(np.mean(all_returns[-20:])),
        "eval_success": eval_success,
        "eval_return": float(np.mean(eval_returns)),
        "mean_vloss": float(np.mean(vlosses)),
        "mean_grad": float(np.mean(grads)),
        "seconds": time.time() - t0,
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="第007章：PPO 奖励归一化实现")
    p.add_argument("--modes", type=str, nargs="+", default=["none", "scale", "clip"],
                   choices=["none", "scale", "clip"], help="奖励处理方式")
    p.add_argument("--max-steps", type=int, default=80000, help="每种方式的环境步数预算")
    p.add_argument("--steps-per-update", type=int, default=2048, help="每次更新前采样步数")
    p.add_argument("--epochs", type=int, default=10, help="每批数据的更新轮数")
    p.add_argument("--batch-size", type=int, default=64, help="minibatch 大小")
    p.add_argument("--lr", type=float, default=3e-4, help="学习率")
    p.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
    p.add_argument("--lam", type=float, default=0.95, help="GAE 的 lambda")
    p.add_argument("--clip", type=float, default=0.2, help="PPO 裁剪范围")
    p.add_argument("--ent-coef", type=float, default=0.01, help="熵正则系数")
    p.add_argument("--vf-coef", type=float, default=0.5, help="价值损失系数")
    p.add_argument("--eval-episodes", type=int, default=20, help="评估回合数")
    p.add_argument("--seed", type=int, default=0, help="随机种子")
    p.add_argument("--save-dir", type=str, default="runs/ch007", help="结果保存目录")
    p.add_argument("--quick", action="store_true", help="快速模式：3 种方式、各 1.2 万步")
    return p.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.max_steps = 12000
        args.steps_per_update = 512
        args.epochs = 4

    print("=" * 96)
    print(f"奖励归一化对比 | 处理方式 {args.modes} | 每个 {args.max_steps} 步 | 种子 {args.seed}")
    print("环境：5x5 GridWorld，每回合奖励尺度 s ~ U[0.2, 5.0]，奖励量级相差 25 倍")
    print("=" * 96)

    results = []
    for mode in args.modes:
        res = train_one(mode, args)
        results.append(res)
        print(f"[完成] {mode}: 近100成功 {res['success100']:.2f} | 评估成功 {res['eval_success']:.2f} | "
              f"平均价值损失 {res['mean_vloss']:.3f} | 用时 {res['seconds']:.1f}s")

    print("\n" + "=" * 96)
    print("对比总结（同种子、同步数预算、同一环境分布）")
    print("-" * 96)
    print(f"{'处理':>7} | {'首达80%成功步数':>14} | {'近100成功率':>11} | {'评估成功率':>10} | "
          f"{'评估回报(尺度=1)':>15} | {'平均价值损失':>12} | {'平均梯度范数':>12} | {'用时(s)':>8}")
    for r in results:
        reached = str(r["reached_at"]) if r["reached_at"] is not None else "未达到"
        print(f"{r['mode']:>7} | {reached:>14} | {r['success100']:>11.2f} | {r['eval_success']:>10.2f} | "
              f"{r['eval_return']:>15.2f} | {r['mean_vloss']:>12.2f} | {r['mean_grad']:>12.3f} | "
              f"{r['seconds']:>8.1f}")

    print("\n观察提示：")
    print("  - none 的价值损失会随奖励尺度波动，梯度也更大；scale 把两者压回稳定区间；")
    print("  - clip 会削掉大尺度回合的奖励差异，成功率通常介于两者之间；")
    print("  - 评估统一用尺度 1.0，保证三种处理在同一把尺子下比较。")

    os.makedirs(args.save_dir, exist_ok=True)
    path = os.path.join(args.save_dir, "reward_norm.txt")
    with open(path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(f"mode={r['mode']} reached80={r['reached_at']} success100={r['success100']:.3f} "
                    f"eval_success={r['eval_success']:.3f} eval_return={r['eval_return']:.3f} "
                    f"vloss={r['mean_vloss']:.4f} grad={r['mean_grad']:.4f}\n")
    print(f"\n结果已写入 {path}")


if __name__ == "__main__":
    main()
