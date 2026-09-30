"""
第012章 PPO的离散动作空间的分类策略

离散策略的进阶问题不是"选 logits 还是 probs"，而是"有些动作在某些状态下非法"。
本章自建带障碍的 MaskedMaze 环境：走进墙里或障碍格的动作是无效动作。
对比两种策略处理（同种子、同步数预算）：
  - nomask：普通 Categorical，无效动作照采样，但执行时原地不动、浪费一步
  - mask  ：非法动作的 logits 置为 −1e8（无效动作掩码，invalid action masking），
            采样与更新时的对数概率都基于掩码后的分布

记录：成功率（近100回合）、首达 90% 成功步数、无效动作比例、
熵（有效动作上的分布）、KL 与评估成功率。

运行：
    python code/ch012.py             # 2 种处理，各 5 万步，CPU 约 1-3 分钟
    python code/ch012.py --quick     # 快速跑通（约 15-30 秒）
    python code/ch012.py --arms mask

预期：mask 版通常在 1-2 万步内把成功率推到 90% 以上、无效动作比例为 0；
nomask 版更慢，且平均无效动作比例收敛到 5%～15% 后很难继续下降。
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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_tensor(obs, device):
    return torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)


# ---------------------------------------------------------------------------
# 自定义环境：带障碍的网格 + 无效动作
# ---------------------------------------------------------------------------
class MaskedMaze(gym.Env):
    """size×size 网格，从 (0,0) 走到 (size−1, size−1)。

    动作：0=上 1=右 2=下 3=左。走出边界或撞上障碍的动作是"无效动作"：
    原地不动、仍消耗一步。观测为归一化坐标 [row, col]/(size−1)。
    """

    def __init__(self, size: int = 6, max_steps: int = 60):
        super().__init__()
        self.size = size
        self.max_steps = max_steps
        self.obstacles = {(1, 1), (2, 2), (3, 1), (1, 4), (4, 2), (2, 4)}
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(4)
        self.pos = None
        self.steps = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.pos = np.array([0, 0], dtype=np.int64)
        self.steps = 0
        return self._obs(), {}

    def _obs(self):
        return (self.pos / (self.size - 1)).astype(np.float32)

    def action_mask(self):
        """返回长度 4 的 0/1 掩码（1=可执行）。掩码只依赖当前状态。"""
        deltas = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        mask = np.zeros(4, dtype=np.float32)
        for a, (dr, dc) in enumerate(deltas):
            nr, nc = self.pos[0] + dr, self.pos[1] + dc
            in_bounds = 0 <= nr < self.size and 0 <= nc < self.size
            blocked = (nr, nc) in self.obstacles
            mask[a] = 1.0 if (in_bounds and not blocked) else 0.0
        return mask

    def step(self, action):
        deltas = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        mask = self.action_mask()
        valid = bool(mask[int(action)] > 0.5)
        if valid:
            dr, dc = deltas[int(action)]
            self.pos = self.pos + np.array([dr, dc], dtype=np.int64)
        self.steps += 1

        success = bool(self.pos[0] == self.size - 1 and self.pos[1] == self.size - 1)
        r = 1.0 if success else -0.01
        terminated = success
        truncated = self.steps >= self.max_steps
        info = {"success": success, "invalid_action": (not valid)}
        return self._obs(), float(r), terminated, truncated, info


def mask_from_obs(obs: np.ndarray, size: int, obstacles) -> np.ndarray:
    """由观测（归一化坐标）反推动作掩码，便于在训练更新时重建。"""
    pos = np.round(np.asarray(obs, dtype=np.float64) * (size - 1)).astype(np.int64)
    deltas = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    mask = np.zeros(4, dtype=np.float32)
    for a, (dr, dc) in enumerate(deltas):
        nr, nc = pos[0] + dr, pos[1] + dc
        in_bounds = 0 <= nr < size and 0 <= nc < size
        blocked = (int(nr), int(nc)) in obstacles
        mask[a] = 1.0 if (in_bounds and not blocked) else 0.0
    return mask


def apply_mask(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """把非法动作的 logits 置为 −1e8（用大负数代替 −inf，避免 NaN）。"""
    return logits.masked_fill(mask < 0.5, -1e8)


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
    def act(self, obs: torch.Tensor, mask: torch.Tensor):
        """mask 形状 (1, act_dim)；masked 版用掩码后的分布采样。"""
        logits, value = self.forward(obs)
        dist = Categorical(logits=apply_mask(logits, mask))
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
# 采样：无论哪种策略，执行前都把无效动作记下来统计
# ---------------------------------------------------------------------------
def collect_rollout(env, net, steps, obs_t, device, gamma, lam, use_mask: bool):
    obs_buf, act_buf, logp_buf, mask_buf = [], [], [], []
    rew_buf, val_buf, term_buf = [], [], []
    ep_returns, ep_success, invalid_count, total_count = [], [], 0, 0

    ep_ret = 0.0
    success_flag = False
    obs_after = obs_t
    for _ in range(steps):
        mask_i = env.action_mask()
        mask_t = torch.as_tensor(mask_i, dtype=torch.float32, device=device).unsqueeze(0)
        if use_mask:
            a, logp, v = net.act(obs_t, mask_t)
        else:
            logits, value = net(obs_t)
            dist = Categorical(logits=logits)
            action = dist.sample()
            a, logp, v = action.item(), dist.log_prob(action).item(), value.item()

        next_obs, r, terminated, truncated, info = env.step(a)

        obs_buf.append(obs_t.squeeze(0).cpu().numpy())
        act_buf.append(a)
        logp_buf.append(logp)
        mask_buf.append(mask_i)
        rew_buf.append(r)
        val_buf.append(v)
        term_buf.append(float(terminated))

        ep_ret += r
        success_flag = success_flag or bool(info["success"])
        invalid_count += int(info["invalid_action"])
        total_count += 1
        obs_t = to_tensor(next_obs, device)
        obs_after = obs_t

        if terminated or truncated:
            ep_returns.append(ep_ret)
            ep_success.append(1.0 if success_flag else 0.0)
            ep_ret, success_flag = 0.0, False
            reset_obs, _ = env.reset()
            obs_t = to_tensor(reset_obs, device)

    last_value = net.value(obs_after)
    advantages, returns = compute_gae(rew_buf, val_buf, term_buf, last_value, gamma, lam)
    batch = {
        "obs": torch.as_tensor(np.asarray(obs_buf), dtype=torch.float32, device=device),
        "act": torch.as_tensor(np.asarray(act_buf), dtype=torch.long, device=device),
        "logp_old": torch.as_tensor(np.asarray(logp_buf), dtype=torch.float32, device=device),
        "mask": torch.as_tensor(np.asarray(mask_buf), dtype=torch.float32, device=device),
        "adv": torch.as_tensor(advantages, dtype=torch.float32, device=device),
        "ret": torch.as_tensor(returns, dtype=torch.float32, device=device),
    }
    invalid_rate = invalid_count / max(total_count, 1)
    return batch, ep_returns, ep_success, invalid_rate, obs_t


# ---------------------------------------------------------------------------
# PPO 更新：masked 版在重算 logp 时应用同一张掩码
# ---------------------------------------------------------------------------
def ppo_update(net, optimizer, batch, args, use_mask: bool):
    obs, act = batch["obs"], batch["act"]
    logp_old, adv, ret = batch["logp_old"], batch["adv"], batch["ret"]
    mask = batch["mask"]
    N = obs.shape[0]
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    stats = {"approx_kl": 0.0, "clip_frac": 0.0, "entropy": 0.0, "n_updates": 0}
    for _ in range(args.epochs):
        idx = torch.randperm(N, device=obs.device)
        for start in range(0, N, args.batch_size):
            mb = idx[start:start + args.batch_size]
            logits, value = net(obs[mb])
            if use_mask:
                dist = Categorical(logits=apply_mask(logits, mask[mb]))
            else:
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
            nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            optimizer.step()

            with torch.no_grad():
                stats["approx_kl"] += (logp_old[mb] - logp).mean().item()
                stats["clip_frac"] += ((ratio - 1.0).abs() > args.clip).float().mean().item()
                stats["entropy"] += entropy.item()
                stats["n_updates"] += 1

    for k in stats:
        if k != "n_updates":
            stats[k] /= max(stats["n_updates"], 1)
    return stats


# ---------------------------------------------------------------------------
# 评估：贪婪策略也需要掩码（否则会选无效动作）
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(env, net, episodes: int, device, use_mask: bool):
    returns, success, invalid = [], [], 0
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_ret = 0.0
        ok = False
        while not done:
            logits, _ = net(to_tensor(obs, device))
            if use_mask:
                mask_t = torch.as_tensor(env.action_mask(), dtype=torch.float32,
                                         device=device).unsqueeze(0)
                logits = apply_mask(logits, mask_t)
            action = int(torch.argmax(logits, dim=-1).item())
            obs, r, terminated, truncated, info = env.step(action)
            ep_ret += r
            ok = ok or bool(info["success"])
            invalid += int(info["invalid_action"])
            done = terminated or truncated
        returns.append(ep_ret)
        success.append(1.0 if ok else 0.0)
    return returns, float(np.mean(success)), invalid


# ---------------------------------------------------------------------------
# 训练一个 arm
# ---------------------------------------------------------------------------
def train_one(arm: str, args):
    use_mask = (arm == "mask")
    set_seed(args.seed)
    device = torch.device("cpu")
    env = MaskedMaze()
    obs, _ = env.reset(seed=args.seed)

    net = ActorCritic(env.observation_space.shape[0], env.action_space.n)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    all_returns, all_success, kls = [], [], []
    invalid_rates = []
    total_steps, n_updates, obs_t = 0, 0, to_tensor(obs, device)
    reached_at = None
    t0 = time.time()

    print(f"\n[开始] arm={arm}（使用掩码：{use_mask}）")
    while total_steps < args.max_steps:
        batch, ep_returns, ep_success, invalid_rate, obs_t = collect_rollout(
            env, net, args.steps_per_update, obs_t, device, args.gamma, args.lam, use_mask)
        total_steps += args.steps_per_update
        n_updates += 1
        all_returns.extend(ep_returns)
        all_success.extend(ep_success)
        invalid_rates.append(invalid_rate)

        stats = ppo_update(net, optimizer, batch, args, use_mask)
        kls.append(stats["approx_kl"])

        if reached_at is None and len(all_success) >= 100 and np.mean(all_success[-100:]) >= 0.9:
            reached_at = total_steps

        if n_updates % 5 == 0 and len(all_success) >= 20:
            print(f"  [{arm:6}] 步数 {total_steps:6d} | 回合 {len(all_success):4d} | "
                  f"近20成功 {np.mean(all_success[-20:]):.2f} | 无效动作率 {invalid_rate:.3f} | "
                  f"KL {stats['approx_kl']:.4f} | 熵 {stats['entropy']:.3f}")

    eval_returns, eval_success, eval_invalid = evaluate(
        MaskedMaze(), net, args.eval_episodes, device, use_mask)
    tail_s = all_success[-100:] if len(all_success) >= 100 else all_success
    return {
        "arm": arm,
        "steps": total_steps,
        "reached_at": reached_at,
        "success100": float(np.mean(tail_s)),
        "last20_return": float(np.mean(all_returns[-20:])),
        "invalid_rate": float(np.mean(invalid_rates)),
        "eval_success": eval_success,
        "eval_invalid": eval_invalid,
        "mean_kl": float(np.mean(kls)),
        "seconds": time.time() - t0,
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="第012章：PPO 离散动作空间的分类策略（动作掩码）")
    p.add_argument("--arms", type=str, nargs="+", default=["nomask", "mask"],
                   choices=["nomask", "mask"], help="是否使用无效动作掩码")
    p.add_argument("--max-steps", type=int, default=50000, help="每个 arm 的环境步数预算")
    p.add_argument("--steps-per-update", type=int, default=1024, help="每次更新前采样步数")
    p.add_argument("--epochs", type=int, default=8, help="每批数据的更新轮数")
    p.add_argument("--batch-size", type=int, default=64, help="minibatch 大小")
    p.add_argument("--lr", type=float, default=3e-4, help="学习率")
    p.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
    p.add_argument("--lam", type=float, default=0.95, help="GAE 的 lambda")
    p.add_argument("--clip", type=float, default=0.2, help="PPO 裁剪范围")
    p.add_argument("--ent-coef", type=float, default=0.01, help="熵正则系数")
    p.add_argument("--vf-coef", type=float, default=0.5, help="价值损失系数")
    p.add_argument("--eval-episodes", type=int, default=20, help="评估回合数")
    p.add_argument("--seed", type=int, default=0, help="随机种子")
    p.add_argument("--save-dir", type=str, default="runs/ch012", help="模型保存目录")
    p.add_argument("--quick", action="store_true", help="快速模式：2 个 arm、各 8000 步")
    return p.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.max_steps = 8000
        args.steps_per_update = 512
        args.epochs = 4

    print("=" * 96)
    print(f"离散动作掩码对比 | arms {args.arms} | 每个 {args.max_steps} 步 | 种子 {args.seed}")
    print("环境：6x6 MaskedMaze，撞墙/撞障碍的动作非法；非法动作原地不动仍消耗一步")
    print("=" * 96)

    results = []
    for arm in args.arms:
        res = train_one(arm, args)
        results.append(res)
        print(f"[完成] {arm}: 近100成功 {res['success100']:.2f} | 评估成功 {res['eval_success']:.2f} | "
              f"平均无效率 {res['invalid_rate']:.3f} | 用时 {res['seconds']:.1f}s")

    print("\n" + "=" * 96)
    print("对比总结（同种子、同步数预算）")
    print("-" * 96)
    print(f"{'arm':>7} | {'首达90%成功步数':>13} | {'近100成功率':>11} | {'平均无效动作率':>12} | "
          f"{'评估成功率':>9} | {'评估非法次数':>11} | {'平均KL':>8} | {'用时(s)':>8}")
    for r in results:
        reached = str(r["reached_at"]) if r["reached_at"] is not None else "未达到"
        print(f"{r['arm']:>7} | {reached:>13} | {r['success100']:>11.2f} | {r['invalid_rate']:>12.3f} | "
              f"{r['eval_success']:>9.2f} | {r['eval_invalid']:>11d} | {r['mean_kl']:>8.4f} | "
              f"{r['seconds']:>8.1f}")

    print("\n观察提示：")
    print("  - mask 版无效动作率恒为 0（采样时就被排除），学习信号不会浪费在撞墙动作上；")
    print("  - nomask 版要靠负奖励慢慢学会躲墙，无效动作率通常停在 5%～15%；")
    print("  - 掩码必须同时作用于采样与更新重算的两个分布，否则比率会失真。")

    os.makedirs(args.save_dir, exist_ok=True)
    path = os.path.join(args.save_dir, "mask_compare.txt")
    with open(path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(f"arm={r['arm']} reached90={r['reached_at']} success100={r['success100']:.3f} "
                    f"invalid_rate={r['invalid_rate']:.4f} eval_success={r['eval_success']:.3f} "
                    f"eval_invalid={r['eval_invalid']} kl={r['mean_kl']:.5f}\n")
    print(f"\n结果已写入 {path}")


if __name__ == "__main__":
    main()
