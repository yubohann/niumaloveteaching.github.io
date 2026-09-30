"""
第015章 PPO的迁移学习：从简单到复杂

同一套 CartPole 动力学，改两个参数就变成一个"更难"的任务：
    easy：默认 CartPole-v1（重力 9.8、推力 10）
    hard：重力 11.5、推力 9.0（恢复力更弱、更容易倒）

对比三种在 hard 任务上的学习方式（评价口径统一为 hard 上的环境步数）：
  1) scratch        ：随机初始化，直接在 hard 上训练（基线）
  2) transfer_full  ：先在 easy 上预训练，再全参数微调 hard
  3) transfer_frozen：先在 easy 上预训练，然后冻结躯干、只训练策略/价值头

同时评估每个 arm 在 easy 与 hard 上的最终成绩，观察"迁移收益"与"遗忘代价"。

运行：
    python code/ch015.py            # 完整实验，CPU 约 4-8 分钟
    python code/ch015.py --quick    # 快速跑通（约 30-60 秒）
    python code/ch015.py --arms scratch transfer_full

预期：transfer_full 在 hard 上首达 400 的步数明显少于 scratch（约省 30%～60%）；
transfer_frozen 起步快但上限低；两个迁移臂在 easy 上都有小幅遗忘。
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


def make_env(kind: str):
    """easy=默认；hard=重力更大、推力更小。"""
    env = gym.make("CartPole-v1")
    if kind == "hard":
        env.unwrapped.gravity = 11.5
        env.unwrapped.force_mag = 9.0
    return env


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
# 采样
# ---------------------------------------------------------------------------
def collect_rollout(env, net, steps, obs_t, device, gamma, lam):
    obs_buf, act_buf, logp_buf = [], [], []
    rew_buf, val_buf, term_buf = [], [], []
    ep_returns = []

    ep_ret = 0.0
    obs_after = obs_t
    for _ in range(steps):
        a, logp, v = net.act(obs_t)
        next_obs, r, terminated, truncated, _ = env.step(a)

        obs_buf.append(obs_t.squeeze(0).cpu().numpy())
        act_buf.append(a)
        logp_buf.append(logp)
        rew_buf.append(r)
        val_buf.append(v)
        term_buf.append(float(terminated))

        ep_ret += r
        obs_t = to_tensor(next_obs, device)
        obs_after = obs_t

        if terminated or truncated:
            ep_returns.append(ep_ret)
            ep_ret = 0.0
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
    return batch, ep_returns, obs_t


# ---------------------------------------------------------------------------
# PPO 更新
# ---------------------------------------------------------------------------
def ppo_update(net, optimizer, batch, args):
    obs, act = batch["obs"], batch["act"]
    logp_old, adv, ret = batch["logp_old"], batch["adv"], batch["ret"]
    N = obs.shape[0]
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    stats = {"approx_kl": 0.0, "clip_frac": 0.0, "entropy": 0.0, "n_updates": 0}
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
# 评估
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(env_kind: str, net, episodes: int, device):
    env = make_env(env_kind)
    returns = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            logits, _ = net(to_tensor(obs, device))
            action = int(torch.argmax(logits, dim=-1).item())
            obs, r, terminated, truncated, _ = env.step(action)
            ep_ret += r
            done = terminated or truncated
        returns.append(ep_ret)
    return returns


# ---------------------------------------------------------------------------
# 训练一个阶段（返回轨迹与首达阈值步数）
# ---------------------------------------------------------------------------
def train_phase(kind: str, net, optimizer, args, budget: int, tag: str):
    device = torch.device("cpu")
    env = make_env(kind)
    obs, _ = env.reset(seed=args.seed)
    obs_t = to_tensor(obs, device)

    returns_hist, kls = [], []
    total_steps, n_updates = 0, 0
    reached_at = None
    t0 = time.time()

    print(f"\n[阶段] {tag} | 环境 {kind} | 预算 {budget} 步")
    while total_steps < budget:
        batch, ep_returns, obs_t = collect_rollout(
            env, net, args.steps_per_update, obs_t, device, args.gamma, args.lam)
        total_steps += args.steps_per_update
        n_updates += 1
        returns_hist.extend(ep_returns)

        stats = ppo_update(net, optimizer, batch, args)
        kls.append(stats["approx_kl"])

        if reached_at is None and len(returns_hist) >= 100 and np.mean(returns_hist[-100:]) >= args.stage_reward:
            reached_at = total_steps

        if n_updates % 5 == 0 and len(returns_hist) >= 20:
            print(f"  [{tag:14}] 步数 {total_steps:6d} | 回合 {len(returns_hist):4d} | "
                  f"近20均 {np.mean(returns_hist[-20:]):7.1f} | KL {stats['approx_kl']:.4f}")

    tail = returns_hist[-100:] if len(returns_hist) >= 100 else returns_hist
    return {
        "steps": total_steps,
        "history": returns_hist,
        "reached_at": reached_at,
        "last100": float(np.mean(tail)),
        "mean_kl": float(np.mean(kls)),
        "seconds": time.time() - t0,
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="第015章：PPO 迁移学习（从简单到复杂）")
    p.add_argument("--arms", type=str, nargs="+",
                   default=["scratch", "transfer_full", "transfer_frozen"],
                   choices=["scratch", "transfer_full", "transfer_frozen"],
                   help="要对比的学习方式")
    p.add_argument("--pretrain-steps", type=int, default=50000, help="easy 预训练步数")
    p.add_argument("--hard-steps", type=int, default=100000, help="每个 arm 在 hard 上的训练步数")
    p.add_argument("--steps-per-update", type=int, default=2048, help="每次更新前采样步数")
    p.add_argument("--epochs", type=int, default=10, help="每批数据的更新轮数")
    p.add_argument("--batch-size", type=int, default=64, help="minibatch 大小")
    p.add_argument("--lr", type=float, default=3e-4, help="学习率")
    p.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
    p.add_argument("--lam", type=float, default=0.95, help="GAE 的 lambda")
    p.add_argument("--clip", type=float, default=0.2, help="PPO 裁剪范围")
    p.add_argument("--ent-coef", type=float, default=0.01, help="熵正则系数")
    p.add_argument("--vf-coef", type=float, default=0.5, help="价值损失系数")
    p.add_argument("--stage-reward", type=float, default=400.0, help="阶段目标（100 回合滑窗）")
    p.add_argument("--eval-episodes", type=int, default=10, help="评估回合数")
    p.add_argument("--seed", type=int, default=0, help="随机种子")
    p.add_argument("--save-dir", type=str, default="runs/ch015", help="模型保存目录")
    p.add_argument("--quick", action="store_true", help="快速模式：各阶段 8 千步、目标 250")
    return p.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.pretrain_steps = 8000
        args.hard_steps = 8000
        args.steps_per_update = 512
        args.epochs = 4
        args.stage_reward = 250.0
        args.eval_episodes = 5

    print("=" * 96)
    print(f"迁移学习 | arms {args.arms} | easy 预训练 {args.pretrain_steps} 步 | "
          f"hard 阶段各 {args.hard_steps} 步 | 种子 {args.seed}")
    print("easy=默认 CartPole-v1；hard=重力 11.5 + 推力 9.0")
    print("=" * 96)

    # 预训练一次，供迁移臂共享（同一随机种子 => 同一份预训练权重）
    set_seed(args.seed)
    device = torch.device("cpu")
    probe = make_env("easy")
    base = ActorCritic(probe.observation_space.shape[0], probe.action_space.n)
    pre_opt = torch.optim.Adam(base.parameters(), lr=args.lr)
    print("\n[共享预训练] 供所有迁移臂使用")
    pre = train_phase("easy", base, pre_opt, args, args.pretrain_steps, "shared-pre")
    pretrained_state = {k: v.clone() for k, v in base.state_dict().items()}
    pre_easy_eval = float(np.mean(evaluate("easy", base, args.eval_episodes, device)))
    print(f"[共享预训练] easy 贪婪评估 {pre_easy_eval:.1f} | 首达 {pre['reached_at']}")

    results = []
    for arm in args.arms:
        set_seed(args.seed)
        net = ActorCritic(probe.observation_space.shape[0], probe.action_space.n)
        if arm != "scratch":
            net.load_state_dict(pretrained_state)
        device = torch.device("cpu")

        info = {"arm": arm}
        if arm != "scratch":
            info["pretrain_easy_eval"] = pre_easy_eval

        if arm == "transfer_frozen":
            for p in net.body.parameters():
                p.requires_grad_(False)
            params = list(net.actor.parameters()) + list(net.critic.parameters())
            optimizer = torch.optim.Adam(params, lr=args.lr)
        else:
            optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

        phase = train_phase("hard", net, optimizer, args, args.hard_steps, arm)
        info["hard_reached_at"] = phase["reached_at"]
        info["hard_last100"] = phase["last100"]
        info["seconds"] = phase["seconds"]
        info["hard_eval"] = float(np.mean(evaluate("hard", net, args.eval_episodes, device)))
        info["easy_eval"] = float(np.mean(evaluate("easy", net, args.eval_episodes, device)))
        if arm != "scratch":
            info["forget"] = info["easy_eval"] - pre_easy_eval
        results.append(info)

        os.makedirs(args.save_dir, exist_ok=True)
        torch.save(net.state_dict(), os.path.join(args.save_dir, f"{arm}.pt"))

    print("\n" + "=" * 96)
    print("迁移对比总结（hard 阶段预算相同；首达=100 回合滑窗≥400）")
    print("-" * 96)
    print(f"{'arm':>16} | {'hard首达步数':>11} | {'hard近100均':>10} | {'hard评估':>8} | "
          f"{'easy评估':>8} | {'easy遗忘':>9} | {'hard用时(s)':>10}")
    for r in results:
        reached = str(r["hard_reached_at"]) if r["hard_reached_at"] is not None else "未达到"
        forget = f"{r['forget']:+.1f}" if "forget" in r else "—"
        print(f"{r['arm']:>16} | {reached:>11} | {r['hard_last100']:>10.1f} | "
              f"{r['hard_eval']:>8.1f} | {r['easy_eval']:>8.1f} | {forget:>9} | "
              f"{r['seconds']:>10.1f}")

    print(f"\n共享预训练（easy {args.pretrain_steps} 步）的 easy 贪婪评估：{pre_easy_eval:.1f}")
    print("观察提示：")
    print("  - transfer_full 通常比 scratch 更快在 hard 上达标，省下的步数就是迁移收益；")
    print("  - transfer_frozen 起步快但上限低：特征可复用，动作细节要重新学；")
    print("  - easy 评估列持续下降说明发生了遗忘，可考虑低学习率微调缓解。")

    os.makedirs(args.save_dir, exist_ok=True)
    path = os.path.join(args.save_dir, "transfer.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"pretrain_easy_eval={pre_easy_eval:.2f}\n")
        for r in results:
            f.write(f"arm={r['arm']} hard_reached={r['hard_reached_at']} "
                    f"hard_last100={r['hard_last100']:.2f} hard_eval={r['hard_eval']:.2f} "
                    f"easy_eval={r['easy_eval']:.2f} seconds={r['seconds']:.1f}\n")
    print(f"\n结果已写入 {path}")


if __name__ == "__main__":
    main()
