"""
第019章 PPO的课程学习

目标任务：重版 CartPole（重力 13.0、推力 9.0），直接学比较吃力。
两种训练方式（同种子、同环境步数预算）：
  - direct     ：一直在目标任务上训练
  - curriculum ：从重力 7.0 起步，当前难度连续 2 次更新（近 20 回合均值≥350）后
                 重力 +0.5，直到到达 13.0 再继续在目标上训练

公平比较的口径：每 6 次更新在"目标环境"上做一次贪婪评估，
记录首达目标成绩 400 的步数、评估曲线与最终成绩；
课程臂额外报告难度（重力）随步数的推进轨迹。

运行：
    python code/ch019.py            # 2 种方式，各 15 万步，CPU 约 5-9 分钟
    python code/ch019.py --quick    # 快速跑通（约 30-60 秒）
    python code/ch019.py --arms curriculum

预期：curriculum 在目标环境上的评估曲线更早抬升（首达 400 步数更少），
最终两者趋同；难度推进太快或太慢都会削弱课程收益。
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


def make_env(gravity: float):
    env = gym.make("CartPole-v1")
    env.unwrapped.gravity = gravity
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
# 采样与更新
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
# 评估：任务环境用目标重力，两种方式共用
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_target(net, episodes: int, device, target_gravity: float):
    env = make_env(target_gravity)
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
    return float(np.mean(returns))


# ---------------------------------------------------------------------------
# 训练一臂
# ---------------------------------------------------------------------------
def train_one(arm: str, args):
    set_seed(args.seed)
    device = torch.device("cpu")

    curriculum = (arm == "curriculum")
    gravity = args.start_gravity if curriculum else args.target_gravity
    env = make_env(gravity)
    obs, _ = env.reset(seed=args.seed)
    obs_t = to_tensor(obs, device)

    net = ActorCritic(env.observation_space.shape[0], env.action_space.n)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    all_returns, kls = [], []
    eval_curve = []                 # (步数, 目标环境评估)
    gravity_trace = [(0, gravity)]
    eval_at_400 = None
    total_steps, n_updates = 0, 0
    updates_since_advance = 0
    t0 = time.time()

    print(f"\n[开始] arm={arm} | 起始重力 {gravity} | 目标重力 {args.target_gravity}")
    while total_steps < args.max_steps:
        batch, ep_returns, obs_t = collect_rollout(
            env, net, args.steps_per_update, obs_t, device, args.gamma, args.lam)
        total_steps += args.steps_per_update
        n_updates += 1
        all_returns.extend(ep_returns)
        updates_since_advance += 1

        stats = ppo_update(net, optimizer, batch, args)
        kls.append(stats["approx_kl"])

        # 课程推进：当前难度表现达标且距上次推进至少 2 次更新
        if curriculum and gravity < args.target_gravity and updates_since_advance >= args.advance_patience:
            if len(all_returns) >= 20 and np.mean(all_returns[-20:]) >= args.advance_reward:
                gravity = min(args.target_gravity, gravity + args.gravity_step)
                gravity_trace.append((total_steps, gravity))
                env = make_env(gravity)
                reset_obs, _ = env.reset()
                obs_t = to_tensor(reset_obs, device)
                updates_since_advance = 0
                print(f"  [curriculum] 推进难度：重力 → {gravity:.1f}（步数 {total_steps}）")

        # 目标环境评估
        if n_updates % args.eval_every == 0 or total_steps >= args.max_steps:
            ev = evaluate_target(net, args.eval_episodes, device, args.target_gravity)
            eval_curve.append((total_steps, ev))
            if eval_at_400 is None and ev >= 400.0:
                eval_at_400 = total_steps
            print(f"  [{arm:10}] 步数 {total_steps:6d} | 训练难度 g={gravity:.1f} | "
                  f"近20均 {np.mean(all_returns[-20:]):7.1f} | 目标评估 {ev:6.1f} | "
                  f"KL {stats['approx_kl']:.4f}")

    final_eval = evaluate_target(net, args.eval_episodes, device, args.target_gravity)
    return {
        "arm": arm,
        "gravity_final": gravity,
        "gravity_trace": gravity_trace,
        "eval_at_400": eval_at_400,
        "final_eval": final_eval,
        "eval_curve": eval_curve,
        "mean_kl": float(np.mean(kls)),
        "seconds": time.time() - t0,
        "net": net,
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="第019章：PPO 课程学习")
    p.add_argument("--arms", type=str, nargs="+", default=["direct", "curriculum"],
                   choices=["direct", "curriculum"], help="训练方式")
    p.add_argument("--start-gravity", type=float, default=7.0, help="课程起始重力")
    p.add_argument("--target-gravity", type=float, default=13.0, help="目标重力")
    p.add_argument("--gravity-step", type=float, default=0.5, help="每次推进的重力增量")
    p.add_argument("--advance-reward", type=float, default=350.0, help="推进难度所需的近20回合均值")
    p.add_argument("--advance-patience", type=int, default=2, help="两次推进之间至少经过的更新次数")
    p.add_argument("--max-steps", type=int, default=150000, help="每个 arm 的环境步数预算")
    p.add_argument("--steps-per-update", type=int, default=2048, help="每次更新前采样步数")
    p.add_argument("--epochs", type=int, default=10, help="每批数据的更新轮数")
    p.add_argument("--batch-size", type=int, default=64, help="minibatch 大小")
    p.add_argument("--lr", type=float, default=3e-4, help="学习率")
    p.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
    p.add_argument("--lam", type=float, default=0.95, help="GAE 的 lambda")
    p.add_argument("--clip", type=float, default=0.2, help="PPO 裁剪范围")
    p.add_argument("--ent-coef", type=float, default=0.01, help="熵正则系数")
    p.add_argument("--vf-coef", type=float, default=0.5, help="价值损失系数")
    p.add_argument("--eval-every", type=int, default=6, help="每隔多少次更新评估一次目标环境")
    p.add_argument("--eval-episodes", type=int, default=5, help="每次评估的回合数")
    p.add_argument("--seed", type=int, default=0, help="随机种子")
    p.add_argument("--save-dir", type=str, default="runs/ch019", help="模型保存目录")
    p.add_argument("--quick", action="store_true", help="快速模式：1.2 万步、评估更密")
    return p.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.max_steps = 12000
        args.steps_per_update = 512
        args.epochs = 4
        args.advance_reward = 250.0
        args.eval_every = 2
        args.eval_episodes = 3

    print("=" * 96)
    print(f"课程学习对比 | arms {args.arms} | 预算 {args.max_steps} 步 | 种子 {args.seed}")
    print(f"课程：重力 {args.start_gravity} → {args.target_gravity}，每步 +{args.gravity_step}，"
          f"推进条件 近20均 ≥ {args.advance_reward}")
    print("=" * 96)

    results = []
    for arm in args.arms:
        res = train_one(arm, args)
        results.append(res)
        print(f"[完成] {arm}: 目标评估 {res['final_eval']:.1f} | "
              f"首达400评估步数 {res['eval_at_400']} | 用时 {res['seconds']:.1f}s")

    print("\n" + "=" * 96)
    print("对比总结（评估口径：目标环境、贪婪策略）")
    print("-" * 96)
    print(f"{'arm':>10} | {'目标环境首达400步数':>17} | {'最终目标评估':>11} | "
          f"{'终点训练难度':>11} | {'平均KL':>8} | {'用时(s)':>8}")
    for r in results:
        reached = str(r["eval_at_400"]) if r["eval_at_400"] is not None else "未达到"
        print(f"{r['arm']:>10} | {reached:>17} | {r['final_eval']:>11.1f} | "
              f"{r['gravity_final']:>11.1f} | {r['mean_kl']:>8.4f} | {r['seconds']:>8.1f}")

    print("\n目标环境评估曲线（步数: 回报）")
    for r in results:
        pts = " ".join(f"{s // 1000}k:{v:.0f}" for s, v in r["eval_curve"])
        print(f"  {r['arm']:>10}  {pts}")
    for r in results:
        if r["arm"] == "curriculum":
            trace = " ".join(f"{s // 1000}k:g={g:.1f}" for s, g in r["gravity_trace"])
            print(f"\n  课程难度轨迹：{trace}")

    print("\n观察提示：")
    print("  - 课程臂的训练近20均不可与直训臂直接比（难度不同），要看目标环境评估列；")
    print("  - 推进太快（advance_reward 太低）课程退化为直训，太慢则浪费预算；")
    print("  - 两条评估曲线最终趋同是正常现象，课程的收益在前期。")

    os.makedirs(args.save_dir, exist_ok=True)
    path = os.path.join(args.save_dir, "curriculum.txt")
    with open(path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(f"arm={r['arm']} eval_at_400={r['eval_at_400']} "
                    f"final_eval={r['final_eval']:.2f} gravity_final={r['gravity_final']:.1f} "
                    f"seconds={r['seconds']:.1f}\n")
            f.write("  curve: " + " ".join(f"{s}:{v:.0f}" for s, v in r["eval_curve"]) + "\n")
            if r["arm"] == "curriculum":
                f.write("  gravity: " + " ".join(f"{s}:{g:.1f}" for s, g in r["gravity_trace"]) + "\n")
    print(f"\n结果已写入 {path}")


if __name__ == "__main__":
    main()
