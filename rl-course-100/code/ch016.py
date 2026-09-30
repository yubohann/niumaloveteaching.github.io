"""
第016章 PPO的多任务学习

同一个 CartPole 环境，三种重力 = 三个任务：
    gravity ∈ {7.0（轻）, 9.8（标准）, 12.5（重）}
对比两种策略：
  1) multitask：一个共享网络，观测里拼接任务 one-hot 标识（7 维输入），
                按回合轮流在三个任务上采样，总预算 3 × 每任务步数
  2) separate ：每个任务一个独立网络，依次训练，各用同样的每任务步数

两者总环境步数相同，对比：每任务的达标步数、最终评估回报、参数量与总耗时。
本实现按"完整回合"收集数据，并用回合末观测的价值做 bootstrap，
避免截断回合跨任务串台（这是多任务采样最常见的正确性坑）。

运行：
    python code/ch016.py            # 两臂各 15 万步，CPU 约 4-8 分钟
    python code/ch016.py --quick    # 快速跑通（约 30-60 秒）
    python code/ch016.py --arms multitask --steps-per-task 80000

预期：共享网络在三个任务上都能学会；轻/标准任务与专精版差距很小，
最重任务上共享网络通常略慢（负迁移），但只花 1/3 参数量。
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


GRAVITIES = [7.0, 9.8, 12.5]
TASK_NAMES = ["轻(7.0)", "标准(9.8)", "重(12.5)"]


# ---------------------------------------------------------------------------
# 任务环境：可改重力；可选把任务 one-hot 拼到观测尾部
# ---------------------------------------------------------------------------
class TaskObsWrapper(gym.ObservationWrapper):
    def __init__(self, env, task_id: int, n_tasks: int):
        super().__init__(env)
        self.task_id = task_id
        self.onehot = np.zeros(n_tasks, dtype=np.float32)
        self.onehot[task_id] = 1.0
        low = np.concatenate([env.observation_space.low, np.zeros(n_tasks, dtype=np.float32)])
        high = np.concatenate([env.observation_space.high, np.ones(n_tasks, dtype=np.float32)])
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def observation(self, obs):
        return np.concatenate([obs, self.onehot]).astype(np.float32)


def make_task_env(task_id: int, with_id: bool):
    env = gym.make("CartPole-v1")
    env.unwrapped.gravity = GRAVITIES[task_id]
    if with_id:
        env = TaskObsWrapper(env, task_id, len(GRAVITIES))
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
# 按回合收集：轮流使用各任务的 env；每回合用自己的末观测 bootstrap
# ---------------------------------------------------------------------------
def collect_episodes(envs, net, steps_target, start_task, device, gamma, lam):
    obs_buf, act_buf, logp_buf, adv_buf, ret_buf = [], [], [], [], []
    ep_returns = {i: [] for i in range(len(envs))}
    total_steps = 0
    task = start_task

    while total_steps < steps_target:
        env = envs[task]
        obs, _ = env.reset()
        obs_t = to_tensor(obs, device)

        obs_l, act_l, logp_l, rew_l, val_l, term_l = [], [], [], [], [], []
        ep_ret = 0.0
        done = False
        while not done:
            a, logp, v = net.act(obs_t)
            next_obs, r, terminated, truncated, _ = env.step(a)

            obs_l.append(obs_t.squeeze(0).cpu().numpy())
            act_l.append(a)
            logp_l.append(logp)
            rew_l.append(r)
            val_l.append(v)
            term_l.append(float(terminated))

            ep_ret += r
            obs_t = to_tensor(next_obs, device)
            done = terminated or truncated

        # 用本回合真正结束时的观测做 bootstrap（截断时有效、终止时被掩码置零）
        last_value = net.value(obs_t)
        adv, ret = compute_gae(rew_l, val_l, term_l, last_value, gamma, lam)
        obs_buf.extend(obs_l)
        act_buf.extend(act_l)
        logp_buf.extend(logp_l)
        adv_buf.append(adv)
        ret_buf.append(ret)
        ep_returns[task].append(ep_ret)

        total_steps += len(obs_l)
        task = (task + 1) % len(envs)

    batch = {
        "obs": torch.as_tensor(np.asarray(obs_buf), dtype=torch.float32, device=device),
        "act": torch.as_tensor(np.asarray(act_buf), dtype=torch.long, device=device),
        "logp_old": torch.as_tensor(np.asarray(logp_buf), dtype=torch.float32, device=device),
        "adv": torch.as_tensor(np.concatenate(adv_buf), dtype=torch.float32, device=device),
        "ret": torch.as_tensor(np.concatenate(ret_buf), dtype=torch.float32, device=device),
    }
    return batch, ep_returns, task


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
def evaluate(envs, net, episodes: int, device):
    results = []
    for env in envs:
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
        results.append(float(np.mean(returns)))
    return results


# ---------------------------------------------------------------------------
# 训练一臂
# ---------------------------------------------------------------------------
def train_arm(arm: str, args):
    set_seed(args.seed)
    device = torch.device("cpu")
    with_id = (arm == "multitask")

    if with_id:
        envs = [make_task_env(i, True) for i in range(len(GRAVITIES))]
        net = ActorCritic(envs[0].observation_space.shape[0], envs[0].action_space.n)
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
        nets = [net]
    else:
        envs = [make_task_env(i, False) for i in range(len(GRAVITIES))]
        nets = [ActorCritic(envs[0].observation_space.shape[0], envs[0].action_space.n)
                for _ in GRAVITIES]
        net = None

    all_ep_returns = {i: [] for i in range(len(GRAVITIES))}
    reached_at = {i: None for i in range(len(GRAVITIES))}
    spent_steps = {i: 0 for i in range(len(GRAVITIES))}
    kl_hist = []
    t0 = time.time()
    task_cursor = 0

    print(f"\n[开始] arm={arm}（任务标识：{with_id}）")

    if with_id:
        total_budget = args.steps_per_task * len(GRAVITIES)
        while sum(spent_steps.values()) < total_budget:
            batch, ep_returns, task_cursor = collect_episodes(
                envs, net, args.steps_per_update, task_cursor, device, args.gamma, args.lam)
            # 统计各任务新增步数与回报（从本批的批次结构里无法区分任务，
            # 因此以回合回报为主指标；步数按回合平均长度估算）
            for i, rs in ep_returns.items():
                all_ep_returns[i].extend(rs)
            step_share = args.steps_per_update / len(GRAVITIES)
            for i in range(len(GRAVITIES)):
                spent_steps[i] += step_share
            stats = ppo_update(net, optimizer, batch, args)
            kl_hist.append(stats["approx_kl"])
            for i in range(len(GRAVITIES)):
                if reached_at[i] is None and len(all_ep_returns[i]) >= 100 and \
                        np.mean(all_ep_returns[i][-100:]) >= args.stage_reward:
                    reached_at[i] = spent_steps[i]
            if (sum(len(v) for v in all_ep_returns.values())) % 200 < max(len(GRAVITIES), 1):
                msg = " | ".join(f"T{i}:{np.mean(all_ep_returns[i][-20:]):6.1f}"
                                 if all_ep_returns[i] else f"T{i}:  --  "
                                 for i in range(len(GRAVITIES)))
                print(f"  [multitask] 回合 {sum(len(v) for v in all_ep_returns.values()):5d} | {msg} | "
                      f"KL {stats['approx_kl']:.4f}")
        eval_scores = evaluate(envs, net, args.eval_episodes, device)
    else:
        for i, (env, net_i) in enumerate(zip(envs, nets)):
            optimizer = torch.optim.Adam(net_i.parameters(), lr=args.lr)
            done_steps = 0
            print(f"  [separate] 开始任务 T{i} {TASK_NAMES[i]}")
            while done_steps < args.steps_per_task:
                batch, ep_returns, _ = collect_episodes(
                    [env], net_i, args.steps_per_update, 0, device, args.gamma, args.lam)
                all_ep_returns[i].extend(ep_returns[0])
                done_steps += args.steps_per_update
                spent_steps[i] = done_steps
                stats = ppo_update(net_i, optimizer, batch, args)
                kl_hist.append(stats["approx_kl"])
                if reached_at[i] is None and len(all_ep_returns[i]) >= 100 and \
                        np.mean(all_ep_returns[i][-100:]) >= args.stage_reward:
                    reached_at[i] = done_steps
                if done_steps % (args.steps_per_update * 5) == 0 and all_ep_returns[i]:
                    print(f"  [separate] T{i} 步数 {done_steps:6d} | "
                          f"近20均 {np.mean(all_ep_returns[i][-20:]):7.1f} | "
                          f"KL {stats['approx_kl']:.4f}")
        # 统一评估：每个任务用各自的网络
        eval_scores = [evaluate([env], net_i, args.eval_episodes, device)[0]
                       for env, net_i in zip(envs, nets)]

    n_params = sum(p.numel() for p in nets[0].parameters()) if not with_id else \
        sum(p.numel() for p in net.parameters())
    total_params = n_params if with_id else n_params * len(GRAVITIES)

    return {
        "arm": arm,
        "reached_at": reached_at,
        "last100": {i: (float(np.mean(all_ep_returns[i][-100:]))
                        if len(all_ep_returns[i]) >= 100 else float("nan"))
                    for i in range(len(GRAVITIES))},
        "eval": eval_scores,
        "n_params": n_params,
        "total_params": total_params,
        "mean_kl": float(np.mean(kl_hist)),
        "seconds": time.time() - t0,
        "nets": nets,
        "envs": envs,
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="第016章：PPO 多任务学习")
    p.add_argument("--arms", type=str, nargs="+", default=["multitask", "separate"],
                   choices=["multitask", "separate"], help="共享网络或每任务独立网络")
    p.add_argument("--steps-per-task", type=int, default=50000, help="每个任务的环境步数预算")
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
    p.add_argument("--save-dir", type=str, default="runs/ch016", help="模型保存目录")
    p.add_argument("--quick", action="store_true", help="快速模式：每任务 8 千步、目标 250")
    return p.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.steps_per_task = 8000
        args.steps_per_update = 512
        args.epochs = 4
        args.stage_reward = 250.0
        args.eval_episodes = 5

    print("=" * 96)
    print(f"多任务学习 | arms {args.arms} | 每任务 {args.steps_per_task} 步 | 种子 {args.seed}")
    print(f"任务：重力 {GRAVITIES}（T0/T1/T2）")
    print("=" * 96)

    results = []
    for arm in args.arms:
        res = train_arm(arm, args)
        results.append(res)
        print(f"[完成] {arm}: 参数 {res['total_params']} | "
              f"评估 {[round(v) for v in res['eval']]} | 用时 {res['seconds']:.1f}s")

    print("\n" + "=" * 96)
    print("多任务 vs 单任务专精（同总预算；首达=各任务 100 回合滑窗≥目标）")
    print("-" * 96)
    header = f"{'arm':>10} | {'参数量':>8} |"
    for i, name in enumerate(TASK_NAMES):
        header += f" T{i} 首达 | T{i} 近100 | T{i} 评估 |"
    header += f" {'用时(s)':>8}"
    print(header)
    for r in results:
        line = f"{r['arm']:>10} | {r['total_params']:>8d} |"
        for i in range(len(GRAVITIES)):
            reached = str(r["reached_at"][i]) if r["reached_at"][i] is not None else "未达"
            line += f" {reached:>7} | {r['last100'][i]:>8.1f} | {r['eval'][i]:>7.1f} |"
        line += f" {r['seconds']:>8.1f}"
        print(line)

    print("\n观察提示：")
    print("  - 共享网络用 1/3 参数处理三个任务，任务标识让同一套权重按上下文复用；")
    print("  - 重任务通常最难：共享网络可能略慢（负迁移），轻任务最容易被学好；")
    print("  - 按整回合采样 + 回合末 bootstrap，保证截断回合不与其它任务串台。")

    os.makedirs(args.save_dir, exist_ok=True)
    path = os.path.join(args.save_dir, "multitask.txt")
    with open(path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(f"arm={r['arm']} params={r['total_params']} seconds={r['seconds']:.1f}\n")
            for i in range(len(GRAVITIES)):
                f.write(f"  T{i} reached={r['reached_at'][i]} last100={r['last100'][i]:.2f} "
                        f"eval={r['eval'][i]:.2f}\n")
    print(f"\n结果已写入 {path}")


if __name__ == "__main__":
    main()
