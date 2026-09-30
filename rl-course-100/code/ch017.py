"""
第017章 PPO的元学习初始化

元学习的目标不是"学会一个任务"，而是"学会怎么快速学会一个新任务"。
本章用 Reptile（一阶元学习算法）在 CartPole 的"重力任务族"上训练初始化：
    任务分布：重力 g ~ U[8.0, 13.0]，每个元迭代抽 2 个任务做内循环
    内循环  ：从当前元参数出发，用 PPO 在任务上做少量更新
    外循环  ：θ ← θ + β · mean(θ_inner − θ)（Reptile 插值）

元测试（held-out 重力 {8.5, 10.0, 11.5, 13.0}）：
    对比"元初始化"与"随机初始化"在新任务上做 0/1/2/3 轮 PPO 适配的成绩。

运行：
    python code/ch017.py            # 元训练 60 迭代 + 适配评估，CPU 约 4-8 分钟
    python code/ch017.py --quick    # 快速跑通（约 30-60 秒）
    python code/ch017.py --meta-iters 120

预期：元初始化在前 1-2 轮适配后成绩明显领先随机初始化；
随着适配轮数增加，两者差距缩小（随机初始化用更多数据也能追上）。
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
    return env


def clone_state(net):
    return {k: v.detach().clone() for k, v in net.state_dict().items()}


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
# 采样与更新（供内循环与适配共用）
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


def ppo_update(net, optimizer, batch, args, epochs: int):
    obs, act = batch["obs"], batch["act"]
    logp_old, adv, ret = batch["logp_old"], batch["adv"], batch["ret"]
    N = obs.shape[0]
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    stats = {"approx_kl": 0.0, "clip_frac": 0.0, "n_updates": 0}
    for _ in range(epochs):
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
                stats["n_updates"] += 1

    for k in stats:
        if k != "n_updates":
            stats[k] /= max(stats["n_updates"], 1)
    return stats


# ---------------------------------------------------------------------------
# 内循环：在单个任务上从当前参数做少量 PPO 更新
# ---------------------------------------------------------------------------
def inner_loop(net, gravity, args, steps: int, epochs: int, rng_seed: int):
    set_seed(rng_seed)
    device = torch.device("cpu")
    env = make_env(gravity)
    obs, _ = env.reset(seed=rng_seed)
    obs_t = to_tensor(obs, device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.inner_lr)

    batch, ep_returns, obs_t = collect_rollout(
        env, net, steps, obs_t, device, args.gamma, args.lam)
    stats = ppo_update(net, optimizer, batch, args, epochs)
    return stats, ep_returns


def reptile_step(net, state_before, states_after, beta: float):
    """θ ← θ_before + β · mean(θ_after − θ_before)。"""
    with torch.no_grad():
        for name, p in net.state_dict().items():
            deltas = [sa[name] - state_before[name] for sa in states_after]
            avg_delta = sum(deltas) / len(deltas)
            p.copy_(state_before[name] + beta * avg_delta)


# ---------------------------------------------------------------------------
# 评估
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(env, net, episodes: int, device):
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
# 适配实验：元初始化 vs 随机初始化
# ---------------------------------------------------------------------------
def adaptation_curve(init_state, gravity, args):
    """从给定初始化出发，做 args.adapt_rounds 轮适配，记录每轮后的贪婪评估。"""
    set_seed(args.seed + int(gravity * 100))
    device = torch.device("cpu")
    env = make_env(gravity)
    net = ActorCritic(env.observation_space.shape[0], env.action_space.n)
    net.load_state_dict(init_state)
    obs, _ = env.reset(seed=args.seed)
    obs_t = to_tensor(obs, device)

    curve = [evaluate(env, net, args.eval_episodes, device)]
    for r in range(args.adapt_rounds):
        optimizer = torch.optim.Adam(net.parameters(), lr=args.adapt_lr)
        batch, _, obs_t = collect_rollout(
            env, net, args.adapt_steps, obs_t, device, args.gamma, args.lam)
        ppo_update(net, optimizer, batch, args, args.adapt_epochs)
        curve.append(evaluate(env, net, args.eval_episodes, device))
    return curve


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="第017章：PPO 元学习初始化（Reptile）")
    p.add_argument("--meta-iters", type=int, default=60, help="元迭代次数")
    p.add_argument("--meta-batch", type=int, default=2, help="每次元迭代采样的任务数")
    p.add_argument("--meta-steps", type=int, default=2048, help="内循环每任务采样步数")
    p.add_argument("--inner-epochs", type=int, default=2, help="内循环的 PPO 更新轮数")
    p.add_argument("--meta-beta", type=float, default=0.5, help="Reptile 插值系数 β")
    p.add_argument("--inner-lr", type=float, default=1e-3, help="内循环学习率")
    p.add_argument("--adapt-rounds", type=int, default=3, help="适配评估的轮数")
    p.add_argument("--adapt-steps", type=int, default=1024, help="每轮适配采样步数")
    p.add_argument("--adapt-epochs", type=int, default=1, help="每轮适配的更新轮数")
    p.add_argument("--adapt-lr", type=float, default=3e-4, help="适配学习率")
    p.add_argument("--batch-size", type=int, default=64, help="minibatch 大小")
    p.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
    p.add_argument("--lam", type=float, default=0.95, help="GAE 的 lambda")
    p.add_argument("--clip", type=float, default=0.2, help="PPO 裁剪范围")
    p.add_argument("--ent-coef", type=float, default=0.01, help="熵正则系数")
    p.add_argument("--vf-coef", type=float, default=0.5, help="价值损失系数")
    p.add_argument("--eval-episodes", type=int, default=5, help="每次评估回合数")
    p.add_argument("--seed", type=int, default=0, help="随机种子")
    p.add_argument("--save-dir", type=str, default="runs/ch017", help="模型保存目录")
    p.add_argument("--quick", action="store_true", help="快速模式：8 次元迭代、1 轮适配")
    return p.parse_args()


def main():
    args = parse_args()
    eval_gravities = [8.5, 10.0, 11.5, 13.0]
    if args.quick:
        args.meta_iters = 8
        args.meta_steps = 512
        args.inner_epochs = 1
        args.adapt_rounds = 1
        args.adapt_steps = 512
        args.eval_episodes = 3
        eval_gravities = [8.5, 12.0]

    set_seed(args.seed)
    device = torch.device("cpu")
    probe = gym.make("CartPole-v1")
    obs_dim = probe.observation_space.shape[0]
    act_dim = probe.action_space.n

    net = ActorCritic(obs_dim, act_dim)
    rng = random.Random(args.seed)
    t0 = time.time()

    print("=" * 96)
    print(f"Reptile 元训练 | 迭代 {args.meta_iters} | 任务批 {args.meta_batch} | "
          f"内循环 {args.meta_steps} 步 × {args.inner_epochs} epoch | β={args.meta_beta}")
    print("任务分布：重力 g ~ U[8.0, 13.0]")
    print("=" * 96)

    for it in range(args.meta_iters):
        state_before = clone_state(net)
        states_after = []
        for k in range(args.meta_batch):
            task_seed = rng.randint(0, 10**9)
            gravity = rng.uniform(8.0, 13.0)
            net.load_state_dict(state_before)          # 每个任务从同一元参数出发
            stats, ep_returns = inner_loop(
                net, gravity, args, args.meta_steps, args.inner_epochs, task_seed)
            states_after.append(clone_state(net))
            if k == 0:
                last_stats, last_task = stats, (gravity, ep_returns)
        reptile_step(net, state_before, states_after, args.meta_beta)

        if (it + 1) % 5 == 0:
            g, rets = last_task
            tail = np.mean(rets[-5:]) if rets else float("nan")
            print(f"  [元迭代 {it + 1:3d}] 末任务 g={g:4.2f} | 内循环近5回合 {tail:7.1f} | "
                  f"KL {last_stats['approx_kl']:.4f}")

    meta_time = time.time() - t0
    meta_state = clone_state(net)

    # 适配评估
    print("\n" + "=" * 96)
    print(f"适配评估 | held-out 重力 {eval_gravities} | 每轮 {args.adapt_steps} 步 × "
          f"{args.adapt_epochs} epoch")
    print("=" * 96)

    set_seed(args.seed + 12345)
    random_state = clone_state(ActorCritic(obs_dim, act_dim))

    curves = {"meta": {}, "random": {}}
    for g in eval_gravities:
        curves["meta"][g] = adaptation_curve(meta_state, g, args)
        curves["random"][g] = adaptation_curve(random_state, g, args)
        print(f"  重力 {g}: 元初始化 {[round(v) for v in curves['meta'][g]]} | "
              f"随机初始化 {[round(v) for v in curves['random'][g]]}")

    print("\n" + "=" * 96)
    print("适配曲线（0 轮 = 直接评估；每轮 = 采样 1024 步 + 1 次 PPO 更新）")
    print("-" * 96)
    header = f"{'重力':>6} | {'方法':>6} | " + " | ".join(f"轮{r}" for r in range(args.adapt_rounds + 1))
    print(header)
    for g in eval_gravities:
        for method in ["meta", "random"]:
            cells = " | ".join(f"{v:6.1f}" for v in curves[method][g])
            print(f"{g:>6.1f} | {method:>6} | {cells}")

    meta_avg = [np.mean([curves["meta"][g][r] for g in eval_gravities])
                for r in range(args.adapt_rounds + 1)]
    rand_avg = [np.mean([curves["random"][g][r] for g in eval_gravities])
                for r in range(args.adapt_rounds + 1)]
    print("\n平均曲线：")
    print(f"  元初始化   ：{' '.join(f'{v:6.1f}' for v in meta_avg)}")
    print(f"  随机初始化 ：{' '.join(f'{v:6.1f}' for v in rand_avg)}")
    print(f"  优势       ：{' '.join(f'{m - r:+6.1f}' for m, r in zip(meta_avg, rand_avg))}")

    print(f"\n元训练用时 {meta_time:.1f}s | 元训练环境步数约 "
          f"{args.meta_iters * args.meta_batch * args.meta_steps}")

    os.makedirs(args.save_dir, exist_ok=True)
    for name, state in [("meta_init", meta_state), ("random_init", random_state)]:
        torch.save(state, os.path.join(args.save_dir, f"{name}.pt"))
    path = os.path.join(args.save_dir, "meta_adapt.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"meta_iters={args.meta_iters} meta_time={meta_time:.1f}\n")
        for g in eval_gravities:
            f.write(f"g={g} meta={[round(v) for v in curves['meta'][g]]} "
                    f"random={[round(v) for v in curves['random'][g]]}\n")
        f.write(f"avg_meta={[round(v) for v in meta_avg]}\n")
        f.write(f"avg_random={[round(v) for v in rand_avg]}\n")
    print(f"模型与结果已保存到 {args.save_dir}")


if __name__ == "__main__":
    main()
