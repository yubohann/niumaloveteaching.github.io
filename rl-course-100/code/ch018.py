"""
第018章 PPO的模仿学习初始化

先训练一个 PPO 专家，再把它的状态-动作记为示范数据，做行为克隆（BC）：
    BC 损失 = − E[ log π_θ(a_expert | s) ]（交叉熵 / 负对数似然）

然后对比两种微调起点（同样的 PPO 预算与种子）：
  - scratch：从随机初始化开始
  - bc     ：从行为克隆权重开始
另外报告 BC 本身的精度与闭环成绩（复合误差会让它低于专家）。

流程：
  1) 训练专家（达标提前停止），生成 N 个示范回合
  2) 有监督训练 BC，报告训练精度与贪婪评估
  3) 两臂各微调 8 万步，比较首达 475 的步数与最终评估

运行：
    python code/ch018.py            # 完整实验，CPU 约 5-9 分钟
    python code/ch018.py --quick    # 快速跑通（约 30-60 秒）
    python code/ch018.py --demo-episodes 50

预期：BC 精度 85%～95%，闭环评估 200～450（复合误差）；
bc 臂首达 475 的步数明显少于 scratch 臂，最终评估持平或略高。
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
    return returns


# ---------------------------------------------------------------------------
# 训练一个 PPO 臂（可指定初始权重与预算）
# ---------------------------------------------------------------------------
def train_ppo(arm: str, init_state, args, budget: int):
    set_seed(args.seed)
    device = torch.device("cpu")
    env = gym.make("CartPole-v1")
    obs, _ = env.reset(seed=args.seed)
    net = ActorCritic(env.observation_space.shape[0], env.action_space.n)
    if init_state is not None:
        net.load_state_dict(init_state)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    all_returns, kls = [], []
    total_steps, n_updates, obs_t = 0, 0, to_tensor(obs, device)
    reached_at = None
    t0 = time.time()
    initial_eval = float(np.mean(evaluate(gym.make("CartPole-v1"), net, args.eval_episodes, device)))

    print(f"\n[微调] arm={arm} | 初始评估 {initial_eval:.1f} | 预算 {budget} 步")
    while total_steps < budget:
        batch, ep_returns, obs_t = collect_rollout(
            env, net, args.steps_per_update, obs_t, device, args.gamma, args.lam)
        total_steps += args.steps_per_update
        n_updates += 1
        all_returns.extend(ep_returns)

        stats = ppo_update(net, optimizer, batch, args)
        kls.append(stats["approx_kl"])

        if reached_at is None and len(all_returns) >= 100 and \
                np.mean(all_returns[-100:]) >= args.target_reward:
            reached_at = total_steps
            print(f"  [{arm}] 达到 {args.target_reward}：步数 {reached_at}")
            break

        if n_updates % 5 == 0 and len(all_returns) >= 20:
            print(f"  [{arm}] 步数 {total_steps:7d} | 回合 {len(all_returns):4d} | "
                  f"近20均 {np.mean(all_returns[-20:]):7.1f} | KL {stats['approx_kl']:.4f}")

    final_eval = float(np.mean(evaluate(gym.make("CartPole-v1"), net, args.eval_episodes, device)))
    return {
        "arm": arm,
        "initial_eval": initial_eval,
        "reached_at": reached_at,
        "steps": total_steps,
        "last100": float(np.mean(all_returns[-100:])) if len(all_returns) >= 100
        else float(np.mean(all_returns)),
        "final_eval": final_eval,
        "mean_kl": float(np.mean(kls)) if kls else 0.0,
        "seconds": time.time() - t0,
        "net": net,
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="第018章：PPO 模仿学习初始化")
    p.add_argument("--expert-budget", type=int, default=120000, help="专家训练步数上限")
    p.add_argument("--demo-episodes", type=int, default=30, help="采集的示范回合数")
    p.add_argument("--bc-epochs", type=int, default=80, help="行为克隆训练轮数")
    p.add_argument("--bc-lr", type=float, default=1e-3, help="行为克隆学习率")
    p.add_argument("--bc-batch", type=int, default=256, help="行为克隆批量大小")
    p.add_argument("--finetune-budget", type=int, default=80000, help="每个微调臂的步数预算")
    p.add_argument("--steps-per-update", type=int, default=2048, help="每次更新前采样步数")
    p.add_argument("--epochs", type=int, default=10, help="每批数据的更新轮数")
    p.add_argument("--batch-size", type=int, default=64, help="minibatch 大小")
    p.add_argument("--lr", type=float, default=3e-4, help="PPO 学习率")
    p.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
    p.add_argument("--lam", type=float, default=0.95, help="GAE 的 lambda")
    p.add_argument("--clip", type=float, default=0.2, help="PPO 裁剪范围")
    p.add_argument("--ent-coef", type=float, default=0.01, help="熵正则系数")
    p.add_argument("--vf-coef", type=float, default=0.5, help="价值损失系数")
    p.add_argument("--target-reward", type=float, default=475.0, help="专家与微调的达标阈值")
    p.add_argument("--eval-episodes", type=int, default=10, help="评估回合数")
    p.add_argument("--seed", type=int, default=0, help="随机种子")
    p.add_argument("--save-dir", type=str, default="runs/ch018", help="模型保存目录")
    p.add_argument("--quick", action="store_true", help="快速模式：专家 1.2 万步、BC 20 轮、微调 1 万步")
    return p.parse_args()


def collect_demos(env, expert, episodes: int, device):
    """用专家的随机策略采样，记录 (obs, action) 对；同时记录专家实际回合回报。"""
    obs_list, act_list = [], []
    ep_returns = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            obs_t = to_tensor(obs, device)
            a, _, _ = expert.act(obs_t)          # 按专家分布采样，增加状态覆盖
            obs_list.append(obs_t.squeeze(0).cpu().numpy())
            act_list.append(a)
            obs, r, terminated, truncated, _ = env.step(a)
            ep_ret += r
            done = terminated or truncated
        ep_returns.append(ep_ret)
    return np.asarray(obs_list, dtype=np.float32), np.asarray(act_list, dtype=np.int64), ep_returns


def behavior_clone(net, obs, act, args, device):
    """有监督训练策略头（价值头不参与 BC）。"""
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
    act_t = torch.as_tensor(act, dtype=torch.long, device=device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.bc_lr)
    n = obs_t.shape[0]
    for epoch in range(args.bc_epochs):
        idx = torch.randperm(n, device=device)
        total_loss, correct = 0.0, 0
        for start in range(0, n, args.bc_batch):
            mb = idx[start:start + args.bc_batch]
            logits, _ = net(obs_t[mb])
            loss = F.cross_entropy(logits, act_t[mb])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                total_loss += loss.item() * mb.shape[0]
                correct += (logits.argmax(dim=-1) == act_t[mb]).sum().item()
        if (epoch + 1) % 20 == 0 or epoch == args.bc_epochs - 1:
            print(f"  [BC] epoch {epoch + 1:3d} | 损失 {total_loss / n:.4f} | "
                  f"精度 {correct / n:.3f}")
    return total_loss / n, correct / n


def main():
    args = parse_args()
    if args.quick:
        args.expert_budget = 12000
        args.demo_episodes = 8
        args.bc_epochs = 20
        args.finetune_budget = 10000
        args.steps_per_update = 512
        args.epochs = 4
        args.target_reward = 300.0
        args.eval_episodes = 5

    device = torch.device("cpu")
    set_seed(args.seed)
    probe = gym.make("CartPole-v1")
    obs_dim, act_dim = probe.observation_space.shape[0], probe.action_space.n

    print("=" * 96)
    print(f"模仿学习初始化 | 专家预算 {args.expert_budget} | 示范 {args.demo_episodes} 回合 | "
          f"微调各 {args.finetune_budget} 步 | 种子 {args.seed}")
    print("=" * 96)

    # 1) 训练专家
    expert = ActorCritic(obs_dim, act_dim)
    expert_result = train_ppo("expert", None, args, args.expert_budget)
    expert = expert_result["net"]

    # 2) 采集示范
    demo_env = gym.make("CartPole-v1")
    demo_obs, demo_act, demo_returns = collect_demos(
        demo_env, expert, args.demo_episodes, device)
    print(f"\n[示范] {len(demo_obs)} 个状态-动作对 | 专家采样平均回报 "
          f"{np.mean(demo_returns):.1f}")

    # 3) 行为克隆
    bc_net = ActorCritic(obs_dim, act_dim)
    bc_loss, bc_acc = behavior_clone(bc_net, demo_obs, demo_act, args, device)
    bc_eval = float(np.mean(evaluate(gym.make("CartPole-v1"), bc_net, args.eval_episodes, device)))
    print(f"[BC] 训练精度 {bc_acc:.3f} | 闭环贪婪评估 {bc_eval:.1f}")
    bc_state = {k: v.clone() for k, v in bc_net.state_dict().items()}

    # 4) 两臂微调
    scratch_result = train_ppo("scratch", None, args, args.finetune_budget)
    bc_result = train_ppo("bc", bc_state, args, args.finetune_budget)

    # 5) 结果
    print("\n" + "=" * 96)
    print("模仿学习初始化对比（PPO 微调臂共享同一种子与预算）")
    print("-" * 96)
    print(f"专家：最后近100均 {expert_result['last100']:.1f} | 评估 "
          f"{expert_result['final_eval']:.1f} | 首达阈值 {expert_result['reached_at']}")
    print(f"BC  ：训练精度 {bc_acc:.3f} | 闭环评估 {bc_eval:.1f}（复合误差低于专家）")
    print(f"{'arm':>8} | {'初始评估':>8} | {'首达475步数':>11} | {'近100均':>8} | "
          f"{'最终评估':>8} | {'平均KL':>8} | {'用时(s)':>8}")
    for r in [scratch_result, bc_result]:
        reached = str(r["reached_at"]) if r["reached_at"] is not None else "未达到"
        print(f"{r['arm']:>8} | {r['initial_eval']:>8.1f} | {reached:>11} | {r['last100']:>8.1f} | "
              f"{r['final_eval']:>8.1f} | {r['mean_kl']:>8.4f} | {r['seconds']:>8.1f}")

    if scratch_result["reached_at"] and bc_result["reached_at"]:
        saved = scratch_result["reached_at"] - bc_result["reached_at"]
        print(f"\nBC 初始化节省的达标步数：{saved}（约 "
              f"{100.0 * saved / scratch_result['reached_at']:.0f}%）")

    print("\n观察提示：")
    print("  - BC 是从专家状态分布上拟合的，闭环执行会出现分布偏移（复合误差）；")
    print("  - BC 初始化让微调从更接近好状态的地方起步，通常更快达标；")
    print("  - 若 BC 精度很高但闭环很差，示范数据的状态覆盖不足（可多采回合）。")

    os.makedirs(args.save_dir, exist_ok=True)
    torch.save(expert.state_dict(), os.path.join(args.save_dir, "expert.pt"))
    torch.save(bc_state, os.path.join(args.save_dir, "bc_init.pt"))
    torch.save(scratch_result["net"].state_dict(), os.path.join(args.save_dir, "scratch.pt"))
    torch.save(bc_result["net"].state_dict(), os.path.join(args.save_dir, "bc_finetuned.pt"))
    path = os.path.join(args.save_dir, "imitation.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"expert_eval={expert_result['final_eval']:.2f} bc_acc={bc_acc:.3f} "
                f"bc_eval={bc_eval:.2f}\n")
        for r in [scratch_result, bc_result]:
            f.write(f"arm={r['arm']} initial_eval={r['initial_eval']:.2f} "
                    f"reached={r['reached_at']} last100={r['last100']:.2f} "
                    f"final_eval={r['final_eval']:.2f}\n")
    print(f"\n结果已写入 {path}")


if __name__ == "__main__":
    main()
