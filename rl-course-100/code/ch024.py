"""
第024章 PPO的分布式实现

教学简化版的"分布式"PPO：单进程 + 多环境（向量化采样）+ 共享网络。
真实的分布式 PPO 会把 actor 与 learner 拆到不同进程/机器上，
本章用"单进程多环境"复现它最关键的两件事：

  1) 同步屏障：一次更新前，N 个环境各采集 steps_per_env 步，
     所有环境到齐后才做一次 PPO 更新（等价于 mini-batch 式的数据并行）。
  2) 批量前向：N 个环境每步的观测拼成 (N, obs_dim) 张量，
     一次前向同时产出 N 个动作，显著摊薄网络调用开销。

对比实验：--workers-list 1,8
  - 两臂的总采样预算相同（--max-steps），每次更新的样本数也相同
  - 分别记录墙钟时间、吞吐量（步/秒）与学习曲线
  - 关键教学点：优势估计必须"按环境分段"做 GAE，绝不能跨环境边界递推

运行：
    python code/ch024.py            # workers=1 vs 8，各 12 万步（CPU 约 3-6 分钟）
    python code/ch024.py --quick    # 快速跑通（约 30-60 秒）
    python code/ch024.py --workers-list 1,4,8
    python code/ch024.py --workers-list 8 --max-steps 200000

预期：8 环境同步采样的吞吐通常是单环境的 1.5～3 倍（瓶颈从网络调用
转移到环境步进与 PPO 更新），学习曲线相当或略稳；请以实跑数据为准。
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
    """固定随机种子，保证实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# 网络：共享躯干 + 策略头 + 价值头
# ---------------------------------------------------------------------------
class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 64):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.actor = nn.Linear(hidden, act_dim)   # 输出离散动作的 logits
        self.critic = nn.Linear(hidden, 1)        # 输出状态价值 V(s)

    def forward(self, obs: torch.Tensor):
        h = self.body(obs)
        return self.actor(h), self.critic(h).squeeze(-1)


# ---------------------------------------------------------------------------
# 多环境运行器：N 个环境轮流推进，共享同一个策略
# ---------------------------------------------------------------------------
class VecRunner:
    """单进程多环境（教学简化）：N 个 gym 环境 + 每个环境的当前观测与累计回报。"""

    def __init__(self, env_id: str, n_envs: int, base_seed: int):
        self.n_envs = n_envs
        self.envs = [gym.make(env_id) for _ in range(n_envs)]
        self.obs = []
        for i, env in enumerate(self.envs):
            o, _ = env.reset(seed=base_seed + i)   # 每个环境用不同种子
            self.obs.append(o)
        self.ep_ret = np.zeros(n_envs, dtype=np.float64)
        self.obs_dim = self.envs[0].observation_space.shape[0]

    def step_all(self, actions: np.ndarray):
        """让所有环境各走一步。

        返回：
            rewards    : (N,) 本步奖励
            terminated : (N,) 是否真实终止（如杆子倒下）
            truncated  : (N,) 是否时间截断（如达到 500 步上限）
            finished   : 本步结束时完成的回合回报列表
        """
        rewards = np.zeros(self.n_envs, dtype=np.float32)
        terminated = np.zeros(self.n_envs, dtype=bool)
        truncated = np.zeros(self.n_envs, dtype=bool)
        finished = []
        for i, env in enumerate(self.envs):
            o, r, term, trunc, _ = env.step(int(actions[i]))
            rewards[i] = r
            self.ep_ret[i] += r
            if term or trunc:
                finished.append(float(self.ep_ret[i]))
                self.ep_ret[i] = 0.0
                o, _ = env.reset()                 # 该环境单独重置，不影响其他环境
            self.obs[i] = o
            terminated[i] = term
            truncated[i] = trunc
        return rewards, terminated, truncated, finished

    def close(self):
        for env in self.envs:
            env.close()


# ---------------------------------------------------------------------------
# GAE：按"单个环境的一条片段"递推
# ---------------------------------------------------------------------------
def compute_gae(rewards, values, terminateds, last_value, gamma: float, lam: float):
    """从后向前递推 GAE；reward/value/terminated 必须来自同一环境的时间序列。"""
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(T)):
        next_value = last_value if t == T - 1 else values[t + 1]
        non_terminal = 1.0 - terminateds[t]       # 真实终止时价值为 0
        delta = rewards[t] + gamma * next_value * non_terminal - values[t]
        last_gae = delta + gamma * lam * non_terminal * last_gae
        advantages[t] = last_gae
    returns = advantages + np.asarray(values, dtype=np.float32)
    return advantages, returns


# ---------------------------------------------------------------------------
# 同步采样：一次前向给全部环境选动作；优势按环境分段计算
# ---------------------------------------------------------------------------
@torch.no_grad()
def collect_vec_rollout(runner: VecRunner, net, steps_per_env: int, device,
                        gamma: float, lam: float):
    """采集 steps_per_env 步 × N 环境的数据，返回 batch 与本批完成的回合回报。"""
    n = runner.n_envs
    obs_seg = [[] for _ in range(n)]     # 每个环境一条时间序列
    act_seg = [[] for _ in range(n)]
    logp_seg = [[] for _ in range(n)]
    rew_seg = [[] for _ in range(n)]
    val_seg = [[] for _ in range(n)]
    term_seg = [[] for _ in range(n)]
    ep_returns = []

    for _ in range(steps_per_env):
        # 1) 批量前向：N 个环境的观测一次前向，摊薄网络开销
        obs_batch = torch.as_tensor(np.stack(runner.obs), dtype=torch.float32, device=device)
        logits, values = net(obs_batch)
        dist = Categorical(logits=logits)
        acts = dist.sample()
        logps = dist.log_prob(acts)

        # 2) 记录"步进前"的状态与动作信息
        for i in range(n):
            obs_seg[i].append(runner.obs[i])
            act_seg[i].append(int(acts[i].item()))
            logp_seg[i].append(float(logps[i].item()))
            val_seg[i].append(float(values[i].item()))

        # 3) 所有环境各走一步
        actions = acts.cpu().numpy()
        rewards, terminated, truncated, finished = runner.step_all(actions)
        for i in range(n):
            rew_seg[i].append(float(rewards[i]))
            # 只有真实终止才切断 bootstrap；时间截断仍用 V(s)
            term_seg[i].append(float(terminated[i]))
        ep_returns.extend(finished)

    # 4) 每个环境的序列末端各做一次 bootstrap 估值（批量前向）
    with torch.no_grad():
        obs_last = torch.as_tensor(np.stack(runner.obs), dtype=torch.float32, device=device)
        _, last_values = net(obs_last)
        last_values = last_values.cpu().numpy()

    # 5) 按环境分段做 GAE，再按"环境优先"的顺序拼接成一个大 batch
    adv_parts, ret_parts = [], []
    for i in range(n):
        adv_i, ret_i = compute_gae(rew_seg[i], val_seg[i], term_seg[i],
                                   float(last_values[i]), gamma, lam)
        adv_parts.append(adv_i)
        ret_parts.append(ret_i)
    advantages = np.concatenate(adv_parts, axis=0)
    returns = np.concatenate(ret_parts, axis=0)

    obs_flat = np.asarray([o for seg in obs_seg for o in seg], dtype=np.float32)
    act_flat = np.asarray([a for seg in act_seg for a in seg], dtype=np.int64)
    logp_flat = np.asarray([p for seg in logp_seg for p in seg], dtype=np.float32)

    batch = {
        "obs": torch.as_tensor(obs_flat, dtype=torch.float32, device=device),
        "act": torch.as_tensor(act_flat, dtype=torch.long, device=device),
        "logp_old": torch.as_tensor(logp_flat, dtype=torch.float32, device=device),
        "adv": torch.as_tensor(advantages, dtype=torch.float32, device=device),
        "ret": torch.as_tensor(returns, dtype=torch.float32, device=device),
    }
    return batch, ep_returns


# ---------------------------------------------------------------------------
# PPO 更新：裁剪目标 + 多轮小批量
# ---------------------------------------------------------------------------
def ppo_update(net, optimizer, batch, args):
    """在收集到的 batch 上做 epochs 轮小批量更新，返回平均诊断指标。"""
    obs, act = batch["obs"], batch["act"]
    logp_old, adv, ret = batch["logp_old"], batch["adv"], batch["ret"]
    N = obs.shape[0]

    adv = (adv - adv.mean()) / (adv.std() + 1e-8)  # 优势标准化

    stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
             "approx_kl": 0.0, "clip_frac": 0.0, "n_updates": 0}

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
                approx_kl = (logp_old[mb] - logp).mean().item()
                clip_frac = ((ratio - 1.0).abs() > args.clip).float().mean().item()
            stats["policy_loss"] += policy_loss.item()
            stats["value_loss"] += value_loss.item()
            stats["entropy"] += entropy.item()
            stats["approx_kl"] += approx_kl
            stats["clip_frac"] += clip_frac
            stats["n_updates"] += 1

    for k in stats:
        if k != "n_updates":
            stats[k] /= max(stats["n_updates"], 1)
    return stats


# ---------------------------------------------------------------------------
# 评估：贪婪策略
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(env, net, episodes: int, device):
    returns = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            logits, _ = net(obs_t)
            action = int(torch.argmax(logits, dim=-1).item())
            obs, r, terminated, truncated, _ = env.step(action)
            ep_ret += r
            done = terminated or truncated
        returns.append(ep_ret)
    return returns


# ---------------------------------------------------------------------------
# 单臂训练：workers 个环境的同步采样 PPO
# ---------------------------------------------------------------------------
def run_arm(workers: int, args, device):
    """训练一个 workers 配置，返回吞吐与学习结果。"""
    set_seed(args.seed)

    runner = VecRunner("CartPole-v1", workers, base_seed=args.seed)
    n_actions = runner.envs[0].action_space.n

    net = ActorCritic(runner.obs_dim, n_actions).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    # 每次更新共采集 steps_per_update 个样本，均分给各环境
    steps_per_env = max(1, args.steps_per_update // workers)
    samples_per_update = steps_per_env * workers

    all_returns = []
    total_steps = 0
    update_idx = 0
    solved_at = None
    t_collect = 0.0
    t_update = 0.0
    t_start = time.time()

    while total_steps < args.max_steps and solved_at is None:
        # 1) 同步采样：所有环境各走 steps_per_env 步（同步屏障的简化体现）
        t0 = time.time()
        batch, ep_returns = collect_vec_rollout(
            runner, net, steps_per_env, device, args.gamma, args.lam)
        t_collect += time.time() - t0
        total_steps += samples_per_update
        all_returns.extend(ep_returns)

        # 2) 中心化更新（所有环境到齐后才允许更新）
        t0 = time.time()
        stats = ppo_update(net, optimizer, batch, args)
        t_update += time.time() - t0
        update_idx += 1

        # 3) 日志
        if update_idx % args.log_every == 0 or update_idx == 1:
            recent = all_returns[-20:] if all_returns else [0.0]
            print(f"  [workers={workers}] 更新 {update_idx:3d} | 步数 {total_steps:7d} | "
                  f"回合 {len(all_returns):4d} | 近20均 {np.mean(recent):6.1f} | "
                  f"KL {stats['approx_kl']:.4f} | 裁剪比例 {stats['clip_frac']:.3f} | "
                  f"熵 {stats['entropy']:.3f}")

        if solved_at is None and len(all_returns) >= 100 and \
                np.mean(all_returns[-100:]) >= args.target_reward:
            solved_at = total_steps

    elapsed = time.time() - t_start
    runner.close()

    eval_env = gym.make("CartPole-v1")
    eval_returns = evaluate(eval_env, net, args.eval_episodes, device)
    eval_env.close()

    os.makedirs(args.save_dir, exist_ok=True)
    torch.save(net.state_dict(), os.path.join(args.save_dir, f"ppo_w{workers}.pt"))

    last100 = float(np.mean(all_returns[-100:])) if all_returns else 0.0
    result = {
        "workers": workers,
        "steps_per_env": steps_per_env,
        "samples_per_update": samples_per_update,
        "total_steps": total_steps,
        "time": elapsed,
        "steps_per_sec": total_steps / max(elapsed, 1e-6),
        "collect_time": t_collect,
        "update_time": t_update,
        "solved_at": solved_at,
        "last100": last100,
        "eval_mean": float(np.mean(eval_returns)),
    }
    print(f"  [workers={workers}] 结束：总步数 {total_steps} | 用时 {elapsed:.1f}s | "
          f"吞吐 {result['steps_per_sec']:.0f} 步/秒 | 采样耗时 {t_collect:.1f}s | "
          f"更新耗时 {t_update:.1f}s | 首达 "
          f"{('步数 ' + str(solved_at)) if solved_at else '未达成'} | "
          f"近100均 {last100:.1f} | 评估 {np.mean(eval_returns):.1f}")
    return result


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="第024章：同步多环境 PPO（分布式教学简化）")
    p.add_argument("--workers-list", type=str, default="1,8",
                   help="要对比的环境数量，逗号分隔（如 1,4,8）")
    p.add_argument("--max-steps", type=int, default=120000, help="每个臂的环境步数上限")
    p.add_argument("--steps-per-update", type=int, default=2048,
                   help="每次更新前所有环境采样的总步数（均分给各环境）")
    p.add_argument("--epochs", type=int, default=10, help="每批数据的更新轮数")
    p.add_argument("--batch-size", type=int, default=64, help="小批量大小")
    p.add_argument("--lr", type=float, default=3e-4, help="学习率")
    p.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
    p.add_argument("--lam", type=float, default=0.95, help="GAE 的 lambda")
    p.add_argument("--clip", type=float, default=0.2, help="PPO 裁剪范围")
    p.add_argument("--ent-coef", type=float, default=0.01, help="熵正则系数")
    p.add_argument("--vf-coef", type=float, default=0.5, help="价值损失系数")
    p.add_argument("--log-every", type=int, default=5, help="每隔多少次更新打印日志")
    p.add_argument("--target-reward", type=float, default=475.0, help="提前停的滑窗平均回报")
    p.add_argument("--eval-episodes", type=int, default=10, help="每个臂的评估回合数")
    p.add_argument("--seed", type=int, default=0, help="随机种子")
    p.add_argument("--save-dir", type=str, default="runs/ch024", help="模型保存目录")
    p.add_argument("--quick", action="store_true", help="快速模式：缩小训练规模，几十秒跑完")
    return p.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.max_steps = min(args.max_steps, 24000)
        args.steps_per_update = 768
        args.epochs = 5
        args.eval_episodes = 3

    workers_list = [int(x) for x in args.workers_list.split(",") if x.strip()]
    workers_list = [w for w in workers_list if w >= 1]
    if not workers_list:
        raise SystemExit("--workers-list 为空，请检查参数")

    device = torch.device("cpu")  # 纯 CPU
    print(f"设备: {device} | 环境: CartPole-v1 | 种子: {args.seed}")
    print(f"配置: workers={workers_list}, max_steps={args.max_steps}, "
          f"steps_per_update={args.steps_per_update}")

    results = []
    for workers in workers_list:
        print("-" * 80)
        print(f"开始训练：workers={workers}")
        results.append(run_arm(workers, args, device))

    # 吞吐汇总表：以第一个臂为基准算加速比
    base = results[0]
    print("=" * 92)
    print(f"{'workers':>8}{'每环境步数':>12}{'总步数':>10}{'用时(s)':>10}"
          f"{'步/秒':>10}{'加速比':>8}{'首达目标':>10}{'评估均':>9}")
    for r in results:
        solved = str(r["solved_at"]) if r["solved_at"] else "未达成"
        speedup = r["steps_per_sec"] / max(base["steps_per_sec"], 1e-6)
        print(f"{r['workers']:>8}{r['steps_per_env']:>12}{r['total_steps']:>10}"
              f"{r['time']:>10.1f}{r['steps_per_sec']:>10.0f}{speedup:>8.2f}"
              f"{solved:>10}{r['eval_mean']:>9.1f}")
    print("=" * 92)
    print("注意：这里加速的是采样与前向，PPO 更新本身并不并行；workers 越多，"
          "每次更新的样本里'每个环境的时间片段'越短，GAE 的可用视野也随之变短。")


if __name__ == "__main__":
    main()
