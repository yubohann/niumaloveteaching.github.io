"""
第068章 多智能体PPO：独立PPO

本章进入多智能体强化学习（MARL）。先用纯 numpy 实现一个轻量合作导航环境：
  - 3 个智能体、3 个地标随机分布在单位方格内
  - 奖励 = 负的"二分匹配最小总距离"，任务对称（谁去哪个地标都行）
  - 碰撞有惩罚，鼓励智能体彼此避让
然后实现最朴素的基线——独立 PPO（IPPO）：每个智能体把其他智能体当作环境的一
部分，各自维护 Actor-Critic、各自估计优势、各自更新。

运行：
    python code/ch068.py           # 完整训练（CPU 约 3-8 分钟）
    python code/ch068.py --quick   # 快速跑通（约 30-60 秒）

预期（以实跑为准）：随机策略约 -25 分；训练后 IPPO 的评估回报约 -6 ~ -3 分，
平均末距降到 0.05~0.12，碰撞接近 0。各智能体的学习速度会有先后差异。
"""

import argparse
import itertools
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ===========================================================================
# 自研多智能体环境：合作导航（纯 numpy，无任何多智能体库依赖）
# ===========================================================================
class CooperativeNavEnv:
    """3+ 智能体合作导航环境。

    观测（每个智能体，14 维）= 自身位置(2) + 自身速度(2) + 各地标相对位置(2N)
                              + 其他智能体相对位置(2(N-1))
    动作：连续 2 维力，范围 [-1, 1]
    奖励：团队奖励 = -二分匹配最小总距离/N；碰撞双方各扣 0.3
    回合：固定 episode_len 步截断
    """

    def __init__(self, n_agents=3, episode_len=50, damping=0.9, accel=0.2,
                 collision_dist=0.1, collision_penalty=0.3, seed=0):
        self.n_agents = n_agents
        self.episode_len = episode_len
        self.damping = damping
        self.accel = accel
        self.collision_dist = collision_dist
        self.collision_penalty = collision_penalty
        self.obs_dim = 4 + 2 * n_agents + 2 * (n_agents - 1)
        self.state_dim = 6 * n_agents + 1
        self.act_dim = 2
        self._rng = np.random.RandomState(seed)
        self._perms = list(itertools.permutations(range(n_agents)))
        self.pos = np.zeros((n_agents, 2), np.float32)
        self.vel = np.zeros((n_agents, 2), np.float32)
        self.landmarks = np.zeros((n_agents, 2), np.float32)
        self.t = 0

    def reset(self):
        self.pos = self._rng.uniform(0.15, 0.85, (self.n_agents, 2)).astype(np.float32)
        self.vel = np.zeros((self.n_agents, 2), np.float32)
        self.landmarks = self._rng.uniform(0.10, 0.90, (self.n_agents, 2)).astype(np.float32)
        self.t = 0
        return self._obs()

    def _obs(self):
        obs = np.zeros((self.n_agents, self.obs_dim), np.float32)
        for i in range(self.n_agents):
            rel_land = (self.landmarks - self.pos[i]).ravel()
            rel_others = [(self.pos[j] - self.pos[i]) for j in range(self.n_agents) if j != i]
            others = np.concatenate(rel_others) if rel_others else np.zeros(0, np.float32)
            obs[i] = np.concatenate([self.pos[i], self.vel[i], rel_land, others])
        return obs

    def global_state(self) -> np.ndarray:
        """集中式观察：所有智能体位置/速度 + 所有地标位置 + 归一化时间。"""
        return np.concatenate([self.pos.ravel(), self.vel.ravel(),
                               self.landmarks.ravel(),
                               [self.t / self.episode_len]]).astype(np.float32)

    def _min_matching_distance(self) -> float:
        """枚举全部配对，返回最小总距离的平均值（任务对称的奖励核心）。"""
        best = np.inf
        for perm in self._perms:
            total = 0.0
            for i, j in enumerate(perm):
                total += float(np.linalg.norm(self.pos[i] - self.landmarks[j]))
            if total < best:
                best = total
        return best / self.n_agents

    def step(self, actions):
        actions = np.clip(np.asarray(actions, dtype=np.float32), -1.0, 1.0)
        self.vel = self.damping * self.vel + self.accel * actions
        self.pos = np.clip(self.pos + self.vel, 0.0, 1.0)
        self.t += 1

        mean_dist = self._min_matching_distance()
        per_reward = np.full(self.n_agents, -mean_dist, dtype=np.float32)
        collisions = 0
        for i in range(self.n_agents):
            for j in range(i + 1, self.n_agents):
                if np.linalg.norm(self.pos[i] - self.pos[j]) < self.collision_dist:
                    per_reward[i] -= self.collision_penalty
                    per_reward[j] -= self.collision_penalty
                    collisions += 1
        done = self.t >= self.episode_len
        info = {"mean_dist": mean_dist, "collisions": collisions}
        return self._obs(), per_reward, float(per_reward.mean()), done, info


# ===========================================================================
# 网络与 GAE
# ===========================================================================
class ActorCritic(nn.Module):
    """单个智能体的 Actor-Critic：共享躯干 + 高斯策略 + 价值头。"""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 64):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(),
                                  nn.Linear(hidden, hidden), nn.Tanh())
        self.actor_mean = nn.Linear(hidden, act_dim)
        self.actor_log_std = nn.Parameter(torch.full((act_dim,), -0.7))
        self.critic = nn.Linear(hidden, 1)

    def forward(self, obs):
        h = self.body(obs)
        mean = self.actor_mean(h)
        log_std = torch.clamp(self.actor_log_std, -4.0, 0.5).expand_as(mean)
        return mean, log_std, self.critic(h).squeeze(-1)

    @torch.no_grad()
    def act(self, obs_t):
        """obs_t: (N, obs_dim) -> 每行采样一个动作，返回 (actions, logps, values)。"""
        mean, log_std, value = self.forward(obs_t)
        dist = Normal(mean, log_std.exp())
        raw = dist.sample()
        action = torch.clamp(raw, -1.0, 1.0)
        logp = dist.log_prob(action).sum(-1)
        return action, logp, value

    @torch.no_grad()
    def mean_action(self, obs_t):
        mean, _, _ = self.forward(obs_t)
        return torch.clamp(mean, -1.0, 1.0)

    @torch.no_grad()
    def value(self, obs_t):
        _, _, v = self.forward(obs_t)
        return v


def compute_gae(rewards, values, cuts, last_value, gamma, lam):
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(T)):
        next_value = last_value if t == T - 1 else values[t + 1]
        non_terminal = 1.0 - cuts[t]
        delta = rewards[t] + gamma * next_value * non_terminal - values[t]
        last_gae = delta + gamma * lam * non_terminal * last_gae
        adv[t] = last_gae
    return adv, adv + np.asarray(values, dtype=np.float32)


# ===========================================================================
# 独立 PPO：每个智能体一套网络与优化器
# ===========================================================================
class IPPO:
    def __init__(self, obs_dim, act_dim, n_agents, args, device):
        self.n_agents = n_agents
        self.device = device
        self.nets = [ActorCritic(obs_dim, act_dim, args.hidden).to(device)
                     for _ in range(n_agents)]
        self.optims = [torch.optim.Adam(net.parameters(), lr=args.lr) for net in self.nets]

    @torch.no_grad()
    def act_all(self, obs: np.ndarray):
        """整批前向：一次算出所有智能体的动作、对数概率与价值。"""
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        actions, logps, values = [], [], []
        for i, net in enumerate(self.nets):
            a, lp, v = net.act(obs_t[i:i + 1])
            actions.append(a.squeeze(0).cpu().numpy())
            logps.append(float(lp.item()))
            values.append(float(v.item()))
        return np.asarray(actions, np.float32), logps, values

    @torch.no_grad()
    def mean_actions_all(self, obs: np.ndarray):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        acts = []
        for i, net in enumerate(self.nets):
            acts.append(net.mean_action(obs_t[i:i + 1]).squeeze(0).cpu().numpy())
        return np.asarray(acts, np.float32)

    def update(self, batches, args) -> dict:
        """batches: 每个智能体一个字典，包含 obs/act/logp_old/adv/ret。"""
        stats = {"policy_loss": 0.0, "value_loss": 0.0, "approx_kl": 0.0,
                 "clip_frac": 0.0, "n": 0}
        for i in range(self.n_agents):
            b = batches[i]
            adv = (b["adv"] - b["adv"].mean()) / (b["adv"].std() + 1e-8)
            N = b["obs"].shape[0]
            net, optim = self.nets[i], self.optims[i]
            for _ in range(args.epochs):
                idx = torch.randperm(N, device=self.device)
                for start in range(0, N, args.batch_size):
                    mb = idx[start:start + args.batch_size]
                    mean, log_std, value = net(b["obs"][mb])
                    dist = Normal(mean, log_std.exp())
                    logp = dist.log_prob(b["act"][mb]).sum(-1)
                    ratio = torch.exp(logp - b["logp_old"][mb])
                    surr1 = ratio * adv[mb]
                    surr2 = torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * adv[mb]
                    policy_loss = -torch.min(surr1, surr2).mean()
                    value_loss = F.mse_loss(value, b["ret"][mb])
                    loss = policy_loss + args.vf_coef * value_loss
                    optim.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                    optim.step()
                    with torch.no_grad():
                        stats["policy_loss"] += policy_loss.item()
                        stats["value_loss"] += value_loss.item()
                        stats["approx_kl"] += (b["logp_old"][mb] - logp).mean().item()
                        stats["clip_frac"] += ((ratio - 1.0).abs() > args.clip).float().mean().item()
                        stats["n"] += 1
        for k in stats:
            if k != "n":
                stats[k] /= max(stats["n"], 1)
        return stats


# ===========================================================================
# 采样、评估与主循环
# ===========================================================================
def collect_rollout(env, agents, steps_agent, args, device, obs):
    """采样 steps_agent 个"智能体步"（即 steps_agent / N 个环境步）。"""
    n = env.n_agents
    buf = [{"obs": [], "act": [], "logp": [], "rew": [], "val": [], "cut": []}
           for _ in range(n)]
    team_rets = []
    ep_team_ret = 0.0
    env_steps = steps_agent // n
    for _ in range(env_steps):
        actions, logps, values = agents.act_all(obs)
        next_obs, per_r, team_r, done, _ = env.step(actions)
        for i in range(n):
            buf[i]["obs"].append(obs[i])
            buf[i]["act"].append(actions[i])
            buf[i]["logp"].append(logps[i])
            buf[i]["rew"].append(per_r[i])
            buf[i]["val"].append(values[i])
            buf[i]["cut"].append(float(done))
        ep_team_ret += team_r
        obs = next_obs
        if done:
            team_rets.append(ep_team_ret)
            ep_team_ret = 0.0
            obs = env.reset()

    # 每个智能体用自己的批评家估计 bootstrap 价值，再做 GAE
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
    batches = []
    for i in range(n):
        last_value = float(agents.nets[i].value(obs_t[i:i + 1]).item())
        adv, ret = compute_gae(buf[i]["rew"], buf[i]["val"], buf[i]["cut"],
                               last_value, args.gamma, args.lam)
        batches.append({
            "obs": torch.as_tensor(np.asarray(buf[i]["obs"]), dtype=torch.float32, device=device),
            "act": torch.as_tensor(np.asarray(buf[i]["act"]), dtype=torch.float32, device=device),
            "logp_old": torch.as_tensor(np.asarray(buf[i]["logp"]), dtype=torch.float32, device=device),
            "adv": torch.as_tensor(adv, device=device),
            "ret": torch.as_tensor(ret, device=device),
        })
    return batches, team_rets, obs


@torch.no_grad()
def evaluate(env, agents, episodes, device):
    """用确定性动作评估：返回平均回报、平均末距、平均碰撞。"""
    rets, dists, colls = [], [], []
    for _ in range(episodes):
        obs = env.reset()
        done = False
        ep_ret = 0.0
        info = {"mean_dist": 0.0, "collisions": 0}
        while not done:
            actions = agents.mean_actions_all(obs)
            obs, _, team_r, done, info = env.step(actions)
            ep_ret += team_r
        rets.append(ep_ret)
        dists.append(info["mean_dist"])
        colls.append(info["collisions"])
    return float(np.mean(rets)), float(np.mean(dists)), float(np.mean(colls))


def parse_args():
    p = argparse.ArgumentParser(description="第068章：多智能体 PPO 之独立 PPO")
    p.add_argument("--n-agents", type=int, default=3, help="智能体数量")
    p.add_argument("--episode-len", type=int, default=50, help="每个回合的环境步数")
    p.add_argument("--max-steps", type=int, default=180000, help="总智能体步数预算")
    p.add_argument("--steps-per-update", type=int, default=3000, help="每次更新前采样的智能体步数")
    p.add_argument("--eval-every", type=int, default=30000, help="每多少智能体步评估一次")
    p.add_argument("--epochs", type=int, default=8, help="每批数据的更新轮数")
    p.add_argument("--batch-size", type=int, default=256, help="小批量大小")
    p.add_argument("--hidden", type=int, default=64, help="隐藏层宽度")
    p.add_argument("--lr", type=float, default=3e-4, help="学习率")
    p.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
    p.add_argument("--lam", type=float, default=0.95, help="GAE 的 lambda")
    p.add_argument("--clip", type=float, default=0.2, help="PPO 裁剪范围")
    p.add_argument("--vf-coef", type=float, default=0.5, help="价值损失系数")
    p.add_argument("--eval-episodes", type=int, default=5, help="每次评估的回合数")
    p.add_argument("--seed", type=int, default=0, help="随机种子")
    p.add_argument("--save-dir", type=str, default="runs/ch068", help="模型保存目录")
    p.add_argument("--quick", action="store_true", help="快速模式：缩小规模，几十秒跑完")
    return p.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.max_steps = min(args.max_steps, 48000)
        args.steps_per_update = 1500
        args.eval_every = 12000
        args.epochs = 5
        args.eval_episodes = 3

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = CooperativeNavEnv(n_agents=args.n_agents, episode_len=args.episode_len,
                            seed=args.seed)
    eval_env = CooperativeNavEnv(n_agents=args.n_agents, episode_len=args.episode_len,
                                 seed=args.seed + 1)
    agents = IPPO(env.obs_dim, env.act_dim, env.n_agents, args, device)

    print(f"设备: {device} | 环境: 自研合作导航（{env.n_agents} 智能体）| 种子 {args.seed}")
    print(f"观测维度 {env.obs_dim}（每个智能体）| 动作维度 {env.act_dim} | "
          f"回合长度 {env.episode_len} | 预算 {args.max_steps} 智能体步")

    obs = env.reset()
    total_steps = 0
    update_idx = 0
    all_team_rets = []
    t0 = time.time()
    next_eval = args.eval_every

    # 随机策略基线
    rand_ret, rand_dist, rand_coll = evaluate(eval_env, agents, 3, device)
    print(f"[基线] 未训练网络的随机策略：回报 {rand_ret:.1f} | 末距 {rand_dist:.3f} | "
          f"碰撞 {rand_coll:.1f}（网络随机初始化）")

    while total_steps < args.max_steps:
        batches, team_rets, obs = collect_rollout(
            env, agents, args.steps_per_update, args, device, obs)
        total_steps += args.steps_per_update
        stats = agents.update(batches, args)
        all_team_rets.extend(team_rets)
        update_idx += 1

        recent = np.mean(all_team_rets[-20:]) if all_team_rets else float("nan")
        print(f"[更新 {update_idx:3d}] 智能体步 {total_steps:7d} | 回合 {len(all_team_rets):4d} | "
              f"近20回合团队回报 {recent:7.2f} | KL {stats['approx_kl']:.4f} | "
              f"裁剪 {stats['clip_frac']:.3f} | 用时 {time.time() - t0:5.1f}s")

        if total_steps >= next_eval:
            ret, dist, coll = evaluate(eval_env, agents, args.eval_episodes, device)
            print(f"    [评估] 回报 {ret:7.2f} | 平均末距 {dist:.3f} | "
                  f"平均碰撞 {coll:.2f}/回合")
            next_eval += args.eval_every

    elapsed = time.time() - t0
    ret, dist, coll = evaluate(eval_env, agents, args.eval_episodes * 2, device)
    print("-" * 72)
    print(f"训练结束 | 用时 {elapsed:.1f}s | 总智能体步 {total_steps} | "
          f"训练回合数 {len(all_team_rets)}")
    print(f"最终评估（确定性动作）：回报 {ret:.2f} | 平均末距 {dist:.3f} | "
          f"平均碰撞 {coll:.2f}/回合")

    os.makedirs(args.save_dir, exist_ok=True)
    for i, net in enumerate(agents.nets):
        torch.save(net.state_dict(), os.path.join(args.save_dir, f"agent_{i}.pt"))
    print(f"每个智能体的模型已保存到 {args.save_dir}")


if __name__ == "__main__":
    main()
