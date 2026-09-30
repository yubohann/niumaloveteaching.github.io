"""
第076章 多智能体SAC：独立SAC

进入第五篇：把前面写过的多智能体机制迁移到 SAC 上。本章是起点——独立 SAC（ISAC）：
每个智能体拥有自己的 Actor、双 Q、目标网络与自动温度，共享一个联合经验回放池，
彼此把对方当作环境的一部分。同时对照两种奖励信号：
  - team：所有智能体的批评家都用团队奖励训练
  - own ：每个智能体用自己的奖励（团队项 + 自己的碰撞惩罚）
环境沿用第 068 章的合作导航（连续动作、对称二分匹配奖励）。

运行：
    python code/ch076.py           # 完整对照（CPU 约 4-8 分钟）
    python code/ch076.py --quick   # 快速跑通（约 40-80 秒）
    python code/ch076.py --reward own

预期（以实跑为准）：ISAC 在 2 万环境步内把评估回报从约 -25 分拉到 -8 ~ -4 分；
两种奖励信号下差异通常不大，own 可能在某些种子下更快（碰撞惩罚是个体化的）。
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
# 环境：合作导航（与第 068 章相同）
# ===========================================================================
class CooperativeNavEnv:
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

    def _min_matching_distance(self) -> float:
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
# 联合经验回放池：一次存 N 个智能体的转移
# ===========================================================================
class JointReplayBuffer:
    def __init__(self, capacity, n_agents, obs_dim, act_dim):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        self.obs = np.zeros((capacity, n_agents, obs_dim), np.float32)
        self.act = np.zeros((capacity, n_agents, act_dim), np.float32)
        self.rew = np.zeros((capacity, n_agents), np.float32)
        self.team_rew = np.zeros((capacity,), np.float32)
        self.next_obs = np.zeros((capacity, n_agents, obs_dim), np.float32)
        self.done = np.zeros((capacity,), np.float32)

    def add(self, obs, act, per_rew, team_rew, next_obs, done):
        i = self.ptr
        self.obs[i] = obs
        self.act[i] = act
        self.rew[i] = per_rew
        self.team_rew[i] = team_rew
        self.next_obs[i] = next_obs
        self.done[i] = done
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, device):
        idx = np.random.randint(0, self.size, size=batch_size)
        return {
            "obs": torch.as_tensor(self.obs[idx], device=device),
            "act": torch.as_tensor(self.act[idx], device=device),
            "rew": torch.as_tensor(self.rew[idx], device=device),
            "team_rew": torch.as_tensor(self.team_rew[idx], device=device),
            "next_obs": torch.as_tensor(self.next_obs[idx], device=device),
            "done": torch.as_tensor(self.done[idx], device=device),
        }


# ===========================================================================
# 单智能体 SAC 组件（与第 062 章同构）
# ===========================================================================
class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=64, act_limit=1.0):
        super().__init__()
        self.act_limit = act_limit
        self.body = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(),
                                  nn.Linear(hidden, hidden), nn.ReLU())
        self.mean = nn.Linear(hidden, act_dim)
        self.log_std = nn.Linear(hidden, act_dim)

    def forward(self, obs):
        h = self.body(obs)
        return self.mean(h), torch.clamp(self.log_std(h), -4.0, 0.5)

    def sample(self, obs):
        mean, log_std = self.forward(obs)
        dist = Normal(mean, log_std.exp())
        x = dist.rsample()
        action = torch.tanh(x)
        logp = dist.log_prob(x) - torch.log(1.0 - action.pow(2) + 1e-6)
        return action * self.act_limit, logp.sum(-1, keepdim=True)

    @torch.no_grad()
    def mean_action(self, obs):
        mean, _ = self.forward(obs)
        return torch.tanh(mean) * self.act_limit


class TwinQ(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=64):
        super().__init__()
        def make():
            return nn.Sequential(nn.Linear(obs_dim + act_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1))
        self.q1, self.q2 = make(), make()

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.q1(x), self.q2(x)


class IndependentSAC:
    """一个智能体的完整 SAC：策略 + 双 Q + 目标网络 + 自动温度。"""

    def __init__(self, obs_dim, act_dim, args, device):
        self.device = device
        self.policy = GaussianPolicy(obs_dim, act_dim, args.hidden).to(device)
        self.q = TwinQ(obs_dim, act_dim, args.hidden).to(device)
        self.q_target = TwinQ(obs_dim, act_dim, args.hidden).to(device)
        self.q_target.load_state_dict(self.q.state_dict())
        self.policy_opt = torch.optim.Adam(self.policy.parameters(), lr=args.lr)
        self.q_opt = torch.optim.Adam(self.q.parameters(), lr=args.lr)
        self.log_alpha = torch.tensor(np.log(args.alpha), requires_grad=True, device=device)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=args.lr)
        self.target_entropy = -float(act_dim)
        self.gamma, self.tau = args.gamma, args.tau

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def update(self, obs, act, rew, next_obs, done):
        """用单智能体的 (obs, act, rew, next_obs, done) 做一次完整更新。"""
        with torch.no_grad():
            next_act, next_logp = self.policy.sample(next_obs)
            q1t, q2t = self.q_target(next_obs, next_act)
            target = rew + self.gamma * (1.0 - done) * (
                torch.min(q1t, q2t) - self.alpha * next_logp)
        q1, q2 = self.q(obs, act)
        q_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.q_opt.zero_grad()
        q_loss.backward()
        self.q_opt.step()

        new_act, logp = self.policy.sample(obs)
        q1n, q2n = self.q(obs, new_act)
        policy_loss = (self.alpha.detach() * logp - torch.min(q1n, q2n)).mean()
        self.policy_opt.zero_grad()
        policy_loss.backward()
        self.policy_opt.step()

        alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        with torch.no_grad():
            for p, p_t in zip(self.q.parameters(), self.q_target.parameters()):
                p_t.data.mul_(1.0 - self.tau).add_(self.tau * p.data)
        return {"q_loss": q_loss.item(), "policy_loss": policy_loss.item(),
                "alpha": self.alpha.item()}


# ===========================================================================
# 评估
# ===========================================================================
@torch.no_grad()
def evaluate(env, agents, episodes, device):
    rets, dists, colls = [], [], []
    n = env.n_agents
    for _ in range(episodes):
        obs = env.reset()
        done = False
        ep_ret = 0.0
        info = {"mean_dist": 0.0, "collisions": 0}
        while not done:
            actions = np.stack([
                agents[i].policy.mean_action(
                    torch.as_tensor(obs[i], dtype=torch.float32, device=device).unsqueeze(0)
                ).squeeze(0).cpu().numpy() for i in range(n)])
            obs, _, team_r, done, info = env.step(actions)
            ep_ret += team_r
        rets.append(ep_ret)
        dists.append(info["mean_dist"])
        colls.append(info["collisions"])
    return float(np.mean(rets)), float(np.mean(dists)), float(np.mean(colls))


# ===========================================================================
# 训练
# ===========================================================================
def run_training(reward_mode, args, device):
    """reward_mode: team 或 own。"""
    set_seed(args.seed)
    env = CooperativeNavEnv(args.n_agents, args.episode_len, seed=args.seed)
    eval_env = CooperativeNavEnv(args.n_agents, args.episode_len, seed=args.seed + 1)
    n = env.n_agents
    agents = [IndependentSAC(env.obs_dim, env.act_dim, args, device) for _ in range(n)]
    buffer = JointReplayBuffer(args.buffer_size, n, env.obs_dim, env.act_dim)
    label = "TEAM" if reward_mode == "team" else "OWN"

    obs = env.reset()
    curve = []
    t0 = time.time()
    last_stats = {"alpha": args.alpha}

    for step in range(1, args.max_steps + 1):
        # ---- 选动作：预热期随机，之后用各自策略采样 ----
        if step <= args.start_steps:
            actions = np.random.uniform(-1.0, 1.0, (n, env.act_dim)).astype(np.float32)
        else:
            actions = []
            for i in range(n):
                with torch.no_grad():
                    a = agents[i].policy.sample(torch.as_tensor(
                        obs[i], dtype=torch.float32, device=device).unsqueeze(0))[0]
                actions.append(a.squeeze(0).cpu().numpy())
            actions = np.stack(actions)

        next_obs, per_rew, team_rew, done, info = env.step(actions)
        buffer.add(obs, actions, per_rew, team_rew, next_obs, float(done))
        obs = next_obs
        if done:
            obs = env.reset()

        # ---- 每个智能体各自更新 ----
        if step > args.start_steps and step % args.update_every == 0:
            batch = buffer.sample(args.batch_size, device)
            for i in range(n):
                rew_i = batch["team_rew"] if reward_mode == "team" else batch["rew"][:, i]
                last_stats = agents[i].update(
                    batch["obs"][:, i], batch["act"][:, i], rew_i,
                    batch["next_obs"][:, i], batch["done"])

        if step % args.eval_every == 0:
            ret, dist, coll = evaluate(eval_env, agents, args.eval_episodes, device)
            curve.append((step, ret))
            alphas = np.mean([float(a.alpha.item()) for a in agents])
            print(f"[{label} 步 {step:6d}] 评估回报 {ret:7.2f} | 平均末距 {dist:.3f} | "
                  f"碰撞 {coll:.2f}/回合 | α均值 {alphas:.3f} | "
                  f"用时 {time.time() - t0:5.1f}s")

    elapsed = time.time() - t0
    ret, dist, coll = evaluate(eval_env, agents, args.eval_episodes * 2, device)
    print(f"[{label}] 完成：用时 {elapsed:.1f}s | 最终评估回报 {ret:.2f} | "
          f"末距 {dist:.3f} | 碰撞 {coll:.2f}")
    return agents, curve, elapsed, (ret, dist, coll)


def parse_args():
    p = argparse.ArgumentParser(description="第076章：多智能体 SAC 之独立 SAC")
    p.add_argument("--reward", type=str, default="both", choices=["both", "team", "own"],
                   help="批评家使用的奖励信号")
    p.add_argument("--n-agents", type=int, default=3, help="智能体数量")
    p.add_argument("--episode-len", type=int, default=50, help="回合长度")
    p.add_argument("--max-steps", type=int, default=20000, help="环境步预算")
    p.add_argument("--update-every", type=int, default=1, help="每多少个环境步做一次更新")
    p.add_argument("--batch-size", type=int, default=128, help="采样批量")
    p.add_argument("--buffer-size", type=int, default=200000, help="回放池容量")
    p.add_argument("--start-steps", type=int, default=1000, help="随机预热步数")
    p.add_argument("--hidden", type=int, default=64, help="隐藏层宽度")
    p.add_argument("--lr", type=float, default=3e-4, help="学习率")
    p.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
    p.add_argument("--tau", type=float, default=0.005, help="目标网络软更新系数")
    p.add_argument("--alpha", type=float, default=0.2, help="温度初值")
    p.add_argument("--eval-every", type=int, default=4000, help="评估间隔（环境步）")
    p.add_argument("--eval-episodes", type=int, default=5, help="评估回合数")
    p.add_argument("--seed", type=int, default=0, help="随机种子")
    p.add_argument("--save-dir", type=str, default="runs/ch076", help="保存目录")
    p.add_argument("--quick", action="store_true", help="快速模式：5000 步/配置")
    return p.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.max_steps = min(args.max_steps, 5000)
        args.eval_every = 1500
        args.eval_episodes = 3
        args.start_steps = 300

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device} | 环境: 合作导航（{args.n_agents} 智能体）| 种子 {args.seed} | "
          f"预算 {args.max_steps} 环境步 | 更新频率 每 {args.update_every} 步")

    results = {}
    if args.reward in ("team", "both"):
        agents_t, curve_t, dt_t, final_t = run_training("team", args, device)
        results["TEAM"] = (curve_t, dt_t, final_t)
    if args.reward in ("own", "both"):
        agents_o, curve_o, dt_o, final_o = run_training("own", args, device)
        results["OWN"] = (curve_o, dt_o, final_o)

    print("=" * 76)
    for name, (curve, dt, final) in results.items():
        best = max(r for _, r in curve) if curve else float("nan")
        print(f"[{name:5s}] 用时 {dt:6.1f}s | 最佳评估 {best:7.2f} | "
              f"最终（回报/末距/碰撞） {final[0]:7.2f} / {final[1]:.3f} / {final[2]:.2f}")
    if len(results) == 2:
        print("提示：独立 SAC 每个智能体一条学习曲线；如果某个智能体先学会、"
              "另一个看似原地踏步，属于独立学习的正常现象，看团队指标。")

    os.makedirs(args.save_dir, exist_ok=True)
    for name, (curve, _, _) in results.items():
        path = os.path.join(args.save_dir, f"{name.lower()}_curve.txt")
        with open(path, "w", encoding="utf-8") as f:
            for step, ret in curve:
                f.write(f"{step}\t{ret:.3f}\n")
    if "TEAM" in results:
        for i, a in enumerate(agents_t):
            torch.save(a.policy.state_dict(), os.path.join(args.save_dir, f"team_agent_{i}.pt"))
    if "OWN" in results:
        for i, a in enumerate(agents_o):
            torch.save(a.policy.state_dict(), os.path.join(args.save_dir, f"own_agent_{i}.pt"))
    print(f"评估曲线与策略已保存到 {args.save_dir}")


if __name__ == "__main__":
    main()
