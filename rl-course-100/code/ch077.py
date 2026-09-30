"""
第077章 多智能体SAC：集中Q函数

在第 076 章独立 SAC 的基础上，把每个智能体的批评家升级为"集中 Q"：
  Q_i(s_global, a_1:N)
批评家能看到全局状态与所有智能体的联合动作，策略仍然只用局部观测执行。
对照两种配置：
  - indep  ：Q_i(o_i, a_i)（独立 SAC，第 076 章）
  - central：Q_i(s_global, a_1:N)（本章主角，MASAC 风格）
其余（环境、预算、网络宽度、评估协议）完全相同。

运行：
    python code/ch077.py           # 完整对照（CPU 约 5-10 分钟）
    python code/ch077.py --quick   # 快速跑通（约 40-80 秒）
    python code/ch077.py --mode central

预期（以实跑为准）：本任务较简单，两种配置最终水平接近；集中 Q 通常在中前期
更稳（协调期优势明显），后期收敛到相近的评估回报。
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
# 环境（与第 068/076 章相同）
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

    def global_state(self) -> np.ndarray:
        return np.concatenate([self.pos.ravel(), self.vel.ravel(),
                               self.landmarks.ravel(),
                               [self.t / self.episode_len]]).astype(np.float32)

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
# 联合回放池（带全局状态）
# ===========================================================================
class JointReplayBuffer:
    def __init__(self, capacity, n_agents, obs_dim, state_dim, act_dim):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        self.obs = np.zeros((capacity, n_agents, obs_dim), np.float32)
        self.act = np.zeros((capacity, n_agents, act_dim), np.float32)
        self.rew = np.zeros((capacity, n_agents), np.float32)
        self.team_rew = np.zeros((capacity,), np.float32)
        self.next_obs = np.zeros((capacity, n_agents, obs_dim), np.float32)
        self.state = np.zeros((capacity, state_dim), np.float32)
        self.next_state = np.zeros((capacity, state_dim), np.float32)
        self.done = np.zeros((capacity,), np.float32)

    def add(self, obs, act, per_rew, team_rew, next_obs, state, next_state, done):
        i = self.ptr
        self.obs[i] = obs
        self.act[i] = act
        self.rew[i] = per_rew
        self.team_rew[i] = team_rew
        self.next_obs[i] = next_obs
        self.state[i] = state
        self.next_state[i] = next_state
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
            "state": torch.as_tensor(self.state[idx], device=device),
            "next_state": torch.as_tensor(self.next_state[idx], device=device),
            "done": torch.as_tensor(self.done[idx], device=device),
        }


# ===========================================================================
# 网络
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


class LocalTwinQ(nn.Module):
    """独立 Q：输入单个智能体的观测与动作。"""

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


class JointTwinQ(nn.Module):
    """集中 Q：输入全局状态与所有智能体的联合动作。"""

    def __init__(self, state_dim, n_agents, act_dim, hidden=64):
        super().__init__()
        in_dim = state_dim + n_agents * act_dim
        def make():
            return nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1))
        self.q1, self.q2 = make(), make()

    def forward(self, state, joint_act):
        x = torch.cat([state, joint_act], dim=-1)
        return self.q1(x), self.q2(x)


# ===========================================================================
# 学习器：同一份代码支持 indep / central 两种批评家
# ===========================================================================
class Learner:
    def __init__(self, obs_dim, state_dim, n_agents, act_dim, args, device, mode):
        self.device = device
        self.mode = mode
        self.policy = GaussianPolicy(obs_dim, act_dim, args.hidden).to(device)
        if mode == "central":
            self.q = JointTwinQ(state_dim, n_agents, act_dim, args.hidden).to(device)
            self.q_target = JointTwinQ(state_dim, n_agents, act_dim, args.hidden).to(device)
        else:
            self.q = LocalTwinQ(obs_dim, act_dim, args.hidden).to(device)
            self.q_target = LocalTwinQ(obs_dim, act_dim, args.hidden).to(device)
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

    def soft_update(self):
        with torch.no_grad():
            for p, p_t in zip(self.q.parameters(), self.q_target.parameters()):
                p_t.data.mul_(1.0 - self.tau).add_(self.tau * p.data)

    def update(self, i, batch, peers, args):
        """更新第 i 个智能体；peers 是所有学习器的列表（用于取队友动作）。"""
        obs_i = batch["obs"][:, i]
        act_i = batch["act"][:, i]
        next_obs_i = batch["next_obs"][:, i]
        done = batch["done"]

        # ---- 批评家目标 ----
        with torch.no_grad():
            next_act_i, next_logp_i = self.policy.sample(next_obs_i)
            if self.mode == "central":
                # 下一状态上，所有智能体的动作都由各自当前策略采样
                next_acts = [next_act_i] * len(peers)
                for j, peer in enumerate(peers):
                    if j == i:
                        continue
                    a_j, _ = peer.policy.sample(batch["next_obs"][:, j])
                    next_acts[j] = a_j
                joint_next = torch.cat(next_acts, dim=-1)
                q1t, q2t = self.q_target(batch["next_state"], joint_next)
            else:
                q1t, q2t = self.q_target(next_obs_i, next_act_i)
            target = batch["rew"][:, i].unsqueeze(-1) + self.gamma * (1.0 - done).unsqueeze(-1) * (
                torch.min(q1t, q2t) - self.alpha * next_logp_i)

        # ---- 批评家更新 ----
        if self.mode == "central":
            joint_cur = batch["act"].reshape(batch["act"].shape[0], -1)
            q1, q2 = self.q(batch["state"], joint_cur)
        else:
            q1, q2 = self.q(obs_i, act_i)
        q_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.q_opt.zero_grad()
        q_loss.backward()
        self.q_opt.step()

        # ---- 策略更新：只对自己的动作求梯度，队友动作 detach ----
        if self.mode == "central":
            own_act, logp = self.policy.sample(obs_i)
            others = []
            for j, peer in enumerate(peers):
                if j == i:
                    others.append(own_act)
                else:
                    a_j, _ = peer.policy.sample(batch["obs"][:, j])
                    others.append(a_j.detach())
            joint_cur2 = torch.cat(others, dim=-1)
            q1n, q2n = self.q(batch["state"], joint_cur2)
        else:
            own_act, logp = self.policy.sample(obs_i)
            q1n, q2n = self.q(obs_i, own_act)
        policy_loss = (self.alpha.detach() * logp - torch.min(q1n, q2n)).mean()
        self.policy_opt.zero_grad()
        policy_loss.backward()
        self.policy_opt.step()

        # ---- 温度 ----
        alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        self.soft_update()
        return {"q_loss": q_loss.item(), "policy_loss": policy_loss.item(),
                "alpha": self.alpha.item()}


# ===========================================================================
# 评估与训练
# ===========================================================================
@torch.no_grad()
def evaluate(env, learners, episodes, device):
    rets, dists, colls = [], [], []
    for _ in range(episodes):
        obs = env.reset()
        done = False
        ep_ret = 0.0
        info = {"mean_dist": 0.0, "collisions": 0}
        while not done:
            actions = np.stack([
                learners[i].policy.mean_action(torch.as_tensor(
                    obs[i], dtype=torch.float32, device=device).unsqueeze(0)
                ).squeeze(0).cpu().numpy() for i in range(env.n_agents)])
            obs, _, team_r, done, info = env.step(actions)
            ep_ret += team_r
        rets.append(ep_ret)
        dists.append(info["mean_dist"])
        colls.append(info["collisions"])
    return float(np.mean(rets)), float(np.mean(dists)), float(np.mean(colls))


def run_training(mode, args, device):
    set_seed(args.seed)
    env = CooperativeNavEnv(args.n_agents, args.episode_len, seed=args.seed)
    eval_env = CooperativeNavEnv(args.n_agents, args.episode_len, seed=args.seed + 1)
    n = env.n_agents
    learners = [Learner(env.obs_dim, env.state_dim, n, env.act_dim, args, device, mode)
                for _ in range(n)]
    buffer = JointReplayBuffer(args.buffer_size, n, env.obs_dim, env.state_dim, env.act_dim)
    label = "CENTRAL" if mode == "central" else "INDEP"

    obs = env.reset()
    curve = []
    t0 = time.time()

    for step in range(1, args.max_steps + 1):
        if step <= args.start_steps:
            actions = np.random.uniform(-1.0, 1.0, (n, env.act_dim)).astype(np.float32)
        else:
            actions = []
            for i in range(n):
                with torch.no_grad():
                    a = learners[i].policy.sample(torch.as_tensor(
                        obs[i], dtype=torch.float32, device=device).unsqueeze(0))[0]
                actions.append(a.squeeze(0).cpu().numpy())
            actions = np.stack(actions)

        state = env.global_state()
        next_obs, per_rew, team_rew, done, info = env.step(actions)
        next_state = env.global_state()
        buffer.add(obs, actions, per_rew, team_rew, next_obs, state, next_state, float(done))
        obs = next_obs
        if done:
            obs = env.reset()

        if step > args.start_steps and step % args.update_every == 0:
            batch = buffer.sample(args.batch_size, device)
            for i in range(n):
                learners[i].update(i, batch, learners, args)

        if step % args.eval_every == 0:
            ret, dist, coll = evaluate(eval_env, learners, args.eval_episodes, device)
            curve.append((step, ret))
            alphas = np.mean([float(l.alpha.item()) for l in learners])
            print(f"[{label} 步 {step:6d}] 评估回报 {ret:7.2f} | 平均末距 {dist:.3f} | "
                  f"碰撞 {coll:.2f}/回合 | α均值 {alphas:.3f} | 用时 {time.time() - t0:5.1f}s")

    elapsed = time.time() - t0
    ret, dist, coll = evaluate(eval_env, learners, args.eval_episodes * 2, device)
    print(f"[{label}] 完成：用时 {elapsed:.1f}s | 最终评估回报 {ret:.2f} | "
          f"末距 {dist:.3f} | 碰撞 {coll:.2f}")
    return learners, curve, elapsed, (ret, dist, coll)


def parse_args():
    p = argparse.ArgumentParser(description="第077章：多智能体 SAC 的集中 Q 函数")
    p.add_argument("--mode", type=str, default="both", choices=["both", "indep", "central"])
    p.add_argument("--n-agents", type=int, default=3, help="智能体数量")
    p.add_argument("--episode-len", type=int, default=50, help="回合长度")
    p.add_argument("--max-steps", type=int, default=20000, help="环境步预算")
    p.add_argument("--update-every", type=int, default=1, help="每多少步更新一次")
    p.add_argument("--batch-size", type=int, default=128, help="采样批量")
    p.add_argument("--buffer-size", type=int, default=200000, help="回放池容量")
    p.add_argument("--start-steps", type=int, default=1000, help="随机预热步数")
    p.add_argument("--hidden", type=int, default=64, help="隐藏层宽度")
    p.add_argument("--lr", type=float, default=3e-4, help="学习率")
    p.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
    p.add_argument("--tau", type=float, default=0.005, help="软更新系数")
    p.add_argument("--alpha", type=float, default=0.2, help="温度初值")
    p.add_argument("--eval-every", type=int, default=4000, help="评估间隔")
    p.add_argument("--eval-episodes", type=int, default=5, help="评估回合数")
    p.add_argument("--seed", type=int, default=0, help="随机种子")
    p.add_argument("--save-dir", type=str, default="runs/ch077", help="保存目录")
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
    print(f"设备: {device} | 环境: 合作导航 | 种子 {args.seed} | "
          f"预算 {args.max_steps} 环境步/配置")

    results = {}
    if args.mode in ("indep", "both"):
        results["INDEP"] = run_training("indep", args, device)
    if args.mode in ("central", "both"):
        results["CENTRAL"] = run_training("central", args, device)

    print("=" * 76)
    for name, (learners, curve, dt, final) in results.items():
        best = max(r for _, r in curve) if curve else float("nan")
        print(f"[{name:7s}] 用时 {dt:6.1f}s | 最佳评估 {best:7.2f} | "
              f"最终（回报/末距/碰撞） {final[0]:7.2f} / {final[1]:.3f} / {final[2]:.2f}")
    if len(results) == 2:
        print("提示：集中 Q 用全局状态+联合动作估计价值，策略仍只用局部观测执行。"
              "如果两者差距不明显，说明本任务的局部观测已经足够充分。")

    os.makedirs(args.save_dir, exist_ok=True)
    for name, (learners, curve, _, _) in results.items():
        for i, l in enumerate(learners):
            torch.save(l.policy.state_dict(),
                       os.path.join(args.save_dir, f"{name.lower()}_agent_{i}.pt"))
        path = os.path.join(args.save_dir, f"{name.lower()}_curve.txt")
        with open(path, "w", encoding="utf-8") as f:
            for step, ret in curve:
                f.write(f"{step}\t{ret:.3f}\n")
    print(f"模型与曲线已保存到 {args.save_dir}")


if __name__ == "__main__":
    main()
