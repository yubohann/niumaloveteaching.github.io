"""
第078章 多智能体SAC：参数共享

同质智能体是否该共用一套 SAC？本章对照两种组织方式：
  - indep ：每个智能体一套 Actor + 双 Q + 自动温度（第 076 章）
  - shared：三个智能体共享一套 Actor 与一对 Q（把三份转移拼成一个大 batch 更新）
奖励统一使用每个智能体自己的奖励（团队项 + 自身碰撞惩罚），回放池与预算完全一致。

运行：
    python code/ch078.py           # 完整对照（CPU 约 4-8 分钟）
    python code/ch078.py --quick   # 快速跑通（约 40-80 秒）
    python code/ch078.py --mode shared

预期（以实跑为准）：任务对称，参数共享通常学得更快更稳；最终两者接近，但共享版
的参数量约为 1/3，且早期评估点领先。
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
# 环境（与前面章节相同）
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
# 回放池
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
            "next_obs": torch.as_tensor(self.next_obs[idx], device=device),
            "done": torch.as_tensor(self.done[idx], device=device),
        }


# ===========================================================================
# 网络与学习器
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


def sac_update(policy, q, q_target, policy_opt, q_opt, log_alpha, alpha_opt,
               obs, act, rew, next_obs, done, gamma, tau, target_entropy):
    """通用单网络 SAC 更新：批评家 -> 策略 -> 温度 -> 软更新。"""
    alpha = log_alpha.exp()
    with torch.no_grad():
        next_act, next_logp = policy.sample(next_obs)
        q1t, q2t = q_target(next_obs, next_act)
        target = rew.unsqueeze(-1) + gamma * (1.0 - done).unsqueeze(-1) * (
            torch.min(q1t, q2t) - alpha * next_logp)
    q1, q2 = q(obs, act)
    q_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
    q_opt.zero_grad()
    q_loss.backward()
    q_opt.step()

    new_act, logp = policy.sample(obs)
    q1n, q2n = q(obs, new_act)
    policy_loss = (alpha.detach() * logp - torch.min(q1n, q2n)).mean()
    policy_opt.zero_grad()
    policy_loss.backward()
    policy_opt.step()

    alpha_loss = -(log_alpha * (logp.detach() + target_entropy)).mean()
    alpha_opt.zero_grad()
    alpha_loss.backward()
    alpha_opt.step()

    with torch.no_grad():
        for p, p_t in zip(q.parameters(), q_target.parameters()):
            p_t.data.mul_(1.0 - tau).add_(tau * p.data)
    return {"q_loss": q_loss.item(), "policy_loss": policy_loss.item(),
            "alpha": float(alpha.item())}


class IndependentLearner:
    """独立学习器：一套完整 SAC（indep 模式）。"""

    def __init__(self, obs_dim, act_dim, args, device):
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

    def update(self, i, batch):
        return sac_update(self.policy, self.q, self.q_target, self.policy_opt,
                          self.q_opt, self.log_alpha, self.alpha_opt,
                          batch["obs"][:, i], batch["act"][:, i], batch["rew"][:, i],
                          batch["next_obs"][:, i], batch["done"],
                          self.gamma, self.tau, self.target_entropy)

    @property
    def alpha(self):
        return float(self.log_alpha.exp().item())


class SharedLearner:
    """共享学习器：一套 SAC 服务所有智能体（shared 模式）。"""

    def __init__(self, obs_dim, act_dim, args, device):
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

    def update(self, batch):
        """把 (B, N, ·) 展平成 (B·N, ·) 后做一次大更新。"""
        B, N = batch["obs"].shape[0], batch["obs"].shape[1]
        obs = batch["obs"].reshape(B * N, -1)
        act = batch["act"].reshape(B * N, -1)
        rew = batch["rew"].reshape(B * N)
        next_obs = batch["next_obs"].reshape(B * N, -1)
        done = batch["done"].unsqueeze(1).repeat(1, N).reshape(B * N)
        return sac_update(self.policy, self.q, self.q_target, self.policy_opt,
                          self.q_opt, self.log_alpha, self.alpha_opt,
                          obs, act, rew, next_obs, done,
                          self.gamma, self.tau, self.target_entropy)

    @property
    def alpha(self):
        return float(self.log_alpha.exp().item())


# ===========================================================================
# 评估与训练
# ===========================================================================
@torch.no_grad()
def evaluate(env, policies, episodes, device):
    """policies: 长度 N 的策略列表（共享模式下可以全是同一个对象）。"""
    rets, dists, colls = [], [], []
    for _ in range(episodes):
        obs = env.reset()
        done = False
        ep_ret = 0.0
        info = {"mean_dist": 0.0, "collisions": 0}
        while not done:
            actions = np.stack([
                policies[i].mean_action(torch.as_tensor(
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
    if mode == "shared":
        learner = SharedLearner(env.obs_dim, env.act_dim, args, device)
        learners, policies = [learner], [learner.policy] * n
        n_params = sum(p.numel() for m in [learner.policy, learner.q]
                       for p in m.parameters())
    else:
        learners = [IndependentLearner(env.obs_dim, env.act_dim, args, device)
                    for _ in range(n)]
        policies = [l.policy for l in learners]
        n_params = sum(p.numel() for m in
                       [l.policy for l in learners] + [l.q for l in learners]
                       for p in m.parameters())
    label = "SHARED" if mode == "shared" else "INDEP"
    print(f"[{label}] 可训练参数量 {n_params}")

    buffer = JointReplayBuffer(args.buffer_size, n, env.obs_dim, env.act_dim)
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
                    a = policies[i].sample(torch.as_tensor(
                        obs[i], dtype=torch.float32, device=device).unsqueeze(0))[0]
                actions.append(a.squeeze(0).cpu().numpy())
            actions = np.stack(actions)

        next_obs, per_rew, team_rew, done, info = env.step(actions)
        buffer.add(obs, actions, per_rew, team_rew, next_obs, float(done))
        obs = next_obs
        if done:
            obs = env.reset()

        if step > args.start_steps and step % args.update_every == 0:
            batch = buffer.sample(args.batch_size, device)
            if mode == "shared":
                learners[0].update(batch)
            else:
                for i in range(n):
                    learners[i].update(i, batch)

        if step % args.eval_every == 0:
            ret, dist, coll = evaluate(eval_env, policies, args.eval_episodes, device)
            curve.append((step, ret))
            if mode == "shared":
                alpha_mean = learners[0].alpha
            else:
                alpha_mean = float(np.mean([l.alpha for l in learners]))
            print(f"[{label} 步 {step:6d}] 评估回报 {ret:7.2f} | 平均末距 {dist:.3f} | "
                  f"碰撞 {coll:.2f}/回合 | α均值 {alpha_mean:.3f} | "
                  f"用时 {time.time() - t0:5.1f}s")

    elapsed = time.time() - t0
    ret, dist, coll = evaluate(eval_env, policies, args.eval_episodes * 2, device)
    print(f"[{label}] 完成：用时 {elapsed:.1f}s | 最终评估回报 {ret:.2f} | "
          f"末距 {dist:.3f} | 碰撞 {coll:.2f}")
    return learners, policies, curve, elapsed, (ret, dist, coll)


def parse_args():
    p = argparse.ArgumentParser(description="第078章：多智能体 SAC 的参数共享")
    p.add_argument("--mode", type=str, default="both", choices=["both", "indep", "shared"])
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
    p.add_argument("--save-dir", type=str, default="runs/ch078", help="保存目录")
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
    if args.mode in ("shared", "both"):
        results["SHARED"] = run_training("shared", args, device)

    print("=" * 76)
    for name, (learners, policies, curve, dt, final) in results.items():
        best = max(r for _, r in curve) if curve else float("nan")
        print(f"[{name:6s}] 用时 {dt:6.1f}s | 最佳评估 {best:7.2f} | "
              f"最终（回报/末距/碰撞） {final[0]:7.2f} / {final[1]:.3f} / {final[2]:.2f}")
    if len(results) == 2:
        print("提示：参数共享把 N 个智能体的数据都喂给同一套网络，样本效率更高；"
              "任务不对称时应改用条件共享（把角色信息拼进观测）。")

    os.makedirs(args.save_dir, exist_ok=True)
    for name, (learners, policies, curve, _, _) in results.items():
        if name == "SHARED":
            torch.save(learners[0].policy.state_dict(),
                       os.path.join(args.save_dir, "shared_policy.pt"))
        else:
            for i, l in enumerate(learners):
                torch.save(l.policy.state_dict(),
                           os.path.join(args.save_dir, f"indep_agent_{i}.pt"))
        path = os.path.join(args.save_dir, f"{name.lower()}_curve.txt")
        with open(path, "w", encoding="utf-8") as f:
            for step, ret in curve:
                f.write(f"{step}\t{ret:.3f}\n")
    print(f"模型与曲线已保存到 {args.save_dir}")


if __name__ == "__main__":
    main()
