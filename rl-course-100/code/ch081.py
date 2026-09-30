"""
第081章 多智能体SAC：信用分配

第五篇收尾：在离散动作（5 个）的合作导航上，给共享 SAC 的演员更新引入
"信用信号"，对照两种方案：
  - shared：逐动作反事实 Q 减去"随机联合动作的平均 Q"（全队共用基线）
  - coma  ：逐动作反事实 Q 减去"只替换自己动作求期望"的基线（逐智能体）
批评家是集中式联合双 Q Q(s, a_1..a_N)，用 SARSA 目标训练；演员与温度按
SAC 的熵最大化准则更新（无重参数化，直接对分类分布求梯度）。

运行：
    python code/ch081.py           # 完整对照（CPU 约 3-6 分钟）
    python code/ch081.py --quick   # 快速跑通（约 30-60 秒）
    python code/ch081.py --credit coma

预期（以实跑为准）：COMA 基线的逐智能体信用信号让协调行为出现得更早、更稳定；
shared 基线实现更省（只需采样若干联合动作），但把功劳混在一起。
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
from torch.distributions import Categorical


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ===========================================================================
# 离散动作版合作导航环境（与第 073 章同构）
# ===========================================================================
class DiscreteNavEnv:
    """5 动作（停/上下左右）合作导航，对称二分匹配奖励 + 碰撞惩罚。"""

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
        self.n_actions = 5
        self._dirs = np.array([[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]], np.float32)
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
        actions = np.asarray(actions, dtype=np.int64)
        forces = self._dirs[actions]
        self.vel = self.damping * self.vel + self.accel * forces
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
# 网络：共享离散策略 + 联合双 Q
# ===========================================================================
class DiscretePolicy(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=64):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(),
                                  nn.Linear(hidden, hidden), nn.Tanh())
        self.logits = nn.Linear(hidden, n_actions)

    def forward(self, obs):
        return self.logits(self.body(obs))

    @torch.no_grad()
    def sample(self, obs):
        return Categorical(logits=self.forward(obs)).sample()

    @torch.no_grad()
    def greedy(self, obs):
        return torch.argmax(self.forward(obs), dim=-1)


class JointTwinQ(nn.Module):
    def __init__(self, state_dim, n_agents, n_actions, hidden=128):
        super().__init__()
        in_dim = state_dim + n_agents * n_actions
        self.in_dim = in_dim

        def make():
            return nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1))
        self.q1, self.q2 = make(), make()

    def forward(self, state, joint_onehot):
        x = torch.cat([state, joint_onehot], dim=-1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)


def joint_onehot(actions: torch.Tensor, n_agents: int, n_actions: int) -> torch.Tensor:
    """(T, N) 动作索引 -> (T, N·A) one-hot 拼接。"""
    T = actions.shape[0]
    out = torch.zeros(T, n_agents, n_actions, device=actions.device)
    out.scatter_(2, actions.unsqueeze(-1), 1.0)
    return out.view(T, n_agents * n_actions)


def counterfactual_alternatives(actions: torch.Tensor, n_agents: int,
                                n_actions: int) -> torch.Tensor:
    """(T, N) -> (T, N, A, N·A)：只替换第 i 个智能体的动作块。"""
    T = actions.shape[0]
    device = actions.device
    base = joint_onehot(actions, n_agents, n_actions)
    alts = base.unsqueeze(1).unsqueeze(2).repeat(1, n_agents, n_actions, 1)
    eye = torch.eye(n_actions, device=device)
    for i in range(n_agents):
        block = alts[:, i, :, i * n_actions:(i + 1) * n_actions]
        block.copy_(eye.unsqueeze(0).expand(T, -1, -1))
    return alts


# ===========================================================================
# 训练器
# ===========================================================================
class Trainer:
    def __init__(self, env, args, device, credit):
        self.n = env.n_agents
        self.A = env.n_actions
        self.device = device
        self.credit = credit
        self.policy = DiscretePolicy(env.obs_dim, env.n_actions, args.hidden).to(device)
        self.q = JointTwinQ(env.state_dim, env.n_agents, env.n_actions,
                            args.q_hidden).to(device)
        self.q_target = JointTwinQ(env.state_dim, env.n_agents, env.n_actions,
                                   args.q_hidden).to(device)
        self.q_target.load_state_dict(self.q.state_dict())
        self.policy_opt = torch.optim.Adam(self.policy.parameters(), lr=args.lr)
        self.q_opt = torch.optim.Adam(self.q.parameters(), lr=args.lr)
        self.log_alpha = torch.tensor(np.log(args.alpha), requires_grad=True, device=device)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=args.lr)
        self.target_entropy = -float(env.n_actions) * 0.5   # 分类分布的一半熵作为目标
        self.gamma, self.tau = args.gamma, args.tau

    @property
    def alpha(self):
        return float(self.log_alpha.exp().item())

    @torch.no_grad()
    def act_all(self, obs):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        return self.policy.sample(obs_t).cpu().numpy()

    @torch.no_grad()
    def greedy_all(self, obs):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        return self.policy.greedy(obs_t).cpu().numpy()

    def update(self, batch, args):
        device = self.device
        B, N, A = batch["obs"].shape[0], self.n, self.A
        state = torch.as_tensor(batch["state"], device=device)
        next_state = torch.as_tensor(batch["next_state"], device=device)
        act = torch.as_tensor(batch["act"], device=device)
        next_act = torch.as_tensor(batch["next_act"], device=device)
        team_rew = torch.as_tensor(batch["team_rew"], device=device)
        done = torch.as_tensor(batch["done"], device=device)
        obs = torch.as_tensor(batch["obs"], device=device)

        joint = joint_onehot(act, N, A)
        next_joint = joint_onehot(next_act, N, A)

        # ---- 联合双 Q：SARSA 目标，软更新目标网络 ----
        with torch.no_grad():
            q1t, q2t = self.q_target(next_state, next_joint)
            target = team_rew + self.gamma * (1.0 - done) * torch.min(q1t, q2t)
        q1, q2 = self.q(state, joint)
        q_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.q_opt.zero_grad()
        q_loss.backward()
        self.q_opt.step()
        with torch.no_grad():
            for p, p_t in zip(self.q.parameters(), self.q_target.parameters()):
                p_t.data.mul_(1.0 - self.tau).add_(self.tau * p.data)

        # ---- 反事实 Q：对每个智能体枚举其全部动作 ----
        with torch.no_grad():
            alts = counterfactual_alternatives(act, N, A)          # (B,N,A,N·A)
            state_rep = state.repeat_interleave(N * A, dim=0)
            q1a, q2a = self.q(state_rep, alts.reshape(B * N * A, N * A))
            q_cf = torch.min(q1a, q2a).reshape(B, N, A)            # 反事实 Q
            if self.credit == "coma":
                logits = self.policy(obs.reshape(B * N, -1))
                probs = torch.softmax(logits, dim=-1).reshape(B, N, A)
                baseline = (probs * q_cf).sum(dim=-1, keepdim=True)  # (B,N,1)
                adv = q_cf - baseline                                # 逐智能体信用
            else:
                k = args.baseline_samples
                rand_acts = torch.randint(0, A, (B, k, N), device=device)
                rand_joint = torch.zeros(B, k, N * A, device=device)
                idx = (torch.arange(N, device=device)[None, None, :] * A + rand_acts)
                rand_joint.scatter_(2, idx, 1.0)
                q1r, q2r = self.q(state.repeat_interleave(k, dim=0),
                                  rand_joint.reshape(B * k, N * A))
                q_rand = torch.min(q1r, q2r).reshape(B, k)
                baseline = q_rand.mean(dim=1, keepdim=True)          # (B,1)
                adv = q_cf - baseline.unsqueeze(1)                   # 全队共享基线
            adv_std = float(adv.std(dim=1).mean().item())

        # ---- 演员：最大化 Σ_a π(a)·Â(a) + α·H(π) ----
        obs_flat = obs.reshape(B * N, -1)
        adv_flat = adv.reshape(B * N, A)                             # (B·N, A)，已 detach
        logits = self.policy(obs_flat)
        probs = torch.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
        actor_loss = -(probs * adv_flat).sum(dim=-1).mean() \
            - self.log_alpha.exp().detach() * entropy
        self.policy_opt.zero_grad()
        actor_loss.backward()
        self.policy_opt.step()

        # ---- 温度：把平均熵拉向目标熵 ----
        with torch.no_grad():
            sampled = Categorical(probs=probs).sample()
            logp = torch.log(probs.gather(1, sampled.unsqueeze(-1)).squeeze(-1) + 1e-8)
        alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        return {"q_loss": q_loss.item(), "actor_loss": actor_loss.item(),
                "adv_std": adv_std, "alpha": self.alpha,
                "q_mean": float(q1.mean().item())}


# ===========================================================================
# 采样、评估、训练循环
# ===========================================================================
def run_training(credit, args, device):
    set_seed(args.seed)
    env = DiscreteNavEnv(args.n_agents, args.episode_len, seed=args.seed)
    eval_env = DiscreteNavEnv(args.n_agents, args.episode_len, seed=args.seed + 1)
    tr = Trainer(env, args, device, credit)
    label = "COMA" if credit == "coma" else "SHARED"

    cap = args.buffer_size
    buf = {k: np.zeros((cap,) + shape, dtype=dt) for k, shape, dt in [
        ("obs", (env.n_agents, env.obs_dim), np.float32),
        ("act", (env.n_agents,), np.int64),
        ("team_rew", (), np.float32),
        ("next_obs", (env.n_agents, env.obs_dim), np.float32),
        ("next_act", (env.n_agents,), np.int64),
        ("state", (env.state_dim,), np.float32),
        ("next_state", (env.state_dim,), np.float32),
        ("done", (), np.float32),
    ]}
    ptr, size = 0, 0
    prev_act = None
    prev_ptr = -1

    obs = env.reset()
    curve = []
    t0 = time.time()
    stats = {"q_loss": 0.0, "adv_std": 0.0, "alpha": args.alpha, "q_mean": 0.0}

    for step in range(1, args.max_steps + 1):
        state = env.global_state()
        if step <= args.start_steps:
            actions = np.random.randint(0, env.n_actions,
                                        size=env.n_agents).astype(np.int64)
        else:
            actions = tr.act_all(obs)
        next_obs, per_rew, team_rew, done, info = env.step(actions)
        next_state = env.global_state()

        # 写池：next_act 由下一步动作回填（SARSA）
        buf["obs"][ptr], buf["act"][ptr] = obs, actions
        buf["team_rew"][ptr] = team_rew
        buf["next_obs"][ptr] = next_obs
        buf["state"][ptr], buf["next_state"][ptr], buf["done"][ptr] = state, next_state, float(done)
        if prev_act is not None and prev_ptr >= 0:
            buf["next_act"][prev_ptr] = actions
        prev_ptr = ptr
        prev_act = actions
        ptr = (ptr + 1) % cap
        size = min(size + 1, cap)
        obs = next_obs
        if done:
            obs = env.reset()
            prev_act = None

        if step > args.start_steps and size > args.batch_size:
            idx = np.random.randint(0, size, size=args.batch_size)
            valid = idx[idx != prev_ptr]     # 丢弃没有后继动作的最后一条
            if len(valid) < args.batch_size // 2:
                valid = idx
            batch = {k: buf[k][valid] for k in buf}
            stats = tr.update(batch, args)

        if step % args.eval_every == 0:
            ret, dist, coll = evaluate(eval_env, tr, args.eval_episodes, device)
            curve.append((step, ret))
            print(f"[{label} 步 {step:6d}] 评估回报 {ret:7.2f} | 平均末距 {dist:.3f} | "
                  f"碰撞 {coll:.2f}/回合 | Q均值 {stats['q_mean']:7.2f} | "
                  f"信用(智能体间std) {stats['adv_std']:.3f} | α {stats['alpha']:.3f} | "
                  f"用时 {time.time() - t0:5.1f}s")

    elapsed = time.time() - t0
    ret, dist, coll = evaluate(eval_env, tr, args.eval_episodes * 2, device)
    print(f"[{label}] 完成：用时 {elapsed:.1f}s | 最终评估回报 {ret:.2f} | "
          f"末距 {dist:.3f} | 碰撞 {coll:.2f}")
    return tr, curve, elapsed, (ret, dist, coll)


@torch.no_grad()
def evaluate(env, tr, episodes, device):
    rets, dists, colls = [], [], []
    for _ in range(episodes):
        obs = env.reset()
        done = False
        ep_ret = 0.0
        info = {"mean_dist": 0.0, "collisions": 0}
        while not done:
            actions = tr.greedy_all(obs)
            obs, _, team_r, done, info = env.step(actions)
            ep_ret += team_r
        rets.append(ep_ret)
        dists.append(info["mean_dist"])
        colls.append(info["collisions"])
    return float(np.mean(rets)), float(np.mean(dists)), float(np.mean(colls))


def parse_args():
    p = argparse.ArgumentParser(description="第081章：多智能体 SAC 的信用分配")
    p.add_argument("--credit", type=str, default="both", choices=["both", "shared", "coma"])
    p.add_argument("--n-agents", type=int, default=3, help="智能体数量")
    p.add_argument("--episode-len", type=int, default=50, help="回合长度")
    p.add_argument("--baseline-samples", type=int, default=16, help="共享基线的随机联合动作采样数")
    p.add_argument("--max-steps", type=int, default=20000, help="环境步预算")
    p.add_argument("--batch-size", type=int, default=128, help="采样批量")
    p.add_argument("--buffer-size", type=int, default=50000, help="回放池容量")
    p.add_argument("--start-steps", type=int, default=1000, help="随机预热步数")
    p.add_argument("--hidden", type=int, default=64, help="策略隐藏层宽度")
    p.add_argument("--q-hidden", type=int, default=128, help="联合 Q 隐藏层宽度")
    p.add_argument("--lr", type=float, default=3e-4, help="学习率")
    p.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
    p.add_argument("--tau", type=float, default=0.01, help="目标网络软更新系数")
    p.add_argument("--alpha", type=float, default=0.2, help="温度初值")
    p.add_argument("--eval-every", type=int, default=4000, help="评估间隔")
    p.add_argument("--eval-episodes", type=int, default=5, help="评估回合数")
    p.add_argument("--seed", type=int, default=0, help="随机种子")
    p.add_argument("--save-dir", type=str, default="runs/ch081", help="保存目录")
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
    print(f"设备: {device} | 环境: 离散动作合作导航（5 动作）| 种子 {args.seed} | "
          f"预算 {args.max_steps} 环境步/配置")

    results = {}
    if args.credit in ("shared", "both"):
        results["SHARED"] = run_training("shared", args, device)
    if args.credit in ("coma", "both"):
        results["COMA"] = run_training("coma", args, device)

    print("=" * 76)
    for name, (tr, curve, dt, final) in results.items():
        best = max(r for _, r in curve) if curve else float("nan")
        print(f"[{name:6s}] 用时 {dt:6.1f}s | 最佳评估 {best:7.2f} | "
              f"最终（回报/末距/碰撞） {final[0]:7.2f} / {final[1]:.3f} / {final[2]:.2f}")
    if len(results) == 2:
        print("提示：两种方案共用同一个联合双 Q 与同一批数据，唯一区别是演员收到的"
              "信用信号；COMA 的智能体间优势差异（std）大于 0，shared 恒为 0。")

    os.makedirs(args.save_dir, exist_ok=True)
    for name, (tr, curve, _, _) in results.items():
        torch.save(tr.policy.state_dict(),
                   os.path.join(args.save_dir, f"{name.lower()}_policy.pt"))
        path = os.path.join(args.save_dir, f"{name.lower()}_curve.txt")
        with open(path, "w", encoding="utf-8") as f:
            for step, ret in curve:
                f.write(f"{step}\t{ret:.3f}\n")
    print(f"模型与曲线已保存到 {args.save_dir}")


if __name__ == "__main__":
    main()
