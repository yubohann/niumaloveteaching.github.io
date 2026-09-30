"""
第080章 多智能体SAC：角色对称

任务对称，但观测向量里的槽位顺序是人为约定：地标 1/2/3、队友 1/2。本章让环境
每回合随机洗牌槽位顺序，对照两种共享 SAC 的编码结构：
  - naive：普通 MLP 直接吃拼接向量
  - sym  ：置换不变编码器（Deep Sets：逐元素编码 + 集合均值池化）
训练结束后再用"置换探针"直接测量策略对槽位顺序的敏感度。

运行：
    python code/ch080.py           # 完整对照（CPU 约 4-8 分钟）
    python code/ch080.py --quick   # 快速跑通（约 40-80 秒）
    python code/ch080.py --encoder sym

预期（以实跑为准）：sym 编码器训练早期领先；置换探针下 sym 的动作偏差为 0，
naive 有明显偏差（这正是"结构保证"与"数据学会"的区别）。
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
# 环境：每回合洗牌槽位的合作导航（与第 072 章同构）
# ===========================================================================
class CooperativeNavEnv:
    def __init__(self, n_agents=3, episode_len=50, damping=0.9, accel=0.2,
                 collision_dist=0.1, collision_penalty=0.3, seed=0, shuffle_slots=True):
        self.n_agents = n_agents
        self.episode_len = episode_len
        self.damping = damping
        self.accel = accel
        self.collision_dist = collision_dist
        self.collision_penalty = collision_penalty
        self.shuffle_slots = shuffle_slots
        self.obs_dim = 4 + 2 * n_agents + 2 * (n_agents - 1)
        self.act_dim = 2
        self._rng = np.random.RandomState(seed)
        self._perms = list(itertools.permutations(range(n_agents)))
        self.pos = np.zeros((n_agents, 2), np.float32)
        self.vel = np.zeros((n_agents, 2), np.float32)
        self.landmarks = np.zeros((n_agents, 2), np.float32)
        self.agent_order = list(range(n_agents))
        self.landmark_order = list(range(n_agents))
        self.t = 0

    def reset(self):
        self.pos = self._rng.uniform(0.15, 0.85, (self.n_agents, 2)).astype(np.float32)
        self.vel = np.zeros((self.n_agents, 2), np.float32)
        self.landmarks = self._rng.uniform(0.10, 0.90, (self.n_agents, 2)).astype(np.float32)
        if self.shuffle_slots:
            self.agent_order = list(self._rng.permutation(self.n_agents))
            self.landmark_order = list(self._rng.permutation(self.n_agents))
        else:
            self.agent_order = list(range(self.n_agents))
            self.landmark_order = list(range(self.n_agents))
        self.t = 0
        return self._obs()

    def _obs(self):
        obs = np.zeros((self.n_agents, self.obs_dim), np.float32)
        for i in range(self.n_agents):
            rel_land = np.concatenate([self.landmarks[j] - self.pos[i]
                                       for j in self.landmark_order])
            rel_others = [self.pos[j] - self.pos[i] for j in self.agent_order if j != i]
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
# 两种编码器 + 共享策略/双 Q
# ===========================================================================
class NaiveEncoder(nn.Module):
    def __init__(self, obs_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(),
                                 nn.Linear(hidden, hidden), nn.Tanh())

    def forward(self, obs):
        return self.net(obs)


class SymmetricEncoder(nn.Module):
    """置换不变编码器：自身向量单独编码，地标/队友集合分别均值池化。"""

    def __init__(self, n_agents, hidden=64):
        super().__init__()
        self.n = n_agents
        self.own_enc = nn.Sequential(nn.Linear(4, hidden), nn.Tanh(),
                                     nn.Linear(hidden, hidden), nn.Tanh())
        self.land_enc = nn.Sequential(nn.Linear(2, hidden), nn.Tanh(),
                                      nn.Linear(hidden, hidden), nn.Tanh())
        self.other_enc = nn.Sequential(nn.Linear(2, hidden), nn.Tanh(),
                                       nn.Linear(hidden, hidden), nn.Tanh())
        self.trunk = nn.Sequential(nn.Linear(3 * hidden, hidden), nn.Tanh(),
                                   nn.Linear(hidden, hidden), nn.Tanh())

    def forward(self, obs):
        own = obs[:, :4]
        land = obs[:, 4:4 + 2 * self.n].reshape(-1, self.n, 2)
        other = obs[:, 4 + 2 * self.n:].reshape(-1, self.n - 1, 2)
        h_own = self.own_enc(own)
        h_land = self.land_enc(land).mean(dim=1)
        h_other = self.other_enc(other).mean(dim=1)
        return self.trunk(torch.cat([h_own, h_land, h_other], dim=-1))


class EncoderPolicy(nn.Module):
    def __init__(self, encoder, act_dim, hidden=64, act_limit=1.0):
        super().__init__()
        self.encoder = encoder
        self.act_limit = act_limit
        self.mean = nn.Linear(hidden, act_dim)
        self.log_std = nn.Linear(hidden, act_dim)

    def forward(self, obs):
        h = self.encoder(obs)
        return self.mean(h), torch.clamp(self.log_std(h), -4.0, 0.5)

    def sample(self, obs):
        mean, log_std = self.forward(obs)
        dist = Normal(mean, log_std.exp())
        z = dist.rsample()
        action = torch.tanh(z)
        logp = dist.log_prob(z) - torch.log(1.0 - action.pow(2) + 1e-6)
        return action * self.act_limit, logp.sum(-1, keepdim=True)

    @torch.no_grad()
    def mean_action(self, obs):
        mean, _ = self.forward(obs)
        return torch.tanh(mean) * self.act_limit


class EncoderTwinQ(nn.Module):
    def __init__(self, encoder, act_dim, hidden=64):
        super().__init__()
        self.encoder = encoder

        def make():
            return nn.Sequential(nn.Linear(hidden + act_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1))
        self.q1, self.q2 = make(), make()

    def forward(self, obs, act):
        h = self.encoder(obs)
        x = torch.cat([h, act], dim=-1)
        return self.q1(x), self.q2(x)


# ===========================================================================
# 共享 SAC 学习器（编码器可插拔）
# ===========================================================================
class SharedLearner:
    def __init__(self, obs_dim, act_dim, n_agents, args, device, encoder_type):
        if encoder_type == "sym":
            encoder_p = SymmetricEncoder(n_agents, args.hidden)
            encoder_q = SymmetricEncoder(n_agents, args.hidden)
            encoder_q_t = SymmetricEncoder(n_agents, args.hidden)
        else:
            encoder_p = NaiveEncoder(obs_dim, args.hidden)
            encoder_q = NaiveEncoder(obs_dim, args.hidden)
            encoder_q_t = NaiveEncoder(obs_dim, args.hidden)
        self.policy = EncoderPolicy(encoder_p, act_dim, args.hidden).to(device)
        self.q = EncoderTwinQ(encoder_q, act_dim, args.hidden).to(device)
        # 目标网络使用独立的编码器实例，避免与在线网络共享参数导致软更新互相干扰
        self.q_target = EncoderTwinQ(encoder_q_t, act_dim, args.hidden).to(device)
        self.q_target.load_state_dict(self.q.state_dict())
        self.policy_opt = torch.optim.Adam(self.policy.parameters(), lr=args.lr)
        self.q_opt = torch.optim.Adam(self.q.parameters(), lr=args.lr)
        self.log_alpha = torch.tensor(np.log(args.alpha), requires_grad=True, device=device)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=args.lr)
        self.target_entropy = -float(act_dim)
        self.gamma, self.tau = args.gamma, args.tau

    @property
    def alpha(self):
        return float(self.log_alpha.exp().item())

    def update(self, batch):
        obs = batch["obs"]
        act = batch["act"]
        rew = batch["rew"]
        next_obs = batch["next_obs"]
        done = batch["done"]
        with torch.no_grad():
            next_act, next_logp = self.policy.sample(next_obs)
            q1t, q2t = self.q_target(next_obs, next_act)
            target = rew.unsqueeze(-1) + self.gamma * (1.0 - done).unsqueeze(-1) * (
                torch.min(q1t, q2t) - self.log_alpha.exp() * next_logp)
        q1, q2 = self.q(obs, act)
        q_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.q_opt.zero_grad()
        q_loss.backward()
        self.q_opt.step()

        new_act, logp = self.policy.sample(obs)
        q1n, q2n = self.q(obs, new_act)
        policy_loss = (self.log_alpha.exp().detach() * logp - torch.min(q1n, q2n)).mean()
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
                "alpha": float(self.log_alpha.exp().item())}


# ===========================================================================
# 评估与置换探针
# ===========================================================================
@torch.no_grad()
def evaluate(env, policy, episodes, device):
    rets, dists, colls = [], [], []
    for _ in range(episodes):
        obs = env.reset()
        done = False
        ep_ret = 0.0
        info = {"mean_dist": 0.0, "collisions": 0}
        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
            actions = policy.mean_action(obs_t).cpu().numpy()
            obs, _, team_r, done, info = env.step(actions)
            ep_ret += team_r
        rets.append(ep_ret)
        dists.append(info["mean_dist"])
        colls.append(info["collisions"])
    return float(np.mean(rets)), float(np.mean(dists)), float(np.mean(colls))


def permute_obs_row(row, n_agents, rng):
    """打乱单个智能体观测里的地标槽位与队友槽位顺序。"""
    head = row[:4]
    land = row[4:4 + 2 * n_agents].reshape(n_agents, 2)
    other = row[4 + 2 * n_agents:].reshape(n_agents - 1, 2)
    land_p = land[rng.permutation(n_agents)].ravel()
    other_p = other[rng.permutation(n_agents - 1)].ravel()
    return np.concatenate([head, land_p, other_p]).astype(np.float32)


@torch.no_grad()
def permutation_probe(env, policy, device, n_states=100, n_perms=10, seed=0):
    """对同一状态施加随机槽位置换，测量策略输出的最大与平均偏差。"""
    rng = np.random.RandomState(seed)
    obs = env.reset()
    states = []
    for _ in range(200):
        states.append(obs.copy())
        actions = rng.uniform(-1.0, 1.0, (env.n_agents, env.act_dim)).astype(np.float32)
        obs, _, _, done, _ = env.step(actions)
        if done:
            obs = env.reset()
    states = [states[i] for i in rng.choice(len(states), size=n_states, replace=False)]

    max_dev, mean_dev = 0.0, 0.0
    for state in states:
        base = policy.mean_action(torch.as_tensor(
            state, dtype=torch.float32, device=device)).cpu().numpy()
        for _ in range(n_perms):
            perm_state = np.stack([permute_obs_row(state[i], env.n_agents, rng)
                                   for i in range(env.n_agents)])
            out = policy.mean_action(torch.as_tensor(
                perm_state, dtype=torch.float32, device=device)).cpu().numpy()
            dev = float(np.abs(out - base).mean())
            max_dev = max(max_dev, dev)
            mean_dev += dev
    mean_dev /= (len(states) * n_perms)
    return mean_dev, max_dev


# ===========================================================================
# 训练
# ===========================================================================
def run_training(encoder_type, args, device):
    set_seed(args.seed)
    env = CooperativeNavEnv(args.n_agents, args.episode_len, seed=args.seed,
                            shuffle_slots=not args.no_shuffle)
    eval_env = CooperativeNavEnv(args.n_agents, args.episode_len, seed=args.seed + 1,
                                 shuffle_slots=not args.no_shuffle)
    learner = SharedLearner(env.obs_dim, env.act_dim, env.n_agents, args, device, encoder_type)
    label = "SYM" if encoder_type == "sym" else "NAIVE"
    n_params = sum(p.numel() for m in [learner.policy, learner.q] for p in m.parameters())
    print(f"[{label}] 参数量 {n_params} | 槽位洗牌: {not args.no_shuffle}")

    # 简易回放池
    capacity = args.buffer_size
    obs_buf = np.zeros((capacity, env.n_agents, env.obs_dim), np.float32)
    act_buf = np.zeros((capacity, env.n_agents, env.act_dim), np.float32)
    rew_buf = np.zeros((capacity, env.n_agents), np.float32)
    nobs_buf = np.zeros((capacity, env.n_agents, env.obs_dim), np.float32)
    done_buf = np.zeros((capacity,), np.float32)
    ptr, size = 0, 0

    obs = env.reset()
    curve = []
    t0 = time.time()

    for step in range(1, args.max_steps + 1):
        if step <= args.start_steps:
            actions = np.random.uniform(-1.0, 1.0, (env.n_agents, env.act_dim)).astype(np.float32)
        else:
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
                actions = learner.policy.sample(obs_t)[0].cpu().numpy()
        next_obs, per_rew, team_rew, done, info = env.step(actions)
        obs_buf[ptr], act_buf[ptr], rew_buf[ptr] = obs, actions, per_rew
        nobs_buf[ptr], done_buf[ptr] = next_obs, float(done)
        ptr = (ptr + 1) % capacity
        size = min(size + 1, capacity)
        obs = next_obs
        if done:
            obs = env.reset()

        if step > args.start_steps and step % args.update_every == 0 and size > args.batch_size:
            idx = np.random.randint(0, size, size=args.batch_size)
            B, N = args.batch_size, env.n_agents
            batch = {
                "obs": torch.as_tensor(obs_buf[idx].reshape(B * N, -1), device=device),
                "act": torch.as_tensor(act_buf[idx].reshape(B * N, -1), device=device),
                "rew": torch.as_tensor(rew_buf[idx].reshape(B * N), device=device),
                "next_obs": torch.as_tensor(nobs_buf[idx].reshape(B * N, -1), device=device),
                "done": torch.as_tensor(np.repeat(done_buf[idx], N), device=device),
            }
            learner.update(batch)

        if step % args.eval_every == 0:
            ret, dist, coll = evaluate(eval_env, learner.policy, args.eval_episodes, device)
            curve.append((step, ret))
            print(f"[{label} 步 {step:6d}] 评估回报 {ret:7.2f} | 平均末距 {dist:.3f} | "
                  f"碰撞 {coll:.2f}/回合 | α {learner.alpha:.3f} | "
                  f"用时 {time.time() - t0:5.1f}s")

    elapsed = time.time() - t0
    ret, dist, coll = evaluate(eval_env, learner.policy, args.eval_episodes * 2, device)
    mean_dev, max_dev = permutation_probe(eval_env, learner.policy, device, seed=args.seed)
    print(f"[{label}] 完成：用时 {elapsed:.1f}s | 最终评估回报 {ret:.2f} | "
          f"末距 {dist:.3f} | 碰撞 {coll:.2f}")
    print(f"[{label} 置换探针] 平均动作偏差 {mean_dev:.4f} | 最大偏差 {max_dev:.4f}")
    return learner, curve, elapsed, (ret, dist, coll), (mean_dev, max_dev)


def parse_args():
    p = argparse.ArgumentParser(description="第080章：多智能体 SAC 的角色对称")
    p.add_argument("--encoder", type=str, default="both", choices=["both", "naive", "sym"])
    p.add_argument("--no-shuffle", action="store_true", help="关闭槽位洗牌（对照用）")
    p.add_argument("--n-agents", type=int, default=3, help="智能体数量")
    p.add_argument("--episode-len", type=int, default=50, help="回合长度")
    p.add_argument("--max-steps", type=int, default=20000, help="环境步预算")
    p.add_argument("--update-every", type=int, default=1, help="每多少步更新一次")
    p.add_argument("--batch-size", type=int, default=128, help="采样批量")
    p.add_argument("--buffer-size", type=int, default=50000, help="回放池容量")
    p.add_argument("--start-steps", type=int, default=1000, help="随机预热步数")
    p.add_argument("--hidden", type=int, default=64, help="隐藏层宽度")
    p.add_argument("--lr", type=float, default=3e-4, help="学习率")
    p.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
    p.add_argument("--tau", type=float, default=0.005, help="软更新系数")
    p.add_argument("--alpha", type=float, default=0.2, help="温度初值")
    p.add_argument("--eval-every", type=int, default=4000, help="评估间隔")
    p.add_argument("--eval-episodes", type=int, default=5, help="评估回合数")
    p.add_argument("--seed", type=int, default=0, help="随机种子")
    p.add_argument("--save-dir", type=str, default="runs/ch080", help="保存目录")
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
    print(f"设备: {device} | 环境: 槽位随机洗牌的合作导航 | 种子 {args.seed} | "
          f"预算 {args.max_steps} 环境步/配置")

    results = {}
    if args.encoder in ("naive", "both"):
        results["NAIVE"] = run_training("naive", args, device)
    if args.encoder in ("sym", "both"):
        results["SYM"] = run_training("sym", args, device)

    print("=" * 76)
    for name, (learner, curve, dt, final, probe) in results.items():
        best = max(r for _, r in curve) if curve else float("nan")
        print(f"[{name:6s}] 用时 {dt:6.1f}s | 最佳评估 {best:7.2f} | "
              f"最终（回报/末距/碰撞） {final[0]:7.2f} / {final[1]:.3f} / {final[2]:.2f} | "
              f"置换探针（均值/最大偏差） {probe[0]:.4f} / {probe[1]:.4f}")
    if len(results) == 2:
        print("提示：置换探针下 sym 的偏差应为 0（浮点误差级），naive 会出现可见偏差；"
              "这正是结构不变性与数据学习不变性的区别。")

    os.makedirs(args.save_dir, exist_ok=True)
    for name, (learner, curve, _, _, _) in results.items():
        torch.save(learner.policy.state_dict(),
                   os.path.join(args.save_dir, f"{name.lower()}_policy.pt"))
        path = os.path.join(args.save_dir, f"{name.lower()}_curve.txt")
        with open(path, "w", encoding="utf-8") as f:
            for step, ret in curve:
                f.write(f"{step}\t{ret:.3f}\n")
    print(f"模型与曲线已保存到 {args.save_dir}")


if __name__ == "__main__":
    main()
