"""
第073章 多智能体PPO：信用分配

团队奖励是一大锅饭：赢了人人有份，输了人人有责。本章把信用分配拆开做对照：
  - shared：优势 = Q(s, a) − 平均联合动作的 Q（所有智能体共用一个基线）
  - coma  ：优势 = Q(s, a) − Σ_{a'_i} π_i(a'_i|o_i) Q(s, a_{-i}, a'_i)
            （反事实基线：只替换智能体 i 的动作求期望，得到"你的这一步比你的平均
            水平好多少"）
为了能枚举反事实动作，本章把动作空间改成离散 5 动作（停/上下左右），并用一个
集中式联合 Q 网络 Q(s_global, a_1..a_N) 估计团队价值。

运行：
    python code/ch073.py           # 完整对照（CPU 约 4-10 分钟）
    python code/ch073.py --quick   # 快速跑通（约 40-90 秒）
    python code/ch073.py --credit coma

预期（以实跑为准）：COMA 的逐智能体优势更能区分"谁的贡献大"，训练通常更稳、
末距更低；shared 基线简单但把功劳混在一起，优势信号更粗。
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
# 离散动作版合作导航环境
# ===========================================================================
class DiscreteNavEnv:
    """5 动作版合作导航：0=停, 1=右(+x), 2=左(−x), 3=上(+y), 4=下(−y)。

    其余与第 068 章一致：对称的二分匹配奖励 + 碰撞惩罚，固定 50 步回合。
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
# 网络：离散共享策略 + 联合 Q
# ===========================================================================
class DiscreteActor(nn.Module):
    """共享的离散策略网络（三个智能体共用）。"""

    def __init__(self, obs_dim, n_actions, hidden=64):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(),
                                  nn.Linear(hidden, hidden), nn.Tanh())
        self.logits = nn.Linear(hidden, n_actions)

    def forward(self, obs):
        return self.logits(self.body(obs))

    @torch.no_grad()
    def sample(self, obs):
        logits = self.forward(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action)

    @torch.no_grad()
    def greedy(self, obs):
        return torch.argmax(self.forward(obs), dim=-1)


class JointQ(nn.Module):
    """集中式联合 Q：输入全局状态 + 所有智能体的动作 one-hot。"""

    def __init__(self, state_dim, n_agents, n_actions, hidden=128):
        super().__init__()
        in_dim = state_dim + n_agents * n_actions
        self.in_dim = in_dim
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1))

    def forward(self, state, joint_onehot):
        x = torch.cat([state, joint_onehot.view(-1, self.in_dim - state.shape[-1])], dim=-1)
        return self.net(x).squeeze(-1)


def joint_onehot(actions: torch.Tensor, n_agents: int, n_actions: int) -> torch.Tensor:
    """(T, N) 的动作索引 -> (T, N*A) 的 one-hot 拼接。"""
    T = actions.shape[0]
    out = torch.zeros(T, n_agents, n_actions, device=actions.device)
    out.scatter_(2, actions.unsqueeze(-1), 1.0)
    return out.view(T, n_agents * n_actions)


def alternative_joint(actions: torch.Tensor, n_agents: int, n_actions: int) -> torch.Tensor:
    """为每个智能体枚举其全部动作，返回 (T, N, A, N*A)。

    第 i 个智能体的动作块被逐一替换成 A 个 one-hot，其余智能体保持原动作。
    """
    T = actions.shape[0]
    device = actions.device
    base = joint_onehot(actions, n_agents, n_actions)           # (T, N*A)
    alts = base.unsqueeze(1).unsqueeze(2).repeat(1, n_agents, n_actions, 1)  # (T,N,A,N*A)
    eye = torch.eye(n_actions, device=device)                   # (A, A)
    for i in range(n_agents):
        block = alts[:, i, :, i * n_actions:(i + 1) * n_actions]  # (T, A, A)
        block.copy_(eye.unsqueeze(0).expand(T, -1, -1))
    return alts


def explained_variance(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    var_y = y_true.var()
    if float(var_y.item()) < 1e-8:
        return 0.0
    return float((1.0 - (y_true - y_pred).var() / var_y).item())


# ===========================================================================
# 管理器：两种信用分配方案共用同一套网络结构
# ===========================================================================
class CreditManager:
    def __init__(self, env, args, device, credit_mode):
        self.n = env.n_agents
        self.A = env.n_actions
        self.device = device
        self.credit = credit_mode
        self.actor = DiscreteActor(env.obs_dim, env.n_actions, args.hidden).to(device)
        self.q = JointQ(env.state_dim, env.n_agents, env.n_actions, args.q_hidden).to(device)
        self.q_target = JointQ(env.state_dim, env.n_agents, env.n_actions, args.q_hidden).to(device)
        self.q_target.load_state_dict(self.q.state_dict())
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=args.lr)
        self.q_opt = torch.optim.Adam(self.q.parameters(), lr=args.lr)

    @torch.no_grad()
    def act_all(self, obs):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        actions, logps = self.actor.sample(obs_t)
        return actions.cpu().numpy(), [float(x) for x in logps]

    @torch.no_grad()
    def greedy_all(self, obs):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        return self.actor.greedy(obs_t).cpu().numpy()

    # ------------------------------------------------------------------
    # 优势计算：本章的核心对照
    # ------------------------------------------------------------------
    def compute_advantages(self, state, acts, obs, args):
        device = self.device
        T, N, A = state.shape[0], self.n, self.A
        joint = joint_onehot(acts, N, A)
        with torch.no_grad():
            q_sa = self.q(state, joint)                            # (T,)
            if self.credit == "coma":
                # 每个智能体的反事实基线：替换它的动作，对其策略分布求期望
                alts = alternative_joint(acts, N, A)               # (T,N,A,N*A)
                q_alts = self.q(state.repeat_interleave(N * A, dim=0),
                                alts.reshape(T * N * A, N * A)).reshape(T, N, A)
                logits = self.actor(obs.reshape(T * N, -1))
                probs = torch.softmax(logits, dim=-1).reshape(T, N, A)
                baseline = (probs * q_alts).sum(dim=-1)             # (T,N)
                adv = q_sa.unsqueeze(1) - baseline                  # (T,N)
            else:
                # 共享基线：随机采 K 组联合动作，用平均 Q 当所有人的基线
                k = args.baseline_samples
                rand_acts = torch.randint(0, A, (T, k, N), device=device)
                flat = rand_acts.permute(0, 2, 1).reshape(T * N * k, 1)
                rand_joint = torch.zeros(T, k, N * A, device=device)
                idx = (torch.arange(N, device=device)[None, None, :] * A
                       + rand_acts)                                 # (T,k,N)
                rand_joint.scatter_(2, idx, 1.0)
                rand_joint = rand_joint.reshape(T, k, N * A)
                q_rand = self.q(state.repeat_interleave(k, dim=0),
                                rand_joint.reshape(T * k, N * A)).reshape(T, k)
                baseline = q_rand.mean(dim=1)                        # (T,)
                adv = (q_sa - baseline).unsqueeze(1).repeat(1, N)    # (T,N)
            adv_std = float(adv.std(dim=1).mean().item())            # 智能体间差异
        return adv, float(q_sa.mean().item()), adv_std

    # ------------------------------------------------------------------
    def update(self, buf, args) -> dict:
        device = self.device
        T, N, A = len(buf["rew"]), self.n, self.A
        obs = torch.as_tensor(np.asarray(buf["obs"]), dtype=torch.float32, device=device)
        acts = torch.as_tensor(np.asarray(buf["act"]), dtype=torch.long, device=device)
        logp_old = torch.as_tensor(np.asarray(buf["logp"]), dtype=torch.float32, device=device)
        state = torch.as_tensor(np.asarray(buf["state"]), dtype=torch.float32, device=device)
        next_state = torch.as_tensor(np.asarray(buf["next_state"]), dtype=torch.float32,
                                     device=device)
        next_acts = torch.as_tensor(np.asarray(buf["next_act"]), dtype=torch.long, device=device)
        rew = torch.as_tensor(np.asarray(buf["rew"]), dtype=torch.float32, device=device)
        done = torch.as_tensor(np.asarray(buf["done"]), dtype=torch.float32, device=device)

        joint = joint_onehot(acts, N, A)
        next_joint = joint_onehot(next_acts, N, A)

        # ---- 1) 联合 Q 更新（SARSA 目标；最后一条样本缺少后继动作，丢弃）----
        stats = {"q_loss": 0.0, "q_ev": 0.0, "policy_loss": 0.0, "approx_kl": 0.0,
                 "clip_frac": 0.0, "adv_std": 0.0, "q_mean": 0.0, "n": 0}
        with torch.no_grad():
            next_q = self.q_target(next_state[:-1], next_joint[:-1])
            target = rew[:-1] + args.gamma * (1.0 - done[:-1]) * next_q
        for _ in range(args.q_epochs):
            pred = self.q(state[:-1], joint[:-1])
            q_loss = F.mse_loss(pred, target)
            self.q_opt.zero_grad()
            q_loss.backward()
            self.q_opt.step()
            stats["q_loss"] += q_loss.item()
            stats["n"] += 1
        stats["q_loss"] /= max(stats["n"], 1)

        # 目标网络软更新
        with torch.no_grad():
            for p, p_t in zip(self.q.parameters(), self.q_target.parameters()):
                p_t.data.mul_(1.0 - args.tau_q).add_(args.tau_q * p.data)
        with torch.no_grad():
            stats["q_ev"] = explained_variance(self.q(state[:-1], joint[:-1]), target)
            stats["q_mean"] = float(self.q(state, joint).mean().item())

        # ---- 2) 信用分配：计算逐智能体优势 ----
        adv, q_mean, adv_std = self.compute_advantages(state, acts, obs, args)
        stats["adv_std"] = adv_std
        adv_flat = adv.reshape(T * N)
        adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)

        # ---- 3) 策略更新（PPO 裁剪目标，共享策略）----
        obs_flat = obs.reshape(T * N, -1)
        act_flat = acts.reshape(T * N)
        logp_flat = logp_old.reshape(T * N)
        M = T * N
        n_steps = 0
        for _ in range(args.epochs):
            idx = torch.randperm(M, device=device)
            for start in range(0, M, args.batch_size):
                mb = idx[start:start + args.batch_size]
                logits = self.actor(obs_flat[mb])
                dist = Categorical(logits=logits)
                logp = dist.log_prob(act_flat[mb])
                ratio = torch.exp(logp - logp_flat[mb])
                surr1 = ratio * adv_flat[mb]
                surr2 = torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * adv_flat[mb]
                loss = -torch.min(surr1, surr2).mean()
                self.actor_opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.actor_opt.step()
                with torch.no_grad():
                    stats["policy_loss"] += loss.item()
                    stats["approx_kl"] += (logp_flat[mb] - logp).mean().item()
                    stats["clip_frac"] += ((ratio - 1.0).abs() > args.clip).float().mean().item()
                    n_steps += 1
        for k in ["policy_loss", "approx_kl", "clip_frac"]:
            stats[k] /= max(n_steps, 1)
        return stats


# ===========================================================================
# 采样与评估
# ===========================================================================
def collect(env, mgr, steps_agent, obs):
    n = env.n_agents
    obs_list, act_list, logp_list = [], [], []
    state_list, next_state_list = [], []
    rew_list, done_list = [], []
    team_rets = []
    ep_team_ret = 0.0
    for _ in range(steps_agent // n):
        gs = env.global_state()
        actions, logps = mgr.act_all(obs)
        next_obs, per_r, team_r, done, _ = env.step(actions)
        obs_list.append(obs)
        act_list.append(actions)
        logp_list.append(logps)
        state_list.append(gs)
        next_state_list.append(env.global_state())
        rew_list.append(team_r)
        done_list.append(float(done))
        ep_team_ret += team_r
        obs = next_obs
        if done:
            team_rets.append(ep_team_ret)
            ep_team_ret = 0.0
            obs = env.reset()
    # SARSA 需要 a_{t+1}：用相邻两步动作构造，最后一条样本在更新时丢弃
    acts_np = np.asarray(act_list, dtype=np.int64)
    next_acts = np.concatenate([acts_np[1:], acts_np[-1:]], axis=0)
    buf = {"obs": obs_list, "act": act_list, "logp": logp_list,
           "state": state_list, "next_state": next_state_list,
           "next_act": next_acts, "rew": rew_list, "done": done_list}
    return buf, team_rets, obs


@torch.no_grad()
def evaluate(env, mgr, episodes):
    rets, dists, colls = [], [], []
    for _ in range(episodes):
        obs = env.reset()
        done = False
        ep_ret = 0.0
        info = {"mean_dist": 0.0, "collisions": 0}
        while not done:
            actions = mgr.greedy_all(obs)
            obs, _, team_r, done, info = env.step(actions)
            ep_ret += team_r
        rets.append(ep_ret)
        dists.append(info["mean_dist"])
        colls.append(info["collisions"])
    return float(np.mean(rets)), float(np.mean(dists)), float(np.mean(colls))


def run_training(credit, args, device):
    set_seed(args.seed)
    env = DiscreteNavEnv(args.n_agents, args.episode_len, seed=args.seed)
    eval_env = DiscreteNavEnv(args.n_agents, args.episode_len, seed=args.seed + 1)
    mgr = CreditManager(env, args, device, credit)
    label = "COMA" if credit == "coma" else "SHARED"

    obs = env.reset()
    total_steps, update_idx = 0, 0
    curve = []
    all_team_rets = []
    t0 = time.time()
    next_eval = args.eval_every

    while total_steps < args.max_steps:
        buf, team_rets, obs = collect(env, mgr, args.steps_per_update, obs)
        total_steps += args.steps_per_update
        stats = mgr.update(buf, args)
        all_team_rets.extend(team_rets)
        update_idx += 1
        recent = np.mean(all_team_rets[-20:]) if all_team_rets else float("nan")
        print(f"[{label} 更新 {update_idx:3d}] 智能体步 {total_steps:7d} | "
              f"近20回合团队回报 {recent:7.2f} | Q损失 {stats['q_loss']:.2f} | "
              f"Q均值 {stats['q_mean']:7.2f} | Q_EV {stats['q_ev']:.3f} | "
              f"优势(智能体间std) {stats['adv_std']:.3f} | KL {stats['approx_kl']:.4f} | "
              f"用时 {time.time() - t0:5.1f}s")
        if total_steps >= next_eval:
            ret, dist, coll = evaluate(eval_env, mgr, args.eval_episodes)
            curve.append((total_steps, ret))
            print(f"    [评估] 回报 {ret:7.2f} | 平均末距 {dist:.3f} | "
                  f"平均碰撞 {coll:.2f}/回合")
            next_eval += args.eval_every

    elapsed = time.time() - t0
    ret, dist, coll = evaluate(eval_env, mgr, args.eval_episodes * 2)
    print(f"[{label}] 完成：用时 {elapsed:.1f}s | 最终评估回报 {ret:.2f} | "
          f"末距 {dist:.3f} | 碰撞 {coll:.2f}")
    return curve, elapsed, (ret, dist, coll)


def parse_args():
    p = argparse.ArgumentParser(description="第073章：多智能体 PPO 的信用分配")
    p.add_argument("--credit", type=str, default="both", choices=["both", "shared", "coma"])
    p.add_argument("--n-agents", type=int, default=3, help="智能体数量")
    p.add_argument("--episode-len", type=int, default=50, help="回合长度")
    p.add_argument("--baseline-samples", type=int, default=16, help="共享基线的随机联合动作采样数")
    p.add_argument("--max-steps", type=int, default=120000, help="每种方案的智能体步预算")
    p.add_argument("--steps-per-update", type=int, default=3000, help="每次更新采样步数")
    p.add_argument("--eval-every", type=int, default=30000, help="评估间隔")
    p.add_argument("--q-epochs", type=int, default=4, help="联合 Q 每批数据的更新轮数")
    p.add_argument("--epochs", type=int, default=8, help="策略每批数据的更新轮数")
    p.add_argument("--batch-size", type=int, default=256, help="小批量大小")
    p.add_argument("--hidden", type=int, default=64, help="策略隐藏层宽度")
    p.add_argument("--q-hidden", type=int, default=128, help="联合 Q 隐藏层宽度")
    p.add_argument("--lr", type=float, default=3e-4, help="学习率")
    p.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
    p.add_argument("--tau-q", type=float, default=0.01, help="Q 目标网络软更新系数")
    p.add_argument("--clip", type=float, default=0.2, help="PPO 裁剪范围")
    p.add_argument("--eval-episodes", type=int, default=5, help="评估回合数")
    p.add_argument("--seed", type=int, default=0, help="随机种子")
    p.add_argument("--save-dir", type=str, default="runs/ch073", help="保存目录")
    p.add_argument("--quick", action="store_true", help="快速模式：24000 步/方案")
    return p.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.max_steps = min(args.max_steps, 24000)
        args.steps_per_update = 1500
        args.eval_every = 8000
        args.epochs = 5
        args.q_epochs = 3
        args.eval_episodes = 3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device} | 环境: 离散动作合作导航（5 动作）| 种子 {args.seed} | "
          f"每方案预算 {args.max_steps} 智能体步")

    results = {}
    if args.credit in ("shared", "both"):
        results["SHARED"] = run_training("shared", args, device)
    if args.credit in ("coma", "both"):
        results["COMA"] = run_training("coma", args, device)

    print("=" * 76)
    for name, (curve, elapsed, final) in results.items():
        best = max(r for _, r in curve) if curve else float("nan")
        print(f"[{name:6s}] 用时 {elapsed:6.1f}s | 最佳评估 {best:7.2f} | "
              f"最终（回报/末距/碰撞） {final[0]:7.2f} / {final[1]:.3f} / {final[2]:.2f}")
    if len(results) == 2:
        print("提示：两种方案共用同一套网络与数据，唯一区别是优势里的基线。"
              "COMA 的逐智能体基线通常让协调行为出现得更早。")

    os.makedirs(args.save_dir, exist_ok=True)
    for name, (curve, _, _) in results.items():
        path = os.path.join(args.save_dir, f"{name.lower()}_curve.txt")
        with open(path, "w", encoding="utf-8") as f:
            for step, ret in curve:
                f.write(f"{step}\t{ret:.3f}\n")
    print(f"评估曲线已保存到 {args.save_dir}")


if __name__ == "__main__":
    main()
