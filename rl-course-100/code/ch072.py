"""
第072章 多智能体PPO：角色对称性

任务本身完全对称（谁去哪个地标都行），但观测向量里的"槽位"有固定顺序：
地标 1、地标 2、地标 3 与队友 1、队友 2。本章让环境在每个回合随机打乱这些
槽位顺序，然后对照两种策略输入结构：
  - naive：普通 MLP 直接吃拼接向量，要靠数据自己学会"槽位顺序无关"
  - sym  ：置换不变编码器（Deep Sets 风格）——每个地标/队友单独编码后取均值，
           结构上对槽位顺序完全不敏感

运行：
    python code/ch072.py           # 完整对照（CPU 约 5-12 分钟）
    python code/ch072.py --quick   # 快速跑通（约 1 分钟）
    python code/ch072.py --encoder sym

预期（以实跑为准）：在每回合洗牌槽位的设置下，sym 编码器通常学得更快、更稳，
因为它把"顺序无关"直接写进了结构；naive 也能学会但需要更多数据。
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
# 环境：每回合随机打乱队友/地标槽位顺序
# ===========================================================================
class CooperativeNavEnv:
    """合作导航 + 槽位洗牌。

    观测布局固定为 [自身位置(2), 自身速度(2), 地标相对位置(2N), 队友相对位置(2(N-1))]，
    但地标与队友在列表中的顺序每回合被随机置换。任务奖励（二分匹配）不受影响，
    因此"槽位顺序"是纯粹的干扰变量：好策略不该依赖它。
    """

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
        self.state_dim = 6 * n_agents + 1
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
# 两种策略网络
# ===========================================================================
def mlp(in_dim, out_dim, hidden, act=nn.Tanh):
    return nn.Sequential(nn.Linear(in_dim, hidden), act(),
                         nn.Linear(hidden, hidden), act(),
                         nn.Linear(hidden, out_dim))


class NaiveActor(nn.Module):
    """普通 MLP：直接把 14 维拼接向量喂进去。"""

    def __init__(self, obs_dim, act_dim, hidden=64):
        super().__init__()
        self.net = mlp(obs_dim, act_dim, hidden)
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.7))

    def forward(self, obs):
        mean = self.net(obs)
        log_std = torch.clamp(self.log_std, -4.0, 0.5).expand_as(mean)
        return mean, log_std

    @torch.no_grad()
    def act_batch(self, obs):
        mean, log_std = self.forward(obs)
        dist = Normal(mean, log_std.exp())
        action = torch.clamp(dist.sample(), -1.0, 1.0)
        return action, dist.log_prob(action).sum(-1)

    @torch.no_grad()
    def mean_actions(self, obs):
        mean, _ = self.forward(obs)
        return torch.clamp(mean, -1.0, 1.0)


class SymmetricActor(nn.Module):
    """置换不变编码器：自身编码 + 地标均值池化 + 队友均值池化。

    池化把每个地标（或队友）的相对位置先单独编码，再对集合取均值，
    因此输出对槽位顺序完全不敏感——"角色对称"被写进结构而不是靠数据学。
    """

    def __init__(self, obs_dim, act_dim, n_agents, hidden=64):
        super().__init__()
        self.n = n_agents
        self.own_enc = mlp(4, hidden, hidden)
        self.land_enc = mlp(2, hidden, hidden)
        self.other_enc = mlp(2, hidden, hidden)
        self.trunk = nn.Sequential(nn.Linear(3 * hidden, hidden), nn.Tanh(),
                                   nn.Linear(hidden, hidden), nn.Tanh())
        self.mean = nn.Linear(hidden, act_dim)
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.7))

    def _split(self, obs):
        own = obs[:, :4]
        land = obs[:, 4:4 + 2 * self.n].reshape(-1, self.n, 2)
        other = obs[:, 4 + 2 * self.n:].reshape(-1, self.n - 1, 2)
        return own, land, other

    def forward(self, obs):
        own, land, other = self._split(obs)
        h_own = self.own_enc(own)
        h_land = self.land_enc(land).mean(dim=1)
        h_other = self.other_enc(other).mean(dim=1)
        h = self.trunk(torch.cat([h_own, h_land, h_other], dim=-1))
        mean = self.mean(h)
        log_std = torch.clamp(self.log_std, -4.0, 0.5).expand_as(mean)
        return mean, log_std

    @torch.no_grad()
    def act_batch(self, obs):
        mean, log_std = self.forward(obs)
        dist = Normal(mean, log_std.exp())
        action = torch.clamp(dist.sample(), -1.0, 1.0)
        return action, dist.log_prob(action).sum(-1)

    @torch.no_grad()
    def mean_actions(self, obs):
        mean, _ = self.forward(obs)
        return torch.clamp(mean, -1.0, 1.0)


class CentralCritic(nn.Module):
    def __init__(self, state_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, hidden), nn.Tanh(),
                                 nn.Linear(hidden, hidden), nn.Tanh(),
                                 nn.Linear(hidden, 1))

    def forward(self, state):
        return self.net(state).squeeze(-1)


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


def explained_variance(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    var_y = y_true.var()
    if float(var_y.item()) < 1e-8:
        return 0.0
    return float((1.0 - (y_true - y_pred).var() / var_y).item())


def ppo_actor_update(actor, optim, obs, act, logp_old, adv, args) -> dict:
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    N = obs.shape[0]
    stats = {"policy_loss": 0.0, "approx_kl": 0.0, "clip_frac": 0.0, "n": 0}
    for _ in range(args.epochs):
        idx = torch.randperm(N, device=obs.device)
        for start in range(0, N, args.batch_size):
            mb = idx[start:start + args.batch_size]
            mean, log_std = actor(obs[mb])
            dist = Normal(mean, log_std.exp())
            logp = dist.log_prob(act[mb]).sum(-1)
            ratio = torch.exp(logp - logp_old[mb])
            surr1 = ratio * adv[mb]
            surr2 = torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * adv[mb]
            loss = -torch.min(surr1, surr2).mean()
            optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
            optim.step()
            with torch.no_grad():
                stats["policy_loss"] += loss.item()
                stats["approx_kl"] += (logp_old[mb] - logp).mean().item()
                stats["clip_frac"] += ((ratio - 1.0).abs() > args.clip).float().mean().item()
                stats["n"] += 1
    for k in stats:
        if k != "n":
            stats[k] /= max(stats["n"], 1)
    return stats


# ===========================================================================
# 管理器：共享 Actor（两种编码器之一）+ 中央批评家
# ===========================================================================
class Manager:
    def __init__(self, env, args, device, encoder):
        self.n = env.n_agents
        self.device = device
        if encoder == "naive":
            self.actor = NaiveActor(env.obs_dim, env.act_dim, args.hidden).to(device)
        else:
            self.actor = SymmetricActor(env.obs_dim, env.act_dim, env.n_agents,
                                        args.hidden).to(device)
        self.central = CentralCritic(env.state_dim, args.hidden).to(device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=args.lr)
        self.central_opt = torch.optim.Adam(self.central.parameters(), lr=args.lr)

    @torch.no_grad()
    def act_all(self, obs, global_state):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        actions, logps = self.actor.act_batch(obs_t)
        state_t = torch.as_tensor(global_state, dtype=torch.float32,
                                  device=self.device).unsqueeze(0)
        value = float(self.central(state_t).item())
        return actions.cpu().numpy(), [float(x) for x in logps], value

    @torch.no_grad()
    def mean_actions_all(self, obs):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        return self.actor.mean_actions(obs_t).cpu().numpy()

    def update(self, buf, args) -> dict:
        device = self.device
        agent_bufs, cuts = buf["agents"], buf["cuts"]
        states = torch.as_tensor(np.asarray(buf["states"]), dtype=torch.float32, device=device)
        last_state = torch.as_tensor(buf["last_state"], dtype=torch.float32,
                                     device=device).unsqueeze(0)
        with torch.no_grad():
            last_value = float(self.central(last_state).item())
        adv, ret = compute_gae(buf["team_rew"], buf["val"], cuts, last_value,
                               args.gamma, args.lam)
        ret_t = torch.as_tensor(ret, device=device)
        for _ in range(args.epochs):
            loss = F.mse_loss(self.central(states), ret_t)
            self.central_opt.zero_grad()
            loss.backward()
            self.central_opt.step()
        with torch.no_grad():
            ev = explained_variance(self.central(states), ret_t)

        obs_cat = torch.as_tensor(
            np.concatenate([np.asarray(agent_bufs[i]["obs"]) for i in range(self.n)], axis=0),
            dtype=torch.float32, device=device)
        act_cat = torch.as_tensor(
            np.concatenate([np.asarray(agent_bufs[i]["act"]) for i in range(self.n)], axis=0),
            dtype=torch.float32, device=device)
        logp_cat = torch.as_tensor(
            np.concatenate([np.asarray(agent_bufs[i]["logp"]) for i in range(self.n)], axis=0),
            dtype=torch.float32, device=device)
        adv_cat = torch.as_tensor(np.tile(adv, self.n), device=device)
        ps = ppo_actor_update(self.actor, self.actor_opt, obs_cat, act_cat,
                              logp_cat, adv_cat, args)
        return {"policy_loss": ps["policy_loss"], "approx_kl": ps["approx_kl"],
                "clip_frac": ps["clip_frac"], "ev": ev}


# ===========================================================================
# 采样、评估、训练
# ===========================================================================
def collect(env, mgr, steps_agent, obs):
    n = env.n_agents
    agent_bufs = [{"obs": [], "act": [], "logp": []} for _ in range(n)]
    states, team_rews, values, cuts = [], [], [], []
    team_rets = []
    ep_team_ret = 0.0
    for _ in range(steps_agent // n):
        global_state = env.global_state()
        actions, logps, value = mgr.act_all(obs, global_state)
        next_obs, per_r, team_r, done, _ = env.step(actions)
        for i in range(n):
            agent_bufs[i]["obs"].append(obs[i])
            agent_bufs[i]["act"].append(actions[i])
            agent_bufs[i]["logp"].append(logps[i])
        states.append(global_state)
        team_rews.append(team_r)
        values.append(value)
        cuts.append(float(done))
        ep_team_ret += team_r
        obs = next_obs
        if done:
            team_rets.append(ep_team_ret)
            ep_team_ret = 0.0
            obs = env.reset()
    buf = {"agents": agent_bufs, "states": states, "team_rew": team_rews,
           "val": values, "cuts": cuts, "last_state": env.global_state()}
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
            actions = mgr.mean_actions_all(obs)
            obs, _, team_r, done, info = env.step(actions)
            ep_ret += team_r
        rets.append(ep_ret)
        dists.append(info["mean_dist"])
        colls.append(info["collisions"])
    return float(np.mean(rets)), float(np.mean(dists)), float(np.mean(colls))


def run_training(encoder, args, device):
    set_seed(args.seed)
    env = CooperativeNavEnv(args.n_agents, args.episode_len, seed=args.seed,
                            shuffle_slots=not args.no_shuffle)
    eval_env = CooperativeNavEnv(args.n_agents, args.episode_len, seed=args.seed + 1,
                                 shuffle_slots=not args.no_shuffle)
    mgr = Manager(env, args, device, encoder)
    n_params = sum(p.numel() for p in mgr.actor.parameters())
    label = "SYM" if encoder == "sym" else "NAIVE"
    print(f"[{label}] 策略参数量 {n_params} | 槽位洗牌: {not args.no_shuffle}")

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
              f"近20回合团队回报 {recent:7.2f} | KL {stats['approx_kl']:.4f} | "
              f"裁剪 {stats['clip_frac']:.3f} | 批评家EV {stats['ev']:.3f} | "
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
    p = argparse.ArgumentParser(description="第072章：多智能体 PPO 的角色对称性")
    p.add_argument("--encoder", type=str, default="both", choices=["both", "naive", "sym"])
    p.add_argument("--no-shuffle", action="store_true",
                   help="关闭每回合槽位洗牌（用于对照实验）")
    p.add_argument("--n-agents", type=int, default=3, help="智能体数量")
    p.add_argument("--episode-len", type=int, default=50, help="回合长度")
    p.add_argument("--max-steps", type=int, default=120000, help="每种配置的智能体步预算")
    p.add_argument("--steps-per-update", type=int, default=3000, help="每次更新采样步数")
    p.add_argument("--eval-every", type=int, default=30000, help="评估间隔")
    p.add_argument("--epochs", type=int, default=8, help="更新轮数")
    p.add_argument("--batch-size", type=int, default=256, help="小批量大小")
    p.add_argument("--hidden", type=int, default=64, help="隐藏层宽度")
    p.add_argument("--lr", type=float, default=3e-4, help="学习率")
    p.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
    p.add_argument("--lam", type=float, default=0.95, help="GAE lambda")
    p.add_argument("--clip", type=float, default=0.2, help="PPO 裁剪范围")
    p.add_argument("--eval-episodes", type=int, default=5, help="评估回合数")
    p.add_argument("--seed", type=int, default=0, help="随机种子")
    p.add_argument("--save-dir", type=str, default="runs/ch072", help="保存目录")
    p.add_argument("--quick", action="store_true", help="快速模式：24000 步/配置")
    return p.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.max_steps = min(args.max_steps, 24000)
        args.steps_per_update = 1500
        args.eval_every = 8000
        args.epochs = 5
        args.eval_episodes = 3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device} | 环境: 槽位随机洗牌的合作导航 | 种子 {args.seed} | "
          f"每配置预算 {args.max_steps} 智能体步")

    results = {}
    if args.encoder in ("naive", "both"):
        results["NAIVE"] = run_training("naive", args, device)
    if args.encoder in ("sym", "both"):
        results["SYM"] = run_training("sym", args, device)

    print("=" * 76)
    for name, (curve, elapsed, final) in results.items():
        best = max(r for _, r in curve) if curve else float("nan")
        print(f"[{name:6s}] 用时 {elapsed:6.1f}s | 最佳评估 {best:7.2f} | "
              f"最终（回报/末距/碰撞） {final[0]:7.2f} / {final[1]:.3f} / {final[2]:.2f}")
    if len(results) == 2:
        print("提示：sym 编码器把'顺序无关'写进结构，naive 则要靠数据学会它。"
              "两者最终可能接近，差距通常体现在前 1/3 的训练窗口。")

    os.makedirs(args.save_dir, exist_ok=True)
    for name, (curve, _, _) in results.items():
        path = os.path.join(args.save_dir, f"{name.lower()}_curve.txt")
        with open(path, "w", encoding="utf-8") as f:
            for step, ret in curve:
                f.write(f"{step}\t{ret:.3f}\n")
    print(f"评估曲线已保存到 {args.save_dir}")


if __name__ == "__main__":
    main()
