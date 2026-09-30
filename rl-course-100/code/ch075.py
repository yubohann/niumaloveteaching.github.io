"""
第075章 多智能体PPO：混合合作竞争

2 个追捕者（合作小队）对抗 1 个逃跑者，奖励零和：
  - 每步：追捕队得 -min距离，逃跑者得 +min距离
  - 被抓（距离 < 0.1）：追捕队 +5，逃跑者 -5，回合结束
对抗两种训练方式：
  - frozen  ：只有追捕队学习，逃跑者固定为脚本"远离"策略
  - selfplay：两个阵营同时学习（混合合作竞争：队内合作、队间竞争）
评估时把追捕队放到三种逃跑者面前：脚本随机 / 脚本逃跑 / 训练出的逃跑者。

运行：
    python code/ch075.py           # 完整对照（CPU 约 5-12 分钟）
    python code/ch075.py --quick   # 快速跑通（约 1 分钟）
    python code/ch075.py --mode selfplay

预期（以实跑为准）：selfplay 训练出的追捕队在面对"会学习的逃跑者"时不落下风，
且对脚本逃跑者保持不错成绩；frozen 追捕队在脚本对手上分数很高，但面对新风格
对手时泛化通常更弱。selfplay 的曲线会有明显的此消彼长（非平稳性）。
"""

import argparse
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
# 环境：2 追 1 混合合作竞争
# ===========================================================================
class TeamTagEnv:
    """2 追 1 追捕环境，零和奖励，双方均在单位方格内。

    追捕者观测（每个 12 维）：自身位置/速度(4) + 逃跑者相对位置/速度(4)
                            + 队友相对位置/速度(4)
    逃跑者观测（12 维）    ：自身位置/速度(4) + 追捕者 1 相对位置/速度(4)
                            + 追捕者 2 相对位置/速度(4)
    全局状态（13 维）      ：所有位置/速度 + 归一化时间
    """

    def __init__(self, episode_len=60, catch_dist=0.1, seed=0):
        self.episode_len = episode_len
        self.catch_dist = catch_dist
        self.obs_dim = 12
        self.state_dim = 13
        self.act_dim = 2
        self.damping, self.accel = 0.9, 0.2
        self._rng = np.random.RandomState(seed)
        self.chaser_p = np.zeros((2, 2), np.float32)
        self.chaser_v = np.zeros((2, 2), np.float32)
        self.runner_p = np.zeros(2, np.float32)
        self.runner_v = np.zeros(2, np.float32)
        self.t = 0

    def reset(self):
        self.chaser_p = self._rng.uniform(0.15, 0.45, (2, 2)).astype(np.float32)
        self.runner_p = self._rng.uniform(0.55, 0.85, 2).astype(np.float32)
        self.chaser_v = np.zeros((2, 2), np.float32)
        self.runner_v = np.zeros(2, np.float32)
        self.t = 0
        return self._obs()

    def _obs(self):
        obs_c = np.zeros((2, self.obs_dim), np.float32)
        rel_runner = self.runner_p - self.chaser_p
        for i in range(2):
            rel_teammate = self.chaser_p[1 - i] - self.chaser_p[i]
            obs_c[i] = np.concatenate([
                self.chaser_p[i], self.chaser_v[i],
                rel_runner[i], self.runner_v - self.chaser_v[i],
                rel_teammate, self.chaser_v[1 - i] - self.chaser_v[i]])
        obs_r = np.concatenate([
            self.runner_p, self.runner_v,
            self.chaser_p[0] - self.runner_p, self.chaser_v[0] - self.runner_v,
            self.chaser_p[1] - self.runner_p, self.chaser_v[1] - self.runner_v])
        return obs_c, obs_r.astype(np.float32)

    def global_state(self) -> np.ndarray:
        return np.concatenate([self.chaser_p.ravel(), self.chaser_v.ravel(),
                               self.runner_p, self.runner_v,
                               [self.t / self.episode_len]]).astype(np.float32)

    def step(self, chaser_actions, runner_action):
        chaser_actions = np.clip(np.asarray(chaser_actions, np.float32), -1.0, 1.0)
        runner_action = np.clip(np.asarray(runner_action, np.float32), -1.0, 1.0)
        self.chaser_v = self.damping * self.chaser_v + self.accel * chaser_actions
        self.runner_v = self.damping * self.runner_v + self.accel * runner_action
        self.chaser_p = np.clip(self.chaser_p + self.chaser_v, 0.0, 1.0)
        self.runner_p = np.clip(self.runner_p + self.runner_v, 0.0, 1.0)
        self.t += 1

        dists = np.linalg.norm(self.chaser_p - self.runner_p, axis=1)
        min_dist = float(dists.min())
        caught = min_dist < self.catch_dist
        # 零和奖励：追捕队拿 -距离，逃跑者拿 +距离；捕获给 ±5
        team_reward = -min_dist + (5.0 if caught else 0.0)
        runner_reward = min_dist + (-5.0 if caught else 0.0)
        done = caught or self.t >= self.episode_len
        info = {"min_dist": min_dist, "caught": bool(caught)}
        obs_c, obs_r = self._obs()
        return obs_c, obs_r, team_reward, runner_reward, done, info


def scripted_flee_action(env, mode="flee", rng=None, t=0):
    """脚本对手：flee 远离追捕者；random 随机游走。"""
    rng = rng if rng is not None else np.random
    if mode == "random":
        return rng.uniform(-1.0, 1.0, 2).astype(np.float32)
    d = env.runner_p - env.chaser_p.mean(axis=0)
    norm = np.linalg.norm(d) + 1e-6
    action = d / norm + rng.normal(0.0, 0.15, 2)
    return np.clip(action, -1.0, 1.0).astype(np.float32)


# ===========================================================================
# 网络
# ===========================================================================
class GaussianPolicy(nn.Module):
    """连续动作高斯策略（同一阵营的成员共享）。"""

    def __init__(self, obs_dim, act_dim, hidden=64):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(),
                                  nn.Linear(hidden, hidden), nn.Tanh())
        self.mean = nn.Linear(hidden, act_dim)
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.7))

    def forward(self, obs):
        h = self.body(obs)
        mean = self.mean(h)
        log_std = torch.clamp(self.log_std, -4.0, 0.5).expand_as(mean)
        return mean, log_std

    @torch.no_grad()
    def sample(self, obs):
        mean, log_std = self.forward(obs)
        dist = Normal(mean, log_std.exp())
        action = torch.clamp(dist.sample(), -1.0, 1.0)
        return action, dist.log_prob(action).sum(-1)

    @torch.no_grad()
    def mean_action(self, obs):
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


def ppo_update(policy, optim, obs, act, logp_old, adv, args) -> dict:
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    N = obs.shape[0]
    stats = {"policy_loss": 0.0, "approx_kl": 0.0, "clip_frac": 0.0, "n": 0}
    for _ in range(args.epochs):
        idx = torch.randperm(N, device=obs.device)
        for start in range(0, N, args.batch_size):
            mb = idx[start:start + args.batch_size]
            mean, log_std = policy(obs[mb])
            dist = Normal(mean, log_std.exp())
            logp = dist.log_prob(act[mb]).sum(-1)
            ratio = torch.exp(logp - logp_old[mb])
            surr1 = ratio * adv[mb]
            surr2 = torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * adv[mb]
            loss = -torch.min(surr1, surr2).mean()
            optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
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


class Team:
    """一个阵营：策略 + 中央批评家 + 优化器 + 采样缓冲。"""

    def __init__(self, obs_dim, state_dim, act_dim, args, device):
        self.policy = GaussianPolicy(obs_dim, act_dim, args.hidden).to(device)
        self.critic = CentralCritic(state_dim, args.hidden).to(device)
        self.policy_opt = torch.optim.Adam(self.policy.parameters(), lr=args.lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=args.lr)
        self.device = device

    @torch.no_grad()
    def value_of(self, states_t):
        return self.critic(states_t)


# ===========================================================================
# 采样、更新与评估（单次训练运行）
# ===========================================================================
def policy_step(policy, obs, device):
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
    if obs_t.dim() == 1:
        obs_t = obs_t.unsqueeze(0)
    actions, logps = policy.sample(obs_t)
    return actions.cpu().numpy(), logps.cpu().numpy()


def run_training(mode, args, device):
    """mode: frozen（逃跑者脚本）或 selfplay（双方学习）。"""
    set_seed(args.seed)
    env = TeamTagEnv(args.episode_len, seed=args.seed)
    chasers = Team(env.obs_dim, env.state_dim, env.act_dim, args, device)
    runner = Team(env.obs_dim, env.state_dim, env.act_dim, args, device)
    rng = np.random.RandomState(args.seed + 7)

    obs_c, obs_r = env.reset()
    total_steps, update_idx = 0, 0
    catches_hist = []
    t0 = time.time()
    caption = "SELFPLAY" if mode == "selfplay" else "FROZEN"

    # 采样缓冲
    cb = {k: [] for k in ["obs", "act", "logp", "rew", "state", "val", "cut"]}
    rb = {k: [] for k in ["obs", "act", "logp", "rew", "state", "val", "cut"]}

    while total_steps < args.max_steps:
        ep_catch = []
        ep_ret_c = 0.0
        for _ in range(args.steps_per_update):
            gs = env.global_state()
            chaser_actions, chaser_logps = policy_step(chasers.policy, obs_c, device)
            if mode == "selfplay":
                runner_action, runner_logp = policy_step(runner.policy, obs_r, device)
                runner_logp = float(runner_logp[0])
            else:
                runner_action = scripted_flee_action(env, "flee", rng)
                runner_logp = 0.0
            next_obs_c, next_obs_r, r_c, r_r, done, info = env.step(chaser_actions, runner_action)

            cb["obs"].append(obs_c)
            cb["act"].append(chaser_actions)
            cb["logp"].append(chaser_logps)
            cb["rew"].append(np.full(2, r_c, np.float32))
            cb["state"].append(gs)
            cb["val"].append(float(chasers.value_of(torch.as_tensor(
                gs, dtype=torch.float32, device=device).unsqueeze(0)).item()))
            cb["cut"].append(float(done))
            rb["obs"].append(obs_r)
            rb["act"].append(runner_action)
            rb["logp"].append(runner_logp)
            rb["rew"].append(r_r)
            rb["state"].append(gs)
            rb["val"].append(float(runner.value_of(torch.as_tensor(
                gs, dtype=torch.float32, device=device).unsqueeze(0)).item()))
            rb["cut"].append(float(done))

            ep_ret_c += r_c
            obs_c, obs_r = next_obs_c, next_obs_r
            if done:
                catches_hist.append(1.0 if info["caught"] else 0.0)
                ep_catch.append(info["caught"])
                ep_ret_c = 0.0
                obs_c, obs_r = env.reset()

        # ---- 追捕队更新：团队奖励 + 中央价值 ----
        last_gs = env.global_state()
        last_val_c = float(chasers.value_of(torch.as_tensor(
            last_gs, dtype=torch.float32, device=device).unsqueeze(0)).item())
        adv_c, ret_c = compute_gae([float(x[0]) for x in cb["rew"]], cb["val"], cb["cut"],
                                   last_val_c, args.gamma, args.lam)
        adv_c2 = np.stack([adv_c, adv_c], axis=1)          # 队友共享同一优势
        cb["adv"] = adv_c2
        cb["ret"] = np.stack([ret_c, ret_c], axis=1)
        stats_c = ppo_update(
            chasers.policy, chasers.policy_opt,
            torch.as_tensor(np.asarray(cb["obs"]), dtype=torch.float32,
                            device=device).reshape(-1, env.obs_dim),
            torch.as_tensor(np.asarray(cb["act"]), dtype=torch.float32,
                            device=device).reshape(-1, env.act_dim),
            torch.as_tensor(np.asarray(cb["logp"]), dtype=torch.float32,
                            device=device).reshape(-1),
            torch.as_tensor(adv_c2.reshape(-1), dtype=torch.float32, device=device),
            args)
        # 追捕队中央批评家
        states_t = torch.as_tensor(np.asarray(cb["state"]), dtype=torch.float32, device=device)
        ret_t = torch.as_tensor(ret_c, dtype=torch.float32, device=device)
        for _ in range(args.epochs):
            loss = F.mse_loss(chasers.critic(states_t), ret_t)
            chasers.critic_opt.zero_grad()
            loss.backward()
            chasers.critic_opt.step()

        # ---- 逃跑者更新（仅 selfplay）----
        if mode == "selfplay":
            last_val_r = float(runner.value_of(torch.as_tensor(
                last_gs, dtype=torch.float32, device=device).unsqueeze(0)).item())
            adv_r, ret_r = compute_gae([float(x) for x in rb["rew"]], rb["val"], rb["cut"],
                                       last_val_r, args.gamma, args.lam)
            rb["adv"], rb["ret"] = adv_r, ret_r
            stats_r = ppo_update(
                runner.policy, runner.policy_opt,
                torch.as_tensor(np.asarray(rb["obs"]), dtype=torch.float32, device=device),
                torch.as_tensor(np.asarray(rb["act"]), dtype=torch.float32, device=device),
                torch.as_tensor(np.asarray(rb["logp"]), dtype=torch.float32, device=device),
                torch.as_tensor(adv_r, dtype=torch.float32, device=device), args)
            states_t = torch.as_tensor(np.asarray(rb["state"]), dtype=torch.float32, device=device)
            ret_t = torch.as_tensor(ret_r, dtype=torch.float32, device=device)
            for _ in range(args.epochs):
                loss = F.mse_loss(runner.critic(states_t), ret_t)
                runner.critic_opt.zero_grad()
                loss.backward()
                runner.critic_opt.step()
        else:
            stats_r = {"approx_kl": float("nan"), "clip_frac": float("nan")}

        for k in ["obs", "act", "logp", "rew", "state", "val", "cut"]:
            cb[k].clear()
            rb[k].clear()
        total_steps += args.steps_per_update
        update_idx += 1
        recent_catch = np.mean(catches_hist[-40:]) if catches_hist else float("nan")
        print(f"[{caption} 更新 {update_idx:3d}] 环境步 {total_steps:6d} | "
              f"近40回合捕获率 {recent_catch:.2f} | 追捕KL {stats_c['approx_kl']:.4f} | "
              f"逃跑KL {stats_r['approx_kl']:.4f} | 用时 {time.time() - t0:5.1f}s")

    # ---- 评估矩阵 ----
    results = {}
    for opp in ["random", "flee"]:
        results[opp] = evaluate(env, chasers.policy, None, opp, args.eval_episodes * 2, device, rng)
        print(f"[{caption} 评估] 追捕 vs {opp:6s} 捕获率 {results[opp][0]:.2f} | "
              f"平均捕获步数 {results[opp][1]:.1f} | 末距 {results[opp][2]:.3f}")
    if mode == "selfplay":
        results["learned"] = evaluate(env, chasers.policy, runner.policy, None,
                                      args.eval_episodes * 2, device, rng)
        print(f"[{caption} 评估] 追捕 vs 学习逃跑者 捕获率 {results['learned'][0]:.2f} | "
              f"平均捕获步数 {results['learned'][1]:.1f} | 末距 {results['learned'][2]:.3f}")

    os.makedirs(args.save_dir, exist_ok=True)
    torch.save(chasers.policy.state_dict(),
               os.path.join(args.save_dir, f"chasers_{mode}.pt"))
    if mode == "selfplay":
        torch.save(runner.policy.state_dict(),
                   os.path.join(args.save_dir, "runner_selfplay.pt"))
    return results, time.time() - t0


@torch.no_grad()
def evaluate(env, chaser_policy, runner_policy, script_mode, episodes, device, rng):
    """评估：追捕队用确定性动作；逃跑者可用学习策略或脚本策略。"""
    catches, steps_list, dists = [], [], []
    for _ in range(episodes):
        obs_c, obs_r = env.reset()
        done = False
        steps = 0
        info = {"min_dist": 1.0, "caught": False}
        while not done:
            c_obs_t = torch.as_tensor(obs_c, dtype=torch.float32, device=device)
            chaser_actions = chaser_policy.mean_action(c_obs_t).cpu().numpy()
            if runner_policy is not None:
                r_obs_t = torch.as_tensor(obs_r, dtype=torch.float32, device=device).unsqueeze(0)
                runner_action = runner_policy.mean_action(r_obs_t).squeeze(0).cpu().numpy()
            else:
                runner_action = scripted_flee_action(env, script_mode, rng)
            obs_c, obs_r, _, _, done, info = env.step(chaser_actions, runner_action)
            steps += 1
        catches.append(1.0 if info["caught"] else 0.0)
        steps_list.append(steps if info["caught"] else env.episode_len)
        dists.append(info["min_dist"])
    return float(np.mean(catches)), float(np.mean(steps_list)), float(np.mean(dists))


def parse_args():
    p = argparse.ArgumentParser(description="第075章：多智能体 PPO 的混合合作竞争")
    p.add_argument("--mode", type=str, default="both", choices=["both", "frozen", "selfplay"])
    p.add_argument("--episode-len", type=int, default=60, help="回合长度")
    p.add_argument("--max-steps", type=int, default=80000, help="每种模式的环境步预算")
    p.add_argument("--steps-per-update", type=int, default=2000, help="每次更新的采样步数")
    p.add_argument("--epochs", type=int, default=8, help="更新轮数")
    p.add_argument("--batch-size", type=int, default=256, help="小批量大小")
    p.add_argument("--hidden", type=int, default=64, help="隐藏层宽度")
    p.add_argument("--lr", type=float, default=3e-4, help="学习率")
    p.add_argument("--gamma", type=float, default=0.95, help="折扣因子")
    p.add_argument("--lam", type=float, default=0.95, help="GAE lambda")
    p.add_argument("--clip", type=float, default=0.2, help="PPO 裁剪范围")
    p.add_argument("--eval-episodes", type=int, default=15, help="每种评估对手的回合数")
    p.add_argument("--seed", type=int, default=0, help="随机种子")
    p.add_argument("--save-dir", type=str, default="runs/ch075", help="模型保存目录")
    p.add_argument("--quick", action="store_true", help="快速模式：20000 步/模式")
    return p.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.max_steps = min(args.max_steps, 20000)
        args.steps_per_update = 1000
        args.epochs = 5
        args.eval_episodes = 5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device} | 环境: 2 追 1 混合合作竞争 | 种子 {args.seed} | "
          f"每模式预算 {args.max_steps} 环境步")

    results = {}
    if args.mode in ("frozen", "both"):
        results["FROZEN"], dt = run_training("frozen", args, device)
    if args.mode in ("selfplay", "both"):
        results["SELFPLAY"], dt = run_training("selfplay", args, device)

    print("=" * 76)
    for name, res in results.items():
        parts = " | ".join(
            f"vs {k}: 捕获率 {v[0]:.2f}（{v[1]:.1f} 步）" for k, v in res.items())
        print(f"[{name:8s}] {parts}")
    if len(results) == 2:
        print("提示：selfplay 的曲线会出现此消彼长（非平稳性）；比较两种训练方式的"
              "泛化矩阵，而不是只看对训练对手的成绩。")
    print(f"模型已保存到 {args.save_dir}")


if __name__ == "__main__":
    main()
