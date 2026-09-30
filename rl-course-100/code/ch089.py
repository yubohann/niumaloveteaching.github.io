"""
第089章 多智能体环境：足球

实验内容（2v2 简化足球，环境为脚本内联的纯 numpy 实现）：
  - 学习者：A 队 2 名球员（参数共享的独立 PPO 训练，一组网络服务两人）
  - 脚本对手：B 队 2 名球员——一人绕到球后往左路推进，一人守在球与自家
    球门之间，形成基本攻守分工
  - 球有独立物理：摩擦减速、与球员接触时获得球员速度与踢球冲量、边界反弹
  - 奖励（A 队共享）：-0.05×球到对方球门距离（推进塑形）+ 0.03×控球
    - 0.005 时间成本，进球 +5、失球 -5 并终止
  - 指标：对脚本对手的胜率、场均进球、控球率；随机动作基线对照

运行：
    python code/ch089.py           # 完整训练（CPU 约 3-6 分钟）
    python code/ch089.py --quick   # 快速跑通（约 30-60 秒）

预期：随机基线的胜率约 0.0～0.1；训练后胜率通常升到 0.4～0.7（以实跑为准），
进球主要来自边路带球后的门前推射。
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

LOG_STD_MIN, LOG_STD_MAX = -20.0, 2.0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# 环境：2v2 简化足球
# ---------------------------------------------------------------------------
class SoccerEnv:
    """A 队（两名学习球员）进攻右侧球门，B 队（脚本）进攻左侧球门。

    观测（18 维/学习球员）：自己位置+速度(4)、队友相对位置+速度(4)、
    球相对位置+球速度(4)、两名对手相对位置(4)、对方球门中心相对位置(2)。
    动作：2 维加速度 [-1,1]。控球判定：球员与球距离 < control_radius。
    """

    obs_dim = 18
    act_dim = 2
    n_learners = 2
    goal_y_half = 0.35          # 球门半宽
    field_x, field_y = 1.4, 0.8

    def __init__(self, max_steps: int = 50, friction: float = 0.8,
                 accel: float = 0.12, v_max: float = 1.0,
                 control_radius: float = 0.16, ball_friction: float = 0.96,
                 kick: float = 0.1, ball_v_max: float = 1.3):
        self.max_steps = max_steps
        self.friction = friction
        self.accel = accel
        self.v_max = v_max
        self.control_radius = control_radius
        self.ball_friction = ball_friction
        self.kick = kick
        self.ball_v_max = ball_v_max

    def reset(self, rng: np.random.Generator):
        # 4 名球员：0/1 为 A 队（左半场），2/3 为 B 队（右半场）
        self.pos = np.array([
            [-0.7, -0.3], [-0.7, 0.3],
            [0.7, -0.3], [0.7, 0.3],
        ], dtype=np.float32) + rng.uniform(-0.05, 0.05, size=(4, 2)).astype(np.float32)
        self.vel = np.zeros((4, 2), dtype=np.float32)
        self.ball_pos = np.array([0.0, float(rng.uniform(-0.2, 0.2))], dtype=np.float32)
        self.ball_vel = np.zeros(2, dtype=np.float32)
        self.goals = [0, 0]          # [A 队进球, B 队进球]
        self.steps = 0
        return self._obs()

    def _obs(self) -> np.ndarray:
        obs = []
        opponent_goal = np.array([self.field_x, 0.0], dtype=np.float32)
        for i in range(self.n_learners):
            mate = 1 - i
            opps = [2, 3]
            o = np.concatenate([
                self.pos[i], self.vel[i],
                self.pos[mate] - self.pos[i], self.vel[mate],
                self.ball_pos - self.pos[i], self.ball_vel,
                self.pos[opps[0]] - self.pos[i],
                self.pos[opps[1]] - self.pos[i],
                opponent_goal - self.pos[i],
            ])
            obs.append(o.astype(np.float32))
        return np.stack(obs)

    def _scripted_opponent(self) -> np.ndarray:
        """B 队脚本策略：2 号绕后推进，3 号留守在球与自家球门之间。"""
        target_goal = np.array([-self.field_x, 0.0], dtype=np.float32)  # B 队进攻目标
        own_goal = np.array([self.field_x, 0.0], dtype=np.float32)
        actions = np.zeros((2, 2), dtype=np.float32)

        # 2 号：站到"球的后方"（相对目标球门而言），朝球门方向推
        d = self.ball_pos - target_goal
        approach = self.ball_pos + d / (np.linalg.norm(d) + 1e-8) * 0.18
        actions[0] = np.clip((approach - self.pos[2]) * 4.0, -1.0, 1.0)

        # 3 号：占住球与自家球门的连线中点
        guard = self.ball_pos + (own_goal - self.ball_pos) * 0.35
        actions[1] = np.clip((guard - self.pos[3]) * 4.0, -1.0, 1.0)
        return actions

    def step(self, actions: np.ndarray):
        """actions: (2,2) 学习方动作。返回 obs, rewards, terminated, truncated, info。"""
        a_all = np.zeros((4, 2), dtype=np.float32)
        a_all[:2] = np.clip(actions, -1.0, 1.0)
        a_all[2:] = self._scripted_opponent()

        # 1) 球员动力学
        self.vel = np.clip(self.friction * self.vel + self.accel * a_all,
                           -self.v_max, self.v_max)
        self.pos = np.clip(self.pos + self.vel,
                           [-self.field_x, -self.field_y],
                           [self.field_x, self.field_y])

        # 2) 球物理：摩擦 + 球员接触传递动量
        self.ball_vel *= self.ball_friction
        for i in range(4):
            d = self.ball_pos - self.pos[i]
            dist = float(np.linalg.norm(d)) + 1e-8
            if dist < self.control_radius:
                self.ball_vel = (self.ball_vel + 0.6 * self.vel[i]
                                 + (d / dist) * self.kick)
        self.ball_vel = np.clip(self.ball_vel, -self.ball_v_max, self.ball_v_max)
        self.ball_pos = self.ball_pos + self.ball_vel
        self.steps += 1

        # 3) 进球与边界
        reward_a, reward_b = 0.0, 0.0
        terminated = False
        if self.ball_pos[0] > self.field_x:
            if abs(self.ball_pos[1]) < self.goal_y_half:
                self.goals[0] += 1
                reward_a, reward_b = 5.0, -5.0
                terminated = True
            else:
                self.ball_pos[0] = 2 * self.field_x - self.ball_pos[0]
                self.ball_vel[0] *= -0.5
        elif self.ball_pos[0] < -self.field_x:
            if abs(self.ball_pos[1]) < self.goal_y_half:
                self.goals[1] += 1
                reward_a, reward_b = -5.0, 5.0
                terminated = True
            else:
                self.ball_pos[0] = -2 * self.field_x - self.ball_pos[0]
                self.ball_vel[0] *= -0.5
        if abs(self.ball_pos[1]) > self.field_y:
            self.ball_pos[1] = np.clip(self.ball_pos[1], -self.field_y, self.field_y)
            self.ball_vel[1] *= -0.5

        # 4) 学习方奖励：推进塑形 + 控球 + 时间成本
        dist_to_goal = float(np.linalg.norm(
            self.ball_pos - np.array([self.field_x, 0.0], dtype=np.float32)))
        controls = any(np.linalg.norm(self.ball_pos - self.pos[i]) < self.control_radius
                       for i in range(2))
        reward = -0.005 - 0.05 * dist_to_goal + (0.03 if controls else 0.0)
        reward += reward_a
        rewards = np.array([reward, reward], dtype=np.float32)

        self.ball_vel = np.clip(self.ball_vel, -self.ball_v_max, self.ball_v_max)
        truncated = (not terminated) and self.steps >= self.max_steps
        info = {"goals": tuple(self.goals), "controls": bool(controls),
                "dist_to_goal": dist_to_goal}
        return self._obs(), rewards, terminated, truncated, info


# ---------------------------------------------------------------------------
# 网络与 PPO（与第 087 章同族）
# ---------------------------------------------------------------------------
class PPOActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.mu_head = nn.Linear(hidden, act_dim)
        self.v_head = nn.Linear(hidden, 1)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs: torch.Tensor):
        h = self.body(obs)
        mu = self.mu_head(h)
        std = torch.clamp(self.log_std, LOG_STD_MIN, LOG_STD_MAX).exp()
        v = self.v_head(h).squeeze(-1)
        return mu, std, v

    @torch.no_grad()
    def act_batch(self, obs_np: np.ndarray, device):
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
        mu, std, v = self.forward(obs)
        dist = Normal(mu, std)
        x = dist.sample()
        action = torch.tanh(x)
        logp = dist.log_prob(x) - torch.log(1.0 - action.pow(2) + 1e-6)
        return (action.cpu().numpy().astype(np.float32),
                logp.sum(dim=-1).cpu().numpy().astype(np.float32),
                v.cpu().numpy().astype(np.float32))

    @torch.no_grad()
    def act_deterministic(self, obs_np: np.ndarray, device) -> np.ndarray:
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
        mu, _, _ = self.forward(obs)
        return torch.tanh(mu).cpu().numpy().astype(np.float32)

    def log_prob_and_value(self, obs, act):
        mu, std, v = self.forward(obs)
        dist = Normal(mu, std)
        a = torch.clamp(act, -0.999999, 0.999999)
        x = torch.atanh(a)
        logp = dist.log_prob(x) - torch.log(1.0 - a.pow(2) + 1e-6)
        return logp.sum(dim=-1), v, dist.entropy().sum(dim=-1)


def compute_gae(rewards, values, dones, last_values, gamma, lam):
    T, N = values.shape
    adv = np.zeros((T, N), dtype=np.float32)
    last = np.zeros(N, dtype=np.float32)
    for t in reversed(range(T)):
        next_v = last_values if t == T - 1 else values[t + 1]
        non_term = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_v * non_term - values[t]
        last = delta + gamma * lam * non_term * last
        adv[t] = last
    return adv, adv + values


def collect_rollout(env, net, rng, steps, obs, device, args):
    obs_buf, act_buf, logp_buf, val_buf = [], [], [], []
    rew_buf, done_buf = [], []
    ep_stats = []
    ep_ret, ep_len, ep_controls = 0.0, 0, 0

    for _ in range(steps):
        a, logp, v = net.act_batch(obs, device)
        next_obs, rew, terminated, truncated, info = env.step(a)
        obs_buf.append(obs)
        act_buf.append(a)
        logp_buf.append(logp)
        val_buf.append(v)
        rew_buf.append(float(rew[0]))
        done_buf.append(float(terminated))
        ep_ret += float(rew[0])
        ep_len += 1
        ep_controls += int(info["controls"])
        if terminated or truncated:
            ep_stats.append({"return": ep_ret, "goals": info["goals"],
                             "controls": ep_controls / max(ep_len, 1)})
            obs = env.reset(rng)
            ep_ret, ep_len, ep_controls = 0.0, 0, 0
        else:
            obs = next_obs

    with torch.no_grad():
        _, _, last_v = net.forward(
            torch.as_tensor(obs, dtype=torch.float32, device=device))
        last_values = last_v.cpu().numpy().astype(np.float32)
    values = np.stack(val_buf)
    adv, ret = compute_gae(np.asarray(rew_buf, dtype=np.float32), values,
                           np.asarray(done_buf, dtype=np.float32),
                           last_values, args.gamma, args.lam)
    T, N = adv.shape
    batch = {
        "obs": np.stack(obs_buf).reshape(T * N, -1),
        "act": np.stack(act_buf).reshape(T * N, -1),
        "logp": np.stack(logp_buf).reshape(T * N),
        "adv": adv.reshape(T * N),
        "ret": ret.reshape(T * N),
    }
    return batch, ep_stats, obs


def ppo_update(net, optimizer, batch, args, device):
    obs = torch.as_tensor(batch["obs"], dtype=torch.float32, device=device)
    act = torch.as_tensor(batch["act"], dtype=torch.float32, device=device)
    logp_old = torch.as_tensor(batch["logp"], dtype=torch.float32, device=device)
    adv = torch.as_tensor(batch["adv"], dtype=torch.float32, device=device)
    ret = torch.as_tensor(batch["ret"], dtype=torch.float32, device=device)
    N = obs.shape[0]
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    stats = {"kl": 0.0, "clip_frac": 0.0, "n": 0}
    for _ in range(args.epochs):
        idx = torch.randperm(N, device=device)
        for start in range(0, N, args.batch_size):
            mb = idx[start:start + args.batch_size]
            logp, v, entropy = net.log_prob_and_value(obs[mb], act[mb])
            ratio = torch.exp(logp - logp_old[mb])
            surr1 = ratio * adv[mb]
            surr2 = torch.clamp(ratio, 1.0 - args.clip, 1.0 + args.clip) * adv[mb]
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(v, ret[mb])
            loss = policy_loss + 0.5 * value_loss - args.ent_coef * entropy.mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            optimizer.step()
            with torch.no_grad():
                stats["kl"] += (logp_old[mb] - logp).mean().item()
                stats["clip_frac"] += ((ratio - 1.0).abs() > args.clip).float().mean().item()
                stats["n"] += 1
    stats["kl"] /= max(stats["n"], 1)
    stats["clip_frac"] /= max(stats["n"], 1)
    return stats


# ---------------------------------------------------------------------------
# 评估：学习方 vs 脚本对手
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(net, episodes, device, seed, random_policy=False):
    env = SoccerEnv()
    rng = np.random.default_rng(seed)
    wins, draws, losses, goals_for, goals_against, controls = 0, 0, 0, 0, 0, 0.0
    for _ in range(episodes):
        obs = env.reset(rng)
        done = False
        n, ctrl = 0, 0
        info = {"goals": (0, 0), "controls": False}
        while not done:
            if random_policy:
                a = rng.uniform(-1.0, 1.0, size=(2, 2)).astype(np.float32)
            else:
                a = net.act_deterministic(obs, device)
            obs, rew, terminated, truncated, info = env.step(a)
            ctrl += int(info["controls"])
            n += 1
            done = terminated or truncated
        gf, ga = info["goals"]
        goals_for += gf
        goals_against += ga
        controls += ctrl / max(n, 1)
        if gf > ga:
            wins += 1
        elif gf == ga:
            draws += 1
        else:
            losses += 1
    e = max(episodes, 1)
    return (wins / e, draws / e, losses / e, goals_for / e,
            goals_against / e, controls / e)


def parse_args():
    p = argparse.ArgumentParser(description="第089章：多智能体环境 2v2 足球")
    p.add_argument("--max-steps", type=int, default=40000, help="环境步数上限")
    p.add_argument("--rollout", type=int, default=2048, help="每次更新采样步数")
    p.add_argument("--epochs", type=int, default=4, help="PPO 更新轮数")
    p.add_argument("--batch-size", type=int, default=256, help="小批量大小")
    p.add_argument("--lr", type=float, default=3e-4, help="学习率")
    p.add_argument("--gamma", type=float, default=0.95, help="折扣因子")
    p.add_argument("--lam", type=float, default=0.95, help="GAE lambda")
    p.add_argument("--clip", type=float, default=0.2, help="裁剪范围")
    p.add_argument("--ent-coef", type=float, default=0.01, help="熵正则系数")
    p.add_argument("--hidden", type=int, default=128, help="隐藏层宽度")
    p.add_argument("--eval-every", type=int, default=8000, help="评估间隔（步）")
    p.add_argument("--eval-episodes", type=int, default=20, help="评估回合数")
    p.add_argument("--seed", type=int, default=0, help="随机种子")
    p.add_argument("--save-dir", type=str, default="runs/ch089", help="模型保存目录")
    p.add_argument("--quick", action="store_true", help="快速模式：缩小训练规模")
    return p.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.max_steps = min(args.max_steps, 12000)
        args.rollout = 1024
        args.eval_every = 4000
        args.eval_episodes = 10
    set_seed(args.seed)

    device = torch.device("cpu")
    env = SoccerEnv()
    rng = np.random.default_rng(args.seed)
    net = PPOActorCritic(env.obs_dim, env.act_dim, args.hidden).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    print(f"设备 {device} | 2v2 简化足球 | 学习方 A 队（参数共享）| 种子 {args.seed}")

    # 随机基线
    w, d, l, gf, ga, ctrl = evaluate(net, args.eval_episodes, device,
                                     args.seed + 111, random_policy=True)
    print(f"随机基线：胜/平/负 {w:.2f}/{d:.2f}/{l:.2f} | 场均进球 {gf:.2f} | "
          f"控球率 {ctrl:.2f}")

    obs = env.reset(rng)
    total_steps, update_idx = 0, 0
    all_stats = []
    t_start = time.time()

    while total_steps < args.max_steps:
        batch, ep_stats, obs = collect_rollout(
            env, net, rng, args.rollout, obs, device, args)
        total_steps += args.rollout
        all_stats.extend(ep_stats)
        stats = ppo_update(net, optimizer, batch, args, device)
        update_idx += 1

        if total_steps % args.eval_every < args.rollout:
            w, d, l, gf, ga, ctrl = evaluate(
                net, args.eval_episodes, device, args.seed + 777)
            recent = all_stats[-30:]
            ep_goals = np.mean([s["goals"][0] for s in recent]) if recent else float("nan")
            print(f"[更新 {update_idx:3d}] 步 {total_steps:6d} | 胜率 {w:.2f} | "
                  f"平 {d:.2f} | 负 {l:.2f} | 场均进球 {gf:.2f}（训练 {ep_goals:.2f}）| "
                  f"控球 {ctrl:.2f} | KL {stats['kl']:.4f}")

    elapsed = time.time() - t_start
    w, d, l, gf, ga, ctrl = evaluate(net, args.eval_episodes, device, args.seed + 999)
    print("-" * 74)
    print(f"训练结束 | 用时 {elapsed:.1f}s | 环境步 {total_steps} | 更新 {update_idx} 次")
    print(f"评估（{args.eval_episodes} 回合，确定性策略）：胜/平/负 "
          f"{w:.2f}/{d:.2f}/{l:.2f} | 场均进球 {gf:.2f} | 场均失球 {ga:.2f} | "
          f"控球率 {ctrl:.2f}")

    os.makedirs(args.save_dir, exist_ok=True)
    path = os.path.join(args.save_dir, "ippo_soccer.pt")
    torch.save({"policy": net.state_dict(), "args": vars(args)}, path)
    print(f"模型已保存到 {path}")


if __name__ == "__main__":
    main()
