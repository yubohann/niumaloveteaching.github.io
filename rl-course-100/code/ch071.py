"""
第071章 多智能体PPO：通信机制

在"感知受限"的合作导航环境（队友超出感知半径就看不见）上，给共享策略加一条
通信通道：每个智能体在输出动作的同时广播一个 2 维"意图消息"（预测自己下一步
的位移），队友的消息会拼进自己的策略输入。消息头用自监督 MSE 损失训练。
对照两种配置：
  - off：无通信（队友看不见就真的看不见）
  - on ：有通信（远方队友的意图仍然送达）

运行：
    python code/ch071.py           # 完整对照（CPU 约 5-12 分钟）
    python code/ch071.py --quick   # 快速跑通（约 1 分钟）
    python code/ch071.py --comm on

预期（以实跑为准）：有通信的版本在感知半径 0.4 的设置下碰撞更少、协调更好；
但通信本身不改变任务信息，差距主要体现在"避让时机"而不是绝对上限。
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
# 环境：加入"感知半径"的合作导航
# ===========================================================================
class CooperativeNavEnv:
    """与第 068 章相同，但队友可见性受 sense_radius 限制。

    超出感知半径的队友相对位置被置零；地标保持全可见（任务仍然可解）。
    通信章节以此制造"信息缺口"，用来检验消息通道的价值。
    """

    def __init__(self, n_agents=3, episode_len=50, damping=0.9, accel=0.2,
                 collision_dist=0.1, collision_penalty=0.3, seed=0, sense_radius=3.0):
        self.n_agents = n_agents
        self.episode_len = episode_len
        self.damping = damping
        self.accel = accel
        self.collision_dist = collision_dist
        self.collision_penalty = collision_penalty
        self.sense_radius = sense_radius
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
            rel_others = []
            for j in range(self.n_agents):
                if j == i:
                    continue
                d = self.pos[j] - self.pos[i]
                if float(np.linalg.norm(d)) > self.sense_radius:
                    d = np.zeros(2, np.float32)   # 超出感知半径：队友"隐身"
                rel_others.append(d)
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
# 带通信头的共享策略
# ===========================================================================
class SharedPolicy(nn.Module):
    """共享策略：输入=局部观测(+收到的队友消息)，输出=动作分布 + 意图消息。"""

    def __init__(self, obs_dim, act_dim, n_agents, hidden=64,
                 use_comm=True, msg_dim=2, msg_scale=0.4):
        super().__init__()
        self.use_comm = use_comm
        self.msg_dim = msg_dim
        self.msg_scale = msg_scale
        in_dim = obs_dim + (n_agents - 1) * msg_dim if use_comm else obs_dim
        self.in_dim = in_dim
        self.body = nn.Sequential(nn.Linear(in_dim, hidden), nn.Tanh(),
                                  nn.Linear(hidden, hidden), nn.Tanh())
        self.mean = nn.Linear(hidden, act_dim)
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.7))
        if use_comm:
            self.msg_head = nn.Linear(hidden, msg_dim)

    def forward(self, x):
        h = self.body(x)
        mean = self.mean(h)
        log_std = torch.clamp(self.log_std, -4.0, 0.5).expand_as(mean)
        msg = torch.tanh(self.msg_head(h)) * self.msg_scale if self.use_comm else None
        return mean, log_std, msg

    @torch.no_grad()
    def act_batch(self, x):
        mean, log_std, msg = self.forward(x)
        dist = Normal(mean, log_std.exp())
        action = torch.clamp(dist.sample(), -1.0, 1.0)
        return action, dist.log_prob(action).sum(-1), msg

    @torch.no_grad()
    def mean_actions(self, x):
        mean, _, msg = self.forward(x)
        return torch.clamp(mean, -1.0, 1.0), msg


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


def build_policy_input(obs, messages, use_comm, n_agents):
    """把"自己的观测 + 收到的队友消息"拼成策略输入。"""
    if not use_comm:
        return obs.astype(np.float32)
    rows = []
    for i in range(n_agents):
        parts = [obs[i]]
        for j in range(n_agents):
            if j != i:
                parts.append(messages[j])
        rows.append(np.concatenate(parts))
    return np.asarray(rows, dtype=np.float32)


# ===========================================================================
# 策略管理器：训练与评估共用
# ===========================================================================
class PolicyManager:
    def __init__(self, env, args, device, use_comm):
        self.n = env.n_agents
        self.device = device
        self.use_comm = use_comm
        self.policy = SharedPolicy(env.obs_dim, env.act_dim, env.n_agents,
                                   args.hidden, use_comm, args.msg_dim).to(device)
        self.central = CentralCritic(env.state_dim, args.hidden).to(device)
        self.policy_opt = torch.optim.Adam(self.policy.parameters(), lr=args.lr)
        self.central_opt = torch.optim.Adam(self.central.parameters(), lr=args.lr)

    @torch.no_grad()
    def act_step(self, obs, messages, global_state):
        """训练期：随机采样动作，同时生成新消息。"""
        x = build_policy_input(obs, messages, self.use_comm, self.n)
        x_t = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        actions, logps, msg = self.policy.act_batch(x_t)
        state_t = torch.as_tensor(global_state, dtype=torch.float32,
                                  device=self.device).unsqueeze(0)
        value = float(self.central(state_t).item())
        msg_np = msg.cpu().numpy() if msg is not None else np.zeros((self.n, self.policy.msg_dim),
                                                                    np.float32)
        return x, actions.cpu().numpy(), [float(l) for l in logps], value, msg_np

    @torch.no_grad()
    def eval_step(self, obs, messages):
        """评估期：确定性动作 + 确定性消息。"""
        x = build_policy_input(obs, messages, self.use_comm, self.n)
        x_t = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        actions, msg = self.policy.mean_actions(x_t)
        msg_np = msg.cpu().numpy() if msg is not None else np.zeros((self.n, self.policy.msg_dim),
                                                                    np.float32)
        return actions.cpu().numpy(), msg_np

    def update(self, buf, args) -> dict:
        device = self.device
        # --- 中央批评家 ---
        states = torch.as_tensor(np.asarray(buf["states"]), dtype=torch.float32, device=device)
        last_state = torch.as_tensor(buf["last_state"], dtype=torch.float32,
                                     device=device).unsqueeze(0)
        with torch.no_grad():
            last_value = float(self.central(last_state).item())
        adv, ret = compute_gae(buf["team_rew"], buf["val"], buf["cuts"], last_value,
                               args.gamma, args.lam)
        ret_t = torch.as_tensor(ret, device=device)
        for _ in range(args.epochs):
            loss = F.mse_loss(self.central(states), ret_t)
            self.central_opt.zero_grad()
            loss.backward()
            self.central_opt.step()
        with torch.no_grad():
            ev = explained_variance(self.central(states), ret_t)

        # --- 策略 + 消息头 ---
        x = torch.as_tensor(np.asarray(buf["x"]), dtype=torch.float32, device=device)
        act = torch.as_tensor(np.asarray(buf["act"]), dtype=torch.float32, device=device)
        logp_old = torch.as_tensor(np.asarray(buf["logp"]), dtype=torch.float32, device=device)
        disp = torch.as_tensor(np.asarray(buf["disp"]), dtype=torch.float32, device=device)
        adv_t = torch.as_tensor(adv, device=device)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        T, N = x.shape[0], x.shape[1]
        x_flat = x.reshape(T * N, -1)
        act_flat = act.reshape(T * N, -1)
        logp_flat = logp_old.reshape(T * N)
        disp_flat = disp.reshape(T * N, -1)
        adv_flat = adv_t.unsqueeze(1).repeat(1, N).reshape(T * N)

        stats = {"policy_loss": 0.0, "aux_loss": 0.0, "approx_kl": 0.0,
                 "clip_frac": 0.0, "ev": ev, "n": 0}
        M = x_flat.shape[0]
        for _ in range(args.epochs):
            idx = torch.randperm(M, device=device)
            for start in range(0, M, args.batch_size):
                mb = idx[start:start + args.batch_size]
                mean, log_std, msg = self.policy(x_flat[mb])
                dist = Normal(mean, log_std.exp())
                logp = dist.log_prob(act_flat[mb]).sum(-1)
                ratio = torch.exp(logp - logp_flat[mb])
                surr1 = ratio * adv_flat[mb]
                surr2 = torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * adv_flat[mb]
                policy_loss = -torch.min(surr1, surr2).mean()
                if self.use_comm:
                    aux_loss = F.mse_loss(msg, disp_flat[mb])
                    loss = policy_loss + args.comm_coef * aux_loss
                else:
                    aux_loss = torch.tensor(0.0)
                    loss = policy_loss
                self.policy_opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.policy_opt.step()
                with torch.no_grad():
                    stats["policy_loss"] += policy_loss.item()
                    stats["aux_loss"] += float(aux_loss.item())
                    stats["approx_kl"] += (logp_flat[mb] - logp).mean().item()
                    stats["clip_frac"] += ((ratio - 1.0).abs() > args.clip).float().mean().item()
                    stats["n"] += 1
        for k in ["policy_loss", "aux_loss", "approx_kl", "clip_frac"]:
            stats[k] /= max(stats["n"], 1)
        return stats


# ===========================================================================
# 采样与评估
# ===========================================================================
def collect(env, mgr, steps_agent, obs, messages):
    n = env.n_agents
    xs, acts, logps, msgs_pred, disps = [], [], [], [], []
    states, team_rews, values, cuts = [], [], [], []
    team_rets = []
    ep_team_ret = 0.0
    for _ in range(steps_agent // n):
        global_state = env.global_state()
        x, actions, logps_i, value, msg = mgr.act_step(obs, messages, global_state)
        next_obs, per_r, team_r, done, _ = env.step(actions)
        disp = next_obs[:, :2] - obs[:, :2]          # 本步实际发生的位移（消息监督目标）
        xs.append(x)
        acts.append(actions)
        logps.append(logps_i)
        msgs_pred.append(msg)
        disps.append(disp)
        states.append(global_state)
        team_rews.append(team_r)
        values.append(value)
        cuts.append(float(done))
        messages = msg                               # 下一步收到本步广播的消息
        ep_team_ret += team_r
        obs = next_obs
        if done:
            team_rets.append(ep_team_ret)
            ep_team_ret = 0.0
            obs = env.reset()
            messages = np.zeros((n, mgr.policy.msg_dim), np.float32)
    buf = {"x": xs, "act": acts, "logp": logps, "disp": disps,
           "states": states, "team_rew": team_rews, "val": values, "cuts": cuts,
           "last_state": env.global_state()}
    return buf, team_rets, obs, messages


@torch.no_grad()
def evaluate(env, mgr, episodes):
    rets, dists, colls = [], [], []
    n = env.n_agents
    for _ in range(episodes):
        obs = env.reset()
        messages = np.zeros((n, mgr.policy.msg_dim), np.float32)
        done = False
        ep_ret = 0.0
        info = {"mean_dist": 0.0, "collisions": 0}
        while not done:
            actions, msg = mgr.eval_step(obs, messages)
            obs, _, team_r, done, info = env.step(actions)
            messages = msg
            ep_ret += team_r
        rets.append(ep_ret)
        dists.append(info["mean_dist"])
        colls.append(info["collisions"])
    return float(np.mean(rets)), float(np.mean(dists)), float(np.mean(colls))


def run_training(use_comm, args, device):
    set_seed(args.seed)
    env = CooperativeNavEnv(args.n_agents, args.episode_len, seed=args.seed,
                            sense_radius=args.sense_radius)
    eval_env = CooperativeNavEnv(args.n_agents, args.episode_len, seed=args.seed + 1,
                                 sense_radius=args.sense_radius)
    mgr = PolicyManager(env, args, device, use_comm)
    n_params = sum(p.numel() for p in mgr.policy.parameters())
    label = "COMM-ON" if use_comm else "COMM-OFF"
    print(f"[{label}] 策略参数量 {n_params} | 感知半径 {args.sense_radius} | "
          f"消息维度 {args.msg_dim if use_comm else 0}")

    obs = env.reset()
    messages = np.zeros((env.n_agents, mgr.policy.msg_dim), np.float32)
    total_steps, update_idx = 0, 0
    curve = []
    all_team_rets = []
    t0 = time.time()
    next_eval = args.eval_every

    while total_steps < args.max_steps:
        buf, team_rets, obs, messages = collect(env, mgr, args.steps_per_update, obs, messages)
        total_steps += args.steps_per_update
        stats = mgr.update(buf, args)
        all_team_rets.extend(team_rets)
        update_idx += 1
        recent = np.mean(all_team_rets[-20:]) if all_team_rets else float("nan")
        aux_str = f"消息MSE {stats['aux_loss']:.4f}" if use_comm else "无通信"
        print(f"[{label} 更新 {update_idx:3d}] 智能体步 {total_steps:7d} | "
              f"近20回合团队回报 {recent:7.2f} | KL {stats['approx_kl']:.4f} | "
              f"裁剪 {stats['clip_frac']:.3f} | {aux_str} | 批评家EV {stats['ev']:.3f} | "
              f"用时 {time.time() - t0:5.1f}s")
        if total_steps >= next_eval:
            ret, dist, coll = evaluate(eval_env, mgr, args.eval_episodes)
            curve.append((total_steps, ret))
            print(f"    [评估] 回报 {ret:7.2f} | 平均末距 {dist:.3f} | "
                  f"平均碰撞 {coll:.2f}/回合")
            next_eval += args.eval_every

    elapsed = time.time() - t0
    ret, dist, coll = evaluate(eval_env, mgr, args.eval_episodes * 3)
    print(f"[{label}] 完成：用时 {elapsed:.1f}s | 最终评估回报 {ret:.2f} | "
          f"末距 {dist:.3f} | 碰撞 {coll:.2f}")
    return curve, elapsed, (ret, dist, coll)


def parse_args():
    p = argparse.ArgumentParser(description="第071章：多智能体 PPO 的通信机制")
    p.add_argument("--comm", type=str, default="both", choices=["both", "on", "off"],
                   help="运行哪些通信配置")
    p.add_argument("--n-agents", type=int, default=3, help="智能体数量")
    p.add_argument("--episode-len", type=int, default=50, help="回合长度")
    p.add_argument("--sense-radius", type=float, default=0.4, help="队友可见半径")
    p.add_argument("--msg-dim", type=int, default=2, help="消息维度")
    p.add_argument("--comm-coef", type=float, default=1.0, help="消息自监督损失权重")
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
    p.add_argument("--save-dir", type=str, default="runs/ch071", help="保存目录")
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
    print(f"设备: {device} | 训练: 带感知限制的合作导航 | 种子 {args.seed} | "
          f"每配置预算 {args.max_steps} 智能体步")

    results = {}
    if args.comm in ("off", "both"):
        results["COMM-OFF"] = run_training(False, args, device)
    if args.comm in ("on", "both"):
        results["COMM-ON"] = run_training(True, args, device)

    print("=" * 76)
    for name, (curve, elapsed, final) in results.items():
        best = max(r for _, r in curve) if curve else float("nan")
        print(f"[{name:8s}] 用时 {elapsed:6.1f}s | 最佳评估 {best:7.2f} | "
              f"最终（回报/末距/碰撞） {final[0]:7.2f} / {final[1]:.3f} / {final[2]:.2f}")
    if len(results) == 2:
        print("提示：评估重点看碰撞次数与末距。通信带来的是信息，不是直接奖励；"
              "消息头靠自监督位移预测训练，消费端靠策略梯度学习。")

    os.makedirs(args.save_dir, exist_ok=True)
    for name, (curve, _, _) in results.items():
        path = os.path.join(args.save_dir, f"{name.lower().replace('-', '_')}_curve.txt")
        with open(path, "w", encoding="utf-8") as f:
            for step, ret in curve:
                f.write(f"{step}\t{ret:.3f}\n")
    print(f"评估曲线已保存到 {args.save_dir}")


if __name__ == "__main__":
    main()
