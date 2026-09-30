"""
第085章 多智能体PPO的分布式训练

实验内容（4 个智能体协同围捕 1 个脚本驱动的猎物，环境为纯 numpy 实现）：
  - 同一套 IPPO（参数共享 + GAE + 裁剪目标），两种分布式采样架构：
      vector：单进程多环境（K 个环境步调一致地并行采样，同步合并更新）
      thread：K 个采样线程各自持有一份策略快照，独立产轨迹进队列，
              学习线程异步消费并周期性同步权重（A3C 风格的简化教学版）
  - 对比指标：更新次数、吞吐（步/秒）、最近回合回报、评估回报
  - 这是教学简化实现：真正的分布式 RL 会跨进程/跨机器，并处理通信与容错

运行：
    python code/ch085.py                  # 默认 vector 模式，约 2-4 分钟（CPU）
    python code/ch085.py --dist thread    # 异步线程模式（结果不保证逐位可复现）
    python code/ch085.py --quick          # 快速跑通，约 30-60 秒

预期：两种架构每步的环境交互量相同；vector 的结果可复现、thread 的吞吐通常
略高但回报曲线更抖。具体数字随机器与核心数波动，请以实跑为准。
"""

import argparse
import copy
import os
import queue
import random
import threading
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
# 环境：4 个智能体协同围捕 1 个猎物
# ---------------------------------------------------------------------------
class EncirclEnv:
    """4 个围捕者要在回合结束前把猎物围在中间（所有人靠得足够近）。

    观测（12 维/智能体）：自己的位置(2)+速度(2)、猎物相对坐标(2)、
    另外三名队友的相对位置(6)。
    动作：2 维连续加速度。奖励：团队共享的"到猎物的距离"负均值，
    完成围捕（所有人距离 < 0.22）+5 并终止；猎物按远离围捕者质心的
    脚本策略逃跑（速度上限 0.6）。
    """

    obs_dim = 12
    act_dim = 2
    n_agents = 4

    def __init__(self, capture_radius: float = 0.22, max_steps: int = 60,
                 friction: float = 0.8, accel: float = 0.12,
                 v_max: float = 1.0, bound: float = 1.0, prey_speed: float = 0.6):
        self.capture_radius = capture_radius
        self.max_steps = max_steps
        self.friction = friction
        self.accel = accel
        self.v_max = v_max
        self.bound = bound
        self.prey_speed = prey_speed

    def reset(self, rng: np.random.Generator):
        angles = np.linspace(0.0, 2.0 * np.pi, self.n_agents, endpoint=False)
        self.pos = (0.9 * np.stack([np.cos(angles), np.sin(angles)], axis=1)
                    + rng.uniform(-0.1, 0.1, size=(self.n_agents, 2))
                    ).astype(np.float32)
        self.vel = np.zeros((self.n_agents, 2), dtype=np.float32)
        self.prey = (rng.uniform(-0.2, 0.2, size=2)).astype(np.float32)
        self.steps = 0
        return self._obs()

    def _obs(self) -> np.ndarray:
        obs = []
        for i in range(self.n_agents):
            others = [self.pos[j] - self.pos[i]
                      for j in range(self.n_agents) if j != i]
            o = np.concatenate([self.pos[i], self.vel[i],
                                self.prey - self.pos[i],
                                np.concatenate(others)])
            obs.append(o)
        return np.stack(obs).astype(np.float32)

    def step(self, actions: np.ndarray):
        a = np.clip(actions, -1.0, 1.0).astype(np.float32)
        self.vel = np.clip(self.friction * self.vel + self.accel * a,
                           -self.v_max, self.v_max)
        self.pos = np.clip(self.pos + self.vel, -self.bound, self.bound)

        # 猎物脚本策略：远离围捕者质心，速度大小固定
        centroid = self.pos.mean(axis=0)
        flee = self.prey - centroid
        norm = float(np.linalg.norm(flee)) + 1e-8
        self.prey = np.clip(self.prey + self.prey_speed * flee / norm,
                            -self.bound, self.bound)
        self.steps += 1

        dists = np.linalg.norm(self.pos - self.prey[None, :], axis=1)
        reward = float(-dists.mean())
        terminated = bool(dists.max() < self.capture_radius)
        if terminated:
            reward += 5.0
        truncated = (not terminated) and self.steps >= self.max_steps
        return self._obs(), reward, terminated, truncated, {"max_dist": float(dists.max())}


# ---------------------------------------------------------------------------
# 网络与更新（与第 084 章的连续 PPO 相同骨架）
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


def compute_gae_single_env(rewards, values, dones, last_values, gamma, lam):
    """单个环境、N 个智能体的 GAE：(T,)，(T,N) -> (T,N) 优势与回报。"""
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


def ppo_update(net, optimizer, batch, args, device):
    """在 numpy 形式的 rollout 批次上做裁剪目标更新，返回诊断量。"""
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
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy.mean()
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
# 采样：单环境轨迹（供同步与异步两种架构复用）
# ---------------------------------------------------------------------------
def collect_single_env(env, net, rng, steps, obs, device, args):
    """在单个环境里采 steps 步，返回 PPO 批次与完成的回合回报列表。"""
    obs_buf, act_buf, logp_buf, val_buf = [], [], [], []
    rew_buf, done_buf = [], []
    ep_returns = []
    ep_ret = 0.0

    for _ in range(steps):
        a, logp, v = net.act_batch(obs, device)
        next_obs, rew, terminated, truncated, info = env.step(a)
        obs_buf.append(obs)
        act_buf.append(a)
        logp_buf.append(logp)
        val_buf.append(v)
        rew_buf.append(rew)
        done_buf.append(float(terminated))       # 时间截断不算终止
        ep_ret += rew
        if terminated or truncated:
            ep_returns.append(ep_ret)
            ep_ret = 0.0
            obs = env.reset(rng)
        else:
            obs = next_obs

    with torch.no_grad():
        _, _, last_v = net.forward(
            torch.as_tensor(obs, dtype=torch.float32, device=device))
    last_values = last_v.cpu().numpy().astype(np.float32)

    values = np.stack(val_buf)                   # (T, N)
    adv, ret = compute_gae_single_env(np.asarray(rew_buf, dtype=np.float32),
                                      values,
                                      np.asarray(done_buf, dtype=np.float32),
                                      last_values, args.gamma, args.lam)
    T, N = adv.shape
    batch = {
        "obs": np.stack(obs_buf).reshape(T * N, -1).copy(),
        "act": np.stack(act_buf).reshape(T * N, -1).copy(),
        "logp": np.stack(logp_buf).reshape(T * N).copy(),
        "adv": adv.reshape(T * N).copy(),
        "ret": ret.reshape(T * N).copy(),
    }
    return batch, ep_returns, obs


# ---------------------------------------------------------------------------
# 架构一：单进程多环境（同步）
# ---------------------------------------------------------------------------
def run_vector(args, device):
    envs = [EncirclEnv() for _ in range(args.workers)]
    rngs = [np.random.default_rng(args.seed + 100 * k) for k in range(args.workers)]
    obss = [env.reset(rngs[k]) for k, env in enumerate(envs)]

    net = PPOActorCritic(EncirclEnv.obs_dim, EncirclEnv.act_dim,
                         args.hidden).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    total_steps, update_idx = 0, 0
    all_returns = []
    t_start = time.time()
    last_log_t = t_start
    last_log_steps = 0

    while total_steps < args.max_steps:
        # 1) 并行走 K 个环境：每个时间步对 K*N 个智能体做一次前向
        obs_buf, act_buf, logp_buf, val_buf = [], [], [], []
        rew_buf, done_buf = [], []
        for _ in range(args.rollout):
            obs_batch = np.concatenate(obss, axis=0)
            a, logp, v = net.act_batch(obs_batch, device)
            rews_step, dones_step = [], []
            for k, env in enumerate(envs):
                sl = slice(k * EncirclEnv.n_agents, (k + 1) * EncirclEnv.n_agents)
                next_obs, rew, terminated, truncated, info = env.step(a[sl])
                rews_step.append(rew)
                dones_step.append(float(terminated))
                if terminated or truncated:
                    obss[k] = env.reset(rngs[k])
                else:
                    obss[k] = next_obs
            obs_buf.append(obs_batch)
            act_buf.append(a)
            logp_buf.append(logp)
            val_buf.append(v)
            rew_buf.append(rews_step)
            done_buf.append(dones_step)
            total_steps += args.workers * EncirclEnv.n_agents

        # 2) 逐环境计算 GAE，再合并
        K, N = args.workers, EncirclEnv.n_agents
        values = np.stack(val_buf).reshape(args.rollout, K, N)
        rewards = np.asarray(rew_buf, dtype=np.float32)          # (T, K)
        dones = np.asarray(done_buf, dtype=np.float32)           # (T, K)
        adv = np.zeros((args.rollout, K, N), dtype=np.float32)
        ret = np.zeros_like(adv)
        with torch.no_grad():
            for k in range(K):
                last_v, _, _ = net.forward(torch.as_tensor(
                    obss[k], dtype=torch.float32, device=device))
                last_values = last_v.cpu().numpy().astype(np.float32)
                adv[:, k, :], ret[:, k, :] = compute_gae_single_env(
                    rewards[:, k], values[:, k, :], dones[:, k],
                    last_values, args.gamma, args.lam)
        T = args.rollout
        batch = {
            "obs": np.stack(obs_buf).reshape(T * K * N, -1),
            "act": np.stack(act_buf).reshape(T * K * N, -1),
            "logp": np.stack(logp_buf).reshape(T * K * N),
            "adv": adv.reshape(T * K * N),
            "ret": ret.reshape(T * K * N),
        }

        stats = ppo_update(net, optimizer, batch, args, device)
        update_idx += 1

        # 3) 日志：吞吐 + 最近评估
        now = time.time()
        if now - last_log_t > 2.0 or total_steps >= args.max_steps:
            sps = (total_steps - last_log_steps) / max(now - last_log_t, 1e-6)
            print(f"[vector] 步 {total_steps:6d} | 更新 {update_idx:2d} | "
                  f"吞吐 {sps:7.1f} 步/秒 | KL {stats['kl']:.4f} | "
                  f"裁剪比例 {stats['clip_frac']:.3f}")
            last_log_t, last_log_steps = now, total_steps

    elapsed = time.time() - t_start
    return net, total_steps, elapsed, update_idx


# ---------------------------------------------------------------------------
# 架构二：线程化异步采样
# ---------------------------------------------------------------------------
class Worker(threading.Thread):
    """采样线程：本地策略快照 + 独立环境，产出 rollout 放进队列。"""

    def __init__(self, wid, args, device, global_net, lock, out_queue, stop_event):
        super().__init__(daemon=True)
        self.wid = wid
        self.args = args
        self.device = device
        self.global_net = global_net
        self.lock = lock
        self.out_queue = out_queue
        self.stop_event = stop_event
        self.net = copy.deepcopy(global_net)
        self.env = EncirclEnv()
        self.rng = np.random.default_rng(args.seed + 1000 + 37 * wid)
        self.obs = self.env.reset(self.rng)
        self.samples = 0

    def run(self):
        while not self.stop_event.is_set():
            # 1) 同步权重（拿一份快照）
            with self.lock:
                self.net.load_state_dict(self.global_net.state_dict())
            # 2) 采样一段固定长度的轨迹
            batch, ep_returns, self.obs = collect_single_env(
                self.env, self.net, self.rng, self.args.rollout,
                self.obs, self.device, self.args)
            self.samples += self.args.rollout * EncirclEnv.n_agents
            # 3) 入队；队列满则丢弃（教学简化：表示"数据过期"）
            try:
                self.out_queue.put_nowait(batch)
            except queue.Full:
                pass


def run_thread(args, device):
    net = PPOActorCritic(EncirclEnv.obs_dim, EncirclEnv.act_dim,
                         args.hidden).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    lock = threading.Lock()
    stop_event = threading.Event()
    out_queue = queue.Queue(maxsize=args.queue_size)

    workers = [Worker(wid, args, device, net, lock, out_queue, stop_event)
               for wid in range(args.workers)]
    t_start = time.time()
    for w in workers:
        w.start()

    total_steps, update_idx = 0, 0
    last_log_t, last_log_steps = t_start, 0

    while total_steps < args.max_steps:
        try:
            batch = out_queue.get(timeout=5.0)
        except queue.Empty:
            print("[thread] 等待 rollout 超时，检查采样线程是否卡死")
            break
        stats = ppo_update(net, optimizer, batch, args, device)
        update_idx += 1
        total_steps += batch["obs"].shape[0]

        now = time.time()
        if now - last_log_t > 2.0 or total_steps >= args.max_steps:
            sps = (total_steps - last_log_steps) / max(now - last_log_t, 1e-6)
            produced = sum(w.samples for w in workers)
            print(f"[thread] 步 {total_steps:6d} | 更新 {update_idx:2d} | "
                  f"吞吐 {sps:7.1f} 步/秒（产出 {produced}） | KL {stats['kl']:.4f} | "
                  f"队列 {out_queue.qsize()}/{args.queue_size}")
            last_log_t, last_log_steps = now, total_steps

    stop_event.set()
    for w in workers:
        w.join(timeout=2.0)
    elapsed = time.time() - t_start
    return net, total_steps, elapsed, update_idx


# ---------------------------------------------------------------------------
# 评估
# ---------------------------------------------------------------------------
def evaluate(net, episodes, device, seed):
    env = EncirclEnv()
    rng = np.random.default_rng(seed)
    rets, captures, max_dists = [], [], []
    for _ in range(episodes):
        obs = env.reset(rng)
        done = False
        ep_ret, info = 0.0, {"max_dist": 1.0}
        captured = False
        while not done:
            a = net.act_deterministic(obs, device)
            obs, rew, terminated, truncated, info = env.step(a)
            ep_ret += rew
            done = terminated or truncated
            if terminated:
                captured = True
        rets.append(ep_ret)
        captures.append(1.0 if captured else 0.0)
        max_dists.append(info["max_dist"])
    return float(np.mean(rets)), float(np.mean(captures)), float(np.mean(max_dists))


def parse_args():
    p = argparse.ArgumentParser(description="第085章：多智能体PPO的分布式训练")
    p.add_argument("--dist", type=str, default="vector", choices=["vector", "thread"],
                   help="采样架构：同步多环境 / 异步线程")
    p.add_argument("--workers", type=int, default=4, help="并行环境/采样线程数")
    p.add_argument("--max-steps", type=int, default=40000, help="总环境步数预算")
    p.add_argument("--rollout", type=int, default=1000, help="每个 worker 每次采样步数")
    p.add_argument("--queue-size", type=int, default=8, help="异步模式队列容量")
    p.add_argument("--epochs", type=int, default=4, help="PPO 更新轮数")
    p.add_argument("--batch-size", type=int, default=256, help="更新小批量")
    p.add_argument("--lr", type=float, default=3e-4, help="学习率")
    p.add_argument("--gamma", type=float, default=0.95, help="折扣因子")
    p.add_argument("--lam", type=float, default=0.95, help="GAE lambda")
    p.add_argument("--clip", type=float, default=0.2, help="裁剪范围")
    p.add_argument("--hidden", type=int, default=128, help="隐藏层宽度")
    p.add_argument("--eval-episodes", type=int, default=10, help="评估回合数")
    p.add_argument("--seed", type=int, default=0, help="随机种子")
    p.add_argument("--save-dir", type=str, default="runs/ch085", help="模型保存目录")
    p.add_argument("--quick", action="store_true", help="快速模式：缩小训练规模")
    return p.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.max_steps = min(args.max_steps, 12000)
        args.rollout = 500
        args.workers = min(args.workers, 2)
        args.eval_episodes = 5
    set_seed(args.seed)
    device = torch.device("cpu")

    print(f"架构 {args.dist} | workers={args.workers} | 预算 {args.max_steps} 步 | "
          f"种子 {args.seed}")

    if args.dist == "vector":
        net, steps, elapsed, updates = run_vector(args, device)
    else:
        net, steps, elapsed, updates = run_thread(args, device)

    mean_ret, capture_rate, mean_max_dist = evaluate(
        net, args.eval_episodes, device, args.seed + 777)
    print("-" * 74)
    print(f"训练结束 | 用时 {elapsed:.1f}s | 更新 {updates} 次 | 环境步 {steps} | "
          f"平均吞吐 {steps / max(elapsed, 1e-6):.1f} 步/秒")
    print(f"评估（{args.eval_episodes} 回合，确定性策略）：平均回报 {mean_ret:.2f} | "
          f"围捕成功率 {capture_rate:.2f} | 结束时最大距离 {mean_max_dist:.3f}")

    os.makedirs(args.save_dir, exist_ok=True)
    path = os.path.join(args.save_dir, f"ippo_{args.dist}.pt")
    torch.save({"policy": net.state_dict(), "args": vars(args)}, path)
    print(f"模型已保存到 {path}")


if __name__ == "__main__":
    main()
