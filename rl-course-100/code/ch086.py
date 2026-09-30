"""
第086章 多智能体SAC的分布式训练

实验内容（3 个智能体协同把箱子搬到目标点，环境为脚本内联的纯 numpy 实现）：
  - off-policy 的分布式 SAC，两种采集-学习架构：
      sync ：无重叠的批量模式——K 个采集段依次采完统一入池，学习端再做更新
              （基线，单进程确定性）
      async：采集线程按自己的节奏把转移塞进共享回放池，学习端在主线程持续
              采样更新（线程重叠；Ape-X 风格的简化教学版）
  - 共享回放池线程安全（deque + 锁），采集线程周期性同步策略权重快照
  - 对比指标：吞吐（步/秒）、回放池规模、更新/采样比、评估回报与搬箱成功率

运行：
    python code/ch086.py                  # 默认 async，约 2-4 分钟（CPU）
    python code/ch086.py --mode sync      # 同步基线（可复现）
    python code/ch086.py --quick          # 快速跑通，约 30-60 秒

预期：async 利用采集与学习重叠获得更高吞吐；sync 的结果可复现、作为对照。
具体数字随机器核数波动，请以实跑为准。
"""

import argparse
import collections
import copy
import os
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
# 环境：协同搬运
# ---------------------------------------------------------------------------
class CarryEnv:
    """3 个智能体合力把一个箱子推到目标区域。

    物理：智能体靠近箱子（<0.28）时产生推力，箱子速度 = 摩擦衰减 + 推力之和；
    箱子只有被"从正确一侧"推才能朝目标移动，因此需要包抄与协作。
    观测（14 维/智能体）：自己位置+速度(4)、箱子相对位置与速度(4)、
    目标相对箱子的方向(2)、另外两名队友的相对位置(4)。
    奖励：团队共享的 -‖箱子−目标‖，到达（<0.15）+5 并终止。
    """

    obs_dim = 14
    act_dim = 2
    n_agents = 3

    def __init__(self, touch_radius: float = 0.28, push_strength: float = 0.06,
                 max_steps: int = 80, friction: float = 0.9, accel: float = 0.12,
                 v_max: float = 1.0, bound: float = 1.0):
        self.touch_radius = touch_radius
        self.push_strength = push_strength
        self.max_steps = max_steps
        self.friction = friction
        self.accel = accel
        self.v_max = v_max
        self.bound = bound

    def reset(self, rng: np.random.Generator):
        angles = np.linspace(0.0, 2.0 * np.pi, self.n_agents, endpoint=False)
        self.pos = (0.35 * np.stack([np.cos(angles), np.sin(angles)], axis=1)
                    + np.array([-0.5, -0.5]) + rng.uniform(-0.05, 0.05, size=(3, 2))
                    ).astype(np.float32)
        self.vel = np.zeros((self.n_agents, 2), dtype=np.float32)
        self.box_pos = np.array([-0.5, -0.5], dtype=np.float32)
        self.box_vel = np.zeros(2, dtype=np.float32)
        self.goal = np.array([0.6, 0.6], dtype=np.float32)
        self.steps = 0
        return self._obs()

    def _obs(self) -> np.ndarray:
        obs = []
        for i in range(self.n_agents):
            others = [self.pos[j] - self.pos[i]
                      for j in range(self.n_agents) if j != i]
            o = np.concatenate([self.pos[i], self.vel[i],
                                self.box_pos - self.pos[i], self.box_vel,
                                self.goal - self.box_pos,
                                np.concatenate(others)])
            obs.append(o)
        return np.stack(obs).astype(np.float32)

    def step(self, actions: np.ndarray):
        a = np.clip(actions, -1.0, 1.0).astype(np.float32)
        self.vel = np.clip(self.friction * self.vel + self.accel * a,
                           -self.v_max, self.v_max)
        self.pos = np.clip(self.pos + self.vel, -self.bound, self.bound)

        # 箱子推力：每个贴近箱子的智能体贡献一份朝向箱子外侧的力
        push = np.zeros(2, dtype=np.float32)
        for i in range(self.n_agents):
            d = self.box_pos - self.pos[i]
            dist = float(np.linalg.norm(d)) + 1e-8
            if dist < self.touch_radius:
                push += (d / dist) * (1.0 - dist / self.touch_radius) * self.push_strength
        self.box_vel = np.clip(0.9 * self.box_vel + push, -0.8, 0.8)
        self.box_pos = np.clip(self.box_pos + self.box_vel, -self.bound, self.bound)
        self.steps += 1

        dist_goal = float(np.linalg.norm(self.box_pos - self.goal))
        reward = -dist_goal
        terminated = dist_goal < 0.15
        if terminated:
            reward += 5.0
        truncated = (not terminated) and self.steps >= self.max_steps
        return self._obs(), reward, terminated, truncated, {"dist_goal": dist_goal}


# ---------------------------------------------------------------------------
# 网络与 SAC 学习器
# ---------------------------------------------------------------------------
class SquashedGaussianActor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden, act_dim)
        self.log_std_head = nn.Linear(hidden, act_dim)

    def forward(self, obs: torch.Tensor, deterministic: bool = False):
        h = self.net(obs)
        mu = self.mu_head(h)
        log_std = torch.clamp(self.log_std_head(h), LOG_STD_MIN, LOG_STD_MAX)
        dist = Normal(mu, log_std.exp())
        x = mu if deterministic else dist.rsample()
        action = torch.tanh(x)
        logp = dist.log_prob(x) - torch.log(1.0 - action.pow(2) + 1e-6)
        return action, logp.sum(dim=-1, keepdim=True)

    @torch.no_grad()
    def act_np(self, obs_np: np.ndarray, device) -> np.ndarray:
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
        a, _ = self.forward(obs, deterministic=False)
        return a.cpu().numpy().astype(np.float32)


class QNet(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SACLearner:
    """学习端：双 Q + 目标网络 + 自动温度，从共享回放池采样更新。"""

    def __init__(self, env_cls, args, device):
        self.args = args
        self.device = device
        self.actor = SquashedGaussianActor(env_cls.obs_dim, env_cls.act_dim,
                                           args.hidden).to(device)
        q_in = env_cls.obs_dim + env_cls.act_dim
        self.q1 = QNet(q_in, args.hidden).to(device)
        self.q2 = QNet(q_in, args.hidden).to(device)
        self.q1_target = copy.deepcopy(self.q1)
        self.q2_target = copy.deepcopy(self.q2)
        for net in (self.q1_target, self.q2_target):
            for p in net.parameters():
                p.requires_grad_(False)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=args.lr)
        self.critic_opt = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=args.lr)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=args.lr)
        self.target_entropy = -float(env_cls.act_dim)
        self.updates = 0

    def update(self, buffer, rng):
        args = self.args
        batch = buffer.sample(args.batch_size, rng)

        def stack(key, dtype=torch.float32):
            return torch.as_tensor(np.stack([b[key] for b in batch]),
                                   dtype=dtype, device=self.device)

        obs, act, rew = stack("obs"), stack("act"), stack("rew")
        next_obs, done = stack("next_obs"), stack("done")
        alpha = self.log_alpha.exp().detach()

        with torch.no_grad():
            next_a, next_logp = self.actor(next_obs)
            q_in = torch.cat([next_obs, next_a], dim=-1)
            y = rew + args.gamma * (1.0 - done) * (
                torch.min(self.q1_target(q_in), self.q2_target(q_in)) - alpha * next_logp).squeeze(-1)
        q_in = torch.cat([obs, act], dim=-1)
        critic_loss = (F.mse_loss(self.q1(q_in).squeeze(-1), y)
                       + F.mse_loss(self.q2(q_in).squeeze(-1), y))
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        new_a, logp = self.actor(obs)
        q_in = torch.cat([obs, new_a], dim=-1)
        actor_loss = (alpha * logp - torch.min(self.q1(q_in), self.q2(q_in))).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        with torch.no_grad():
            for net, tnet in ((self.q1, self.q1_target), (self.q2, self.q2_target)):
                for p, tp in zip(net.parameters(), tnet.parameters()):
                    tp.data.mul_(1.0 - args.tau).add_(args.tau * p.data)
        self.updates += 1
        return float(critic_loss.item())


# ---------------------------------------------------------------------------
# 线程安全的共享回放池
# ---------------------------------------------------------------------------
class SharedReplay:
    def __init__(self, capacity: int):
        self.buffer = collections.deque(maxlen=capacity)
        self.lock = threading.Lock()
        self.total = 0          # 累计写入的转移数（不随容量淘汰而减少）

    def push_many(self, entries):
        with self.lock:
            self.buffer.extend(entries)
            self.total += len(entries)

    def sample(self, batch_size: int, rng: np.random.Generator):
        with self.lock:
            n = len(self.buffer)
            idx = rng.integers(0, n, size=batch_size)
            items = list(self.buffer)
        return [items[i] for i in idx]

    def __len__(self):
        with self.lock:
            return len(self.buffer)


# ---------------------------------------------------------------------------
# 采集段：单环境 + 一份策略快照
# ---------------------------------------------------------------------------
@torch.no_grad()
def collect_segment(env, actor, rng, steps, obs, device, random_actions=False):
    """采集 steps 步，返回转移列表、完成的回合统计、最新观测。"""
    entries, ep_stats = [], []
    ep_ret, ep_success = 0.0, False

    for _ in range(steps):
        if random_actions:
            actions = rng.uniform(-1.0, 1.0,
                                  size=(env.n_agents, env.act_dim)).astype(np.float32)
        else:
            actions = actor.act_np(obs, device)
        next_obs, rew, terminated, truncated, info = env.step(actions)
        for i in range(env.n_agents):
            entries.append({
                "obs": obs[i].copy(), "act": actions[i].copy(),
                "rew": np.float32(rew), "next_obs": next_obs[i].copy(),
                "done": np.float32(terminated),
            })
        ep_ret += rew
        if terminated:
            ep_success = True
        if terminated or truncated:
            ep_stats.append({"return": ep_ret, "success": ep_success,
                             "dist_goal": info["dist_goal"]})
            ep_ret, ep_success = 0.0, False
            obs = env.reset(rng)
        else:
            obs = next_obs
    return entries, ep_stats, obs


class Collector(threading.Thread):
    """异步采集线程：定期同步全局策略快照，把转移写入共享回放池。"""

    def __init__(self, cid, args, device, learner_actor, lock, replay, stop_event):
        super().__init__(daemon=True)
        self.cid = cid
        self.args = args
        self.device = device
        self.learner_actor = learner_actor
        self.lock = lock
        self.replay = replay
        self.stop_event = stop_event
        self.actor = copy.deepcopy(learner_actor)
        self.env = CarryEnv()
        self.rng = np.random.default_rng(args.seed + 100 + 13 * cid)
        self.obs = self.env.reset(self.rng)
        self.ep_stats = []

    def run(self):
        while not self.stop_event.is_set():
            with self.lock:            # 同步一份策略权重快照
                self.actor.load_state_dict(self.learner_actor.state_dict())
            entries, stats, self.obs = collect_segment(
                self.env, self.actor, self.rng, self.args.segment,
                self.obs, self.device, random_actions=False)
            self.replay.push_many(entries)
            self.ep_stats.extend(stats)


# ---------------------------------------------------------------------------
# 两种训练架构
# ---------------------------------------------------------------------------
def run_sync(args, device):
    """同步批量：采集段依次执行，全部入池后学习端按比例更新。"""
    replay = SharedReplay(args.buffer_capacity)
    learner = SACLearner(CarryEnv, args, device)
    rng = np.random.default_rng(args.seed)
    env = CarryEnv()
    obs = env.reset(rng)

    total_steps = 0
    t_start = time.time()

    # 预热：随机动作填池（off-policy 需要初始数据）
    entries, stats, obs = collect_segment(env, learner.actor, rng, args.warmup,
                                          obs, device, random_actions=True)
    replay.push_many(entries)
    total_steps += args.warmup
    all_stats = list(stats)

    while total_steps < args.max_steps:
        # 1) 采集 K 段（模拟 K 个采集器，但无重叠、依次执行）
        new_entries = 0
        for _ in range(args.collectors):
            entries, stats, obs = collect_segment(
                env, learner.actor, rng, args.segment, obs, device)
            replay.push_many(entries)
            new_entries += len(entries)
            total_steps += args.segment
            all_stats.extend(stats)
        # 2) 学习端按更新/采样比消费
        n_updates = int(new_entries * args.update_ratio / CarryEnv.n_agents)
        for _ in range(n_updates):
            learner.update(replay, rng)

        # 3) 日志
        recent = all_stats[-20:]
        m_ret = float(np.mean([s["return"] for s in recent])) if recent else float("nan")
        m_succ = float(np.mean([s["success"] for s in recent])) if recent else float("nan")
        print(f"[sync ] 步 {total_steps:6d} | 更新 {learner.updates:5d} | "
              f"池 {len(replay):6d} | 近20局回报 {m_ret:7.2f} | 成功率 {m_succ:.2f}")

    elapsed = time.time() - t_start
    return learner, total_steps, elapsed, all_stats


def run_async(args, device):
    """异步：采集线程与学习线程重叠，学习端持续从池中采样。"""
    replay = SharedReplay(args.buffer_capacity)
    learner = SACLearner(CarryEnv, args, device)
    lock = threading.Lock()
    stop_event = threading.Event()
    rng = np.random.default_rng(args.seed)

    # 预热数据（主线程随机动作）
    warm_env = CarryEnv()
    warm_rng = np.random.default_rng(args.seed)
    obs = warm_env.reset(warm_rng)
    entries, stats, obs = collect_segment(warm_env, learner.actor, warm_rng,
                                          args.warmup, obs, device,
                                          random_actions=True)
    replay.push_many(entries)
    all_stats = list(stats)

    collectors = [Collector(cid, args, device, learner.actor, lock, replay,
                            stop_event) for cid in range(args.collectors)]
    t_start = time.time()
    for c in collectors:
        c.start()

    last_log_t = t_start
    logged_steps = 0
    while True:
        # 学习端：按"更新/采样比"消费数据；暂无新数据时小睡让出 CPU
        steps = replay.total // CarryEnv.n_agents
        target_updates = int(steps * args.update_ratio)
        if learner.updates < target_updates:
            learner.update(replay, rng)
        else:
            time.sleep(0.001)

        now = time.time()
        if now - last_log_t > 3.0:
            s_per_s = (steps - logged_steps) / max(now - last_log_t, 1e-6)
            print(f"[async] 步 {steps:6d} | 更新 {learner.updates:5d} | "
                  f"池 {len(replay):6d} | 吞吐 {s_per_s:7.1f} 步/秒")
            last_log_t = now
            logged_steps = steps
        if steps >= args.max_steps:
            break

    stop_event.set()
    for c in collectors:
        c.join(timeout=2.0)
    # 汇总各线程的回合统计
    for c in collectors:
        all_stats.extend(c.ep_stats)
    elapsed = time.time() - t_start
    total_steps = replay.total // CarryEnv.n_agents
    return learner, total_steps, elapsed, all_stats


# ---------------------------------------------------------------------------
# 评估
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(actor, episodes, device, seed):
    env = CarryEnv()
    rng = np.random.default_rng(seed)
    rets, successes, dists = [], [], []
    for _ in range(episodes):
        obs = env.reset(rng)
        done = False
        ep_ret, info = 0.0, {"dist_goal": 1.0}
        success = False
        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
            a, _ = actor(obs_t, deterministic=True)
            obs, rew, terminated, truncated, info = env.step(a.cpu().numpy())
            ep_ret += rew
            done = terminated or truncated
            if terminated:
                success = True
        rets.append(ep_ret)
        successes.append(1.0 if success else 0.0)
        dists.append(info["dist_goal"])
    return float(np.mean(rets)), float(np.mean(successes)), float(np.mean(dists))


def parse_args():
    p = argparse.ArgumentParser(description="第086章：多智能体SAC的分布式训练")
    p.add_argument("--mode", type=str, default="async", choices=["sync", "async"],
                   help="采集-学习架构")
    p.add_argument("--collectors", type=int, default=3, help="采集器/线程数")
    p.add_argument("--segment", type=int, default=600, help="每个采集段的环境步数")
    p.add_argument("--max-steps", type=int, default=30000, help="总环境步预算")
    p.add_argument("--warmup", type=int, default=2000, help="随机预热步数")
    p.add_argument("--update-ratio", type=float, default=1.0,
                   help="同步模式：每个新样本对应多少次梯度更新")
    p.add_argument("--buffer-capacity", type=int, default=150000, help="回放池容量")
    p.add_argument("--batch-size", type=int, default=256, help="采样批量")
    p.add_argument("--lr", type=float, default=3e-4, help="学习率")
    p.add_argument("--gamma", type=float, default=0.95, help="折扣因子")
    p.add_argument("--tau", type=float, default=0.01, help="软更新系数")
    p.add_argument("--hidden", type=int, default=128, help="隐藏层宽度")
    p.add_argument("--eval-episodes", type=int, default=10, help="评估回合数")
    p.add_argument("--seed", type=int, default=0, help="随机种子")
    p.add_argument("--save-dir", type=str, default="runs/ch086", help="模型保存目录")
    p.add_argument("--quick", action="store_true", help="快速模式：缩小训练规模")
    return p.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.max_steps = min(args.max_steps, 12000)
        args.warmup = min(args.warmup, 800)
        args.segment = 400
        args.collectors = min(args.collectors, 2)
        args.eval_episodes = 5
    set_seed(args.seed)
    device = torch.device("cpu")

    print(f"模式 {args.mode} | collectors={args.collectors} | 预算 {args.max_steps} 步 | "
          f"种子 {args.seed}")

    if args.mode == "sync":
        learner, steps, elapsed, stats = run_sync(args, device)
    else:
        learner, steps, elapsed, stats = run_async(args, device)

    mean_ret, success, mean_dist = evaluate(learner.actor, args.eval_episodes,
                                            device, args.seed + 555)
    recent = stats[-20:]
    print("-" * 74)
    print(f"训练结束 | 用时 {elapsed:.1f}s | 环境步 {steps} | 更新 {learner.updates} 次 | "
          f"吞吐 {steps / max(elapsed, 1e-6):.1f} 步/秒")
    if recent:
        print(f"训练尾段（近20局）：平均回报 "
              f"{np.mean([s['return'] for s in recent]):.2f} | 成功率 "
              f"{np.mean([s['success'] for s in recent]):.2f}")
    print(f"评估（{args.eval_episodes} 回合，确定性策略）：平均回报 {mean_ret:.2f} | "
          f"搬箱成功率 {success:.2f} | 结束时距目标 {mean_dist:.3f}")

    os.makedirs(args.save_dir, exist_ok=True)
    path = os.path.join(args.save_dir, f"masac_{args.mode}.pt")
    torch.save({"actor": learner.actor.state_dict(), "args": vars(args)}, path)
    print(f"模型已保存到 {path}")


if __name__ == "__main__":
    main()
