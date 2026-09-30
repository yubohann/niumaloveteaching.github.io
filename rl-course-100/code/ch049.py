"""
第049章 SAC的分布式实现

生产级 SAC 训练系统通常是"多个采集器 + 一个学习器"的结构：采集器并行跑
环境、把经验推进共享回放池，学习器只负责采样与梯度更新。本章用单进程多
线程实现一个教学版：

  采集阶段：--workers 个线程，每个线程持有独立环境，用当前策略采样并写入
           共享回放池（带锁的环形缓冲）
  学习阶段：主线程扮演学习器，按"新收集多少步就补多少次更新"的节奏训练
  吞吐基准：先用 1 个与 W 个采集器各跑一小段，比较 steps/s

说明：这是教学简化——线程共享内存、受 GIL 限制，真实系统用多进程/多机。
我们额外把 torch 内部线程数设为 1，避免与 Python 线程互相争抢。

运行：
    python code/ch049.py           # 基准 + 4 采集器训练（CPU 约 3-5 分钟）
    python code/ch049.py --quick   # 快速跑通（约 40 秒）

预期：采集吞吐随 worker 数接近线性增长（Pendulum 的 step 是轻量 numpy
计算，线程切换开销小）；学习器可能成为瓶颈，日志会显示更新滞后。
"""
import argparse
import os
import random
import threading
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_STD_MIN, LOG_STD_MAX = -20.0, 2.0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_env(env_id: str, seed: int):
    env = gym.make(env_id)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env


# ---------------------------------------------------------------------------
# 线程安全的共享回放池
# ---------------------------------------------------------------------------
class SharedReplay:
    """用一把锁保护的环形回放池：写入与采样都串行化，保证线程安全。"""

    def __init__(self, capacity: int, obs_dim: int, act_dim: int):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rew = np.zeros((capacity, 1), dtype=np.float32)
        self.done = np.zeros((capacity, 1), dtype=np.float32)
        self.ptr = 0
        self.size = 0
        self.lock = threading.Lock()

    def add(self, obs, act, rew, next_obs, done) -> None:
        with self.lock:
            i = self.ptr
            self.obs[i] = obs
            self.act[i] = act
            self.rew[i] = rew
            self.next_obs[i] = next_obs
            self.done[i] = done
            self.ptr = (self.ptr + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict:
        with self.lock:
            idx = np.random.randint(0, self.size, size=batch_size)
            return {
                "obs": torch.as_tensor(self.obs[idx].copy()),
                "act": torch.as_tensor(self.act[idx].copy()),
                "rew": torch.as_tensor(self.rew[idx].copy()),
                "next_obs": torch.as_tensor(self.next_obs[idx].copy()),
                "done": torch.as_tensor(self.done[idx].copy()),
            }


# ---------------------------------------------------------------------------
# 网络与 SAC
# ---------------------------------------------------------------------------
class QNet(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, act], dim=-1)).squeeze(-1)


class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256,
                 act_scale: float = 1.0):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mean_layer = nn.Linear(hidden, act_dim)
        self.log_std_layer = nn.Linear(hidden, act_dim)
        self.register_buffer("act_scale", torch.tensor(float(act_scale)))

    def forward(self, obs: torch.Tensor):
        h = self.body(obs)
        mean = self.mean_layer(h)
        log_std = torch.clamp(self.log_std_layer(h), LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, obs: torch.Tensor):
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        eps = torch.randn_like(mean)
        u = mean + std * eps
        a = torch.tanh(u)
        log_prob = -0.5 * (eps ** 2 + 2.0 * log_std + np.log(2.0 * np.pi))
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        log_prob = log_prob - torch.log(1.0 - a ** 2 + 1e-6).sum(dim=-1, keepdim=True)
        action = a * self.act_scale
        log_prob = log_prob - torch.log(self.act_scale).sum()
        return action, log_prob

    def mean_action(self, obs: torch.Tensor) -> torch.Tensor:
        mean, _ = self.forward(obs)
        return torch.tanh(mean) * self.act_scale


def soft_update(net: nn.Module, target: nn.Module, tau: float) -> None:
    for tp, p in zip(target.parameters(), net.parameters()):
        tp.data.mul_(1.0 - tau).add_(tau * p.data)


class SAC:
    def __init__(self, obs_dim: int, act_dim: int, action_scale: float, args):
        self.policy = GaussianPolicy(obs_dim, act_dim, args.hidden, action_scale)
        self.q1 = QNet(obs_dim, act_dim, args.hidden)
        self.q2 = QNet(obs_dim, act_dim, args.hidden)
        self.q1_target = QNet(obs_dim, act_dim, args.hidden)
        self.q2_target = QNet(obs_dim, act_dim, args.hidden)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        for p in self.q1_target.parameters():
            p.requires_grad_(False)
        for p in self.q2_target.parameters():
            p.requires_grad_(False)
        self.log_alpha = torch.tensor(np.log(args.alpha_init), requires_grad=True)
        self.target_entropy = -float(act_dim)
        self.opt_q = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=args.lr)
        self.opt_policy = torch.optim.Adam(self.policy.parameters(), lr=args.lr)
        self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=args.lr)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp().clamp(1e-4, 10.0)

    @torch.no_grad()
    def act(self, obs, deterministic: bool = False) -> np.ndarray:
        obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
        if deterministic:
            return self.policy.mean_action(obs_t).squeeze(0).numpy()
        action, _ = self.policy.sample(obs_t)
        return action.squeeze(0).numpy()

    def update(self, batch: dict, args) -> dict:
        obs, act = batch["obs"], batch["act"]
        rew, next_obs, done = batch["rew"], batch["next_obs"], batch["done"]

        with torch.no_grad():
            next_act, next_logp = self.policy.sample(next_obs)
            q_next = torch.minimum(self.q1_target(next_obs, next_act),
                                   self.q2_target(next_obs, next_act))
            backup = rew + args.gamma * (1.0 - done) * (q_next - self.alpha.detach() * next_logp)

        q1 = self.q1(obs, act)
        q2 = self.q2(obs, act)
        q_loss = F.mse_loss(q1, backup) + F.mse_loss(q2, backup)
        self.opt_q.zero_grad()
        q_loss.backward()
        self.opt_q.step()

        new_act, logp = self.policy.sample(obs)
        q_pi = torch.minimum(self.q1(obs, new_act), self.q2(obs, new_act))
        policy_loss = (self.alpha.detach() * logp - q_pi).mean()
        self.opt_policy.zero_grad()
        policy_loss.backward()
        self.opt_policy.step()

        alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()
        self.opt_alpha.zero_grad()
        alpha_loss.backward()
        self.opt_alpha.step()

        with torch.no_grad():
            soft_update(self.q1, self.q1_target, args.tau)
            soft_update(self.q2, self.q2_target, args.tau)

        return {"q_loss": q_loss.item(), "policy_loss": policy_loss.item(),
                "alpha": float(self.alpha.item())}


@torch.no_grad()
def evaluate(agent: SAC, env, episodes: int) -> float:
    returns = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            action = agent.act(obs, deterministic=True)
            obs, rew, terminated, truncated, _ = env.step(action)
            ep_ret += rew
            done = terminated or truncated
        returns.append(ep_ret)
    return float(np.mean(returns))


# ---------------------------------------------------------------------------
# 采集线程
# ---------------------------------------------------------------------------
def collector_worker(worker_id: int, env_id: str, seed: int, agent: SAC,
                     buffer: SharedReplay, stats: dict, args,
                     stop_event: threading.Event) -> None:
    """单个采集器：独立环境 + 共享策略 + 共享回放池。

    随机探索阶段按"每个 worker 贡献 start_steps / workers 步"估算，
    策略推理在 no_grad 下进行，多线程共享同一份网络参数。
    """
    env = make_env(env_id, seed + worker_id * 1000)
    obs, _ = env.reset(seed=seed + worker_id * 1000)
    ep_ret = 0.0
    local_steps = 0
    random_steps = max(args.start_steps // max(args.workers, 1), 10)

    while not stop_event.is_set():
        # 每 20 步检查一次停止标志，减少事件查询开销
        for _ in range(20):
            if stop_event.is_set():
                break
            if local_steps < random_steps:
                action = env.action_space.sample()
            else:
                action = agent.act(obs, deterministic=False)
            next_obs, rew, terminated, truncated, _ = env.step(action)
            buffer.add(obs, action, rew, next_obs, float(terminated))
            obs = next_obs
            ep_ret += rew
            local_steps += 1
            with stats["lock"]:
                stats["total_steps"] += 1
                if terminated or truncated:
                    stats["ep_returns"].append(ep_ret)
                    ep_ret = 0.0
            if terminated or truncated:
                obs, _ = env.reset()
                ep_ret = 0.0


# ---------------------------------------------------------------------------
# 采集吞吐基准：同样步数下 1 个 vs W 个 worker
# ---------------------------------------------------------------------------
def benchmark_throughput(worker_counts, steps_per_worker: int, env_id: str,
                         seed: int):
    results = []
    for w in worker_counts:
        counter = {"n": 0, "lock": threading.Lock()}

        def run(_wid):
            env = make_env(env_id, seed + 7 * _wid)
            obs, _ = env.reset(seed=seed + 7 * _wid)
            for _ in range(steps_per_worker):
                obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
                if terminated or truncated:
                    obs, _ = env.reset()
                with counter["lock"]:
                    counter["n"] += 1

        threads = [threading.Thread(target=run, args=(i,)) for i in range(w)]
        t0 = time.time()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        dt = time.time() - t0
        sps = counter["n"] / max(dt, 1e-9)
        results.append((w, counter["n"], dt, sps))
        print(f"[基准] {w} 个采集器：{counter['n']} 步 / {dt:.2f}s = {sps:.0f} steps/s")
    return results


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="第049章：SAC 的分布式实现（教学版）")
    p.add_argument("--env", type=str, default="Pendulum-v1", help="环境 ID")
    p.add_argument("--workers", type=int, default=4, help="采集器线程数")
    p.add_argument("--bench-workers", type=str, default="1,4",
                   help="吞吐基准的采集器数量列表")
    p.add_argument("--bench-steps", type=int, default=1500, help="每个采集器在基准阶段的步数")
    p.add_argument("--max-steps", type=int, default=16000, help="训练阶段总环境步数上限")
    p.add_argument("--start-steps", type=int, default=1000, help="训练前纯随机探索步数（所有采集器合计）")
    p.add_argument("--eval-every", type=int, default=2000, help="每收集多少步评估一次")
    p.add_argument("--eval-episodes", type=int, default=5, help="评估回合数")
    p.add_argument("--buffer-size", type=int, default=100000, help="共享回放池容量")
    p.add_argument("--batch-size", type=int, default=256, help="批大小")
    p.add_argument("--hidden", type=int, default=256, help="隐藏层宽度")
    p.add_argument("--lr", type=float, default=3e-4, help="学习率")
    p.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
    p.add_argument("--tau", type=float, default=0.005, help="目标网络软更新系数")
    p.add_argument("--alpha-init", type=float, default=0.2, help="温度系数初值")
    p.add_argument("--target-return", type=float, default=-400.0, help="达标阈值")
    p.add_argument("--seed", type=int, default=0, help="随机种子")
    p.add_argument("--save-dir", type=str, default="runs/ch049", help="输出目录")
    p.add_argument("--quick", action="store_true", help="快速模式：缩短训练")
    return p.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.max_steps = min(args.max_steps, 4000)
        args.eval_every = 1000
        args.start_steps = min(args.start_steps, 400)
        args.bench_steps = min(args.bench_steps, 600)

    set_seed(args.seed)
    # 教学约定：限制 torch 的内部线程数，避免与采集线程争抢 CPU
    torch.set_num_threads(1)

    print(f"环境: {args.env} | 种子: {args.seed} | 采集器: {args.workers} | "
          f"torch 内部线程数: {torch.get_num_threads()}")

    # 1) 吞吐基准：随机动作纯采集，比较 1 与 W 个采集器
    bench_workers = [int(x) for x in args.bench_workers.split(",") if x.strip()]
    bench = benchmark_throughput(bench_workers, args.bench_steps, args.env, args.seed)
    if len(bench) >= 2 and bench[0][3] > 0:
        print(f"[基准] 相对单采集器加速比：{bench[-1][3] / bench[0][3]:.2f}x")

    # 2) 训练：W 个采集线程 + 主线程学习器
    env = make_env(args.env, args.seed + 100)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    action_scale = float(env.action_space.high[0])
    agent = SAC(obs_dim, act_dim, action_scale, args)
    buffer = SharedReplay(args.buffer_size, obs_dim, act_dim)
    stats = {"total_steps": 0, "ep_returns": [], "lock": threading.Lock()}
    stop_event = threading.Event()

    threads = [threading.Thread(
        target=collector_worker,
        args=(i, args.env, args.seed, agent, buffer, stats, args, stop_event),
        daemon=True) for i in range(args.workers)]
    print(f"\n===== 启动 {args.workers} 个采集线程，主线程作为学习器 =====")
    t_start = time.time()
    for t in threads:
        t.start()

    n_updates = 0
    last_steps = 0
    eval_returns = []
    solved_at = None
    next_eval_at = args.eval_every

    while stats["total_steps"] < args.max_steps:
        with stats["lock"]:
            total = stats["total_steps"]
        # 学习器节奏：新收集多少步，就补多少次更新（update-to-data ≈ 1）
        backlog = total - last_steps
        if buffer.size >= max(args.start_steps, args.batch_size) and backlog > 0:
            n_todo = min(backlog, 64)  # 每次循环最多补 64 次，避免长时间占住 GIL
            for _ in range(n_todo):
                agent.update(buffer.sample(args.batch_size), args)
                n_updates += 1
            last_steps += n_todo
        else:
            time.sleep(0.002)  # 数据不够时让出 CPU 给采集线程

        if total >= next_eval_at:
            eval_ret = evaluate(agent, env, args.eval_episodes)
            eval_returns.append(eval_ret)
            elapsed = time.time() - t_start
            with stats["lock"]:
                recent20 = np.mean(stats["ep_returns"][-20:]) if stats["ep_returns"] else float("nan")
            print(f"[学习器] 步数 {total:6d} | 更新 {n_updates:6d} | 回放池 {buffer.size:6d} | "
                  f"吞吐 {total / elapsed:7.1f} steps/s | 近20回合回报 {recent20:8.1f} | "
                  f"评估 {eval_ret:8.1f} | 更新滞后 {total - last_steps:5d}")
            if solved_at is None and eval_ret >= args.target_return:
                solved_at = total
                print(f"[学习器] 首次达到评估回报 {args.target_return}（步数 {total}）")
            while next_eval_at <= total:
                next_eval_at += args.eval_every

    stop_event.set()
    for t in threads:
        t.join(timeout=5.0)
    elapsed = time.time() - t_start

    final_eval = evaluate(agent, env, args.eval_episodes)
    all_evals = eval_returns + [final_eval]
    best_eval = max(all_evals)
    with stats["lock"]:
        n_episodes = len(stats["ep_returns"])
    print("-" * 78)
    print(f"训练结束 | 用时 {elapsed:.1f}s | 总步数 {stats['total_steps']} | "
          f"完成回合 {n_episodes} | 更新次数 {n_updates}")
    print(f"学习器平均吞吐 {stats['total_steps'] / max(elapsed, 1e-9):.1f} steps/s | "
          f"最好评估 {best_eval:.1f} | 最终评估 {final_eval:.1f}")
    print("\n" + "=" * 78)
    print("采集吞吐基准汇总（随机动作，纯环境交互）")
    print(f"{'采集器数':>8}{'总步数':>10}{'耗时(s)':>10}{'吞吐(steps/s)':>16}{'加速比':>10}")
    base = bench[0][3] if bench else 1.0
    for w, n, dt, sps in bench:
        print(f"{w:>8}{n:>10}{dt:>10.2f}{sps:>16.0f}{sps / max(base, 1e-9):>10.2f}")
    print("=" * 78)
    print("说明：线程受 GIL 限制，加速比通常达不到线性；真实系统用多进程/多机。")
    print("日志中的'更新滞后'是学习器欠账：环境步数已收集但尚未做完对应更新。")

    os.makedirs(args.save_dir, exist_ok=True)
    print(f"输出目录：{args.save_dir}")


if __name__ == "__main__":
    main()
