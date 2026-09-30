"""
第044章 SAC的优先级经验回放

均匀回放把每条经验一视同仁，但"教训"的分布并不均匀：TD 误差大的样本才藏着
还没学会的因果。本章实现比例型优先级回放（Prioritized Experience Replay, PER）：

  - 采样概率     P(i) ∝ p_i^alpha        （p_i 为该样本的优先级）
  - 重要性权重   w_i = (N * P(i))^(-beta)（beta 从 0.4 退火到 1.0 修正偏差）
  - 优先级更新   p_i ← (|TD 误差| + eps)^alpha

在 Pendulum-v1 上对比 uniform 与 prioritized 两种采样模式，记录评估回报、
达标步数、采样批次的平均 |TD 误差| 与 beta。

运行：
    python code/ch044.py           # 两种模式依次训练（CPU 约 5-8 分钟）
    python code/ch044.py --quick   # 快速跑通（约 1 分钟）

预期：prioritized 早期收敛更快（同样步数下回报更高），采样到的 |TD 误差|
明显大于均匀采样；代价是实现复杂、对优先级参数敏感。
"""
import argparse
import os
import random
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
# 比例优先级回放池：用求和树（SumTree）把采样与优先级更新都做成 O(log N)
# ---------------------------------------------------------------------------
class PrioritizedReplayBuffer:
    """SumTree 结构：内部节点存子树优先级之和，叶子对应一条经验。"""

    def __init__(self, capacity: int, obs_dim: int, act_dim: int,
                 alpha: float = 0.6, beta_start: float = 0.4,
                 beta_end: float = 1.0, beta_steps: int = 20000):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rew = np.zeros((capacity, 1), dtype=np.float32)
        self.done = np.zeros((capacity, 1), dtype=np.float32)
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.ptr = 0
        self.size = 0
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_steps = max(beta_steps, 1)
        self.n_samples = 0          # 供 beta 退火使用
        self.max_priority = 1.0     # 新样本给全局最大优先级，保证至少被抽一次
        self.eps = 1e-6

    def _leaf(self, i: int) -> int:
        return i + self.capacity - 1

    def _update_tree(self, i: int, priority: float) -> None:
        """自底向上更新 leaf i 及其祖先的优先级和。"""
        idx = self._leaf(i)
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        while idx > 0:
            idx = (idx - 1) // 2
            self.tree[idx] += change

    def add(self, obs, act, rew, next_obs, done) -> None:
        i = self.ptr
        self.obs[i] = obs
        self.act[i] = act
        self.rew[i] = rew
        self.next_obs[i] = next_obs
        self.done[i] = done
        self._update_tree(i, self.max_priority ** self.alpha)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _get_leaf_index(self, s: float) -> int:
        """在树上按前缀和 s 找到对应叶子，返回缓冲区下标。"""
        idx = 0
        while idx < self.capacity - 1:
            left = 2 * idx + 1
            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = left + 1
        return idx - self.capacity + 1

    def sample_uniform(self, batch_size: int) -> dict:
        """均匀采样模式：与第 043 章一致，权重恒为 1。"""
        idx = np.random.randint(0, self.size, size=batch_size)
        weights = np.ones(batch_size, dtype=np.float32)
        return self._pack(idx, weights, beta=0.0)

    def sample_prioritized(self, batch_size: int) -> dict:
        """比例优先级采样 + 重要性权重（beta 线性退火）。"""
        total = float(self.tree[0])
        seg = total / batch_size
        indices = np.zeros(batch_size, dtype=np.int64)
        priorities = np.zeros(batch_size, dtype=np.float64)
        for b in range(batch_size):
            s = np.random.uniform(seg * b, seg * (b + 1))
            i = self._get_leaf_index(s)
            indices[b] = i
            priorities[b] = max(self.tree[self._leaf(i)], 1e-12)
        probs = priorities / max(total, 1e-12)
        # beta 从 beta_start 线性退火到 beta_end
        frac = min(1.0, self.n_samples / self.beta_steps)
        beta = self.beta_start + (self.beta_end - self.beta_start) * frac
        self.n_samples += 1
        # 重要性权重：抵消有偏采样带来的梯度偏差，并归一化到最大值为 1
        weights = (self.size * probs) ** (-beta)
        weights = weights / max(weights.max(), 1e-12)
        return self._pack(indices, weights.astype(np.float32), beta=beta)

    def _pack(self, idx: np.ndarray, weights: np.ndarray, beta: float) -> dict:
        return {
            "obs": torch.as_tensor(self.obs[idx]),
            "act": torch.as_tensor(self.act[idx]),
            "rew": torch.as_tensor(self.rew[idx]),
            "next_obs": torch.as_tensor(self.next_obs[idx]),
            "done": torch.as_tensor(self.done[idx]),
            "weights": torch.as_tensor(weights),
            "idx": idx,
            "beta": beta,
        }

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        """用最新 TD 误差刷新优先级；新优先级同时抬高 max_priority。"""
        for i, err in zip(indices, td_errors):
            p = (abs(float(err)) + self.eps) ** self.alpha
            self._update_tree(int(i), p)
            if p > self.max_priority:
                self.max_priority = p


# ---------------------------------------------------------------------------
# 网络与 SAC（与第 043 章相同，但 Q 损失接受重要性权重）
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
        weights = batch["weights"]

        with torch.no_grad():
            next_act, next_logp = self.policy.sample(next_obs)
            q_next = torch.minimum(self.q1_target(next_obs, next_act),
                                   self.q2_target(next_obs, next_act))
            backup = rew + args.gamma * (1.0 - done) * (q_next - self.alpha.detach() * next_logp)

        q1 = self.q1(obs, act)
        q2 = self.q2(obs, act)
        # 加权 MSE：重要性权重抵消有偏采样，未归一化前 bias 类梯度会有偏
        q1_loss = (weights * (q1 - backup) ** 2).mean()
        q2_loss = (weights * (q2 - backup) ** 2).mean()
        q_loss = q1_loss + q2_loss
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

        # 用当前 Q 的 TD 误差作为优先级信号（取双 Q 的平均，更平滑）
        td_error = (0.5 * ((q1 - backup).abs() + (q2 - backup).abs())).detach()
        return {
            "q_loss": q_loss.item(),
            "policy_loss": policy_loss.item(),
            "alpha": float(self.alpha.item()),
            "td_error": td_error.cpu().numpy(),
        }


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
# 训练一种采样模式
# ---------------------------------------------------------------------------
def train_mode(mode: str, args):
    """mode: uniform（均匀采样）或 prioritized（比例优先级）。"""
    set_seed(args.seed)  # 两种模式同种子，控制变量
    env = make_env(args.env, args.seed)
    eval_env = make_env(args.env, args.seed + 100)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    action_scale = float(env.action_space.high[0])
    agent = SAC(obs_dim, act_dim, action_scale, args)
    buffer = PrioritizedReplayBuffer(
        args.buffer_size, obs_dim, act_dim,
        alpha=args.alpha, beta_start=args.beta_start,
        beta_end=args.beta_end, beta_steps=args.beta_steps)

    obs, _ = env.reset(seed=args.seed)
    ep_ret = 0.0
    ep_returns = []
    history = []
    td_recent = []
    solved_at = None
    batch = None
    t_start = time.time()
    print(f"\n===== 模式 {mode}：alpha={args.alpha}, beta {args.beta_start}->{args.beta_end} =====")

    for step in range(1, args.max_steps + 1):
        if step <= args.start_steps:
            action = env.action_space.sample()
        else:
            action = agent.act(obs, deterministic=False)
        next_obs, rew, terminated, truncated, _ = env.step(action)
        buffer.add(obs, action, rew, next_obs, float(terminated))
        obs = next_obs
        ep_ret += rew

        if terminated or truncated:
            ep_returns.append(ep_ret)
            ep_ret = 0.0
            obs, _ = env.reset()

        if step % args.update_every == 0 and buffer.size >= args.start_steps:
            if mode == "prioritized":
                batch = buffer.sample_prioritized(args.batch_size)
            else:
                batch = buffer.sample_uniform(args.batch_size)
            stats = agent.update(batch, args)
            td_mean = float(np.mean(stats["td_error"]))
            td_recent.append(td_mean)
            if mode == "prioritized":
                buffer.update_priorities(batch["idx"], stats["td_error"])

        if step % args.eval_every == 0:
            eval_ret = evaluate(agent, eval_env, args.eval_episodes)
            history.append((step, eval_ret))
            recent20 = np.mean(ep_returns[-20:]) if ep_returns else float("nan")
            mean_td = float(np.mean(td_recent[-500:])) if td_recent else float("nan")
            beta = batch["beta"] if batch is not None else 0.0
            print(f"[{mode}] 步数 {step:6d} | 近20回合回报 {recent20:8.1f} | "
                  f"评估 {eval_ret:8.1f} | 批次|TD|均值 {mean_td:7.2f} | beta {beta:.2f}")
            if solved_at is None and eval_ret >= args.target_return:
                solved_at = step
                print(f"[{mode}] 首次达到评估回报 {args.target_return}（步数 {step}）")

    elapsed = time.time() - t_start
    final_eval = evaluate(agent, eval_env, args.eval_episodes)
    best_eval = max([h[1] for h in history] + [final_eval])
    mean_td = float(np.mean(td_recent)) if td_recent else float("nan")
    print(f"[{mode}] 结束：最好评估 {best_eval:.1f} | 最终评估 {final_eval:.1f} | "
          f"全程批次|TD|均值 {mean_td:.2f} | 用时 {elapsed:.1f}s")
    return {"mode": mode, "best_eval": best_eval, "final_eval": final_eval,
            "solved_at": solved_at, "mean_td": mean_td, "elapsed": elapsed}


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="第044章：SAC 的优先级经验回放")
    p.add_argument("--env", type=str, default="Pendulum-v1", help="环境 ID")
    p.add_argument("--modes", type=str, default="uniform,prioritized",
                   help="逗号分隔：uniform / prioritized")
    p.add_argument("--max-steps", type=int, default=20000, help="每种模式的环境步数上限")
    p.add_argument("--start-steps", type=int, default=1000, help="训练前纯随机探索步数")
    p.add_argument("--update-every", type=int, default=1, help="每多少环境步做一次更新")
    p.add_argument("--eval-every", type=int, default=4000, help="评估间隔")
    p.add_argument("--eval-episodes", type=int, default=5, help="评估回合数")
    p.add_argument("--buffer-size", type=int, default=100000, help="回放池容量")
    p.add_argument("--batch-size", type=int, default=256, help="批大小")
    p.add_argument("--hidden", type=int, default=256, help="隐藏层宽度")
    p.add_argument("--lr", type=float, default=3e-4, help="学习率")
    p.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
    p.add_argument("--tau", type=float, default=0.005, help="目标网络软更新系数")
    p.add_argument("--alpha-init", type=float, default=0.2, help="温度系数初值")
    p.add_argument("--alpha", type=float, default=0.6, help="优先级指数 alpha（0 退化为均匀）")
    p.add_argument("--beta-start", type=float, default=0.4, help="重要性权重指数 beta 初值")
    p.add_argument("--beta-end", type=float, default=1.0, help="beta 退火终值")
    p.add_argument("--beta-steps", type=int, default=10000, help="beta 退火所用的采样批次数")
    p.add_argument("--target-return", type=float, default=-400.0, help="达标阈值")
    p.add_argument("--seed", type=int, default=0, help="随机种子")
    p.add_argument("--save-dir", type=str, default="runs/ch044", help="输出目录")
    p.add_argument("--quick", action="store_true", help="快速模式：缩短训练")
    return p.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.max_steps = min(args.max_steps, 4000)
        args.eval_every = 1000
        args.start_steps = min(args.start_steps, 400)
        args.beta_steps = 1000

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    print(f"环境: {args.env} | 种子: {args.seed} | 模式: {modes}")
    print(f"配置: max_steps={args.max_steps}, batch={args.batch_size}, "
          f"alpha={args.alpha}, beta {args.beta_start}->{args.beta_end}")

    results = [train_mode(m, args) for m in modes]

    print("\n" + "=" * 84)
    print("采样模式对比")
    print(f"{'模式':<14}{'最好评估':>10}{'最终评估':>10}{'达标步数':>10}{'批次|TD|均值':>14}")
    for r in results:
        solved = r["solved_at"] if r["solved_at"] is not None else -1
        print(f"{r['mode']:<14}{r['best_eval']:>10.1f}{r['final_eval']:>10.1f}"
              f"{solved:>10}{r['mean_td']:>14.2f}")
    print("=" * 84)
    print("结论提示：prioritized 的批次 |TD| 会显著高于 uniform（它专挑还没学会的")
    print("样本），早期评估回报通常上升更快；如果 alpha/beta 设置不当，也会出现")
    print("反复训练少数离群样本、后期震荡的问题。")

    os.makedirs(args.save_dir, exist_ok=True)
    print(f"输出目录：{args.save_dir}")


if __name__ == "__main__":
    main()
