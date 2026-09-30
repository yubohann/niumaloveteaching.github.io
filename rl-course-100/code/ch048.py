"""
第048章 SAC的梯度裁剪

梯度裁剪是深度 RL 最常见的保险丝：反传后把梯度范数限制在上限内，防止个别
爆炸样本把一次更新带飞。本章在 Pendulum-v1 上对比三种策略：

  - none : 不裁剪（基线）
  - 0.5  : clip_grad_norm_ 上限 0.5（激进裁剪）
  - 5.0  : clip_grad_norm_ 上限 5.0（温和裁剪）

对每次更新记录裁剪前的 Q 网络梯度总范数，统计其均值/95 分位/最大值，
以及"裁剪被触发的比例"，再结合评估回报与后期波动判断：

  - 阈值远大于梯度常态 → 裁剪几乎不触发，退回 none
  - 阈值小于梯度常态   → 每次都被压缩，等价于改变有效学习率

运行：
    python code/ch048.py           # 三种策略依次训练（CPU 约 5-7 分钟）
    python code/ch048.py --quick   # 快速跑通（约 1 分钟）

预期：Pendulum 的梯度范数大部分时间在 1~10 之间；5.0 的触发率很低（多数
时候等价于 none）；0.5 几乎每次触发，前期学习变慢但曲线可能更平滑。
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


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, act_dim: int):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rew = np.zeros((capacity, 1), dtype=np.float32)
        self.done = np.zeros((capacity, 1), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(self, obs, act, rew, next_obs, done) -> None:
        i = self.ptr
        self.obs[i] = obs
        self.act[i] = act
        self.rew[i] = rew
        self.next_obs[i] = next_obs
        self.done[i] = done
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict:
        idx = np.random.randint(0, self.size, size=batch_size)
        return {
            "obs": torch.as_tensor(self.obs[idx]),
            "act": torch.as_tensor(self.act[idx]),
            "rew": torch.as_tensor(self.rew[idx]),
            "next_obs": torch.as_tensor(self.next_obs[idx]),
            "done": torch.as_tensor(self.done[idx]),
        }


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


def grad_norm(module: nn.Module) -> float:
    """参数梯度总范数（L2），在优化器 step 之前调用。"""
    total = 0.0
    for p in module.parameters():
        if p.grad is not None:
            total += float((p.grad ** 2).sum())
    return float(np.sqrt(total))


def apply_clip(module: nn.Module, clip: float) -> bool:
    """按阈值裁剪梯度范数，返回是否触发了裁剪。"""
    norm = grad_norm(module)
    if clip > 0.0:
        nn.utils.clip_grad_norm_(module.parameters(), clip)
    return clip > 0.0 and norm > clip


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

    def update(self, batch: dict, args, clip: float) -> dict:
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
        # 裁剪前的 Q 网络梯度总范数 + 是否触发裁剪
        q_raw = 0.0
        for p in list(self.q1.parameters()) + list(self.q2.parameters()):
            if p.grad is not None:
                q_raw += float((p.grad ** 2).sum())
        q_raw = float(np.sqrt(q_raw))
        c1 = apply_clip(self.q1, clip)
        c2 = apply_clip(self.q2, clip)
        q_clipped = c1 or c2
        self.opt_q.step()

        new_act, logp = self.policy.sample(obs)
        q_pi = torch.minimum(self.q1(obs, new_act), self.q2(obs, new_act))
        policy_loss = (self.alpha.detach() * logp - q_pi).mean()
        self.opt_policy.zero_grad()
        policy_loss.backward()
        p_raw = grad_norm(self.policy)
        apply_clip(self.policy, clip)
        self.opt_policy.step()

        alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()
        self.opt_alpha.zero_grad()
        alpha_loss.backward()
        self.opt_alpha.step()

        with torch.no_grad():
            soft_update(self.q1, self.q1_target, args.tau)
            soft_update(self.q2, self.q2_target, args.tau)

        return {"q_loss": q_loss.item(), "policy_loss": policy_loss.item(),
                "alpha": float(self.alpha.item()), "q_grad_norm": q_raw,
                "p_grad_norm": p_raw, "clipped": float(q_clipped)}


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
# 训练一种裁剪策略
# ---------------------------------------------------------------------------
def train_clip(clip: float, args):
    set_seed(args.seed)  # 三种策略同种子，控制变量
    env = make_env(args.env, args.seed)
    eval_env = make_env(args.env, args.seed + 100)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    action_scale = float(env.action_space.high[0])
    agent = SAC(obs_dim, act_dim, action_scale, args)
    buffer = ReplayBuffer(args.buffer_size, obs_dim, act_dim)

    obs, _ = env.reset(seed=args.seed)
    ep_ret = 0.0
    ep_returns = []
    eval_returns = []
    q_norms = []
    clip_flags = []
    solved_at = None
    label = "none" if clip <= 0 else f"{clip}"
    t_start = time.time()
    print(f"\n===== 裁剪策略 {label}：开始训练 =====")

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
            stats = agent.update(buffer.sample(args.batch_size), args, clip)
            q_norms.append(stats["q_grad_norm"])
            clip_flags.append(stats["clipped"])

        if step % args.eval_every == 0:
            eval_ret = evaluate(agent, eval_env, args.eval_episodes)
            eval_returns.append(eval_ret)
            recent_norms = q_norms[-2000:]
            mean_n = float(np.mean(recent_norms))
            p95_n = float(np.percentile(recent_norms, 95))
            trig = float(np.mean(clip_flags[-2000:]))
            recent20 = np.mean(ep_returns[-20:]) if ep_returns else float("nan")
            print(f"[clip={label}] 步数 {step:6d} | 近20回合回报 {recent20:8.1f} | "
                  f"评估 {eval_ret:8.1f} | 梯度范数均值 {mean_n:6.2f} | "
                  f"p95 {p95_n:6.2f} | 触发率 {trig:.3f}")
            if solved_at is None and eval_ret >= args.target_return:
                solved_at = step
                print(f"[clip={label}] 首次达到评估回报 {args.target_return}（步数 {step}）")

    elapsed = time.time() - t_start
    final_eval = evaluate(agent, eval_env, args.eval_episodes)
    all_evals = eval_returns + [final_eval]
    best_eval = max(all_evals)
    eval_std = float(np.std(all_evals[-4:]))
    mean_n = float(np.mean(q_norms))
    p95_n = float(np.percentile(q_norms, 95))
    max_n = float(np.max(q_norms))
    trig = float(np.mean(clip_flags))
    print(f"[clip={label}] 结束：最好评估 {best_eval:.1f} | 最终评估 {final_eval:.1f} | "
          f"梯度均值 {mean_n:.2f} p95 {p95_n:.2f} max {max_n:.2f} | "
          f"触发率 {trig:.3f} | 用时 {elapsed:.1f}s")
    return {"clip": clip, "label": label, "best_eval": best_eval,
            "final_eval": final_eval, "solved_at": solved_at, "eval_std": eval_std,
            "mean_norm": mean_n, "p95_norm": p95_n, "max_norm": max_n,
            "trig": trig, "elapsed": elapsed}


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="第048章：SAC 的梯度裁剪")
    p.add_argument("--env", type=str, default="Pendulum-v1", help="环境 ID")
    p.add_argument("--clips", type=str, default="none,0.5,5.0",
                   help="逗号分隔的裁剪阈值：none 或正数（如 0.5,5.0）")
    p.add_argument("--max-steps", type=int, default=16000, help="每种策略的环境步数上限")
    p.add_argument("--start-steps", type=int, default=1000, help="训练前纯随机探索步数")
    p.add_argument("--update-every", type=int, default=1, help="每多少环境步做一次更新")
    p.add_argument("--eval-every", type=int, default=2000, help="评估间隔")
    p.add_argument("--eval-episodes", type=int, default=5, help="评估回合数")
    p.add_argument("--buffer-size", type=int, default=100000, help="回放池容量")
    p.add_argument("--batch-size", type=int, default=256, help="批大小")
    p.add_argument("--hidden", type=int, default=256, help="隐藏层宽度")
    p.add_argument("--lr", type=float, default=3e-4, help="学习率")
    p.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
    p.add_argument("--tau", type=float, default=0.005, help="目标网络软更新系数")
    p.add_argument("--alpha-init", type=float, default=0.2, help="温度系数初值")
    p.add_argument("--target-return", type=float, default=-400.0, help="达标阈值")
    p.add_argument("--seed", type=int, default=0, help="随机种子")
    p.add_argument("--save-dir", type=str, default="runs/ch048", help="输出目录")
    p.add_argument("--quick", action="store_true", help="快速模式：缩短训练")
    return p.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.max_steps = min(args.max_steps, 4000)
        args.eval_every = 1000
        args.start_steps = min(args.start_steps, 400)

    clips = []
    for x in args.clips.split(","):
        x = x.strip()
        if x:
            clips.append(0.0 if x == "none" else float(x))

    print(f"环境: {args.env} | 种子: {args.seed} | 裁剪阈值: {clips}（0 表示不裁剪）")
    print(f"配置: max_steps={args.max_steps}, batch={args.batch_size}, lr={args.lr}")

    results = [train_clip(c, args) for c in clips]

    print("\n" + "=" * 104)
    print("梯度裁剪策略对比（触发率 = 被裁剪的更新占比；规范数取每次更新的 Q 梯度总范数）")
    print(f"{'阈值':>8}{'最好评估':>10}{'最终评估':>10}{'达标步数':>10}"
          f"{'后期评估std':>13}{'范数均值':>10}{'p95':>8}{'max':>9}{'触发率':>9}")
    for r in results:
        solved = r["solved_at"] if r["solved_at"] is not None else -1
        print(f"{r['label']:>8}{r['best_eval']:>10.1f}{r['final_eval']:>10.1f}"
              f"{solved:>10}{r['eval_std']:>13.1f}{r['mean_norm']:>10.2f}"
              f"{r['p95_norm']:>8.2f}{r['max_norm']:>9.2f}{r['trig']:>9.3f}")
    print("=" * 104)
    print("结论提示：阈值远大于梯度常态时裁剪几乎不触发，结果与 none 接近；阈值小")
    print("于梯度常态时会持续压缩更新幅度，等于改变有效学习率，前期学习变慢。选")
    print("阈值的依据是梯度范数的 p95 或 max，而不是凭感觉。")

    os.makedirs(args.save_dir, exist_ok=True)
    print(f"输出目录：{args.save_dir}")


if __name__ == "__main__":
    main()
