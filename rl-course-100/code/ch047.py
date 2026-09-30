"""
第047章 SAC的批量归一化层

把隐藏层换成"线性 + 归一化 + ReLU"的结构，在 Pendulum-v1 上对比三种配置：
  - none : 裸 MLP（第 046 章的默认结构）
  - layer: LayerNorm，对每条样本的特征做归一化，行为与 batch 无关
  - batch: BatchNorm1d，用 batch 统计量归一化，评估时改用滑动统计

记录四项指标：评估回报、梯度范数变异（训练稳定性）、Q 损失变异，
以及"训练模式与评估模式下策略输出差"——后者直接暴露 BatchNorm 的
滑动统计与在线数据分布不一致的问题（对 none/layer 应恒为 0）。

运行：
    python code/ch047.py           # 三种配置依次训练（CPU 约 5-7 分钟）
    python code/ch047.py --quick   # 快速跑通（约 1 分钟）

预期：LayerNorm 与 none 接近或略更稳；BatchNorm 在小批量非独立同分布
数据上出现训练/评估行为不一致，评估回报波动更大。
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

    def probe(self, batch_size: int) -> dict:
        """取一段固定的探针批次（用于结构诊断），前 batch_size 条即可。"""
        n = min(batch_size, max(self.size, 1))
        idx = np.arange(n)
        return {
            "obs": torch.as_tensor(self.obs[idx]),
            "act": torch.as_tensor(self.act[idx]),
        }


def make_hidden(in_dim: int, hidden: int, out_dim: int, norm: str) -> nn.Sequential:
    """构造 'Linear -> [Norm] -> ReLU' 的隐藏层堆叠，尾部接输出层。"""
    layers = []
    last = in_dim
    for _ in range(2):
        layers.append(nn.Linear(last, hidden))
        if norm == "batch":
            layers.append(nn.BatchNorm1d(hidden))
        elif norm == "layer":
            layers.append(nn.LayerNorm(hidden))
        layers.append(nn.ReLU())
        last = hidden
    layers.append(nn.Linear(last, out_dim))
    return nn.Sequential(*layers)


class QNet(nn.Module):
    """Q(s,a) 网络，隐藏层归一化方式由 norm 参数决定。"""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256,
                 norm: str = "none"):
        super().__init__()
        self.net = make_hidden(obs_dim + act_dim, hidden, 1, norm)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, act], dim=-1)).squeeze(-1)


class GaussianPolicy(nn.Module):
    """策略网络：躯干带归一化，均值头与 log_std 头各自线性输出。"""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256,
                 act_scale: float = 1.0, norm: str = "none"):
        super().__init__()
        layers = []
        last = obs_dim
        for _ in range(2):
            layers.append(nn.Linear(last, hidden))
            if norm == "batch":
                layers.append(nn.BatchNorm1d(hidden))
            elif norm == "layer":
                layers.append(nn.LayerNorm(hidden))
            layers.append(nn.ReLU())
            last = hidden
        self.body = nn.Sequential(*layers)
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
    def __init__(self, obs_dim: int, act_dim: int, action_scale: float,
                 norm: str, args):
        self.norm = norm
        self.policy = GaussianPolicy(obs_dim, act_dim, args.hidden, action_scale, norm)
        self.q1 = QNet(obs_dim, act_dim, args.hidden, norm)
        self.q2 = QNet(obs_dim, act_dim, args.hidden, norm)
        self.q1_target = QNet(obs_dim, act_dim, args.hidden, norm)
        self.q2_target = QNet(obs_dim, act_dim, args.hidden, norm)
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
        # 训练时确保网络处于 train 模式（BatchNorm 使用 batch 统计）
        self.policy.train()
        self.q1.train()
        self.q2.train()

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
        # 记录裁剪前的梯度范数（不做裁剪，仅诊断）
        grad_norm = float(torch.sqrt(sum(
            float((p.grad ** 2).sum()) for p in self.q1.parameters() if p.grad is not None
        ) + sum(
            float((p.grad ** 2).sum()) for p in self.q2.parameters() if p.grad is not None
        )))
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
                "alpha": float(self.alpha.item()), "grad_norm": grad_norm}

    @torch.no_grad()
    def train_eval_output_gap(self, probe: dict) -> float:
        """训练模式与评估模式下策略均值动作的差：BatchNorm 会露出马脚。"""
        obs = probe["obs"]
        self.policy.train()
        mean_train, _ = self.policy.forward(obs)
        a_train = torch.tanh(mean_train)
        self.policy.eval()
        mean_eval, _ = self.policy.forward(obs)
        a_eval = torch.tanh(mean_eval)
        return float((a_train - a_eval).abs().mean().item())


@torch.no_grad()
def evaluate(agent: SAC, env, episodes: int) -> float:
    agent.policy.eval()
    agent.q1.eval()
    agent.q2.eval()
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
# 训练一种归一化配置
# ---------------------------------------------------------------------------
def train_norm(norm: str, args):
    set_seed(args.seed)  # 三种配置同种子，控制变量
    env = make_env(args.env, args.seed)
    eval_env = make_env(args.env, args.seed + 100)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    action_scale = float(env.action_space.high[0])
    agent = SAC(obs_dim, act_dim, action_scale, norm, args)
    buffer = ReplayBuffer(args.buffer_size, obs_dim, act_dim)

    obs, _ = env.reset(seed=args.seed)
    ep_ret = 0.0
    ep_returns = []
    eval_returns = []
    grad_hist = []
    q_loss_hist = []
    gap_hist = []
    solved_at = None
    t_start = time.time()
    print(f"\n===== 归一化配置 {norm}：开始训练 =====")

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
            stats = agent.update(buffer.sample(args.batch_size), args)
            grad_hist.append(stats["grad_norm"])
            q_loss_hist.append(stats["q_loss"])

        if step % args.eval_every == 0:
            eval_ret = evaluate(agent, eval_env, args.eval_episodes)
            eval_returns.append(eval_ret)
            gap = agent.train_eval_output_gap(buffer.probe(256))
            gap_hist.append(gap)
            grad_cv = float(np.std(grad_hist[-1000:]) /
                            (np.mean(grad_hist[-1000:]) + 1e-9))
            recent20 = np.mean(ep_returns[-20:]) if ep_returns else float("nan")
            print(f"[{norm}] 步数 {step:6d} | 近20回合回报 {recent20:8.1f} | "
                  f"评估 {eval_ret:8.1f} | 梯度范数变异 {grad_cv:5.3f} | "
                  f"训练/评估输出差 {gap:.4f}")
            if solved_at is None and eval_ret >= args.target_return:
                solved_at = step
                print(f"[{norm}] 首次达到评估回报 {args.target_return}（步数 {step}）")

    elapsed = time.time() - t_start
    final_eval = evaluate(agent, eval_env, args.eval_episodes)
    all_evals = eval_returns + [final_eval]
    best_eval = max(all_evals)
    eval_std = float(np.std(all_evals[-4:]))
    grad_cv = float(np.std(grad_hist[-2000:]) / (np.mean(grad_hist[-2000:]) + 1e-9))
    q_cv = float(np.std(q_loss_hist[-2000:]) / (np.mean(q_loss_hist[-2000:]) + 1e-9))
    mean_gap = float(np.mean(gap_hist)) if gap_hist else 0.0
    print(f"[{norm}] 结束：最好评估 {best_eval:.1f} | 最终评估 {final_eval:.1f} | "
          f"后期评估std {eval_std:.1f} | 梯度变异 {grad_cv:.3f} | "
          f"训练/评估输出差 {mean_gap:.4f} | 用时 {elapsed:.1f}s")
    return {"norm": norm, "best_eval": best_eval, "final_eval": final_eval,
            "solved_at": solved_at, "eval_std": eval_std, "grad_cv": grad_cv,
            "q_cv": q_cv, "gap": mean_gap, "elapsed": elapsed}


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="第047章：SAC 的批量归一化层")
    p.add_argument("--env", type=str, default="Pendulum-v1", help="环境 ID")
    p.add_argument("--norms", type=str, default="none,layer,batch",
                   help="逗号分隔：none / layer / batch")
    p.add_argument("--max-steps", type=int, default=16000, help="每种配置的环境步数上限")
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
    p.add_argument("--save-dir", type=str, default="runs/ch047", help="输出目录")
    p.add_argument("--quick", action="store_true", help="快速模式：缩短训练")
    return p.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.max_steps = min(args.max_steps, 4000)
        args.eval_every = 1000
        args.start_steps = min(args.start_steps, 400)

    norms = [n.strip() for n in args.norms.split(",") if n.strip()]
    print(f"环境: {args.env} | 种子: {args.seed} | 归一化配置: {norms}")
    print(f"配置: max_steps={args.max_steps}, batch={args.batch_size}, hidden={args.hidden}")

    results = [train_norm(n, args) for n in norms]

    print("\n" + "=" * 100)
    print("归一化配置对比（输出差 = 训练模式与评估模式策略输出的平均绝对差）")
    print(f"{'配置':<8}{'最好评估':>10}{'最终评估':>10}{'达标步数':>10}"
          f"{'后期评估std':>13}{'梯度变异':>10}{'Q损失变异':>11}{'输出差':>10}")
    for r in results:
        solved = r["solved_at"] if r["solved_at"] is not None else -1
        print(f"{r['norm']:<8}{r['best_eval']:>10.1f}{r['final_eval']:>10.1f}"
              f"{solved:>10}{r['eval_std']:>13.1f}{r['grad_cv']:>10.3f}"
              f"{r['q_cv']:>11.3f}{r['gap']:>10.4f}")
    print("=" * 100)
    print("结论提示：none 与 layer 的'输出差'应恒为 0（无 batch 相关统计）；batch")
    print("的滑动统计和数据分布不一致，输出差明显为正。RL 的批次非独立同分布，")
    print("BatchNorm 往往损害稳定性；要归一化，优先 LayerNorm 或对观测做归一化。")

    os.makedirs(args.save_dir, exist_ok=True)
    print(f"输出目录：{args.save_dir}")


if __name__ == "__main__":
    main()
