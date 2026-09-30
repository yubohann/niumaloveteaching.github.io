"""
第056章 SAC的元学习

元学习（Meta-Learning）的目标不是学会某个任务，而是学会"如何快速学会新任务"。
本章用 Reptile 算法在"重力参数分布"上训练一个元初始参数：

  任务分布: Pendulum-v1, g ∈ {6, 8, 10, 12, 14}
  元训练  : 每个元迭代采样一个任务，从元参数出发做短程 SAC 训练（内循环），
            再把元参数朝"内循环解"的方向移动一步：
                theta_meta <- theta_meta + beta * (theta_inner - theta_meta)
  元测试  : 在留出任务 g=9 上比较两条适应曲线
            - meta   : 从元参数出发微调
            - scratch: 从随机初始化出发微调

记录：适应前（0 步）与每 500 步的评估回报，对比"几步微调"后的差距。

运行：
    python code/ch056.py           # 元训练 + 两条适应曲线（CPU 约 3-5 分钟）
    python code/ch056.py --quick   # 快速跑通（约 40 秒）

预期：meta 在 0 步时的成绩就明显好于随机策略，适应曲线在前 1000 步
领先 scratch——这就是"学会快速学习"的直接体现。
"""
import argparse
import copy
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


def make_pendulum(seed: int, g: float):
    env = gym.make("Pendulum-v1")
    env.unwrapped.g = g
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

    # ----- 元学习需要的参数接口 -----
    def named_trainable(self):
        """返回 (名字, 张量) 列表：策略 + 双 Q + 温度系数。"""
        out = []
        for name, p in self.policy.named_parameters():
            out.append(("policy." + name, p))
        for name, p in self.q1.named_parameters():
            out.append(("q1." + name, p))
        for name, p in self.q2.named_parameters():
            out.append(("q2." + name, p))
        out.append(("log_alpha", self.log_alpha))
        return out

    def copy_from(self, other: "SAC") -> None:
        """把另一个智能体的全部参数复制过来（含目标网络，就地复制）。"""
        with torch.no_grad():
            self.policy.load_state_dict(other.policy.state_dict())
            self.q1.load_state_dict(other.q1.state_dict())
            self.q2.load_state_dict(other.q2.state_dict())
            self.q1_target.load_state_dict(other.q1_target.state_dict())
            self.q2_target.load_state_dict(other.q2_target.state_dict())
            self.log_alpha.copy_(other.log_alpha)

    def reptile_perturb(self, inner: "SAC", beta: float) -> None:
        """Reptile 元更新：theta_meta <- theta_meta + beta * (theta_inner - theta_meta)。"""
        with torch.no_grad():
            meta_map = dict(self.named_trainable())
            for name, p_inner in inner.named_trainable():
                p_meta = meta_map[name]
                p_meta.add_(beta * (p_inner.data - p_meta.data))
            # 目标网络跟随主网络（避免元参数与目标网络失配）
            self.q1_target.load_state_dict(self.q1.state_dict())
            self.q2_target.load_state_dict(self.q2.state_dict())

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
        rew, done = rew.squeeze(-1), done.squeeze(-1)      # 与 Q 网络的 (B,) 输出对齐

        with torch.no_grad():
            next_act, next_logp = self.policy.sample(next_obs)
            next_logp = next_logp.squeeze(-1)
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
        logp = logp.squeeze(-1)                            # 同上：统一为 (B,)
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

        return {"q_loss": q_loss.item(), "alpha": float(self.alpha.item())}


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
# 短程训练（内循环与适应实验共用）
# ---------------------------------------------------------------------------
def train_short(agent: SAC, g: float, steps: int, seed: int, args,
                eval_env=None, eval_points=None, tag: str = ""):
    """从 current 参数出发，在指定 g 的任务上训练 steps 步。

    eval_points 给定时，在这些步数处评估（含 0 步）并打印。
    返回 [(步数, 评估回报)] 列表。
    """
    env = make_pendulum(seed, g)
    buffer = ReplayBuffer(args.buffer_size, env.observation_space.shape[0],
                          env.action_space.shape[0])
    eval_history = []
    if eval_env is not None and eval_points is not None and 0 in eval_points:
        ret = evaluate(agent, eval_env, args.eval_episodes)
        eval_history.append((0, ret))
        print(f"    [{tag}] 适应步数     0 | 评估 {ret:8.1f}")

    obs, _ = env.reset()
    ep_ret = 0.0
    for step in range(1, steps + 1):
        if step <= args.inner_start:
            action = env.action_space.sample()
        else:
            action = agent.act(obs, deterministic=False)
        next_obs, rew, terminated, truncated, _ = env.step(action)
        buffer.add(obs, action, rew, next_obs, float(terminated))
        obs = next_obs
        ep_ret += rew
        if terminated or truncated:
            ep_ret = 0.0
            obs, _ = env.reset()
        if buffer.size >= min(args.inner_start, 200):
            agent.update(buffer.sample(args.batch_size), args)
        if eval_env is not None and eval_points is not None and step in eval_points:
            ret = evaluate(agent, eval_env, args.eval_episodes)
            eval_history.append((step, ret))
            print(f"    [{tag}] 适应步数 {step:5d} | 评估 {ret:8.1f}")
    return eval_history


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="第056章：SAC 的元学习（Reptile）")
    p.add_argument("--train-gs", type=str, default="6,8,10,12,14",
                   help="元训练任务分布的重力列表")
    p.add_argument("--test-g", type=float, default=9.0, help="留出任务的重力（用于适应实验）")
    p.add_argument("--meta-iters", type=int, default=8, help="元迭代次数（每次采样一个任务）")
    p.add_argument("--inner-steps", type=int, default=1500, help="内循环每个任务的训练步数")
    p.add_argument("--adapt-steps", type=int, default=2000, help="留出任务上的适应步数")
    p.add_argument("--adapt-every", type=int, default=500, help="适应评估间隔")
    p.add_argument("--reptile-beta", type=float, default=0.5, help="Reptile 移动步长")
    p.add_argument("--inner-start", type=int, default=200, help="内循环的随机探索步数")
    p.add_argument("--eval-episodes", type=int, default=3, help="评估回合数")
    p.add_argument("--buffer-size", type=int, default=50000, help="回放池容量")
    p.add_argument("--batch-size", type=int, default=256, help="批大小")
    p.add_argument("--hidden", type=int, default=256, help="隐藏层宽度")
    p.add_argument("--lr", type=float, default=3e-4, help="学习率")
    p.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
    p.add_argument("--tau", type=float, default=0.005, help="目标网络软更新系数")
    p.add_argument("--alpha-init", type=float, default=0.2, help="温度系数初值")
    p.add_argument("--seed", type=int, default=0, help="随机种子")
    p.add_argument("--save-dir", type=str, default="runs/ch056", help="输出目录")
    p.add_argument("--quick", action="store_true", help="快速模式：缩短训练")
    return p.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.meta_iters = min(args.meta_iters, 4)
        args.inner_steps = min(args.inner_steps, 400)
        args.adapt_steps = min(args.adapt_steps, 600)
        args.adapt_every = 200
        args.eval_episodes = 2

    train_gs = [float(x) for x in args.train_gs.split(",") if x.strip()]
    set_seed(args.seed)
    adapt_points = [0] + list(range(args.adapt_every, args.adapt_steps + 1, args.adapt_every))

    print(f"元训练任务: g={train_gs} | 留出任务: g={args.test_g}")
    print(f"元迭代 {args.meta_iters} × 内循环 {args.inner_steps} 步 | "
          f"适应实验 {args.adapt_steps} 步 | reptile beta={args.reptile_beta}")

    # 1) 元训练（Reptile）
    print("\n===== 元训练（Reptile） =====")
    obs_dim, act_dim = 3, 1
    action_scale = 2.0
    meta_agent = SAC(obs_dim, act_dim, action_scale, args)
    t_start = time.time()
    for it in range(args.meta_iters):
        g = train_gs[it % len(train_gs)]
        inner = SAC(obs_dim, act_dim, action_scale, args)
        inner.copy_from(meta_agent)
        train_short(inner, g, args.inner_steps, args.seed + 1000 * it, args, tag=f"内{it}")
        # 元参数朝内循环解移动
        meta_agent.reptile_perturb(inner, args.reptile_beta)
        print(f"[元迭代 {it + 1}/{args.meta_iters}] 任务 g={g} | "
              f"内循环 {args.inner_steps} 步完成，元参数已更新")
    meta_elapsed = time.time() - t_start
    print(f"元训练完成，用时 {meta_elapsed:.1f}s")

    # 2) 留出任务上的适应实验
    print(f"\n===== 留出任务 g={args.test_g}：meta（元参数出发） =====")
    adapt_env = make_pendulum(args.seed + 777, args.test_g)
    meta_agent_adapt = SAC(obs_dim, act_dim, action_scale, args)
    meta_agent_adapt.copy_from(meta_agent)
    curve_meta = train_short(meta_agent_adapt, args.test_g, args.adapt_steps,
                             args.seed + 9000, args,
                             eval_env=adapt_env, eval_points=adapt_points, tag="meta")

    print(f"\n===== 留出任务 g={args.test_g}：scratch（随机初始化） =====")
    scratch_agent = SAC(obs_dim, act_dim, action_scale, args)
    curve_scratch = train_short(scratch_agent, args.test_g, args.adapt_steps,
                                args.seed + 9000, args,
                                eval_env=adapt_env, eval_points=adapt_points, tag="scratch")

    # 3) 汇总适应曲线
    print("\n" + "=" * 76)
    print(f"留出任务 g={args.test_g} 的适应曲线（评估回报）")
    print(f"{'适应步数':>10}{'meta':>12}{'scratch':>12}{'差值(m-s)':>12}")
    meta_hist = dict(curve_meta)
    scratch_hist = dict(curve_scratch)
    for step in adapt_points:
        m = meta_hist.get(step, float("nan"))
        s = scratch_hist.get(step, float("nan"))
        print(f"{step:>10}{m:>12.1f}{s:>12.1f}{m - s:>12.1f}")
    print("=" * 76)
    print("结论提示：meta 在 0 步时（还没适应）就带着'多个重力任务'的先验，")
    print("通常明显好于随机初始化；前几百步的差距最大，之后逐步收敛。")
    print("Reptile 只用到一阶信息、无需二阶梯度，是元学习里最易实现的版本。")

    os.makedirs(args.save_dir, exist_ok=True)
    ckpt_path = os.path.join(args.save_dir, "meta_init.pt")
    torch.save({"policy": meta_agent.policy.state_dict(),
                "q1": meta_agent.q1.state_dict(),
                "q2": meta_agent.q2.state_dict(),
                "log_alpha": meta_agent.log_alpha.detach()}, ckpt_path)
    print(f"元初始化参数已保存到 {ckpt_path}")


if __name__ == "__main__":
    main()
