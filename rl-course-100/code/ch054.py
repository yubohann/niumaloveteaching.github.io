"""
第054章 SAC的迁移学习

先在源任务（Pendulum-v1，重力 g=10）上训练一个 SAC，再把参数迁移到
目标任务（同一环境，重力改为 g=16，动力学更快更难），对比：

  - scratch  : 从随机初始化直接在目标任务上训练
  - transfer : 用源任务的策略与 Q 网络权重初始化，再在目标任务上微调

两种模式共用种子与预算，只差一个"初始参数"。记录：迁移时的初始评估、
达到 −400 / −300 的步数、最终评估与评估曲线。

g=16 的摆杆下落更快、需要更频繁的力矩调整，但"把杆子甩起来"的宏观
技能与 g=10 一致——这正是迁移学习能利用的结构。

运行：
    python code/ch054.py           # 源训练 + 两种目标模式（CPU 约 4-6 分钟）
    python code/ch054.py --quick   # 快速跑通（约 1 分钟）

预期：transfer 的初始评估通常更差（源策略没适配新动力学），但达标步数
明显更少——省下的正是"重新发现摆动策略"的时间。
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


def make_pendulum(seed: int, g: float):
    """创建指定重力的摆杆环境（g 是 Pendulum 的动力学参数）。"""
    env = gym.make("Pendulum-v1")
    env.unwrapped.g = g  # 直接修改底层参数：g=10 为默认，g=16 更难
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

    def state_dict(self) -> dict:
        """打包全部可学习参数，供迁移使用。"""
        return {
            "policy": self.policy.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "q1_target": self.q1_target.state_dict(),
            "q2_target": self.q2_target.state_dict(),
            "log_alpha": self.log_alpha.detach().clone(),
        }

    def load_state(self, ckpt: dict) -> None:
        self.policy.load_state_dict(ckpt["policy"])
        self.q1.load_state_dict(ckpt["q1"])
        self.q2.load_state_dict(ckpt["q2"])
        self.q1_target.load_state_dict(ckpt["q1_target"])
        self.q2_target.load_state_dict(ckpt["q2_target"])
        with torch.no_grad():
            self.log_alpha.copy_(ckpt["log_alpha"])

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
# 通用训练循环（源任务与目标模式共用）
# ---------------------------------------------------------------------------
def train_agent(agent: SAC, env, eval_env, args, steps: int, tag: str,
                log_every: int, threshold: float = None):
    obs, _ = env.reset(seed=args.seed)
    ep_ret = 0.0
    ep_returns = []
    history = []          # (步数, 评估回报)
    solved_at = None
    t_start = time.time()

    for step in range(1, steps + 1):
        if step <= args.start_steps:
            action = env.action_space.sample()
        else:
            action = agent.act(obs, deterministic=False)
        next_obs, rew, terminated, truncated, _ = env.step(action)
        agent.buffer.add(obs, action, rew, next_obs, float(terminated))
        obs = next_obs
        ep_ret += rew
        if terminated or truncated:
            ep_returns.append(ep_ret)
            ep_ret = 0.0
            obs, _ = env.reset()

        if step % args.update_every == 0 and agent.buffer.size >= min(args.start_steps, 1000):
            agent.update(agent.buffer.sample(args.batch_size), args)

        if step % log_every == 0:
            eval_ret = evaluate(agent, eval_env, args.eval_episodes)
            history.append((step, eval_ret))
            recent20 = np.mean(ep_returns[-20:]) if ep_returns else float("nan")
            print(f"[{tag}] 步数 {step:6d} | 近20回合回报 {recent20:8.1f} | 评估 {eval_ret:8.1f}")
            if threshold is not None and solved_at is None and eval_ret >= threshold:
                solved_at = step
                print(f"[{tag}] 首次达到评估回报 {threshold}（步数 {step}）")

    elapsed = time.time() - t_start
    return {"history": history, "solved_at": solved_at, "elapsed": elapsed}


def make_agent_with_buffer(obs_dim, act_dim, action_scale, args):
    agent = SAC(obs_dim, act_dim, action_scale, args)
    agent.buffer = ReplayBuffer(args.buffer_size, obs_dim, act_dim)
    return agent


# ---------------------------------------------------------------------------
# 目标任务的两种模式
# ---------------------------------------------------------------------------
def run_target_mode(mode: str, source_ckpt: dict, args):
    """mode: scratch（随机初始化）/ transfer（从源任务权重初始化）。"""
    set_seed(args.seed)  # 两种模式同种子，控制变量
    env = make_pendulum(args.seed, args.target_g)
    eval_env = make_pendulum(args.seed + 100, args.target_g)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    action_scale = float(env.action_space.high[0])
    agent = make_agent_with_buffer(obs_dim, act_dim, action_scale, args)

    if mode == "transfer":
        agent.load_state(source_ckpt)

    initial_eval = evaluate(agent, eval_env, args.eval_episodes)
    print(f"\n===== 目标模式 {mode}：迁移后初始评估 {initial_eval:.1f} =====")
    stats = train_agent(agent, env, eval_env, args, args.target_steps, mode,
                        log_every=args.eval_every, threshold=args.threshold)
    final_eval = evaluate(agent, eval_env, args.eval_episodes)
    best_eval = max([h[1] for h in stats["history"]] + [final_eval])
    print(f"[{mode}] 结束：初始评估 {initial_eval:.1f} | 最好评估 {best_eval:.1f} | "
          f"最终评估 {final_eval:.1f} | 达标步数 "
          f"{stats['solved_at'] if stats['solved_at'] is not None else -1} | "
          f"用时 {stats['elapsed']:.1f}s")
    return {"mode": mode, "initial_eval": initial_eval, "best_eval": best_eval,
            "final_eval": final_eval, "solved_at": stats["solved_at"],
            "history": stats["history"], "elapsed": stats["elapsed"]}


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="第054章：SAC 的迁移学习")
    p.add_argument("--source-g", type=float, default=10.0, help="源任务重力")
    p.add_argument("--target-g", type=float, default=16.0, help="目标任务重力（更难）")
    p.add_argument("--modes", type=str, default="scratch,transfer",
                   help="逗号分隔：scratch / transfer")
    p.add_argument("--source-steps", type=int, default=20000, help="源任务训练步数")
    p.add_argument("--target-steps", type=int, default=15000, help="每个目标模式的训练步数")
    p.add_argument("--start-steps", type=int, default=1000, help="训练前纯随机探索步数")
    p.add_argument("--update-every", type=int, default=1, help="每多少环境步做一次更新")
    p.add_argument("--eval-every", type=int, default=3000, help="评估间隔")
    p.add_argument("--eval-episodes", type=int, default=5, help="评估回合数")
    p.add_argument("--threshold", type=float, default=-400.0, help="达标阈值")
    p.add_argument("--buffer-size", type=int, default=100000, help="回放池容量")
    p.add_argument("--batch-size", type=int, default=256, help="批大小")
    p.add_argument("--hidden", type=int, default=256, help="隐藏层宽度")
    p.add_argument("--lr", type=float, default=3e-4, help="学习率")
    p.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
    p.add_argument("--tau", type=float, default=0.005, help="目标网络软更新系数")
    p.add_argument("--alpha-init", type=float, default=0.2, help="温度系数初值")
    p.add_argument("--seed", type=int, default=0, help="随机种子")
    p.add_argument("--save-dir", type=str, default="runs/ch054", help="输出目录")
    p.add_argument("--quick", action="store_true", help="快速模式：缩短训练")
    return p.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.source_steps = min(args.source_steps, 4000)
        args.target_steps = min(args.target_steps, 3000)
        args.eval_every = 1000
        args.start_steps = min(args.start_steps, 300)

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    print(f"源任务: Pendulum-v1 (g={args.source_g}) | 目标任务: Pendulum-v1 (g={args.target_g})")
    print(f"种子: {args.seed} | 模式: {modes} | 源步数 {args.source_steps} / "
          f"目标步数 {args.target_steps}")

    # 1) 源任务训练
    set_seed(args.seed)
    src_env = make_pendulum(args.seed, args.source_g)
    src_eval = make_pendulum(args.seed + 100, args.source_g)
    obs_dim = src_env.observation_space.shape[0]
    act_dim = src_env.action_space.shape[0]
    action_scale = float(src_env.action_space.high[0])
    src_agent = make_agent_with_buffer(obs_dim, act_dim, action_scale, args)
    print("\n===== 源任务训练 =====")
    src_stats = train_agent(src_agent, src_env, src_eval, args, args.source_steps,
                            "源任务", log_every=args.eval_every)
    src_final = evaluate(src_agent, src_eval, args.eval_episodes)
    print(f"[源任务] 最终评估 {src_final:.1f} | 用时 {src_stats['elapsed']:.1f}s")
    source_ckpt = src_agent.state_dict()

    # 保存源模型（便于复现）
    os.makedirs(args.save_dir, exist_ok=True)
    ckpt_path = os.path.join(args.save_dir, "source_pendulum_g10.pt")
    torch.save(source_ckpt, ckpt_path)
    print(f"源任务模型已保存到 {ckpt_path}")

    # 2) 目标任务的各模式
    results = [run_target_mode(m, source_ckpt, args) for m in modes]

    print("\n" + "=" * 96)
    print(f"迁移学习对比（目标任务 g={args.target_g}，源 g={args.source_g}，"
          f"源任务最终评估 {src_final:.1f}）")
    print(f"{'模式':<10}{'初始评估':>10}{'最好评估':>10}{'最终评估':>10}"
          f"{'达标步数':>10}{'用时(s)':>10}")
    for r in results:
        solved = r["solved_at"] if r["solved_at"] is not None else -1
        print(f"{r['mode']:<10}{r['initial_eval']:>10.1f}{r['best_eval']:>10.1f}"
              f"{r['final_eval']:>10.1f}{solved:>10}{r['elapsed']:>10.1f}")
    print("=" * 96)
    print("结论提示：transfer 的初始评估不一定更好（源策略未适配新动力学），但")
    print("达标步数通常明显更少：它省掉了'重新探索摆动策略'的阶段。迁移的收益")
    print("取决于两个任务的动力学差距——差距越大，迁移越接近从头训练。")

    print(f"输出目录：{args.save_dir}")


if __name__ == "__main__":
    main()
