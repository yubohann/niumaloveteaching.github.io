"""
第059章 SAC的对抗攻击防御

第 058 章的扰动是随机噪声；本章换更强的威胁模型——对手知道策略参数，会
朝着"让动作偏离最大"的方向构造观测扰动（PGD，L∞ 球内）。实验流程：

  1. 训练两个受害者策略
       clean : 干净训练
       adv   : 对抗训练——交互时把观测先做 3 步 PGD 攻击再喂给策略，
               并把被攻击后的观测存进回放池
  2. 在攻击强度 eps ∈ {0.02, 0.05, 0.10} 下评估两个策略，记录回报、
     动作偏离量与保持率
  3. 附加测一种零成本防御：动作平滑（输出 = 0.5*新动作 + 0.5*旧动作）

运行：
    python code/ch059.py           # 两个策略训练 + 攻击评估矩阵（CPU 约 3-5 分钟）
    python code/ch059.py --quick   # 快速跑通（约 50 秒）

预期：clean 在 eps=0.05 时评估回报显著下降，动作偏离量随 eps 上升；
adv 在攻击下的保持率更高；动作平滑对中等强度攻击有额外缓解。
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


def make_pendulum(seed: int):
    env = gym.make("Pendulum-v1")
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


# ---------------------------------------------------------------------------
# PGD 攻击：在 L∞ 球内最大化"动作偏离"
# ---------------------------------------------------------------------------
def pgd_attack(agent: SAC, obs, eps: float, steps: int) -> np.ndarray:
    """对观测做 PGD 攻击，目标是让确定性动作偏离干净动作最多。

    返回被攻击后的观测（numpy）。eps <= 0 时原样返回。
    """
    if eps <= 0:
        return np.asarray(obs, dtype=np.float32)
    obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
    with torch.no_grad():
        base_action = agent.policy.mean_action(obs_t)
    delta = torch.zeros_like(obs_t, requires_grad=True)
    alpha = 2.5 * eps / max(steps, 1)
    for _ in range(steps):
        adv_action = agent.policy.mean_action(obs_t + delta)
        loss = -((adv_action - base_action) ** 2).sum()  # 最大化动作偏离
        grad = torch.autograd.grad(loss, delta)[0]
        delta = (delta + alpha * grad.sign()).detach()
        delta = delta.clamp(-eps, eps)  # 投影回 L∞ 球
        delta.requires_grad_(True)
    return (obs_t + delta.detach()).squeeze(0).numpy()


def evaluate_attack(agent: SAC, eps: float, smooth: bool, episodes: int,
                    attack_steps: int, seed: int):
    """在攻击下评估：返回 (平均回报, 平均动作偏离)。"""
    env = make_pendulum(seed)
    returns, devs = [], []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_ret = 0.0
        prev_action = np.zeros(env.action_space.shape, dtype=np.float32)
        while not done:
            if eps > 0:
                obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
                with torch.no_grad():
                    a_clean = agent.policy.mean_action(obs_t)
                adv_obs = pgd_attack(agent, obs, eps, attack_steps)
                with torch.no_grad():
                    a_adv = agent.policy.mean_action(
                        torch.as_tensor(adv_obs).unsqueeze(0))
                devs.append(float((a_adv - a_clean).abs().mean().item()))
                action = a_adv.squeeze(0).numpy()
            else:
                action = agent.act(obs, deterministic=True)
            if smooth:
                action = 0.5 * action + 0.5 * prev_action
                prev_action = action.astype(np.float32)
            obs, rew, terminated, truncated, _ = env.step(action)
            ep_ret += rew
            done = terminated or truncated
        returns.append(ep_ret)
    return float(np.mean(returns)), float(np.mean(devs)) if devs else 0.0


# ---------------------------------------------------------------------------
# 训练策略（可开启对抗训练）
# ---------------------------------------------------------------------------
def train_agent(adversarial: bool, args, tag: str) -> SAC:
    set_seed(args.seed)  # 两个策略同种子，控制变量
    env = make_pendulum(args.seed)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    action_scale = float(env.action_space.high[0])
    agent = SAC(obs_dim, act_dim, action_scale, args)
    buffer = ReplayBuffer(args.buffer_size, obs_dim, act_dim)

    obs, _ = env.reset()
    ep_ret = 0.0
    ep_returns = []
    t_start = time.time()
    print(f"\n===== 训练 [{tag}]：对抗训练={adversarial} "
          f"(eps={args.adv_eps if adversarial else 0}) =====")

    for step in range(1, args.max_steps + 1):
        if step <= args.start_steps:
            action = env.action_space.sample()
            obs_in = obs
        else:
            if adversarial:
                # 对抗训练：策略看到的是被攻击后的观测
                obs_in = pgd_attack(agent, obs, args.adv_eps, args.adv_steps)
            else:
                obs_in = obs
            action = agent.act(obs_in, deterministic=False)
        next_obs, rew, terminated, truncated, _ = env.step(action)
        next_in = next_obs
        if adversarial and step > args.start_steps:
            # next_obs 也用同样强度的攻击，保持数据分布一致
            next_in = pgd_attack(agent, next_obs, args.adv_eps, 1)  # 单步近似，降低开销
        buffer.add(obs_in, action, rew, next_in, float(terminated))
        obs = next_obs
        ep_ret += rew
        if terminated or truncated:
            ep_returns.append(ep_ret)
            ep_ret = 0.0
            obs, _ = env.reset()

        if step % args.update_every == 0 and buffer.size >= min(args.start_steps, 1000):
            agent.update(buffer.sample(args.batch_size), args)

        if step % args.eval_every == 0:
            ret_clean, _ = evaluate_attack(agent, 0.0, False, args.eval_episodes,
                                           args.attack_steps, args.seed + 100)
            recent20 = np.mean(ep_returns[-20:]) if ep_returns else float("nan")
            print(f"[{tag}] 步数 {step:6d} | 近20回合回报 {recent20:8.1f} | "
                  f"干净评估 {ret_clean:8.1f}")

    print(f"[{tag}] 训练完成，用时 {time.time() - t_start:.1f}s")
    return agent


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="第059章：SAC 的对抗攻击防御")
    p.add_argument("--max-steps", type=int, default=12000, help="每个策略的训练步数")
    p.add_argument("--start-steps", type=int, default=1000, help="训练前纯随机探索步数")
    p.add_argument("--adv-eps", type=float, default=0.05, help="对抗训练的攻击强度")
    p.add_argument("--adv-steps", type=int, default=3, help="对抗训练中 PGD 的步数")
    p.add_argument("--attack-steps", type=int, default=10, help="评估中 PGD 的步数")
    p.add_argument("--eps-list", type=str, default="0.02,0.05,0.10",
                   help="评估的攻击强度列表")
    p.add_argument("--update-every", type=int, default=1, help="每多少环境步做一次更新")
    p.add_argument("--eval-every", type=int, default=3000, help="训练期间评估间隔")
    p.add_argument("--eval-episodes", type=int, default=3, help="评估回合数")
    p.add_argument("--buffer-size", type=int, default=100000, help="回放池容量")
    p.add_argument("--batch-size", type=int, default=256, help="批大小")
    p.add_argument("--hidden", type=int, default=256, help="隐藏层宽度")
    p.add_argument("--lr", type=float, default=3e-4, help="学习率")
    p.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
    p.add_argument("--tau", type=float, default=0.005, help="目标网络软更新系数")
    p.add_argument("--alpha-init", type=float, default=0.2, help="温度系数初值")
    p.add_argument("--seed", type=int, default=0, help="随机种子")
    p.add_argument("--save-dir", type=str, default="runs/ch059", help="输出目录")
    p.add_argument("--quick", action="store_true", help="快速模式：缩短训练")
    return p.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.max_steps = min(args.max_steps, 3000)
        args.eval_every = 1500
        args.start_steps = min(args.start_steps, 300)
        args.eval_episodes = 2
        args.attack_steps = 6

    eps_list = [float(x) for x in args.eps_list.split(",") if x.strip()]
    print(f"种子: {args.seed} | 训练步数 {args.max_steps} | 攻击强度 {eps_list} | "
          f"评估 PGD 步数 {args.attack_steps}")

    clean = train_agent(False, args, "clean")
    adv = train_agent(True, args, "adv")

    # 评估矩阵：受害者 × 攻击强度 ×（可选平滑）
    rows = []
    for name, agent in (("clean", clean), ("adv", adv)):
        base_ret, _ = evaluate_attack(agent, 0.0, False, args.eval_episodes,
                                      args.attack_steps, args.seed + 300)
        rows.append((f"{name} 基线", base_ret, base_ret, 0.0, 0.0))
        for eps in eps_list:
            ret, dev = evaluate_attack(agent, eps, False, args.eval_episodes,
                                       args.attack_steps, args.seed + 300)
            rows.append((f"{name} 攻击 {eps:.2f}", ret, base_ret, dev, 0.0))
        # 动作平滑防御（对中等强度攻击）
        mid_eps = eps_list[len(eps_list) // 2]
        ret_s, dev_s = evaluate_attack(agent, mid_eps, True, args.eval_episodes,
                                       args.attack_steps, args.seed + 300)
        rows.append((f"{name} 攻击 {mid_eps:.2f}+平滑", ret_s, base_ret, dev_s, 0.0))

    print("\n" + "=" * 92)
    print("对抗攻击评估矩阵（回报越高越好；动作偏离 = 攻击带来的 |Δa| 均值）")
    print(f"{'评估条件':<24}{'回报':>10}{'保持率':>9}{'动作偏离':>10}{'相对 clean 基线':>16}")
    clean_base = rows[0][2]
    for name, ret, base, dev, _ in rows:
        keep = ret / max(base, 1e-9)
        delta = ret - clean_base
        print(f"{name:<24}{ret:>10.1f}{keep:>9.2f}{dev:>10.3f}{delta:>16.1f}")
    print("=" * 92)
    print("结论提示：clean 的回报随 eps 增大快速下滑，动作偏离随 eps 上升；adv")
    print("在同样攻击下的保持率更高，但干净基线可能略低（用一点干净性能换鲁棒）。")
    print("动作平滑是零成本缓解手段：压低高频抖动，对中等强度攻击有效。")

    os.makedirs(args.save_dir, exist_ok=True)
    print(f"输出目录：{args.save_dir}")


if __name__ == "__main__":
    main()
