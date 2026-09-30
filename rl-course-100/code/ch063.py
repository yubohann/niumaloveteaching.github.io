"""
第063章 PPO与SAC性能对比

在同一环境（Pendulum-v1）、同一环境交互预算、同一评估协议下对照 PPO 与 SAC：
  - PPO ：on-policy，每次采样 2048 步后做多轮裁剪更新
  - SAC ：off-policy，每个环境步做一次梯度更新，使用回放池
两条曲线横轴都是环境步数，衡量样本效率；墙钟时间只作辅助参考。

运行：
    python code/ch063.py           # 完整对比（CPU 约 6-15 分钟）
    python code/ch063.py --quick   # 快速跑通（约 1-2 分钟）
    python code/ch063.py --algo ppo
    python code/ch063.py --algo sac

预期（以实跑为准）：SAC 通常在 2 万～4 万步内越过 -400；PPO 在 6 万步预算内
一般只到 -650 ~ -250，样本效率落后约 2～4 倍，但单步计算更省（延长预算也能收敛）。
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
from torch.distributions import Normal

ACT_LIMIT = 2.0  # Pendulum-v1 的力矩上界


def set_seed(seed: int) -> None:
    """固定随机种子，保证两种算法面对同一初始条件。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ===========================================================================
# 公共工具
# ===========================================================================
def obs_tensor(obs, device):
    return torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)


@torch.no_grad()
def evaluate(env, mean_action_fn, episodes: int, device) -> float:
    """用确定性动作评估，两种算法共用同一协议。"""
    returns = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            action = mean_action_fn(obs_tensor(obs, device))
            action = np.clip(action.squeeze(0).cpu().numpy(), -ACT_LIMIT, ACT_LIMIT)
            obs, r, terminated, truncated, _ = env.step(action)
            ep_ret += r
            done = terminated or truncated
        returns.append(ep_ret)
    return float(np.mean(returns))


# ===========================================================================
# 第一部分：PPO（连续动作，对角高斯策略）
# ===========================================================================
class PPOActorCritic(nn.Module):
    """PPO 的 Actor-Critic：共享躯干 + 高斯策略头 + 价值头。"""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 64):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.actor_mean = nn.Linear(hidden, act_dim)
        # 状态无关的对数标准差，是 PPO 探索强度的直接旋钮
        self.actor_log_std = nn.Parameter(torch.full((act_dim,), -0.5))
        self.critic = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor):
        h = self.body(obs)
        mean = self.actor_mean(h)
        log_std = torch.clamp(self.actor_log_std, -4.0, 0.5).expand_as(mean)
        value = self.critic(h).squeeze(-1)
        return mean, log_std, value

    def dist(self, obs: torch.Tensor) -> Normal:
        mean, log_std, _ = self.forward(obs)
        return Normal(mean, log_std.exp())

    @torch.no_grad()
    def act(self, obs: torch.Tensor):
        """采样动作并做边界裁剪；存裁剪后动作的对数概率，保证训练时一致。"""
        mean, log_std, value = self.forward(obs)
        dist = Normal(mean, log_std.exp())
        raw = dist.sample()
        action = torch.clamp(raw, -ACT_LIMIT, ACT_LIMIT)
        logp = dist.log_prob(action).sum(-1)
        return action.squeeze(0).cpu().numpy(), float(logp.item()), float(value.item())

    @torch.no_grad()
    def value(self, obs: torch.Tensor) -> float:
        _, _, v = self.forward(obs)
        return float(v.item())

    @torch.no_grad()
    def mean_action(self, obs: torch.Tensor) -> torch.Tensor:
        mean, _, _ = self.forward(obs)
        return torch.clamp(mean, -ACT_LIMIT, ACT_LIMIT)


def compute_gae(rewards, values, cuts, last_value, gamma: float, lam: float):
    """广义优势估计；cuts[t]=1 表示该步是回合端点（终止或时间截断）。"""
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(T)):
        next_value = last_value if t == T - 1 else values[t + 1]
        non_terminal = 1.0 - cuts[t]
        delta = rewards[t] + gamma * next_value * non_terminal - values[t]
        last_gae = delta + gamma * lam * non_terminal * last_gae
        adv[t] = last_gae
    returns = adv + np.asarray(values, dtype=np.float32)
    return adv, returns


def collect_rollout(env, net, steps, obs, device, gamma, lam):
    """采样固定步数的 rollout，返回 batch 与最新观测。"""
    obs_buf, act_buf, logp_buf, rew_buf, val_buf, cut_buf = [], [], [], [], [], []
    for _ in range(steps):
        action, logp, value = net.act(obs_tensor(obs, device))
        next_obs, reward, terminated, truncated, _ = env.step(action)
        obs_buf.append(obs)
        act_buf.append(action)
        logp_buf.append(logp)
        rew_buf.append(reward)
        val_buf.append(value)
        cut_buf.append(float(terminated or truncated))
        obs = next_obs
        if terminated or truncated:
            obs, _ = env.reset()

    last_value = net.value(obs_tensor(obs, device))
    adv, ret = compute_gae(rew_buf, val_buf, cut_buf, last_value, gamma, lam)
    batch = {
        "obs": torch.as_tensor(np.asarray(obs_buf), dtype=torch.float32, device=device),
        "act": torch.as_tensor(np.asarray(act_buf), dtype=torch.float32, device=device),
        "logp_old": torch.as_tensor(np.asarray(logp_buf), dtype=torch.float32, device=device),
        "adv": torch.as_tensor(adv, dtype=torch.float32, device=device),
        "ret": torch.as_tensor(ret, dtype=torch.float32, device=device),
    }
    return batch, obs


def ppo_update(net, optimizer, batch, args) -> dict:
    """PPO 裁剪目标的小批量更新。"""
    obs, act = batch["obs"], batch["act"]
    logp_old, adv, ret = batch["logp_old"], batch["adv"], batch["ret"]
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    stats = {"policy_loss": 0.0, "value_loss": 0.0, "approx_kl": 0.0,
             "clip_frac": 0.0, "entropy": 0.0, "n": 0}
    N = obs.shape[0]
    for _ in range(args.epochs):
        idx = torch.randperm(N, device=obs.device)
        for start in range(0, N, args.batch_size):
            mb = idx[start:start + args.batch_size]
            mean, log_std, value = net(obs[mb])
            dist = Normal(mean, log_std.exp())
            logp = dist.log_prob(act[mb]).sum(-1)
            ratio = torch.exp(logp - logp_old[mb])
            surr1 = ratio * adv[mb]
            surr2 = torch.clamp(ratio, 1.0 - args.clip, 1.0 + args.clip) * adv[mb]
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(value, ret[mb])
            entropy = dist.entropy().sum(-1).mean()
            loss = policy_loss + args.vf_coef * value_loss - args.ent_coef * entropy
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            optimizer.step()
            with torch.no_grad():
                stats["policy_loss"] += policy_loss.item()
                stats["value_loss"] += value_loss.item()
                stats["entropy"] += entropy.item()
                stats["approx_kl"] += (logp_old[mb] - logp).mean().item()
                stats["clip_frac"] += ((ratio - 1.0).abs() > args.clip).float().mean().item()
                stats["n"] += 1
    for k in stats:
        if k != "n":
            stats[k] /= max(stats["n"], 1)
    return stats


def train_ppo(args, device):
    set_seed(args.seed)
    env = gym.make("Pendulum-v1")
    eval_env = gym.make("Pendulum-v1")
    obs, _ = env.reset(seed=args.seed)
    net = PPOActorCritic(env.observation_space.shape[0], env.action_space.shape[0],
                         args.hidden).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    curve = []
    total_steps = 0
    next_eval = args.eval_every
    t0 = time.time()
    update_idx = 0

    while total_steps < args.max_steps:
        batch, obs = collect_rollout(env, net, args.steps_per_update, obs, device,
                                     args.gamma, args.lam)
        stats = ppo_update(net, optimizer, batch, args)
        total_steps += args.steps_per_update
        update_idx += 1

        if total_steps >= next_eval:
            ret = evaluate(eval_env, net.mean_action, args.eval_episodes, device)
            curve.append((total_steps, ret))
            next_eval += args.eval_every
            print(f"[PPO 更新 {update_idx:3d}] 环境步 {total_steps:6d} | 评估回报 {ret:8.1f} | "
                  f"KL {stats['approx_kl']:.4f} | 裁剪 {stats['clip_frac']:.3f} | "
                  f"熵 {stats['entropy']:.2f} | 用时 {time.time() - t0:5.1f}s")

    return net, curve, time.time() - t0


# ===========================================================================
# 第二部分：SAC（连续动作，双 Q + 自动温度）
# ===========================================================================
class ReplayBuffer:
    """固定容量环形回放池。"""

    def __init__(self, capacity, obs_dim, act_dim):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rew = np.zeros((capacity, 1), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.done = np.zeros((capacity, 1), dtype=np.float32)

    def add(self, obs, act, rew, next_obs, done):
        i = self.ptr
        self.obs[i], self.act[i], self.rew[i] = obs, act, rew
        self.next_obs[i], self.done[i] = next_obs, done
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, device):
        idx = np.random.randint(0, self.size, size=batch_size)
        return {
            "obs": torch.as_tensor(self.obs[idx], device=device),
            "act": torch.as_tensor(self.act[idx], device=device),
            "rew": torch.as_tensor(self.rew[idx], device=device),
            "next_obs": torch.as_tensor(self.next_obs[idx], device=device),
            "done": torch.as_tensor(self.done[idx], device=device),
        }


class SACPolicy(nn.Module):
    """SAC 的高斯策略（tanh 压缩）。"""

    def __init__(self, obs_dim, act_dim, hidden=64):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mean = nn.Linear(hidden, act_dim)
        self.log_std = nn.Linear(hidden, act_dim)

    def forward(self, obs):
        h = self.body(obs)
        return self.mean(h), torch.clamp(self.log_std(h), -4.0, 0.5)

    def sample(self, obs):
        mean, log_std = self.forward(obs)
        dist = Normal(mean, log_std.exp())
        x = dist.rsample()
        action = torch.tanh(x)
        logp = dist.log_prob(x) - torch.log(1.0 - action.pow(2) + 1e-6)
        return action * ACT_LIMIT, logp.sum(-1, keepdim=True)

    @torch.no_grad()
    def mean_action(self, obs):
        mean, _ = self.forward(obs)
        return torch.tanh(mean) * ACT_LIMIT


class TwinQ(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=64):
        super().__init__()
        self.q1 = self._mlp(obs_dim + act_dim, hidden)
        self.q2 = self._mlp(obs_dim + act_dim, hidden)

    @staticmethod
    def _mlp(in_dim, hidden):
        return nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(),
                             nn.Linear(hidden, hidden), nn.ReLU(),
                             nn.Linear(hidden, 1))

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.q1(x), self.q2(x)


class SACAgent:
    def __init__(self, obs_dim, act_dim, args, device):
        self.device = device
        self.policy = SACPolicy(obs_dim, act_dim, args.hidden).to(device)
        self.q = TwinQ(obs_dim, act_dim, args.hidden).to(device)
        self.q_target = TwinQ(obs_dim, act_dim, args.hidden).to(device)
        self.q_target.load_state_dict(self.q.state_dict())
        self.policy_opt = torch.optim.Adam(self.policy.parameters(), lr=args.lr)
        self.q_opt = torch.optim.Adam(self.q.parameters(), lr=args.lr)
        self.log_alpha = torch.tensor(np.log(args.alpha), requires_grad=True, device=device)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=args.lr)
        self.target_entropy = -float(act_dim)
        self.gamma, self.tau = args.gamma, args.tau

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def update(self, batch):
        obs, act = batch["obs"], batch["act"]
        rew, next_obs, done = batch["rew"], batch["next_obs"], batch["done"]
        with torch.no_grad():
            next_act, next_logp = self.policy.sample(next_obs)
            q1t, q2t = self.q_target(next_obs, next_act)
            target = rew + self.gamma * (1.0 - done) * (
                torch.min(q1t, q2t) - self.alpha * next_logp)
        q1, q2 = self.q(obs, act)
        q_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.q_opt.zero_grad()
        q_loss.backward()
        self.q_opt.step()

        new_act, logp = self.policy.sample(obs)
        q1n, q2n = self.q(obs, new_act)
        policy_loss = (self.alpha.detach() * logp - torch.min(q1n, q2n)).mean()
        self.policy_opt.zero_grad()
        policy_loss.backward()
        self.policy_opt.step()

        alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        with torch.no_grad():
            for p, p_t in zip(self.q.parameters(), self.q_target.parameters()):
                p_t.data.mul_(1.0 - self.tau).add_(self.tau * p.data)
        return {"q_loss": q_loss.item(), "policy_loss": policy_loss.item(),
                "alpha": self.alpha.item()}


def train_sac(args, device):
    set_seed(args.seed)
    env = gym.make("Pendulum-v1")
    eval_env = gym.make("Pendulum-v1")
    obs, _ = env.reset(seed=args.seed)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    agent = SACAgent(obs_dim, act_dim, args, device)
    buffer = ReplayBuffer(args.buffer_size, obs_dim, act_dim)
    curve = []
    t0 = time.time()

    for step in range(1, args.max_steps + 1):
        if step <= args.start_steps:
            action = np.random.uniform(-ACT_LIMIT, ACT_LIMIT, size=(act_dim,)).astype(np.float32)
        else:
            with torch.no_grad():
                action = agent.policy.sample(obs_tensor(obs, device))[0]
                action = action.squeeze(0).cpu().numpy()
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        buffer.add(obs, action, (reward,), next_obs, (float(done),))
        obs = next_obs
        if done:
            obs, _ = env.reset()

        if step > args.start_steps:
            stats = agent.update(buffer.sample(args.sac_batch_size, device))

        if step % args.eval_every == 0:
            ret = evaluate(eval_env, agent.policy.mean_action, args.eval_episodes, device)
            curve.append((step, ret))
            extra = (f"| α {stats['alpha']:.3f}" if step > args.start_steps else "| 预热阶段")
            print(f"[SAC 步数 {step:6d}] 评估回报 {ret:8.1f} {extra} | "
                  f"用时 {time.time() - t0:5.1f}s")

    return agent, curve, time.time() - t0


# ===========================================================================
# 汇总与绘图
# ===========================================================================
def cur_steps_to(curve, threshold):
    """返回首次达到阈值时消耗的环境步数，未达到返回 None。"""
    for steps, ret in curve:
        if ret >= threshold:
            return steps
    return None


def save_plot(save_dir, ppo_curve, sac_curve):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("未安装 matplotlib，跳过绘图（不影响实验结论）")
        return
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for curve, label, marker in ((ppo_curve, "PPO", "o"), (sac_curve, "SAC", "s")):
        if curve:
            ax.plot([c[0] for c in curve], [c[1] for c in curve], marker=marker, label=label)
    ax.axhline(-400, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("environment steps")
    ax.set_ylabel("eval return")
    ax.set_title("PPO vs SAC on Pendulum-v1 (same env-step budget, seed=%d)")
    ax.legend()
    ax.grid(alpha=0.3)
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "ppo_vs_sac_curve.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"对比曲线已保存到 {path}")


def summarize(name, curve, elapsed, threshold):
    if not curve:
        print(f"[{name}] 没有可用的评估记录")
        return
    last3 = float(np.mean([c[1] for c in curve[-3:]]))
    best = max(c[1] for c in curve)
    hit = cur_steps_to(curve, threshold)
    hit_str = f"{hit} 步" if hit is not None else "未达到"
    print(f"[{name:3s}] 用时 {elapsed:6.1f}s | 最后3次评估均值 {last3:8.1f} | "
          f"最佳 {best:8.1f} | 首次达到 {threshold:.0f} 分：{hit_str}")


def parse_args():
    p = argparse.ArgumentParser(description="第063章：PPO 与 SAC 性能对比")
    p.add_argument("--algo", type=str, default="both", choices=["both", "ppo", "sac"],
                   help="运行哪些算法")
    p.add_argument("--max-steps", type=int, default=60000, help="每种算法的环境步数预算")
    p.add_argument("--eval-every", type=int, default=5000, help="每多少环境步评估一次")
    p.add_argument("--eval-episodes", type=int, default=10, help="每次评估的回合数")
    p.add_argument("--hidden", type=int, default=64, help="两种算法的隐藏层宽度（对齐）")
    p.add_argument("--lr", type=float, default=3e-4, help="学习率（两种算法对齐）")
    p.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
    p.add_argument("--threshold", type=float, default=-400.0, help="样本效率的参考阈值")
    p.add_argument("--steps-per-update", type=int, default=2048, help="PPO 每次更新前的采样步数")
    p.add_argument("--epochs", type=int, default=10, help="PPO 每批数据的更新轮数")
    p.add_argument("--batch-size", type=int, default=64, help="PPO 小批量大小")
    p.add_argument("--lam", type=float, default=0.95, help="PPO 的 GAE lambda")
    p.add_argument("--clip", type=float, default=0.2, help="PPO 裁剪范围")
    p.add_argument("--ent-coef", type=float, default=0.0, help="PPO 熵正则系数")
    p.add_argument("--vf-coef", type=float, default=0.5, help="PPO 价值损失系数")
    p.add_argument("--sac-batch-size", type=int, default=256, help="SAC 批量大小")
    p.add_argument("--buffer-size", type=int, default=200000, help="SAC 回放池容量")
    p.add_argument("--start-steps", type=int, default=2000, help="SAC 纯随机预热步数")
    p.add_argument("--tau", type=float, default=0.005, help="SAC 目标网络软更新系数")
    p.add_argument("--alpha", type=float, default=0.2, help="SAC 温度系数初值")
    p.add_argument("--seed", type=int, default=0, help="随机种子")
    p.add_argument("--save-dir", type=str, default="runs/ch063", help="模型与曲线保存目录")
    p.add_argument("--quick", action="store_true", help="快速模式：缩小预算，约 1-2 分钟")
    return p.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.max_steps = min(args.max_steps, 15000)
        args.eval_every = 2500
        args.eval_episodes = 3
        args.steps_per_update = 1024
        args.epochs = 6
        args.start_steps = 500
        args.max_steps = max(args.max_steps, 2000)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device} | 环境: Pendulum-v1 | 种子: {args.seed}")
    print(f"预算: 每种算法 {args.max_steps} 环境步 | 每 {args.eval_every} 步评估 "
          f"{args.eval_episodes} 回合 | hidden={args.hidden} | lr={args.lr}")

    ppo_curve, sac_curve = None, None
    if args.algo in ("ppo", "both"):
        net, ppo_curve, dt = train_ppo(args, device)
        ppo_time = dt
    if args.algo in ("sac", "both"):
        agent, sac_curve, dt = train_sac(args, device)
        sac_time = dt

    print("=" * 76)
    if args.algo in ("ppo", "both"):
        summarize("PPO", ppo_curve, ppo_time, args.threshold)
    if args.algo in ("sac", "both"):
        summarize("SAC", sac_curve, sac_time, args.threshold)
    if args.algo == "both":
        print("提示：横轴对齐的是环境步数（样本效率）；CPU 单步计算量 PPO 更高，"
              "墙钟时间只作辅助参考。")

    os.makedirs(args.save_dir, exist_ok=True)
    if args.algo in ("ppo", "both"):
        torch.save(net.state_dict(), os.path.join(args.save_dir, "ppo_actor_critic.pt"))
    if args.algo in ("sac", "both"):
        torch.save(agent.policy.state_dict(), os.path.join(args.save_dir, "sac_policy.pt"))
    print(f"模型已保存到 {args.save_dir}")
    save_plot(args.save_dir, ppo_curve, sac_curve)


if __name__ == "__main__":
    main()
