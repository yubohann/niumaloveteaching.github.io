"""
第066章 PPO与SAC的探索策略对比

同环境、同预算下训练 PPO 与 SAC，并在训练过程中持续记录三类探索指标：
  1) 状态覆盖率：把 (cos θ, sin θ) 离散成 6×6 网格，统计训练轨迹访问过的格子
  2) 策略噪声：PPO 的 log_std（状态无关）对比 SAC 在固定探测状态集上的平均 σ
  3) 动作幅度与 SAC 的温度 α

运行：
    python code/ch066.py           # 完整对比（CPU 约 5-12 分钟）
    python code/ch066.py --quick   # 快速跑通（约 1-2 分钟）

预期（以实跑为准）：SAC 依靠自动温度把平均熵维持在目标附近，前 1 万步的状态覆盖率
通常高于 PPO；PPO 的 σ 会随回报梯度快速收缩，后期探索明显减少。
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

ACT_LIMIT = 2.0
BINS = 6  # 状态离散化格数（6×6 = 36 个格子）


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def obs_tensor(obs, device):
    return torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)


@torch.no_grad()
def evaluate(env, mean_action_fn, episodes, device) -> float:
    returns = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            action = mean_action_fn(obs_tensor(obs, device)).squeeze(0).cpu().numpy()
            action = np.clip(action, -ACT_LIMIT, ACT_LIMIT)
            obs, r, terminated, truncated, _ = env.step(action)
            ep_ret += r
            done = terminated or truncated
        returns.append(ep_ret)
    return float(np.mean(returns))


class CoverageTracker:
    """统计 (cos θ, sin θ) 平面上被访问过的网格，衡量探索覆盖。"""

    def __init__(self, bins=BINS):
        self.bins = bins
        self.counts = np.zeros((bins, bins), dtype=np.int64)

    def add(self, obs) -> None:
        c = int(np.clip((obs[0] + 1.0) / 2.0 * self.bins, 0, self.bins - 1))
        s = int(np.clip((obs[1] + 1.0) / 2.0 * self.bins, 0, self.bins - 1))
        self.counts[c, s] += 1

    @property
    def covered(self) -> int:
        return int((self.counts > 0).sum())

    @property
    def total(self) -> int:
        return self.bins * self.bins


def sigma_probe_states(device):
    """固定探测状态集：6×6 网格中心（角速度为 0），用于比较策略噪声。"""
    cs = np.linspace(-0.85, 0.85, BINS)
    ss = np.linspace(-0.85, 0.85, BINS)
    grid_c, grid_s = np.meshgrid(cs, ss)
    states = np.stack([grid_c.ravel(), grid_s.ravel(),
                       np.zeros(BINS * BINS)], axis=1).astype(np.float32)
    return torch.as_tensor(states, device=device)


# ===========================================================================
# PPO
# ===========================================================================
class PPOActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=64):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(),
                                  nn.Linear(hidden, hidden), nn.Tanh())
        self.actor_mean = nn.Linear(hidden, act_dim)
        self.actor_log_std = nn.Parameter(torch.full((act_dim,), -0.5))
        self.critic = nn.Linear(hidden, 1)

    def forward(self, obs):
        h = self.body(obs)
        mean = self.actor_mean(h)
        log_std = torch.clamp(self.actor_log_std, -4.0, 0.5).expand_as(mean)
        return mean, log_std, self.critic(h).squeeze(-1)

    @torch.no_grad()
    def act(self, obs):
        mean, log_std, value = self.forward(obs)
        dist = Normal(mean, log_std.exp())
        action = torch.clamp(dist.sample(), -ACT_LIMIT, ACT_LIMIT)
        return (action.squeeze(0).cpu().numpy(),
                float(dist.log_prob(action).sum(-1).item()), float(value.item()))

    @torch.no_grad()
    def value(self, obs) -> float:
        _, _, v = self.forward(obs)
        return float(v.item())

    @torch.no_grad()
    def mean_action(self, obs):
        mean, _, _ = self.forward(obs)
        return torch.clamp(mean, -ACT_LIMIT, ACT_LIMIT)

    @torch.no_grad()
    def policy_sigma(self) -> float:
        """PPO 的状态无关策略噪声。"""
        return float(torch.clamp(self.actor_log_std, -4.0, 0.5).exp().mean().item())


def compute_gae(rewards, values, cuts, last_value, gamma, lam):
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(T)):
        next_value = last_value if t == T - 1 else values[t + 1]
        non_terminal = 1.0 - cuts[t]
        delta = rewards[t] + gamma * next_value * non_terminal - values[t]
        last_gae = delta + gamma * lam * non_terminal * last_gae
        adv[t] = last_gae
    return adv, adv + np.asarray(values, dtype=np.float32)


def train_ppo(args, device):
    set_seed(args.seed)
    env = gym.make("Pendulum-v1")
    eval_env = gym.make("Pendulum-v1")
    obs, _ = env.reset(seed=args.seed)
    net = PPOActorCritic(env.observation_space.shape[0], env.action_space.shape[0],
                         args.hidden).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    buffers = ([], [], [], [], [], [])
    tracker = CoverageTracker()
    history = []              # (步数, 评估回报, 覆盖率, 策略σ, α)
    total_steps = 0
    next_eval = args.eval_every
    last_covered = 0
    act_abs_sum, act_count = 0.0, 0
    t0 = time.time()

    while total_steps < args.max_steps:
        obs_buf, act_buf, logp_buf, rew_buf, val_buf, cut_buf = buffers
        for _ in range(args.steps_per_update):
            tracker.add(obs)                      # 记录当前状态覆盖
            action, logp, value = net.act(obs_tensor(obs, device))
            next_obs, reward, terminated, truncated, _ = env.step(action)
            act_abs_sum += float(np.abs(action).sum())
            act_count += 1
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
        adv, ret = compute_gae(rew_buf, val_buf, cut_buf, last_value, args.gamma, args.lam)
        batch = {
            "obs": torch.as_tensor(np.asarray(obs_buf), dtype=torch.float32, device=device),
            "act": torch.as_tensor(np.asarray(act_buf), dtype=torch.float32, device=device),
            "logp_old": torch.as_tensor(np.asarray(logp_buf), dtype=torch.float32, device=device),
            "adv": torch.as_tensor(adv, device=device),
            "ret": torch.as_tensor(ret, device=device),
        }
        for buf in buffers:
            buf.clear()

        adv_t = batch["adv"]
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        N = batch["obs"].shape[0]
        for _ in range(args.epochs):
            idx = torch.randperm(N, device=device)
            for start in range(0, N, args.batch_size):
                mb = idx[start:start + args.batch_size]
                mean, log_std, value = net(batch["obs"][mb])
                dist = Normal(mean, log_std.exp())
                logp = dist.log_prob(batch["act"][mb]).sum(-1)
                ratio = torch.exp(logp - batch["logp_old"][mb])
                surr1 = ratio * adv_t[mb]
                surr2 = torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * adv_t[mb]
                entropy = dist.entropy().sum(-1).mean()
                loss = (-torch.min(surr1, surr2).mean()
                        + args.vf_coef * F.mse_loss(value, batch["ret"][mb])
                        - args.ent_coef * entropy)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                optimizer.step()

        total_steps += args.steps_per_update
        if total_steps >= next_eval:
            ret_eval = evaluate(eval_env, net.mean_action, args.eval_episodes, device)
            sigma = net.policy_sigma()
            covered = tracker.covered
            mean_abs = act_abs_sum / max(act_count, 1)
            history.append((total_steps, ret_eval, covered, sigma, None))
            print(f"[PPO 步 {total_steps:6d}] 评估 {ret_eval:8.1f} | 覆盖 {covered:2d}/{tracker.total} "
                  f"(+{covered - last_covered}) | σ {sigma:.3f} | 平均|a| {mean_abs:.2f} | "
                  f"用时 {time.time() - t0:5.1f}s")
            last_covered = covered
            next_eval += args.eval_every
            act_abs_sum, act_count = 0.0, 0

    return net, history, time.time() - t0


# ===========================================================================
# SAC
# ===========================================================================
class ReplayBuffer:
    def __init__(self, capacity, obs_dim, act_dim):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        self.obs = np.zeros((capacity, obs_dim), np.float32)
        self.act = np.zeros((capacity, act_dim), np.float32)
        self.rew = np.zeros((capacity, 1), np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), np.float32)
        self.done = np.zeros((capacity, 1), np.float32)

    def add(self, obs, act, rew, next_obs, done):
        i = self.ptr
        self.obs[i], self.act[i], self.rew[i] = obs, act, rew
        self.next_obs[i], self.done[i] = next_obs, done
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, device):
        idx = np.random.randint(0, self.size, size=batch_size)
        return {k: torch.as_tensor(v[idx], device=device) for k, v in (
            ("obs", self.obs), ("act", self.act), ("rew", self.rew),
            ("next_obs", self.next_obs), ("done", self.done))}


class SACPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=64):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(),
                                  nn.Linear(hidden, hidden), nn.ReLU())
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
        def make():
            return nn.Sequential(nn.Linear(obs_dim + act_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1))
        self.q1, self.q2 = make(), make()

    def forward(self, obs, act):
        x = torch.cat([obs, act], -1)
        return self.q1(x), self.q2(x)


class SACAgent:
    def __init__(self, obs_dim, act_dim, args, device):
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
        with torch.no_grad():
            next_act, next_logp = self.policy.sample(batch["next_obs"])
            q1t, q2t = self.q_target(batch["next_obs"], next_act)
            target = batch["rew"] + self.gamma * (1.0 - batch["done"]) * (
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

    @torch.no_grad()
    def policy_sigma(self, probe_states) -> float:
        """在固定探测状态集上求状态相关 log_std 的 exp 均值。"""
        _, log_std = self.policy(probe_states)
        return float(log_std.exp().mean().item())


def train_sac(args, device):
    set_seed(args.seed)
    env = gym.make("Pendulum-v1")
    eval_env = gym.make("Pendulum-v1")
    obs, _ = env.reset(seed=args.seed)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    agent = SACAgent(obs_dim, act_dim, args, device)
    buffer = ReplayBuffer(args.buffer_size, obs_dim, act_dim)
    tracker = CoverageTracker()
    probe = sigma_probe_states(device)
    history = []
    last_covered = 0
    act_abs_sum, act_count = 0.0, 0
    t0 = time.time()

    for step in range(1, args.max_steps + 1):
        if step <= args.start_steps:
            action = np.random.uniform(-ACT_LIMIT, ACT_LIMIT, size=(act_dim,)).astype(np.float32)
        else:
            with torch.no_grad():
                action = agent.policy.sample(obs_tensor(obs, device))[0].squeeze(0).cpu().numpy()
        tracker.add(obs)                          # 记录当前状态覆盖
        next_obs, reward, terminated, truncated, _ = env.step(action)
        act_abs_sum += float(np.abs(action).sum())
        act_count += 1
        done = terminated or truncated
        buffer.add(obs, action, (reward,), next_obs, (float(done),))
        obs = next_obs
        if done:
            obs, _ = env.reset()
        if step > args.start_steps:
            agent.update(buffer.sample(args.sac_batch_size, device))

        if step % args.eval_every == 0:
            ret_eval = evaluate(eval_env, agent.policy.mean_action,
                                args.eval_episodes, device)
            sigma = agent.policy_sigma(probe)
            alpha = float(agent.alpha.item()) if step > args.start_steps else None
            covered = tracker.covered
            mean_abs = act_abs_sum / max(act_count, 1)
            history.append((step, ret_eval, covered, sigma, alpha))
            alpha_str = f"{alpha:.3f}" if alpha is not None else "预热"
            print(f"[SAC 步 {step:6d}] 评估 {ret_eval:8.1f} | 覆盖 {covered:2d}/{tracker.total} "
                  f"(+{covered - last_covered}) | σ {sigma:.3f} | α {alpha_str} | "
                  f"平均|a| {mean_abs:.2f} | 用时 {time.time() - t0:5.1f}s")
            last_covered = covered
            act_abs_sum, act_count = 0.0, 0

    return agent, history, time.time() - t0


# ===========================================================================
# 汇总与绘图
# ===========================================================================
def summarize(name, history, elapsed, window_steps):
    """打印覆盖率与噪声的阶段性对比。"""
    if not history:
        print(f"[{name}] 没有记录")
        return
    early = [h for h in history if h[0] <= window_steps] or history[:1]
    last = history[-1]
    sig_mid = history[len(history) // 2][3]
    print(f"[{name:3s}] 用时 {elapsed:6.1f}s | 前 {window_steps} 步覆盖 "
          f"{early[-1][2]:2d}/{BINS * BINS} | 全程覆盖 {last[2]:2d}/{BINS * BINS} | "
          f"σ 中段 {sig_mid:.3f} → 末段 {last[3]:.3f}")


def save_plot(save_dir, ppo_hist, sac_hist):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("未安装 matplotlib，跳过绘图（不影响实验结论）")
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for hist, label in ((ppo_hist, "PPO"), (sac_hist, "SAC")):
        if hist:
            xs = [h[0] for h in hist]
            axes[0].plot(xs, [h[2] / (BINS * BINS) for h in hist], marker="o", label=label)
            axes[1].plot(xs, [h[3] for h in hist], marker="s", label=label)
    axes[0].set_xlabel("env steps")
    axes[0].set_ylabel("state coverage")
    axes[0].set_title("coverage of (cos, sin) grid")
    axes[1].set_xlabel("env steps")
    axes[1].set_ylabel("policy sigma")
    axes[1].set_title("policy noise")
    for ax in axes:
        ax.grid(alpha=0.3)
    axes[0].legend()
    axes[1].legend()
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "exploration_compare.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"探索对比图已保存到 {path}")


def parse_args():
    p = argparse.ArgumentParser(description="第066章：PPO 与 SAC 的探索策略对比")
    p.add_argument("--algo", type=str, default="both", choices=["both", "ppo", "sac"])
    p.add_argument("--max-steps", type=int, default=30000, help="每种算法的环境步预算")
    p.add_argument("--eval-every", type=int, default=3000, help="每多少步记录一次指标")
    p.add_argument("--eval-episodes", type=int, default=10, help="评估回合数")
    p.add_argument("--coverage-window", type=int, default=9000, help="早期覆盖率比较窗口（步数）")
    p.add_argument("--hidden", type=int, default=64, help="隐藏层宽度")
    p.add_argument("--lr", type=float, default=3e-4, help="学习率")
    p.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
    p.add_argument("--steps-per-update", type=int, default=2048, help="PPO 每次采样步数")
    p.add_argument("--epochs", type=int, default=10, help="PPO 更新轮数")
    p.add_argument("--batch-size", type=int, default=64, help="PPO 小批量")
    p.add_argument("--lam", type=float, default=0.95, help="PPO 的 GAE lambda")
    p.add_argument("--clip", type=float, default=0.2, help="PPO 裁剪范围")
    p.add_argument("--ent-coef", type=float, default=0.0, help="PPO 熵正则系数")
    p.add_argument("--vf-coef", type=float, default=0.5, help="PPO 价值损失系数")
    p.add_argument("--sac-batch-size", type=int, default=256, help="SAC 批量")
    p.add_argument("--buffer-size", type=int, default=200000, help="SAC 回放池")
    p.add_argument("--start-steps", type=int, default=2000, help="SAC 预热步数")
    p.add_argument("--tau", type=float, default=0.005, help="SAC 软更新系数")
    p.add_argument("--alpha", type=float, default=0.2, help="SAC 温度初值")
    p.add_argument("--seed", type=int, default=0, help="随机种子")
    p.add_argument("--save-dir", type=str, default="runs/ch066", help="图片保存目录")
    p.add_argument("--quick", action="store_true", help="快速模式：8000 步 / 算法")
    return p.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.max_steps = min(args.max_steps, 8000)
        args.eval_every = 2000
        args.eval_episodes = 3
        args.steps_per_update = 1024
        args.epochs = 6
        args.start_steps = 500
        args.coverage_window = min(args.coverage_window, 4000)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device} | 环境: Pendulum-v1 | 种子 {args.seed}")
    print(f"配置: 每算法 {args.max_steps} 步 | 覆盖网格 {BINS}×{BINS} | "
          f"PPO ent_coef={args.ent_coef} | SAC 自动温度 α 初值 {args.alpha}")

    ppo_hist = sac_hist = None
    if args.algo in ("ppo", "both"):
        net, ppo_hist, ppo_time = train_ppo(args, device)
    if args.algo in ("sac", "both"):
        agent, sac_hist, sac_time = train_sac(args, device)

    print("=" * 76)
    if args.algo in ("ppo", "both"):
        summarize("PPO", ppo_hist, ppo_time, args.coverage_window)
    if args.algo in ("sac", "both"):
        summarize("SAC", sac_hist, sac_time, args.coverage_window)
    if args.algo == "both":
        print("提示：覆盖率与策略噪声要结合回报一起解读——探索多不等于学得好，"
              "但探索太少通常学不到最优策略。")
    save_plot(args.save_dir, ppo_hist, sac_hist)


if __name__ == "__main__":
    main()
