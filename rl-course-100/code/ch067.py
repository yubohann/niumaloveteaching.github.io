"""
第067章 PPO与SAC的集成学习

先分别训练一个 PPO 与一个 SAC（同环境 Pendulum-v1、同预算、同种子），再把两种
策略按不同权重融合，系统评估四种组合方式：
  1) 纯 PPO / 纯 SAC（权重 0 与 1，作为基准）
  2) 动作加权融合：a = w·a_ppo + (1−w)·a_sac
  3) 动作仲裁：用 SAC 的批评家在 {a_ppo, a_sac, a_融合} 中挑 Q 最高的动作
  4) 分歧统计：逐回合平均 |a_ppo − a_sac|，与回合回报做相关分析

所有策略在完全相同的评估回合（固定环境种子）上比较。

运行：
    python code/ch067.py           # 完整实验（CPU 约 5-12 分钟）
    python code/ch067.py --quick   # 快速跑通（约 1-2 分钟）

预期（以实跑为准）：同预算下 SAC 单独通常优于 PPO；融合权重取中间值时往往能贴近
或略超两个单模型（方差降低带来的收益），仲裁策略则偏向 SAC。
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def obs_tensor(obs, device):
    return torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)


def make_eval_seeds(base: int, n: int):
    """固定评估回合的环境种子：所有策略在同一批回合上比较。"""
    return [base + 10000 + i for i in range(n)]


# ===========================================================================
# PPO（与前几章一致）
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


def evaluate_mean_action(env, mean_action_fn, episodes, device):
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
    return returns


def train_ppo(args, device):
    set_seed(args.seed)
    env = gym.make("Pendulum-v1")
    eval_env = gym.make("Pendulum-v1")
    obs, _ = env.reset(seed=args.seed)
    net = PPOActorCritic(env.observation_space.shape[0], env.action_space.shape[0],
                         args.hidden).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    buffers = ([], [], [], [], [], [])
    curve = []
    total_steps = 0
    next_eval = args.eval_every
    t0 = time.time()

    while total_steps < args.max_steps:
        obs_buf, act_buf, logp_buf, rew_buf, val_buf, cut_buf = buffers
        for _ in range(args.steps_per_update):
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
                loss = (-torch.min(surr1, surr2).mean()
                        + args.vf_coef * F.mse_loss(value, batch["ret"][mb]))
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                optimizer.step()

        total_steps += args.steps_per_update
        if total_steps >= next_eval:
            ret = evaluate_mean_action(eval_env, net.mean_action, args.eval_episodes, device)
            curve.append((total_steps, float(np.mean(ret))))
            print(f"[PPO 步 {total_steps:6d}] 评估均值 {np.mean(ret):8.1f} | "
                  f"用时 {time.time() - t0:5.1f}s")
            next_eval += args.eval_every
    return net, curve, time.time() - t0


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
        self.device = device

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
    def q_value(self, obs_t: torch.Tensor, act_t: torch.Tensor) -> float:
        q1, q2 = self.q(obs_t, act_t)
        return float(torch.min(q1, q2).item())


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
                action = agent.policy.sample(obs_tensor(obs, device))[0].squeeze(0).cpu().numpy()
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        buffer.add(obs, action, (reward,), next_obs, (float(done),))
        obs = next_obs
        if done:
            obs, _ = env.reset()
        if step > args.start_steps:
            agent.update(buffer.sample(args.sac_batch_size, device))
        if step % args.eval_every == 0:
            ret = evaluate_mean_action(eval_env, agent.policy.mean_action,
                                       args.eval_episodes, device)
            curve.append((step, float(np.mean(ret))))
            print(f"[SAC 步 {step:6d}] 评估均值 {np.mean(ret):8.1f} | "
                  f"用时 {time.time() - t0:5.1f}s")
    return agent, curve, time.time() - t0


# ===========================================================================
# 集成评估
# ===========================================================================
def run_episode_blend(env, ppo_net, sac_agent, weight, seed, device):
    """一次评估回合：返回（回合回报, 平均动作分歧）。"""
    obs, _ = env.reset(seed=seed)
    done = False
    ep_ret = 0.0
    disagreements = []
    while not done:
        obs_t = obs_tensor(obs, device)
        a_ppo = ppo_net.mean_action(obs_t).squeeze(0).cpu().numpy()
        a_sac = sac_agent.policy.mean_action(obs_t).squeeze(0).cpu().numpy()
        disagreements.append(float(np.mean(np.abs(a_ppo - a_sac))))
        action = np.clip(weight * a_ppo + (1.0 - weight) * a_sac, -ACT_LIMIT, ACT_LIMIT)
        obs, r, terminated, truncated, _ = env.step(action)
        ep_ret += r
        done = terminated or truncated
    return ep_ret, float(np.mean(disagreements))


def run_episode_arbitration(env, ppo_net, sac_agent, seed, device):
    """动作仲裁：在 {a_ppo, a_sac, 0.5 融合} 中选 SAC 批评家 Q 值最高者。"""
    obs, _ = env.reset(seed=seed)
    done = False
    ep_ret = 0.0
    picks = {"ppo": 0, "sac": 0, "mix": 0}
    while not done:
        obs_t = obs_tensor(obs, device)
        a_ppo = ppo_net.mean_action(obs_t).squeeze(0).cpu().numpy()
        a_sac = sac_agent.policy.mean_action(obs_t).squeeze(0).cpu().numpy()
        a_mix = 0.5 * a_ppo + 0.5 * a_sac
        candidates = {"ppo": a_ppo, "sac": a_sac, "mix": a_mix}
        best_name, best_q = None, -np.inf
        for name, a in candidates.items():
            q = sac_agent.q_value(obs_t, torch.as_tensor(a, dtype=torch.float32,
                                                         device=device).unsqueeze(0))
            if q > best_q:
                best_name, best_q = name, q
        picks[best_name] += 1
        action = np.clip(candidates[best_name], -ACT_LIMIT, ACT_LIMIT)
        obs, r, terminated, truncated, _ = env.step(action)
        ep_ret += r
        done = terminated or truncated
    return ep_ret, picks


def summarize_strategy(name, returns, extra: str = ""):
    arr = np.asarray(returns, dtype=np.float64)
    print(f"{name:22s} 平均 {arr.mean():8.1f} | 标准差 {arr.std():7.1f} | "
          f"最小 {arr.min():8.1f} | 最大 {arr.max():8.1f} {extra}")
    return arr.mean(), arr.std()


def save_plot(save_dir, names, means, stds):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("未安装 matplotlib，跳过绘图（不影响实验结论）")
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    xs = np.arange(len(names))
    ax.bar(xs, means, yerr=stds, capsize=5, color="tab:blue", alpha=0.75)
    ax.set_xticks(xs)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("episode return (mean ± std)")
    ax.set_title("PPO + SAC ensemble strategies on Pendulum-v1")
    ax.grid(axis="y", alpha=0.3)
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "ensemble_strategies.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"策略对比图已保存到 {path}")


def parse_args():
    p = argparse.ArgumentParser(description="第067章：PPO 与 SAC 的集成学习")
    p.add_argument("--max-steps", type=int, default=25000, help="每种算法的训练步数")
    p.add_argument("--eval-every", type=int, default=5000, help="训练期评估间隔")
    p.add_argument("--eval-episodes", type=int, default=10, help="训练期评估回合数")
    p.add_argument("--final-episodes", type=int, default=30, help="最终策略对比的回合数")
    p.add_argument("--weights", type=str, default="0,0.25,0.5,0.75,1",
                   help="融合权重列表：w 为 PPO 权重，1−w 为 SAC 权重")
    p.add_argument("--hidden", type=int, default=64, help="隐藏层宽度")
    p.add_argument("--lr", type=float, default=3e-4, help="学习率")
    p.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
    p.add_argument("--steps-per-update", type=int, default=2048, help="PPO 每次采样步数")
    p.add_argument("--epochs", type=int, default=10, help="PPO 更新轮数")
    p.add_argument("--batch-size", type=int, default=64, help="PPO 小批量")
    p.add_argument("--lam", type=float, default=0.95, help="PPO 的 GAE lambda")
    p.add_argument("--clip", type=float, default=0.2, help="PPO 裁剪范围")
    p.add_argument("--vf-coef", type=float, default=0.5, help="PPO 价值损失系数")
    p.add_argument("--sac-batch-size", type=int, default=256, help="SAC 批量")
    p.add_argument("--buffer-size", type=int, default=200000, help="SAC 回放池")
    p.add_argument("--start-steps", type=int, default=2000, help="SAC 预热步数")
    p.add_argument("--tau", type=float, default=0.005, help="SAC 软更新系数")
    p.add_argument("--alpha", type=float, default=0.2, help="SAC 温度初值")
    p.add_argument("--seed", type=int, default=0, help="训练随机种子")
    p.add_argument("--save-dir", type=str, default="runs/ch067", help="图片保存目录")
    p.add_argument("--quick", action="store_true", help="快速模式：8000 步 / 算法")
    return p.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.max_steps = min(args.max_steps, 8000)
        args.eval_every = 2000
        args.eval_episodes = 3
        args.final_episodes = 10
        args.steps_per_update = 1024
        args.epochs = 6
        args.start_steps = 500
    weights = [float(x) for x in args.weights.split(",")]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device} | 环境: Pendulum-v1 | 种子 {args.seed}")
    print(f"配置: PPO 与 SAC 各训练 {args.max_steps} 步 | 最终评估 "
          f"{args.final_episodes} 回合（固定环境种子）| 融合权重 {weights}")

    # ---- 第一步：分别训练两个基模型 ----
    print("-" * 60)
    ppo_net, ppo_curve, ppo_time = train_ppo(args, device)
    print(f"PPO 训练完成，用时 {ppo_time:.1f}s，最后评估 {ppo_curve[-1][1]:.1f}")
    sac_agent, sac_curve, sac_time = train_sac(args, device)
    print(f"SAC 训练完成，用时 {sac_time:.1f}s，最后评估 {sac_curve[-1][1]:.1f}")

    # ---- 第二步：固定回合上的集成评估 ----
    eval_env = gym.make("Pendulum-v1")
    eval_seeds = make_eval_seeds(args.seed, args.final_episodes)
    print("-" * 60)
    print("固定评估回合上的策略对比：")

    names, means, stds = [], [], []
    strategy_returns = {}
    disagreement_per_ep = None
    for w in weights:
        returns, dis = [], []
        for s in eval_seeds:
            ret, d = run_episode_blend(eval_env, ppo_net, sac_agent, w, s, device)
            returns.append(ret)
            dis.append(d)
        label = f"融合 w={w:.2f}" if 0 < w < 1 else ("纯 PPO" if w == 1 else "纯 SAC")
        m, sd = summarize_strategy(label, returns)
        names.append(label)
        means.append(m)
        stds.append(sd)
        strategy_returns[label] = np.asarray(returns)
        if abs(w - 0.5) < 1e-9:
            disagreement_per_ep = np.asarray(dis)

    # 仲裁策略
    arb_returns, picks_total = [], {"ppo": 0, "sac": 0, "mix": 0}
    for s in eval_seeds:
        ret, picks = run_episode_arbitration(eval_env, ppo_net, sac_agent, s, device)
        arb_returns.append(ret)
        for k in picks:
            picks_total[k] += picks[k]
    total_picks = sum(picks_total.values())
    pick_str = "、".join(f"{k} {v / total_picks * 100:.0f}%" for k, v in picks_total.items())
    m, sd = summarize_strategy("仲裁（SAC 批评家）", arb_returns, extra=f"[占比: {pick_str}]")
    names.append("仲裁")
    means.append(m)
    stds.append(sd)
    strategy_returns["仲裁"] = np.asarray(arb_returns)

    # ---- 第三步：分歧与回报的相关性 ----
    if disagreement_per_ep is not None:
        mid_returns = strategy_returns.get("融合 w=0.50")
        if mid_returns is None:
            mid_returns = np.asarray(arb_returns)
        if np.std(disagreement_per_ep) > 1e-6:
            corr = float(np.corrcoef(disagreement_per_ep, mid_returns)[0, 1])
            print(f"动作分歧 |a_ppo − a_sac| 的回合均值 {disagreement_per_ep.mean():.3f}，"
                  f"与回合回报的相关系数 {corr:+.2f}")
        else:
            print("动作分歧几乎为常数，跳过相关性分析")

    # ---- 汇总 ----
    print("=" * 76)
    single_names = [name for name in ("纯 PPO", "纯 SAC") if name in strategy_returns]
    if single_names:
        fused_names = [name for name in strategy_returns if name not in single_names]
        best_single = max(strategy_returns[name].mean() for name in single_names)
        best_fused = max(strategy_returns[name].mean() for name in fused_names)
        print(f"单模型最好成绩 {best_single:.1f}；融合/仲裁的最好成绩 {best_fused:.1f}")
    print("提示：融合的优势通常来自方差降低；仲裁依赖 SAC 批评家，对 PPO 动作的"
          "评分属于分布外推断，解读时要保守。")

    os.makedirs(args.save_dir, exist_ok=True)
    torch.save(ppo_net.state_dict(), os.path.join(args.save_dir, "ppo_actor_critic.pt"))
    torch.save(sac_agent.policy.state_dict(), os.path.join(args.save_dir, "sac_policy.pt"))
    print(f"两个基模型已保存到 {args.save_dir}")
    save_plot(args.save_dir, names, means, stds)


if __name__ == "__main__":
    main()
