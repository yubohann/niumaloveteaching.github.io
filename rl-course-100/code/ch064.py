"""
第064章 PPO与SAC的稳定性对比

同一环境（Pendulum-v1）、同一预算、多个随机种子重复实验，回答两个问题：
  1) 换一个种子，最终性能波动多大（跨种子方差）？
  2) 单次训练过程中，评估曲线抖不抖、会不会崩（回撤与崩溃次数）？

运行：
    python code/ch064.py           # 完整实验（4 种子 × 2 算法，CPU 约 8-16 分钟）
    python code/ch064.py --quick   # 快速跑通（约 1-2 分钟）
    python code/ch064.py --algo sac --seeds 2 --max-steps 15000

预期（以实跑为准）：SAC 的跨种子标准差通常小于 PPO，曲线波动与最大回撤也更小；
PPO 更容易出现"某个种子学得很好、某个种子卡住"的分化。
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


# ===========================================================================
# PPO 部分
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


def train_ppo_once(seed, args, device):
    """训练一次 PPO，返回（评估曲线, 用时）。"""
    set_seed(seed)
    env = gym.make("Pendulum-v1")
    eval_env = gym.make("Pendulum-v1")
    obs, _ = env.reset(seed=seed)
    net = PPOActorCritic(env.observation_space.shape[0], env.action_space.shape[0],
                         args.hidden).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    obs_buf, act_buf, logp_buf, rew_buf, val_buf, cut_buf = [], [], [], [], [], []
    curve = []
    total_steps = 0
    next_eval = args.eval_every
    t0 = time.time()

    while total_steps < args.max_steps:
        # ---- 采样 ----
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
        obs_buf, act_buf, logp_buf, rew_buf, val_buf, cut_buf = [], [], [], [], [], []

        # ---- 更新 ----
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
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(value, batch["ret"][mb])
                loss = policy_loss + args.vf_coef * value_loss
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                optimizer.step()

        total_steps += args.steps_per_update
        if total_steps >= next_eval:
            curve.append((total_steps, evaluate(eval_env, net.mean_action,
                                                args.eval_episodes, device)))
            next_eval += args.eval_every
    return curve, time.time() - t0


# ===========================================================================
# SAC 部分
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
        make = lambda: nn.Sequential(nn.Linear(obs_dim + act_dim, hidden), nn.ReLU(),
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


def train_sac_once(seed, args, device):
    set_seed(seed)
    env = gym.make("Pendulum-v1")
    eval_env = gym.make("Pendulum-v1")
    obs, _ = env.reset(seed=seed)
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
            curve.append((step, evaluate(eval_env, agent.policy.mean_action,
                                         args.eval_episodes, device)))
    return curve, time.time() - t0


# ===========================================================================
# 稳定性指标
# ===========================================================================
def within_run_metrics(curve, crash_drop):
    """单次运行的稳定性指标：波动率、最大回撤、崩溃次数。"""
    rets = [r for _, r in curve]
    half = rets[len(rets) // 2:]
    volatility = float(np.std(half)) if len(half) > 1 else 0.0
    best_so_far = -np.inf
    max_drawdown = 0.0
    crashes = 0
    for r in rets:
        best_so_far = max(best_so_far, r)
        max_drawdown = max(max_drawdown, best_so_far - r)
        if r < best_so_far - crash_drop:
            crashes += 1
    return volatility, max_drawdown, crashes


def run_and_report(name, seed_list, train_once, args, device):
    """跑多个种子并打印汇总表。"""
    curves = []
    finals = []
    vols, dds, crash_list = [], [], []
    time_sum = 0.0
    for seed in seed_list:
        curve, dt = train_once(seed, args, device)
        curves.append(curve)
        time_sum += dt
        final = float(np.mean([r for _, r in curve[-3:]]))
        vol, dd, cr = within_run_metrics(curve, args.crash_drop)
        finals.append(final)
        vols.append(vol)
        dds.append(dd)
        crash_list.append(cr)
        print(f"[{name} 种子 {seed}] 最终 {final:8.1f} | 波动率 {vol:6.1f} | "
              f"最大回撤 {dd:6.1f} | 崩溃点 {cr} | 用时 {dt:6.1f}s")

    finals = np.array(finals)
    print(f"---- {name} 汇总（{len(seed_list)} 个种子）----")
    print(f"  最终性能: 均值 {finals.mean():8.1f} | 标准差 {finals.std():7.1f} | "
          f"最小 {finals.min():8.1f} | 最大 {finals.max():8.1f}")
    print(f"  达标率(≥{args.threshold:.0f}): {np.mean(finals >= args.threshold) * 100:.0f}% | "
          f"平均曲线波动率 {np.mean(vols):6.1f} | 平均最大回撤 {np.mean(dds):6.1f} | "
          f"崩溃点合计 {int(np.sum(crash_list))}")
    print(f"  总用时 {time_sum:.1f}s")
    return curves, finals


def save_plot(save_dir, name, curves):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("未安装 matplotlib，跳过绘图（不影响实验结论）")
        return
    fig, ax = plt.subplots(figsize=(7.5, 4))
    xs = [s for s, _ in curves[0]]
    arr = np.array([[r for _, r in c] for c in curves])
    for i, c in enumerate(curves):
        ax.plot(xs, [r for _, r in c], color="gray", alpha=0.35, linewidth=1,
                label="各次运行" if i == 0 else None)
    ax.plot(xs, arr.mean(axis=0), color="tab:blue", linewidth=2,
            marker="o", label="种子平均")
    ax.fill_between(xs, arr.min(axis=0), arr.max(axis=0), color="tab:blue", alpha=0.12,
                    label="min-max 区间")
    ax.set_xlabel("environment steps")
    ax.set_ylabel("eval return")
    ax.set_title(f"{name} stability over {len(curves)} seeds")
    ax.legend()
    ax.grid(alpha=0.3)
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{name.lower()}_stability.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"{name} 稳定性图已保存到 {path}")


def parse_args():
    p = argparse.ArgumentParser(description="第064章：PPO 与 SAC 的稳定性对比")
    p.add_argument("--algo", type=str, default="both", choices=["both", "ppo", "sac"])
    p.add_argument("--max-steps", type=int, default=30000, help="每次运行的环境步预算")
    p.add_argument("--seeds", type=int, default=4, help="随机种子个数（从 --seed 开始连续取）")
    p.add_argument("--seed", type=int, default=0, help="种子起始值")
    p.add_argument("--eval-every", type=int, default=2500, help="每多少步评估一次")
    p.add_argument("--eval-episodes", type=int, default=10, help="每次评估回合数")
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
    p.add_argument("--buffer-size", type=int, default=200000, help="SAC 回放池容量")
    p.add_argument("--start-steps", type=int, default=2000, help="SAC 预热步数")
    p.add_argument("--tau", type=float, default=0.005, help="SAC 软更新系数")
    p.add_argument("--alpha", type=float, default=0.2, help="SAC 温度初值")
    p.add_argument("--threshold", type=float, default=-400.0, help="达标参考线")
    p.add_argument("--crash-drop", type=float, default=150.0, help="低于历史最佳多少分记一次崩溃点")
    p.add_argument("--save-dir", type=str, default="runs/ch064", help="图片保存目录")
    p.add_argument("--quick", action="store_true", help="快速模式：2 种子 × 8000 步")
    return p.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.max_steps = min(args.max_steps, 8000)
        args.seeds = min(args.seeds, 2)
        args.eval_every = 2000
        args.eval_episodes = 3
        args.steps_per_update = 1024
        args.epochs = 6
        args.start_steps = 500

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = [args.seed + i for i in range(args.seeds)]
    print(f"设备: {device} | 环境: Pendulum-v1 | 种子列表 {seeds}")
    print(f"配置: 每种子 {args.max_steps} 步 | 每 {args.eval_every} 步评估 "
          f"{args.eval_episodes} 回合 | 训练种子与环境种子都随运行种子改变")

    results = {}
    if args.algo in ("ppo", "both"):
        results["PPO"] = run_and_report("PPO", seeds, train_ppo_once, args, device)
    if args.algo in ("sac", "both"):
        results["SAC"] = run_and_report("SAC", seeds, train_sac_once, args, device)

    print("=" * 76)
    if len(results) == 2:
        f_ppo, f_sac = results["PPO"][1], results["SAC"][1]
        print(f"结论速览：跨种子标准差 PPO {f_ppo.std():.1f} vs SAC {f_sac.std():.1f}；"
              f"最低种子 PPO {f_ppo.min():.1f} vs SAC {f_sac.min():.1f}")
        print("提示：最终性能与训练是否充分有关，请结合波动率、回撤与达标率一起解读。")

    save_dir = args.save_dir
    for name, (curves, _) in results.items():
        save_plot(save_dir, name, curves)


if __name__ == "__main__":
    main()
