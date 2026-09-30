"""
第084章 多智能体PPO与SAC对比

实验内容（3 个智能体协同导航到 3 个地标，环境为脚本内联的纯 numpy 实现）：
  - 同一个协作环境、同一套评估协议下训练两种算法：
      ippo：on-policy 的独立 PPO（参数共享，GAE + 裁剪目标）
      isac：off-policy 的独立 SAC（参数共享，双 Q + 目标网络 + 自动温度）
  - 两者的策略网络结构、学习率、折扣因子完全一致，只比较算法本身
  - 对比指标：评估回报曲线、地标覆盖率、首次达到 50% 覆盖率所用的环境步数

运行：
    python code/ch084.py                 # 两个算法都训练（CPU 约 6-12 分钟）
    python code/ch084.py --quick         # 快速跑通，约 50-90 秒
    python code/ch084.py --algo ippo     # 只训练 PPO
    python code/ch084.py --algo isac     # 只训练 SAC

预期：SAC 在样本效率上通常更早达到中等覆盖率，PPO 的曲线更平滑、对新数据
依赖更强。最终结果随随机种子波动，请以同种子多次实跑为准。
"""

import argparse
import copy
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_STD_MIN, LOG_STD_MAX = -20.0, 2.0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# 环境：多智能体协同导航
# ---------------------------------------------------------------------------
class CooperativeNavEnv:
    """3 个智能体需要各自抵达一个地标（共 3 个地标）。

    观测（12 维/智能体）：自己的位置+速度(4)、按距离排序后最近两个地标
    的相对坐标(4)、另外两个智能体的相对位置(4)。
    动作：2 维连续加速度。奖励：所有地标"到最近智能体距离"的负均值
    （团队共享）+ 智能体互相碰撞的惩罚。评估指标：地标覆盖率。
    """

    obs_dim = 12
    act_dim = 2

    def __init__(self, n_agents: int = 3, coverage_radius: float = 0.12,
                 max_steps: int = 60, friction: float = 0.8, accel: float = 0.12,
                 v_max: float = 1.0, bound: float = 1.0):
        self.n_agents = n_agents
        self.coverage_radius = coverage_radius
        self.max_steps = max_steps
        self.friction = friction
        self.accel = accel
        self.v_max = v_max
        self.bound = bound

    def reset(self, rng: np.random.Generator):
        angles = np.linspace(0.0, 2.0 * np.pi, self.n_agents, endpoint=False)
        self.landmarks = (0.6 * np.stack([np.cos(angles), np.sin(angles)], axis=1)
                          ).astype(np.float32)
        self.pos = rng.uniform(-0.25, 0.25, size=(self.n_agents, 2)).astype(np.float32)
        self.vel = np.zeros((self.n_agents, 2), dtype=np.float32)
        self.steps = 0
        return self._obs()

    def _obs(self) -> np.ndarray:
        obs = []
        for i in range(self.n_agents):
            d = np.linalg.norm(self.landmarks - self.pos[i], axis=1)
            order = np.argsort(d)
            rel_landmarks = []
            for k in order[: 2]:                       # 最近两个地标
                rel_landmarks.append(self.landmarks[k] - self.pos[i])
            others = [self.pos[j] - self.pos[i]
                      for j in range(self.n_agents) if j != i]
            o = np.concatenate([self.pos[i], self.vel[i],
                                np.concatenate(rel_landmarks),
                                np.concatenate(others)])
            obs.append(o)
        return np.stack(obs).astype(np.float32)

    def step(self, actions: np.ndarray):
        a = np.clip(actions, -1.0, 1.0).astype(np.float32)
        self.vel = np.clip(self.friction * self.vel + self.accel * a,
                           -self.v_max, self.v_max)
        self.pos = np.clip(self.pos + self.vel, -self.bound, self.bound)
        self.steps += 1

        dists = np.linalg.norm(self.landmarks[:, None, :] - self.pos[None, :, :],
                               axis=-1)          # (地标数, 智能体数)
        min_dists = dists.min(axis=1)            # 每个地标到最近智能体的距离
        reward = float(-min_dists.mean())        # 团队共享回报

        collisions = 0
        for i in range(self.n_agents):
            for j in range(i + 1, self.n_agents):
                if np.linalg.norm(self.pos[i] - self.pos[j]) < 0.08:
                    collisions += 1
        reward -= 0.1 * collisions               # 碰撞惩罚

        coverage = float(np.mean(min_dists < self.coverage_radius))
        truncated = self.steps >= self.max_steps
        return self._obs(), reward, False, truncated, {"coverage": coverage}


# ---------------------------------------------------------------------------
# 网络
# ---------------------------------------------------------------------------
class PPOActorCritic(nn.Module):
    """PPO 的 Actor-Critic：共享躯干 + 状态无关 log_std + 价值头。"""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.mu_head = nn.Linear(hidden, act_dim)
        self.v_head = nn.Linear(hidden, 1)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs: torch.Tensor):
        h = self.body(obs)
        mu = self.mu_head(h)
        std = torch.clamp(self.log_std, LOG_STD_MIN, LOG_STD_MAX).exp()
        v = self.v_head(h).squeeze(-1)
        return mu, std, v

    @torch.no_grad()
    def act_batch(self, obs_np: np.ndarray, device):
        """对一批智能体观测一次前向，返回动作、log 概率、价值（numpy）。"""
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
        mu, std, v = self.forward(obs)
        dist = Normal(mu, std)
        x = dist.sample()
        action = torch.tanh(x)
        logp = dist.log_prob(x) - torch.log(1.0 - action.pow(2) + 1e-6)
        return (action.cpu().numpy().astype(np.float32),
                logp.sum(dim=-1).cpu().numpy().astype(np.float32),
                v.cpu().numpy().astype(np.float32))

    @torch.no_grad()
    def act_deterministic(self, obs_np: np.ndarray, device) -> np.ndarray:
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
        mu, _, _ = self.forward(obs)
        return torch.tanh(mu).cpu().numpy().astype(np.float32)

    def log_prob_and_value(self, obs: torch.Tensor, act: torch.Tensor):
        """训练时重算 log 概率（tanh 压缩 + 逆变换）、状态价值与策略熵。"""
        mu, std, v = self.forward(obs)
        dist = Normal(mu, std)
        a = torch.clamp(act, -0.999999, 0.999999)
        x = torch.atanh(a)
        logp = dist.log_prob(x) - torch.log(1.0 - a.pow(2) + 1e-6)
        return logp.sum(dim=-1), v, dist.entropy().sum(dim=-1)


class SquashedGaussianActor(nn.Module):
    """SAC 的连续策略（重参数化 + tanh 压缩）。"""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden, act_dim)
        self.log_std_head = nn.Linear(hidden, act_dim)

    def forward(self, obs: torch.Tensor, deterministic: bool = False):
        h = self.net(obs)
        mu = self.mu_head(h)
        log_std = torch.clamp(self.log_std_head(h), LOG_STD_MIN, LOG_STD_MAX)
        dist = Normal(mu, log_std.exp())
        x = mu if deterministic else dist.rsample()
        action = torch.tanh(x)
        logp = dist.log_prob(x) - torch.log(1.0 - action.pow(2) + 1e-6)
        return action, logp.sum(dim=-1, keepdim=True)


class QNet(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# 评估（算法无关的公共协议）
# ---------------------------------------------------------------------------
def evaluate(env, act_fn, episodes, rng):
    """act_fn(obs) -> 动作；返回平均回报、平均覆盖率、全地标覆盖成功率。"""
    returns, coverages, successes = [], [], []
    for _ in range(episodes):
        obs = env.reset(rng)
        done = False
        ep_ret, cov_sum, n = 0.0, 0.0, 0
        info = {"coverage": 0.0}
        while not done:
            actions = act_fn(obs)
            obs, rew, terminated, truncated, info = env.step(actions)
            ep_ret += rew
            cov_sum += info["coverage"]
            n += 1
            done = terminated or truncated
        returns.append(ep_ret)
        coverages.append(cov_sum / max(n, 1))
        successes.append(1.0 if info["coverage"] >= 0.999 else 0.0)
    return float(np.mean(returns)), float(np.mean(coverages)), float(np.mean(successes))


# ---------------------------------------------------------------------------
# 算法一：独立 PPO（参数共享）
# ---------------------------------------------------------------------------
def compute_gae_shared(rewards, values, dones, last_values, gamma, lam):
    """共享奖励下的 GAE：rewards/dones 为 (T,)，values 为 (T, N)。"""
    T, N = values.shape
    adv = np.zeros((T, N), dtype=np.float32)
    last = np.zeros(N, dtype=np.float32)
    for t in reversed(range(T)):
        next_v = last_values if t == T - 1 else values[t + 1]
        non_term = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_v * non_term - values[t]
        last = delta + gamma * lam * non_term * last
        adv[t] = last
    return adv, adv + values


def train_ippo(env, args, rng, device):
    net = PPOActorCritic(env.obs_dim, env.act_dim, args.hidden).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    def act_fn(obs):
        return net.act_deterministic(obs, device)

    eval_steps, eval_ret, eval_cov, eval_succ = [], [], [], []
    total_steps, update_idx = 0, 0
    obs = env.reset(rng)
    t_start = time.time()

    while total_steps < args.max_steps:
        # 1) 采样一批联合轨迹
        obs_buf, act_buf, logp_buf, val_buf = [], [], [], []
        rew_buf, done_buf = [], []
        for _ in range(args.rollout):
            a, logp, v = net.act_batch(obs, device)
            next_obs, rew, terminated, truncated, info = env.step(a)
            obs_buf.append(obs)
            act_buf.append(a)
            logp_buf.append(logp)
            val_buf.append(v)
            rew_buf.append(rew)
            done_buf.append(float(terminated))     # 时间截断不算终止
            total_steps += 1
            if terminated or truncated:
                obs = env.reset(rng)
            else:
                obs = next_obs

        with torch.no_grad():
            _, _, last_v = net.forward(
                torch.as_tensor(obs, dtype=torch.float32, device=device))
            last_values = last_v.cpu().numpy().astype(np.float32)
        adv, ret = compute_gae_shared(np.asarray(rew_buf, dtype=np.float32),
                                      np.stack(val_buf),
                                      np.asarray(done_buf, dtype=np.float32),
                                      last_values, args.gamma, args.lam)

        # 2) 展平成 (T*N, ...) 的批量做裁剪目标更新
        T, N = adv.shape
        b_obs = torch.as_tensor(np.stack(obs_buf).reshape(T * N, -1),
                                dtype=torch.float32, device=device)
        b_act = torch.as_tensor(np.stack(act_buf).reshape(T * N, -1),
                                dtype=torch.float32, device=device)
        b_logp = torch.as_tensor(np.stack(logp_buf).reshape(T * N),
                                 dtype=torch.float32, device=device)
        b_adv = torch.as_tensor(adv.reshape(T * N),
                                dtype=torch.float32, device=device)
        b_ret = torch.as_tensor(ret.reshape(T * N),
                                dtype=torch.float32, device=device)
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

        for _ in range(args.epochs):
            idx = torch.randperm(T * N, device=device)
            for start in range(0, T * N, args.batch_size):
                mb = idx[start:start + args.batch_size]
                logp, v, entropy = net.log_prob_and_value(b_obs[mb], b_act[mb])
                ratio = torch.exp(logp - b_logp[mb])
                surr1 = ratio * b_adv[mb]
                surr2 = torch.clamp(ratio, 1.0 - args.clip,
                                    1.0 + args.clip) * b_adv[mb]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(v, b_ret[mb])
                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy.mean()
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                optimizer.step()
        update_idx += 1

        # 3) 定期评估
        if total_steps % args.eval_every < args.rollout:
            m_ret, m_cov, m_succ = evaluate(env, act_fn, args.eval_episodes, rng)
            eval_steps.append(total_steps)
            eval_ret.append(m_ret)
            eval_cov.append(m_cov)
            eval_succ.append(m_succ)
            print(f"[IPPO] 步 {total_steps:6d} | 更新 {update_idx:3d} | "
                  f"回报 {m_ret:7.2f} | 覆盖率 {m_cov:.2f} | 成功率 {m_succ:.2f}")

    elapsed = time.time() - t_start
    return {"net": net, "eval_steps": eval_steps, "eval_ret": eval_ret,
            "eval_cov": eval_cov, "eval_succ": eval_succ, "time": elapsed}


# ---------------------------------------------------------------------------
# 算法二：独立 SAC（参数共享）
# ---------------------------------------------------------------------------
class SharedSAC:
    def __init__(self, env: CooperativeNavEnv, args, device):
        self.args = args
        self.device = device
        self.actor = SquashedGaussianActor(env.obs_dim, env.act_dim,
                                           args.hidden).to(device)
        q_in = env.obs_dim + env.act_dim
        self.q1 = QNet(q_in, args.hidden).to(device)
        self.q2 = QNet(q_in, args.hidden).to(device)
        self.q1_target = copy.deepcopy(self.q1)
        self.q2_target = copy.deepcopy(self.q2)
        for net in (self.q1_target, self.q2_target):
            for p in net.parameters():
                p.requires_grad_(False)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=args.lr)
        self.critic_opt = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=args.lr)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=args.lr)
        self.target_entropy = -float(env.act_dim)
        self.buffer = []

    @torch.no_grad()
    def act_batch(self, obs_np, deterministic=False):
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device)
        a, _ = self.actor(obs, deterministic)
        return a.cpu().numpy().astype(np.float32)

    def update_once(self):
        args = self.args
        idx = np.random.randint(0, len(self.buffer), size=args.batch_size)
        batch = [self.buffer[i] for i in idx]

        def stack(key):
            return torch.as_tensor(np.stack([b[key] for b in batch]),
                                   dtype=torch.float32, device=self.device)

        obs, act, rew = stack("obs"), stack("act"), stack("rew")
        next_obs, done = stack("next_obs"), stack("done")
        alpha = self.log_alpha.exp().detach()

        with torch.no_grad():
            next_a, next_logp = self.actor(next_obs)
            q_in = torch.cat([next_obs, next_a], dim=-1)
            y = rew + args.gamma * (1.0 - done) * (
                torch.min(self.q1_target(q_in), self.q2_target(q_in)) - alpha * next_logp)

        q_in = torch.cat([obs, act], dim=-1)
        critic_loss = F.mse_loss(self.q1(q_in), y) + F.mse_loss(self.q2(q_in), y)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        new_a, logp = self.actor(obs)
        q_in = torch.cat([obs, new_a], dim=-1)
        actor_loss = (alpha * logp - torch.min(self.q1(q_in), self.q2(q_in))).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        with torch.no_grad():
            for net, tnet in ((self.q1, self.q1_target), (self.q2, self.q2_target)):
                for p, tp in zip(net.parameters(), tnet.parameters()):
                    tp.data.mul_(1.0 - args.tau).add_(args.tau * p.data)
        return float(critic_loss.item())


def train_isac(env, args, rng, device):
    sac = SharedSAC(env, args, device)

    def act_fn(obs):
        return sac.act_batch(obs, deterministic=True)

    eval_steps, eval_ret, eval_cov, eval_succ = [], [], [], []
    total_steps = 0
    obs = env.reset(rng)
    t_start = time.time()

    while total_steps < args.max_steps:
        # 1) 选动作并收集联合转移（共享奖励：同一 r 发给所有智能体）
        if total_steps < args.warmup:
            actions = rng.uniform(-1.0, 1.0,
                                  size=(env.n_agents, env.act_dim)).astype(np.float32)
        else:
            actions = sac.act_batch(obs)
        next_obs, rew, terminated, truncated, info = env.step(actions)

        for i in range(env.n_agents):
            sac.buffer.append({
                "obs": obs[i], "act": actions[i],
                "rew": np.float32(rew), "next_obs": next_obs[i],
                "done": np.float32(terminated),    # 截断仍 bootstrap
            })
        total_steps += 1

        # 2) 训练（每步一次梯度更新，共享网络看的是全队经验）
        if total_steps > args.warmup and len(sac.buffer) >= args.batch_size:
            sac.update_once()

        if terminated or truncated:
            obs = env.reset(rng)
        else:
            obs = next_obs

        # 3) 定期评估
        if total_steps % args.eval_every == 0:
            m_ret, m_cov, m_succ = evaluate(env, act_fn, args.eval_episodes, rng)
            eval_steps.append(total_steps)
            eval_ret.append(m_ret)
            eval_cov.append(m_cov)
            eval_succ.append(m_succ)
            print(f"[ISAC] 步 {total_steps:6d} | 回报 {m_ret:7.2f} | "
                  f"覆盖率 {m_cov:.2f} | 成功率 {m_succ:.2f}")

    elapsed = time.time() - t_start
    return {"net": sac.actor, "eval_steps": eval_steps, "eval_ret": eval_ret,
            "eval_cov": eval_cov, "eval_succ": eval_succ, "time": elapsed}


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="第084章：多智能体 PPO 与 SAC 对比")
    p.add_argument("--algo", type=str, default="both",
                   choices=["ippo", "isac", "both"], help="训练哪个算法")
    p.add_argument("--max-steps", type=int, default=30000, help="每个算法的环境步数")
    p.add_argument("--rollout", type=int, default=2048, help="PPO 每次采样的步数")
    p.add_argument("--epochs", type=int, default=4, help="PPO 每批数据的更新轮数")
    p.add_argument("--warmup", type=int, default=2000, help="SAC 随机预热步数")
    p.add_argument("--batch-size", type=int, default=128, help="小批量大小")
    p.add_argument("--lr", type=float, default=3e-4, help="学习率")
    p.add_argument("--gamma", type=float, default=0.95, help="折扣因子")
    p.add_argument("--lam", type=float, default=0.95, help="PPO 的 GAE lambda")
    p.add_argument("--clip", type=float, default=0.2, help="PPO 裁剪范围")
    p.add_argument("--tau", type=float, default=0.01, help="SAC 软更新系数")
    p.add_argument("--hidden", type=int, default=128, help="隐藏层宽度")
    p.add_argument("--eval-every", type=int, default=3000, help="评估间隔（步）")
    p.add_argument("--eval-episodes", type=int, default=10, help="评估回合数")
    p.add_argument("--seed", type=int, default=0, help="随机种子")
    p.add_argument("--save-dir", type=str, default="runs/ch084", help="模型保存目录")
    p.add_argument("--quick", action="store_true", help="快速模式：缩小训练规模")
    return p.parse_args()


def summarize(name, stats):
    """计算首次达到 50% 覆盖率所需步数（样本效率的简单代理指标）。"""
    half_at = None
    for s, c in zip(stats["eval_steps"], stats["eval_cov"]):
        if c >= 0.5:
            half_at = s
            break
    best = max(stats["eval_ret"]) if stats["eval_ret"] else float("nan")
    final = stats["eval_ret"][-1] if stats["eval_ret"] else float("nan")
    print(f"{name:5s} | 用时 {stats['time']:6.1f}s | 最终回报 {final:7.2f} | "
          f"最优回报 {best:7.2f} | 首次覆盖≥0.5 于 {half_at} 步 | "
          f"最终覆盖率 {stats['eval_cov'][-1]:.2f} | 成功率 {stats['eval_succ'][-1]:.2f}")
    return half_at


def main():
    args = parse_args()
    if args.quick:
        args.max_steps = min(args.max_steps, 8000)
        args.rollout = 1024
        args.warmup = 1000
        args.eval_every = 2000
        args.eval_episodes = 5
    set_seed(args.seed)
    device = torch.device("cpu")
    env = CooperativeNavEnv()

    print(f"设备 {device} | 算法 {args.algo} | 种子 {args.seed} | "
          f"环境 3 智能体协同导航（{args.max_steps} 步/算法）")

    results = {}
    if args.algo in ("ippo", "both"):
        print("-" * 74)
        rng_ppo = np.random.default_rng(args.seed)
        results["IPPO"] = train_ippo(env, args, rng_ppo, device)
    if args.algo in ("isac", "both"):
        print("-" * 74)
        rng_sac = np.random.default_rng(args.seed)
        results["ISAC"] = train_isac(env, args, rng_sac, device)

    print("=" * 74)
    print("汇总（同环境、同评估协议、同种子）")
    for name, stats in results.items():
        summarize(name, stats)

    os.makedirs(args.save_dir, exist_ok=True)
    ckpt = {f"{name}_policy": stats["net"].state_dict()
            for name, stats in results.items()}
    ckpt["eval_history"] = {name: stats["eval_ret"] for name, stats in results.items()}
    ckpt["args"] = vars(args)
    path = os.path.join(args.save_dir, "ppo_vs_sac.pt")
    torch.save(ckpt, path)
    print(f"模型与评估历史已保存到 {path}")


if __name__ == "__main__":
    main()
