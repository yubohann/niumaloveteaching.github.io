"""
第030章 PPO的连续控制基准测试

把同一套 PPO 管道扩展成"离散 + 连续"双策略，并在四类经典控制任务上
做统一基准测试：

    CartPole-v1              离散（2 动作，参考基线）
    Acrobot-v1               离散（3 动作，欠驱动摆）
    Pendulum-v1              连续（1 维力矩，[-2, 2]）
    MountainCarContinuous-v0 连续（1 维油门，[-1, 1]，稀疏成功奖励）

连续策略采用 tanh 压缩的高斯策略：
    u ~ N(mu(s), sigma)      # sigma 由可学习的 log_std 参数给出
    a = shift + scale * tanh(u)
    采样时记录"原始高斯变量 u"，PPO 的比率直接在 u 上计算
    （tanh 压缩是确定映射，log 比中雅可比项相消）

统一协议：同网络规模、同超参数、同采样批；每个环境先测随机策略基线，
再训练 --max-steps 步（窗口≥100 回合且均值达标则提前停），最后用
确定性策略评估。输出跨环境基准表，回答"PPO 在各任务上分别需要多少
样本、达到什么水平"——这份表也是下一章切换到 SAC 的起点。

运行：
    python code/ch030.py            # 4 个环境各 6 万步（CPU 约 4-8 分钟）
    python code/ch030.py --quick    # 快速跑通（约 40-80 秒）
    python code/ch030.py --envs Pendulum-v1
    python code/ch030.py --envs CartPole-v1,Acrobot-v1
    python code/ch030.py --max-steps 80000 --steps-per-update 4096

预期（以实跑为准）：CartPole 最快达标；Acrobot 在 6 万步内多数种子能到
-120 上下；Pendulum 平均回报大约 -250～-150，距离 -100 级还远；
MountainCarContinuous 常常整轮跑不出成功回合——on-policy PPO 在这两个
连续任务上的样本效率短板，正是 SAC 登场的理由。
"""

import argparse
import math
import os
import random
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal


# ---------------------------------------------------------------------------
# 环境配置表
# ---------------------------------------------------------------------------
ENV_SPECS = {
    "CartPole-v1": {"discrete": True, "target": 475.0, "unit": "回报（满分 500）"},
    "Acrobot-v1": {"discrete": True, "target": -110.0, "unit": "回报（成功约 -100）"},
    "Pendulum-v1": {"discrete": False, "target": -300.0, "unit": "回报（最优约 -150）"},
    "MountainCarContinuous-v0": {"discrete": False, "target": 90.0,
                                 "unit": "回报（成功回合约 90+）"},
}


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    """固定随机种子，保证实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_tensor(obs, device, batch=False):
    """把 numpy 观测转成张量；(batch=False) 时增加一维变成单样本。"""
    t = torch.as_tensor(obs, dtype=torch.float32, device=device)
    return t if batch else t.unsqueeze(0)


# ---------------------------------------------------------------------------
# 离散策略网络
# ---------------------------------------------------------------------------
class DiscreteActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 64):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.actor = nn.Linear(hidden, act_dim)   # 动作 logits
        self.critic = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor):
        h = self.body(obs)
        return self.actor(h), self.critic(h).squeeze(-1)

    @torch.no_grad()
    def act(self, obs_t):
        logits, value = self.forward(obs_t)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action.item(), action.item(), dist.log_prob(action).item(), value.item()

    @torch.no_grad()
    def deterministic_action(self, obs_t):
        logits, _ = self.forward(obs_t)
        return int(torch.argmax(logits, dim=-1).item())

    @torch.no_grad()
    def value(self, obs_t) -> float:
        _, v = self.forward(obs_t)
        return v.item()


# ---------------------------------------------------------------------------
# 连续策略网络：tanh 压缩高斯
# ---------------------------------------------------------------------------
class ContinuousActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, act_low: np.ndarray,
                 act_high: np.ndarray, hidden: int = 64):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.mu = nn.Linear(hidden, act_dim)             # 均值
        self.log_std = nn.Parameter(torch.zeros(act_dim))  # 可学习方差（状态无关）
        self.critic = nn.Linear(hidden, 1)
        # 动作仿射变换参数：a = shift + scale * tanh(u)
        self.register_buffer("act_shift", torch.as_tensor((act_high + act_low) / 2.0,
                                                          dtype=torch.float32))
        self.register_buffer("act_scale", torch.as_tensor((act_high - act_low) / 2.0,
                                                          dtype=torch.float32))

    def forward(self, obs: torch.Tensor):
        h = self.body(obs)
        return self.mu(h), self.critic(h).squeeze(-1)

    def distribution(self, obs: torch.Tensor) -> Normal:
        """基础高斯分布；log_std 夹在 [-4, 1]，避免方差坍缩或爆炸。"""
        mu, _ = self.forward(obs)
        std = torch.exp(self.log_std.clamp(-4.0, 1.0)).expand_as(mu)
        return Normal(mu, std)

    def to_action(self, u: torch.Tensor) -> torch.Tensor:
        """把原始高斯变量 u 压缩到动作区间。"""
        return self.act_shift + self.act_scale * torch.tanh(u)

    @torch.no_grad()
    def act(self, obs_t):
        dist = self.distribution(obs_t)
        u = dist.sample()
        logp = dist.log_prob(u).sum(-1)
        action = self.to_action(u)
        action_np = action.squeeze(0).cpu().numpy().astype(np.float32)
        return u.item(), action_np, logp.item(), 0.0

    @torch.no_grad()
    def deterministic_action(self, obs_t):
        mu, _ = self.forward(obs_t)
        return self.to_action(mu).squeeze(0).cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def value(self, obs_t) -> float:
        _, v = self.forward(obs_t)
        return v.item()


def build_net(env, hidden: int = 64):
    """按动作空间类型创建对应网络。"""
    obs_dim = env.observation_space.shape[0]
    if isinstance(env.action_space, gym.spaces.Discrete):
        return DiscreteActorCritic(obs_dim, env.action_space.n, hidden)
    low = np.asarray(env.action_space.low, dtype=np.float32)
    high = np.asarray(env.action_space.high, dtype=np.float32)
    return ContinuousActorCritic(obs_dim, env.action_space.shape[0], low, high, hidden)


# ---------------------------------------------------------------------------
# GAE
# ---------------------------------------------------------------------------
def compute_gae(rewards, values, terminateds, last_value, gamma: float, lam: float):
    """从后向前递推 GAE，返回 (advantages, returns)。"""
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(T)):
        next_value = last_value if t == T - 1 else values[t + 1]
        non_terminal = 1.0 - terminateds[t]
        delta = rewards[t] + gamma * next_value * non_terminal - values[t]
        last_gae = delta + gamma * lam * non_terminal * last_gae
        advantages[t] = last_gae
    returns = advantages + np.asarray(values, dtype=np.float32)
    return advantages, returns


# ---------------------------------------------------------------------------
# 采样：离散/连续共用一条通道
# ---------------------------------------------------------------------------
def collect_rollout(env, net, discrete: bool, steps, obs_t, device, gamma, lam):
    """交互 steps 步；连续策略存"原始高斯变量 u"，比率的基准不受 tanh 影响。"""
    obs_buf, act_buf, logp_buf = [], [], []
    rew_buf, val_buf, term_buf = [], [], []
    ep_returns = []

    ep_ret = 0.0
    obs_after = obs_t
    for _ in range(steps):
        sample, action, logp, v = net.act(obs_t)
        next_obs, r, terminated, truncated, _ = env.step(action)

        obs_buf.append(obs_t.squeeze(0).cpu().numpy())
        act_buf.append(sample)          # 离散：动作 id；连续：原始 u
        logp_buf.append(logp)
        rew_buf.append(r)
        val_buf.append(v)
        term_buf.append(float(terminated))

        ep_ret += r
        obs_t = to_tensor(next_obs, device)
        obs_after = obs_t
        if terminated or truncated:
            ep_returns.append(ep_ret)
            ep_ret = 0.0
            reset_obs, _ = env.reset()
            obs_t = to_tensor(reset_obs, device)

    last_value = net.value(obs_after)
    advantages, returns = compute_gae(rew_buf, val_buf, term_buf, last_value, gamma, lam)

    act_dtype = torch.long if discrete else torch.float32
    batch = {
        "obs": torch.as_tensor(np.asarray(obs_buf), dtype=torch.float32, device=device),
        "act": torch.as_tensor(np.asarray(act_buf), dtype=act_dtype, device=device),
        "logp_old": torch.as_tensor(np.asarray(logp_buf), dtype=torch.float32, device=device),
        "adv": torch.as_tensor(advantages, dtype=torch.float32, device=device),
        "ret": torch.as_tensor(returns, dtype=torch.float32, device=device),
    }
    return batch, ep_returns, obs_t


# ---------------------------------------------------------------------------
# PPO 更新：两种策略分支
# ---------------------------------------------------------------------------
def ppo_update(net, optimizer, batch, args, discrete: bool):
    """做 epochs 轮小批量更新，返回平均诊断指标。"""
    obs, act = batch["obs"], batch["act"]
    logp_old, adv, ret = batch["logp_old"], batch["adv"], batch["ret"]
    N = obs.shape[0]
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    sum_kl = sum_clip = sum_entropy = sum_vloss = 0.0
    n_mb = 0

    for _ in range(args.epochs):
        idx = torch.randperm(N, device=obs.device)
        for start in range(0, N, args.batch_size):
            mb = idx[start:start + args.batch_size]
            if discrete:
                logits, value = net(obs[mb])
                dist = Categorical(logits=logits)
                logp = dist.log_prob(act[mb])
                entropy = dist.entropy().mean()
            else:
                dist = net.distribution(obs[mb])
                _, value = net(obs[mb])
                logp = dist.log_prob(act[mb]).sum(-1)
                entropy = dist.entropy().sum(-1).mean()  # 基础高斯熵（近似）

            ratio = torch.exp(logp - logp_old[mb])
            surr1 = ratio * adv[mb]
            surr2 = torch.clamp(ratio, 1.0 - args.clip, 1.0 + args.clip) * adv[mb]
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(value, ret[mb])
            loss = policy_loss + args.vf_coef * value_loss - args.ent_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            optimizer.step()

            with torch.no_grad():
                sum_kl += (logp_old[mb] - logp).mean().item()
                sum_clip += ((ratio - 1.0).abs() > args.clip).float().mean().item()
                sum_entropy += entropy.item()
                sum_vloss += value_loss.item()
            n_mb += 1

    d = max(n_mb, 1)
    return {"approx_kl": sum_kl / d, "clip_frac": sum_clip / d,
            "entropy": sum_entropy / d, "value_loss": sum_vloss / d}


# ---------------------------------------------------------------------------
# 评估与随机基线
# ---------------------------------------------------------------------------
def evaluate(env, net, discrete: bool, episodes: int, device):
    returns = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            action = net.deterministic_action(to_tensor(obs, device))
            obs, r, terminated, truncated, _ = env.step(action)
            ep_ret += r
            done = terminated or truncated
        returns.append(ep_ret)
    return returns


def random_baseline(env, episodes: int):
    returns = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            action = env.action_space.sample()
            obs, r, terminated, truncated, _ = env.step(action)
            ep_ret += r
            done = terminated or truncated
        returns.append(ep_ret)
    return float(np.mean(returns))


# ---------------------------------------------------------------------------
# 单环境训练
# ---------------------------------------------------------------------------
def run_env(env_id: str, args, device):
    """在单个环境上完成"随机基线 → 训练 → 评估"，返回基准条目。"""
    spec = ENV_SPECS[env_id]
    discrete = spec["discrete"]
    set_seed(args.seed)

    env = gym.make(env_id)
    obs, _ = env.reset(seed=args.seed)
    base_env = gym.make(env_id)          # 专用于随机基线，避免污染训练环境
    rand_score = random_baseline(base_env, args.random_episodes)
    base_env.close()

    net = build_net(env, hidden=args.hidden).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    obs_t = to_tensor(obs, device)
    all_returns = []
    total_steps = 0
    update_idx = 0
    solved_at = None
    t_start = time.time()
    act_desc = "离散" if discrete else "连续"

    while total_steps < args.max_steps and solved_at is None:
        batch, ep_returns, obs_t = collect_rollout(
            env, net, discrete, args.steps_per_update, obs_t, device, args.gamma, args.lam)
        total_steps += args.steps_per_update
        all_returns.extend(ep_returns)

        stats = ppo_update(net, optimizer, batch, args, discrete)
        update_idx += 1

        if update_idx % args.log_every == 0 or update_idx == 1:
            recent = all_returns[-100:] if all_returns else [0.0]
            print(f"  [{env_id:26s}] 更新 {update_idx:3d} | 步数 {total_steps:6d} | "
                  f"回合 {len(all_returns):4d} | 近100均 {np.mean(recent):8.1f} | "
                  f"KL {stats['approx_kl']:.4f} | 裁剪 {stats['clip_frac']:.3f} | "
                  f"熵 {stats['entropy']:.3f}")

        if len(all_returns) >= 100 and \
                np.mean(all_returns[-100:]) >= spec["target"]:
            solved_at = total_steps

    elapsed = time.time() - t_start
    eval_returns = evaluate(env, net, discrete, args.eval_episodes, device)

    os.makedirs(args.save_dir, exist_ok=True)
    torch.save(net.state_dict(), os.path.join(args.save_dir, f"{env_id}.pt"))

    last100 = float(np.mean(all_returns[-100:])) if all_returns else 0.0
    result = {
        "env": env_id, "type": act_desc, "random": rand_score,
        "steps": total_steps, "solved_at": solved_at,
        "last100": last100, "eval_mean": float(np.mean(eval_returns)),
        "time": elapsed,
    }
    print(f"  [{env_id:26s}] 结束：步数 {total_steps} | 用时 {elapsed:.1f}s | 随机基线 "
          f"{rand_score:.1f} | 首达 {('步数 ' + str(solved_at)) if solved_at else '未达成'} | "
          f"近100均 {last100:.1f} | 评估 {np.mean(eval_returns):.1f}")
    env.close()
    return result


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="第030章：PPO 连续控制基准测试")
    p.add_argument("--envs", type=str,
                   default="CartPole-v1,Acrobot-v1,Pendulum-v1,MountainCarContinuous-v0",
                   help="要基准测试的环境，逗号分隔")
    p.add_argument("--max-steps", type=int, default=60000, help="每个环境的环境步预算")
    p.add_argument("--steps-per-update", type=int, default=2048, help="每次更新前采样的步数")
    p.add_argument("--epochs", type=int, default=10, help="每批数据的更新轮数")
    p.add_argument("--batch-size", type=int, default=64, help="小批量大小")
    p.add_argument("--hidden", type=int, default=64, help="隐藏层宽度")
    p.add_argument("--lr", type=float, default=3e-4, help="学习率")
    p.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
    p.add_argument("--lam", type=float, default=0.95, help="GAE 的 lambda")
    p.add_argument("--clip", type=float, default=0.2, help="PPO 裁剪范围")
    p.add_argument("--ent-coef", type=float, default=0.01, help="熵正则系数")
    p.add_argument("--vf-coef", type=float, default=0.5, help="价值损失系数")
    p.add_argument("--log-every", type=int, default=10, help="每隔多少次更新打印日志")
    p.add_argument("--eval-episodes", type=int, default=10, help="评估回合数")
    p.add_argument("--random-episodes", type=int, default=5, help="随机策略基线回合数")
    p.add_argument("--seed", type=int, default=0, help="随机种子")
    p.add_argument("--save-dir", type=str, default="runs/ch030", help="模型保存目录")
    p.add_argument("--quick", action="store_true", help="快速模式：缩小训练规模，几十秒跑完")
    return p.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.max_steps = min(args.max_steps, 15000)
        args.steps_per_update = 768
        args.epochs = 5
        args.eval_episodes = 3
        args.random_episodes = 2

    env_ids = [e.strip() for e in args.envs.split(",") if e.strip()]
    env_ids = [e for e in env_ids if e in ENV_SPECS]
    if not env_ids:
        raise SystemExit("--envs 为空或非法，请检查参数")

    device = torch.device("cpu")
    print(f"设备: {device} | 种子: {args.seed}")
    print(f"基准环境: {env_ids}")
    print(f"统一配置: max_steps={args.max_steps}, steps_per_update={args.steps_per_update}, "
          f"epochs={args.epochs}, hidden={args.hidden}, lr={args.lr}")

    results = []
    for env_id in env_ids:
        print("-" * 92)
        print(f"开始环境：{env_id}（目标 {ENV_SPECS[env_id]['target']}，"
              f"{ENV_SPECS[env_id]['unit']}）")
        results.append(run_env(env_id, args, device))

    print("=" * 108)
    print(f"{'环境':<28}{'动作':>6}{'随机基线':>10}{'实际步数':>10}"
          f"{'首达目标':>10}{'近100均':>10}{'评估均':>10}{'用时(s)':>9}")
    for r in results:
        solved = str(r["solved_at"]) if r["solved_at"] else "未达成"
        print(f"{r['env']:<28}{r['type']:>6}{r['random']:>10.1f}{r['steps']:>10}"
              f"{solved:>10}{r['last100']:>10.1f}{r['eval_mean']:>10.1f}"
              f"{r['time']:>9.1f}")
    print("=" * 108)
    print("观察：离散任务（CartPole/Acrobot）PPO 表现稳健；连续任务里 Pendulum "
          "只能到中等水平，MountainCarContinuous 往往整轮不见成功——on-policy 的"
          "样本效率短板在连续控制上被放大。下一章开始用 SAC 处理同一类任务。")


if __name__ == "__main__":
    main()
