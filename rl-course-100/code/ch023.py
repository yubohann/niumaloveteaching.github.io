"""
第023章 PPO的探索策略：熵正则化与噪声结合

把"目标函数内的探索"（熵正则化系数 c_ent）与"行为层的外加探索"
（logits 高斯噪声，σ 线性退火）放在一张 2x3 网格上做对照实验：

    c_ent ∈ {0.0, 0.01, 0.05}  x  noise ∈ {none, logitnoise}

在 CartPole-v1 上同种子、同预算（每格 --max-steps 步）训练，记录：
  - 首达 475 滑窗门槛的步数、末段训练回报、贪婪评估回报
  - 末段策略熵（stats 的后 5 次更新均值）：观察熵正则化留下的"熵底线"
  - 末段偏离率：执行动作不等于策略 argmax 的比例（外部探索的残留）
  - 累计状态覆盖桶（6^4 离散化）

核心问题：熵奖励与动作噪声是"两种可以互相替代的探索燃料"吗？
谁留下持久底线、谁只在早期起作用、叠在一起会不会过度探索？

运行：
    python code/ch023.py            # 6 个配置各 6 万步（CPU 约 4-8 分钟）
    python code/ch023.py --quick    # 快速跑通（约 40-80 秒）
    python code/ch023.py --ent-coefs 0.0,0.05 --noises none
    python code/ch023.py --ent-coefs 0.01 --noises logitnoise

预期：CartPole 上大多数格子都能达标；稳定的差异在"末段熵 / 末段偏离率"
两列——c_ent=0 的格子熵会塌到 0.2 附近，c_ent=0.05 的格子熵长期停在
0.55 以上（训练回报因此偏低）；噪声在退火结束后不留下底噪（偏离率回落），
说明两者的作用层次不同：一个是目标函数的持久偏置，一个是行为层的临时扰动。
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
from torch.distributions import Categorical

# 状态覆盖用的观测范围（CartPole 的常见取值范围）
COVER_LOW = np.array([-2.4, -3.0, -0.21, -2.5], dtype=np.float32)
COVER_HIGH = np.array([2.4, 3.0, 0.21, 2.5], dtype=np.float32)


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    """固定随机种子，保证实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_tensor(obs, device):
    """把 gym 返回的 numpy 观测转成 (1, obs_dim) 的 float32 张量。"""
    return torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)


def anneal_level(progress: float, start: float, end: float, anneal_frac: float) -> float:
    """线性退火：progress 从 0 到 anneal_frac 时，返回值从 start 线性降到 end。"""
    if anneal_frac <= 0.0:
        return end
    p = min(1.0, max(0.0, progress / anneal_frac))
    return start + (end - start) * p


def obs_to_bins(obs: np.ndarray, n_bins: int = 6):
    """把一批观测映射到 n_bins^4 的离散桶，返回访问到的桶集合。"""
    x = np.clip((obs - COVER_LOW) / (COVER_HIGH - COVER_LOW), 0.0, 0.9999)
    idx = (x * n_bins).astype(np.int64)
    return {tuple(int(v) for v in row) for row in idx}


# ---------------------------------------------------------------------------
# 网络：共享躯干 + 策略头 + 价值头
# ---------------------------------------------------------------------------
class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 64):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.actor = nn.Linear(hidden, act_dim)   # 输出离散动作的 logits
        self.critic = nn.Linear(hidden, 1)        # 输出状态价值 V(s)

    def forward(self, obs: torch.Tensor):
        h = self.body(obs)
        return self.actor(h), self.critic(h).squeeze(-1)

    @torch.no_grad()
    def value(self, obs: torch.Tensor) -> float:
        """只计算状态价值，用于 rollout 结尾的 bootstrap。"""
        _, v = self.forward(obs)
        return v.item()


# ---------------------------------------------------------------------------
# 动作选择：熵项不开在采样处，噪声是可选的"外部扰动"
# ---------------------------------------------------------------------------
@torch.no_grad()
def sample_action(net, obs_t, noise_mode: str, sigma: float):
    """返回 (action, logp, value, greedy_action)。

    logp 始终是干净策略对所选动作的对数概率（PPO 比率的基准）；
    logitnoise 只影响动作怎么选，不影响 logp 的算法。
    """
    logits, value = net(obs_t)
    greedy = int(torch.argmax(logits, dim=-1).item())

    if noise_mode == "logitnoise" and sigma > 0.0:
        noisy_logits = logits + sigma * torch.randn_like(logits)
        action = Categorical(logits=noisy_logits).sample()
    else:
        action = Categorical(logits=logits).sample()

    logp = Categorical(logits=logits).log_prob(action)
    return int(action.item()), logp.item(), value.item(), greedy


# ---------------------------------------------------------------------------
# GAE：广义优势估计
# ---------------------------------------------------------------------------
def compute_gae(rewards, values, terminateds, last_value, gamma: float, lam: float):
    """从后向前递推 GAE，返回 (advantages, returns)。"""
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(T)):
        next_value = last_value if t == T - 1 else values[t + 1]
        non_terminal = 1.0 - terminateds[t]       # 真实终止时价值为 0
        delta = rewards[t] + gamma * next_value * non_terminal - values[t]
        last_gae = delta + gamma * lam * non_terminal * last_gae
        advantages[t] = last_gae
    returns = advantages + np.asarray(values, dtype=np.float32)
    return advantages, returns


# ---------------------------------------------------------------------------
# 采样
# ---------------------------------------------------------------------------
def collect_rollout(env, net, noise_mode, sigma, steps, obs_t, device,
                    gamma, lam, n_actions):
    """与环境交互 steps 步，返回 batch、回合回报、偏离率、覆盖桶集合。"""
    obs_buf, act_buf, logp_buf = [], [], []
    rew_buf, val_buf, term_buf = [], [], []
    ep_returns = []
    n_deviate = 0

    ep_ret = 0.0
    obs_after = obs_t
    for _ in range(steps):
        a, logp, v, greedy = sample_action(net, obs_t, noise_mode, sigma)
        if a != greedy:
            n_deviate += 1
        next_obs, r, terminated, truncated, _ = env.step(a)

        obs_buf.append(obs_t.squeeze(0).cpu().numpy())
        act_buf.append(a)
        logp_buf.append(logp)
        rew_buf.append(r)
        val_buf.append(v)
        # 只有真实终止才把价值切断；时间截断（truncated）仍用 bootstrap
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

    obs_array = np.asarray(obs_buf, dtype=np.float32)
    batch = {
        "obs": torch.as_tensor(obs_array, dtype=torch.float32, device=device),
        "act": torch.as_tensor(np.asarray(act_buf), dtype=torch.long, device=device),
        "logp_old": torch.as_tensor(np.asarray(logp_buf), dtype=torch.float32, device=device),
        "adv": torch.as_tensor(advantages, dtype=torch.float32, device=device),
        "ret": torch.as_tensor(returns, dtype=torch.float32, device=device),
    }
    return batch, ep_returns, obs_t, n_deviate / float(steps), obs_to_bins(obs_array)


# ---------------------------------------------------------------------------
# PPO 更新（熵系数从参数传入，便于网格扫描）
# ---------------------------------------------------------------------------
def ppo_update(net, optimizer, batch, args, ent_coef: float):
    """做 epochs 轮小批量裁剪更新；ent_coef 是本次配置的熵正则系数。"""
    obs, act = batch["obs"], batch["act"]
    logp_old, adv, ret = batch["logp_old"], batch["adv"], batch["ret"]
    N = obs.shape[0]

    adv = (adv - adv.mean()) / (adv.std() + 1e-8)  # 优势标准化

    stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
             "approx_kl": 0.0, "clip_frac": 0.0, "n_updates": 0}

    for _ in range(args.epochs):
        idx = torch.randperm(N, device=obs.device)
        for start in range(0, N, args.batch_size):
            mb = idx[start:start + args.batch_size]
            logits, value = net(obs[mb])
            dist = Categorical(logits=logits)
            logp = dist.log_prob(act[mb])

            ratio = torch.exp(logp - logp_old[mb])
            surr1 = ratio * adv[mb]
            surr2 = torch.clamp(ratio, 1.0 - args.clip, 1.0 + args.clip) * adv[mb]
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = F.mse_loss(value, ret[mb])
            entropy = dist.entropy().mean()

            # 总损失：策略 + 价值 - 熵奖励（ent_coef=0 时该项消失）
            loss = policy_loss + args.vf_coef * value_loss - ent_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            optimizer.step()

            with torch.no_grad():
                approx_kl = (logp_old[mb] - logp).mean().item()
                clip_frac = ((ratio - 1.0).abs() > args.clip).float().mean().item()
            stats["policy_loss"] += policy_loss.item()
            stats["value_loss"] += value_loss.item()
            stats["entropy"] += entropy.item()
            stats["approx_kl"] += approx_kl
            stats["clip_frac"] += clip_frac
            stats["n_updates"] += 1

    for k in stats:
        if k != "n_updates":
            stats[k] /= max(stats["n_updates"], 1)
    return stats


# ---------------------------------------------------------------------------
# 评估：贪婪策略
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(env, net, episodes: int, device):
    returns = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            logits, _ = net(to_tensor(obs, device))
            action = int(torch.argmax(logits, dim=-1).item())
            obs, r, terminated, truncated, _ = env.step(action)
            ep_ret += r
            done = terminated or truncated
        returns.append(ep_ret)
    return returns


# ---------------------------------------------------------------------------
# 单个配置的训练流程
# ---------------------------------------------------------------------------
def run_config(ent_coef: float, noise_mode: str, args, device):
    """训练一个 (ent_coef, noise_mode) 配置，返回汇总字典。"""
    set_seed(args.seed)  # 所有配置共享同一初始条件

    env = gym.make("CartPole-v1")
    obs, _ = env.reset(seed=args.seed)
    n_actions = env.action_space.n

    net = ActorCritic(env.observation_space.shape[0], n_actions).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    obs_t = to_tensor(obs, device)
    all_returns = []
    entropy_hist = []      # 每次更新的平均熵
    dev_hist = []          # 每次 rollout 的偏离率
    cover_all = set()
    total_steps = 0
    update_idx = 0
    solved_at = None
    tag = f"ent={ent_coef:.2f},{noise_mode}"
    t_start = time.time()

    while total_steps < args.max_steps and solved_at is None:
        progress = total_steps / max(1, args.max_steps)
        # 噪声强度线性退火；熵系数是常数（它开在目标函数里，不随进度变化）
        sigma = anneal_level(progress, args.sigma_start, args.sigma_end, args.anneal_frac) \
            if noise_mode == "logitnoise" else 0.0

        batch, ep_returns, obs_t, deviate, cover_bins = collect_rollout(
            env, net, noise_mode, sigma, args.steps_per_update, obs_t, device,
            args.gamma, args.lam, n_actions)
        total_steps += args.steps_per_update
        all_returns.extend(ep_returns)
        cover_all |= cover_bins
        dev_hist.append(deviate)

        stats = ppo_update(net, optimizer, batch, args, ent_coef)
        entropy_hist.append(stats["entropy"])
        update_idx += 1

        if update_idx % args.log_every == 0 or update_idx == 1:
            recent = all_returns[-20:] if all_returns else [0.0]
            print(f"  [{tag:22s}] 更新 {update_idx:3d} | 步数 {total_steps:7d} | "
                  f"近20均 {np.mean(recent):6.1f} | 熵 {stats['entropy']:.3f} | "
                  f"偏离率 {deviate:.3f} | 覆盖桶 {len(cover_all):4d} | "
                  f"KL {stats['approx_kl']:.4f} | σ {sigma:.3f}")

        if solved_at is None and len(all_returns) >= 100 and \
                np.mean(all_returns[-100:]) >= args.target_reward:
            solved_at = total_steps

    elapsed = time.time() - t_start

    eval_env = gym.make("CartPole-v1")
    eval_returns = evaluate(eval_env, net, args.eval_episodes, device)
    os.makedirs(args.save_dir, exist_ok=True)
    short = f"e{ent_coef:.2f}_{noise_mode}"
    torch.save(net.state_dict(), os.path.join(args.save_dir, f"{short}.pt"))

    # 末段指标：最后 5 次更新 / 5 次 rollout 的均值
    k = min(5, len(entropy_hist))
    ent_last = float(np.mean(entropy_hist[-k:])) if k else 0.0
    dev_last = float(np.mean(dev_hist[-k:])) if k else 0.0
    last100 = float(np.mean(all_returns[-100:])) if all_returns else 0.0

    result = {
        "tag": tag, "ent_coef": ent_coef, "noise": noise_mode,
        "solved_at": solved_at, "last100": last100,
        "eval_mean": float(np.mean(eval_returns)),
        "ent_last": ent_last, "dev_last": dev_last,
        "coverage": len(cover_all), "time": elapsed,
    }
    print(f"  [{tag:22s}] 结束：总步数 {total_steps} | 首达 "
          f"{('步数 ' + str(solved_at)) if solved_at else '未达成'} | "
          f"近100均 {last100:.1f} | 评估 {np.mean(eval_returns):.1f} | "
          f"末段熵 {ent_last:.3f} | 末段偏离率 {dev_last:.3f} | "
          f"覆盖桶 {len(cover_all)} | 用时 {elapsed:.1f}s")
    env.close()
    eval_env.close()
    return result


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="第023章：熵正则化与动作噪声的 2x3 网格实验")
    p.add_argument("--ent-coefs", type=str, default="0.0,0.01,0.05",
                   help="要扫描的熵系数，逗号分隔")
    p.add_argument("--noises", type=str, default="none,logitnoise",
                   help="要扫描的噪声模式：none / logitnoise")
    p.add_argument("--sigma-start", type=float, default=1.0, help="logits 噪声初始 σ")
    p.add_argument("--sigma-end", type=float, default=0.0, help="logits 噪声最终 σ")
    p.add_argument("--anneal-frac", type=float, default=0.6, help="前多大比例预算内线性退火")
    p.add_argument("--max-steps", type=int, default=60000, help="每个配置的环境步数上限")
    p.add_argument("--steps-per-update", type=int, default=2048, help="每次更新前采样的步数")
    p.add_argument("--epochs", type=int, default=10, help="每批数据的更新轮数")
    p.add_argument("--batch-size", type=int, default=64, help="小批量大小")
    p.add_argument("--lr", type=float, default=3e-4, help="学习率")
    p.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
    p.add_argument("--lam", type=float, default=0.95, help="GAE 的 lambda")
    p.add_argument("--clip", type=float, default=0.2, help="PPO 裁剪范围")
    p.add_argument("--vf-coef", type=float, default=0.5, help="价值损失系数")
    p.add_argument("--log-every", type=int, default=5, help="每隔多少次更新打印日志")
    p.add_argument("--target-reward", type=float, default=475.0, help="提前停的滑窗平均回报")
    p.add_argument("--eval-episodes", type=int, default=10, help="每个配置的评估回合数")
    p.add_argument("--seed", type=int, default=0, help="随机种子")
    p.add_argument("--save-dir", type=str, default="runs/ch023", help="模型保存目录")
    p.add_argument("--quick", action="store_true", help="快速模式：缩小训练规模，几十秒跑完")
    return p.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.max_steps = min(args.max_steps, 8000)
        args.steps_per_update = 768
        args.epochs = 5
        args.eval_episodes = 3

    ent_coefs = [float(x) for x in args.ent_coefs.split(",") if x.strip()]
    noises = [n.strip() for n in args.noises.split(",") if n.strip()]
    noises = [n for n in noises if n in ("none", "logitnoise")]
    if not ent_coefs or not noises:
        raise SystemExit("--ent-coefs 或 --noises 为空，请检查参数")

    device = torch.device("cpu")  # 纯 CPU
    print(f"设备: {device} | 环境: CartPole-v1 | 种子: {args.seed}")
    print(f"网格: ent_coefs={ent_coefs} x noises={noises} | "
          f"每格 {args.max_steps} 步 | anneal_frac={args.anneal_frac}")

    results = []
    for ent_coef in ent_coefs:
        for noise_mode in noises:
            print("-" * 84)
            print(f"开始配置: ent_coef={ent_coef} | noise={noise_mode}")
            results.append(run_config(ent_coef, noise_mode, args, device))

    # 网格汇总表
    print("=" * 96)
    print(f"{'配置':<24}{'首达目标':>10}{'近100均':>10}{'评估均':>9}"
          f"{'末段熵':>8}{'末段偏离率':>11}{'覆盖桶':>8}{'用时(s)':>9}")
    for r in results:
        solved = str(r["solved_at"]) if r["solved_at"] else "未达成"
        print(f"{r['tag']:<24}{solved:>10}{r['last100']:>10.1f}"
              f"{r['eval_mean']:>9.1f}{r['ent_last']:>8.3f}"
              f"{r['dev_last']:>11.3f}{r['coverage']:>8d}{r['time']:>9.1f}")
    print("=" * 96)
    best = max(results, key=lambda r: r["eval_mean"])
    print(f"评估均值最高：{best['tag']}（{best['eval_mean']:.1f}）")
    print("关注三条规律：c_ent=0 的末段熵最低（策略先变确定）；c_ent=0.05 的末段熵"
          "被抬到 0.55 以上（训练回报被随机采样拖低）；噪声退火后末段偏离率与"
          "纯熵配置接近（外部噪声不提供持久底线）。")


if __name__ == "__main__":
    main()
