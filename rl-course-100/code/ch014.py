"""
第014章 PPO的对抗性扰动鲁棒性

策略在"干净"环境里考满分，不代表它进入了现实世界。本章在 CartPole-v1 上做两件事：
  1) 训练两种策略（同种子、同步数预算）：
       - plain：标准训练
       - noisy：训练时给观测叠加 σ=0.05 的高斯噪声（数据增强式防御）
  2) 对两个策略做鲁棒性体检：
       - 高斯观测噪声：σ ∈ {0, 0.01, 0.03, 0.05, 0.1, 0.2}
       - 对抗扰动（FGSM 风格）：沿"降低选中动作概率"的梯度方向扰动观测，
         强度 ε ∈ {0, 0.01, 0.02, 0.05, 0.1}

输出：干净回报、各噪声/扰动档位的回报、以及"回报跌破 400 的档位"。

运行：
    python code/ch014.py            # 2 种训练 × 8 万步 + 完整鲁棒性体检，CPU 约 4-7 分钟
    python code/ch014.py --quick    # 快速跑通（约 30-60 秒）
    python code/ch014.py --arms noisy

预期：noisy 策略干净回报略低，但噪声与对抗档位下的回报明显更高；
对抗扰动的杀伤力通常远大于同量级随机噪声——梯度方向的扰动最危险。
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


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_tensor(obs, device):
    return torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)


# ---------------------------------------------------------------------------
# 网络
# ---------------------------------------------------------------------------
class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 64):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.actor = nn.Linear(hidden, act_dim)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor):
        h = self.body(obs)
        return self.actor(h), self.critic(h).squeeze(-1)

    @torch.no_grad()
    def act(self, obs: torch.Tensor):
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action).item(), value.item()

    @torch.no_grad()
    def value(self, obs: torch.Tensor) -> float:
        _, v = self.forward(obs)
        return v.item()


# ---------------------------------------------------------------------------
# GAE
# ---------------------------------------------------------------------------
def compute_gae(rewards, values, terminateds, last_value, gamma: float, lam: float):
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
# 采样：可选在喂给网络前叠加高斯噪声（训练期数据增强）
# ---------------------------------------------------------------------------
def collect_rollout(env, net, steps, obs_t, device, gamma, lam, noise_sigma, rng):
    obs_buf, act_buf, logp_buf = [], [], []
    rew_buf, val_buf, term_buf = [], [], []
    ep_returns = []

    ep_ret = 0.0
    obs_after = obs_t
    for _ in range(steps):
        if noise_sigma > 0.0:
            noise = rng.normal(0.0, noise_sigma, size=obs_t.shape).astype(np.float32)
            net_in = obs_t + torch.as_tensor(noise, device=device)
        else:
            net_in = obs_t
        a, logp, v = net.act(net_in)
        next_obs, r, terminated, truncated, _ = env.step(a)

        obs_buf.append(net_in.squeeze(0).cpu().numpy())   # 存"策略看到的观测"
        act_buf.append(a)
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
    batch = {
        "obs": torch.as_tensor(np.asarray(obs_buf), dtype=torch.float32, device=device),
        "act": torch.as_tensor(np.asarray(act_buf), dtype=torch.long, device=device),
        "logp_old": torch.as_tensor(np.asarray(logp_buf), dtype=torch.float32, device=device),
        "adv": torch.as_tensor(advantages, dtype=torch.float32, device=device),
        "ret": torch.as_tensor(returns, dtype=torch.float32, device=device),
    }
    return batch, ep_returns, obs_t


# ---------------------------------------------------------------------------
# PPO 更新
# ---------------------------------------------------------------------------
def ppo_update(net, optimizer, batch, args):
    obs, act = batch["obs"], batch["act"]
    logp_old, adv, ret = batch["logp_old"], batch["adv"], batch["ret"]
    N = obs.shape[0]
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    stats = {"approx_kl": 0.0, "clip_frac": 0.0, "entropy": 0.0, "n_updates": 0}
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
            loss = policy_loss + args.vf_coef * value_loss - args.ent_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            optimizer.step()

            with torch.no_grad():
                stats["approx_kl"] += (logp_old[mb] - logp).mean().item()
                stats["clip_frac"] += ((ratio - 1.0).abs() > args.clip).float().mean().item()
                stats["entropy"] += entropy.item()
                stats["n_updates"] += 1

    for k in stats:
        if k != "n_updates":
            stats[k] /= max(stats["n_updates"], 1)
    return stats


# ---------------------------------------------------------------------------
# 鲁棒性评估
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_noise(env, net, episodes, device, sigma, rng):
    """在观测上叠加高斯噪声，检验统计意义上的鲁棒性。"""
    returns = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            obs_t = to_tensor(obs, device)
            if sigma > 0.0:
                noise = rng.normal(0.0, sigma, size=obs_t.shape).astype(np.float32)
                obs_t = obs_t + torch.as_tensor(noise, device=device)
            logits, _ = net(obs_t)
            action = int(torch.argmax(logits, dim=-1).item())
            obs, r, terminated, truncated, _ = env.step(action)
            ep_ret += r
            done = terminated or truncated
        returns.append(ep_ret)
    return float(np.mean(returns))


def evaluate_adversarial(env, net, episodes, device, eps):
    """FGSM 风格对抗扰动：沿 −∇_s log π(a_greedy) 方向走一步，降低贪心动作概率。

    注意：这里评估的是"白盒最坏情况"——攻击者知道策略参数与梯度。
    """
    returns = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            obs_t.requires_grad_(True)
            logits, _ = net(obs_t)
            with torch.no_grad():
                greedy = int(torch.argmax(logits, dim=-1).item())
            loss = -F.log_softmax(logits, dim=-1)[0, greedy]   # 让贪心动作概率下降
            grad = torch.autograd.grad(loss, obs_t)[0]
            with torch.no_grad():
                obs_adv = obs_t + eps * torch.sign(grad)
                logits_adv, _ = net(obs_adv)
                action = int(torch.argmax(logits_adv, dim=-1).item())
            obs, r, terminated, truncated, _ = env.step(action)
            ep_ret += r
            done = terminated or truncated
        returns.append(ep_ret)
    return float(np.mean(returns))


# ---------------------------------------------------------------------------
# 训练一个 arm
# ---------------------------------------------------------------------------
def train_one(arm: str, args):
    set_seed(args.seed)
    device = torch.device("cpu")
    env = gym.make("CartPole-v1")
    obs, _ = env.reset(seed=args.seed)
    net = ActorCritic(env.observation_space.shape[0], env.action_space.n)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    rng = np.random.default_rng(args.seed + 1)
    train_noise = args.train_noise if arm == "noisy" else 0.0

    all_returns, kls = [], []
    total_steps, n_updates, obs_t = 0, 0, to_tensor(obs, device)
    t0 = time.time()

    print(f"\n[开始] arm={arm} | 训练噪声 σ={train_noise}")
    while total_steps < args.max_steps:
        batch, ep_returns, obs_t = collect_rollout(
            env, net, args.steps_per_update, obs_t, device, args.gamma, args.lam,
            train_noise, rng)
        total_steps += args.steps_per_update
        n_updates += 1
        all_returns.extend(ep_returns)

        stats = ppo_update(net, optimizer, batch, args)
        kls.append(stats["approx_kl"])

        if n_updates % 5 == 0 and len(all_returns) >= 20:
            print(f"  [{arm:5}] 步数 {total_steps:7d} | 回合 {len(all_returns):4d} | "
                  f"近20均 {np.mean(all_returns[-20:]):7.1f} | KL {stats['approx_kl']:.4f} | "
                  f"熵 {stats['entropy']:.3f}")

    return {
        "arm": arm,
        "net": net,
        "train_returns": all_returns,
        "mean_kl": float(np.mean(kls)),
        "seconds": time.time() - t0,
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="第014章：PPO 对抗性扰动鲁棒性")
    p.add_argument("--arms", type=str, nargs="+", default=["plain", "noisy"],
                   choices=["plain", "noisy"], help="plain=标准训练；noisy=观测噪声训练")
    p.add_argument("--train-noise", type=float, default=0.05, help="noisy 臂的训练噪声 σ")
    p.add_argument("--max-steps", type=int, default=80000, help="每个 arm 的训练步数")
    p.add_argument("--steps-per-update", type=int, default=2048, help="每次更新前采样步数")
    p.add_argument("--epochs", type=int, default=10, help="每批数据的更新轮数")
    p.add_argument("--batch-size", type=int, default=64, help="minibatch 大小")
    p.add_argument("--lr", type=float, default=3e-4, help="学习率")
    p.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
    p.add_argument("--lam", type=float, default=0.95, help="GAE 的 lambda")
    p.add_argument("--clip", type=float, default=0.2, help="PPO 裁剪范围")
    p.add_argument("--ent-coef", type=float, default=0.01, help="熵正则系数")
    p.add_argument("--vf-coef", type=float, default=0.5, help="价值损失系数")
    p.add_argument("--eval-episodes", type=int, default=10, help="每个档位的评估回合数")
    p.add_argument("--seed", type=int, default=0, help="随机种子")
    p.add_argument("--save-dir", type=str, default="runs/ch014", help="模型保存目录")
    p.add_argument("--quick", action="store_true", help="快速模式：2 个 arm、各 1 万步、少量评估")
    return p.parse_args()


def main():
    args = parse_args()
    noise_levels = [0.0, 0.01, 0.03, 0.05, 0.1, 0.2]
    adv_levels = [0.0, 0.01, 0.02, 0.05, 0.1]
    if args.quick:
        args.max_steps = 10000
        args.steps_per_update = 512
        args.epochs = 4
        args.eval_episodes = 5
        noise_levels = [0.0, 0.05, 0.1]
        adv_levels = [0.0, 0.05]

    print("=" * 96)
    print(f"对抗鲁棒性体检 | arms {args.arms} | 训练 {args.max_steps} 步 | "
          f"噪声训练 σ={args.train_noise} | 种子 {args.seed}")
    print("=" * 96)

    results = []
    for arm in args.arms:
        res = train_one(arm, args)
        results.append(res)
        train_tail = res["train_returns"][-100:] if len(res["train_returns"]) >= 100 else res["train_returns"]
        print(f"[完成] {arm}: 训练近100均 {np.mean(train_tail):.1f} | 用时 {res['seconds']:.1f}s")

    # 体检
    report = {}
    for res in results:
        arm = res["arm"]
        net = res["net"]
        rng = np.random.default_rng(args.seed + 99)
        noise_curve = []
        for sigma in noise_levels:
            r = evaluate_noise(gym.make("CartPole-v1"), net, args.eval_episodes, torch.device("cpu"),
                               sigma, rng)
            noise_curve.append((sigma, r))
        adv_curve = []
        for eps in adv_levels:
            r = evaluate_adversarial(gym.make("CartPole-v1"), net, args.eval_episodes,
                                     torch.device("cpu"), eps)
            adv_curve.append((eps, r))
        report[arm] = {"noise": noise_curve, "adv": adv_curve,
                       "train_tail": float(np.mean(train_tail))}

        os.makedirs(args.save_dir, exist_ok=True)
        torch.save(net.state_dict(), os.path.join(args.save_dir, f"{arm}.pt"))

    print("\n" + "=" * 96)
    print("高斯观测噪声下的回报（贪婪策略）")
    print("-" * 96)
    header = "arm    | 训练近100 | " + " | ".join(f"σ={s:<4}" for s, _ in report[results[0]['arm']]["noise"])
    print(header)
    for arm, rep in report.items():
        cells = " | ".join(f"{r:7.1f}" for _, r in rep["noise"])
        print(f"{arm:6} | {rep['train_tail']:8.1f} | {cells}")

    print("\n对抗扰动（FGSM，ε 为扰动上界）下的回报")
    print("-" * 96)
    print("arm    | " + " | ".join(f"ε={e:<5}" for e, _ in report[results[0]['arm']]["adv"]))
    for arm, rep in report.items():
        cells = " | ".join(f"{r:7.1f}" for _, r in rep["adv"])
        print(f"{arm:6} | {cells}")

    print("\n韧性指标（相对各自的干净成绩）：")
    for arm, rep in report.items():
        clean = rep["noise"][0][1]
        half = None
        for s, r in rep["noise"]:
            if clean - r >= 100 and half is None:
                half = s
        adv_break = None
        for e, r in rep["adv"]:
            if clean - r >= 100 and adv_break is None:
                adv_break = e
        print(f"  {arm:6} 干净 {clean:6.1f} | 噪声损失 100 分对应 σ≈{half} | "
              f"对抗损失 100 分对应 ε≈{adv_break}")

    os.makedirs(args.save_dir, exist_ok=True)
    path = os.path.join(args.save_dir, "robustness.txt")
    with open(path, "w", encoding="utf-8") as f:
        for arm, rep in report.items():
            f.write(f"{arm} train_tail={rep['train_tail']:.2f}\n")
            for s, r in rep["noise"]:
                f.write(f"  noise sigma={s} return={r:.2f}\n")
            for e, r in rep["adv"]:
                f.write(f"  adv eps={e} return={r:.2f}\n")
    print(f"\n结果已写入 {path}")


if __name__ == "__main__":
    main()
