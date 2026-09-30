"""
第027章 PPO的样本效率分析

同一条 PPO 管道，五种"数据使用策略"，回答一个问题：
每单位环境步数到底买到了多少学习进展？五个配置（采样步数:轮数）：

    spu2048_e1   : 2048 步 / 1 epoch   —— 几乎不重复使用数据
    spu2048_e5   : 2048 步 / 5 epochs  —— 常用的数据复用
    spu2048_e20  : 2048 步 / 20 epochs —— 过度复用
    spu512_e10   : 512 步 / 10 epochs  —— 小步快跑（更新频繁）
    spu4096_e10  : 4096 步 / 10 epochs —— 大步慢跑（更新稀少）

所有配置共享 --max-steps 环境步预算，逐次更新记录：
  - 学习曲线的 AUC（曲线下面积，按环境步数归一，越高=样本效率越好）
  - 首达目标步数、末段训练回报、贪婪评估回报
  - 累计梯度步数（计算成本轴）与"每环境步的梯度步数"
  - 每次更新内"第 1 个 epoch 的 KL vs 最后一个 epoch 的 KL"
    ——衡量数据复用带来的 off-policy 漂移

运行：
    python code/ch027.py            # 5 个配置各 5 万步（CPU 约 3-6 分钟）
    python code/ch027.py --quick    # 快速跑通（约 40-80 秒）
    python code/ch027.py --configs 2048:10,512:10
    python code/ch027.py --configs 2048:10 --batch-size 32

预期：小批量频繁更新（512:10）通常首达更快、AUC 更高但 KL 波动更大；
1 epoch 的配置因为每批数据只用一遍，整体最慢；20 epochs 把计算量翻
十几倍，AUC 却往往提升有限——这就是"样本效率"与"计算效率"的分野。
数值区间以实跑为准。
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
from torch.distributions import Categorical


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


def curve_auc(xs, ys, x_max: float) -> float:
    """学习曲线的梯形积分 / x 总长度（手工实现，避免 numpy 版本差异）。"""
    if len(xs) < 2:
        return 0.0
    area = 0.0
    for i in range(1, len(xs)):
        area += 0.5 * (ys[i] + ys[i - 1]) * (xs[i] - xs[i - 1])
    return area / max(x_max, 1e-9)


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
# 采样
# ---------------------------------------------------------------------------
def collect_rollout(env, net, steps, obs_t, device, gamma, lam):
    """与环境交互 steps 步，返回训练 batch、完成的回合回报、最新观测。"""
    obs_buf, act_buf, logp_buf = [], [], []
    rew_buf, val_buf, term_buf = [], [], []
    ep_returns = []

    ep_ret = 0.0
    obs_after = obs_t
    for _ in range(steps):
        a, logp, v = net.act(obs_t)
        next_obs, r, terminated, truncated, _ = env.step(a)

        obs_buf.append(obs_t.squeeze(0).cpu().numpy())
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
# PPO 更新：记录每个 epoch 的 KL（用于测量数据复用漂移）
# ---------------------------------------------------------------------------
def ppo_update(net, optimizer, batch, args, epochs: int):
    """做 epochs 轮小批量更新，返回平均诊断与逐 epoch 的 KL 序列。"""
    obs, act = batch["obs"], batch["act"]
    logp_old, adv, ret = batch["logp_old"], batch["adv"], batch["ret"]
    N = obs.shape[0]
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    sum_kl = sum_clip = sum_entropy = sum_vloss = 0.0
    n_mb = 0
    epoch_kls = []

    for _ in range(epochs):
        epoch_kl, epoch_mb = 0.0, 0
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
                approx_kl = (logp_old[mb] - logp).mean().item()
                clip_frac = ((ratio - 1.0).abs() > args.clip).float().mean().item()
            sum_kl += approx_kl
            sum_clip += clip_frac
            sum_entropy += entropy.item()
            sum_vloss += value_loss.item()
            epoch_kl += approx_kl
            epoch_mb += 1
            n_mb += 1
        epoch_kls.append(epoch_kl / max(epoch_mb, 1))

    d = max(n_mb, 1)
    return {
        "approx_kl": sum_kl / d,
        "clip_frac": sum_clip / d,
        "entropy": sum_entropy / d,
        "value_loss": sum_vloss / d,
        "grad_steps": n_mb,          # 本次更新实际执行的梯度步数
        "epoch_kls": epoch_kls,
    }


# ---------------------------------------------------------------------------
# 评估
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
# 单配置训练
# ---------------------------------------------------------------------------
def run_config(spu: int, epochs: int, args, device):
    """训练一个 (steps_per_update, epochs) 配置，返回样本效率账本。"""
    set_seed(args.seed)
    tag = f"spu{spu}_e{epochs}"

    env = gym.make("CartPole-v1")
    obs, _ = env.reset(seed=args.seed)
    net = ActorCritic(env.observation_space.shape[0], env.action_space.n).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    obs_t = to_tensor(obs, device)
    all_returns = []
    xs, ys = [], []              # 学习曲线采样点（环境步 -> 滑窗均值）
    kl_first, kl_last = [], []   # 每次更新内 epoch1 与最后一个 epoch 的 KL
    total_grad_steps = 0
    total_steps = 0
    update_idx = 0
    solved_at = None
    t_start = time.time()

    while total_steps < args.max_steps and solved_at is None:
        batch, ep_returns, obs_t = collect_rollout(
            env, net, spu, obs_t, device, args.gamma, args.lam)
        total_steps += spu
        all_returns.extend(ep_returns)

        stats = ppo_update(net, optimizer, batch, args, epochs)
        total_grad_steps += stats["grad_steps"]
        update_idx += 1

        window = all_returns[-100:] if all_returns else [0.0]
        xs.append(total_steps)
        ys.append(float(np.mean(window)))
        if len(stats["epoch_kls"]) >= 1:
            kl_first.append(stats["epoch_kls"][0])
            kl_last.append(stats["epoch_kls"][-1])

        if update_idx % args.log_every == 0 or update_idx == 1:
            print(f"  [{tag:12s}] 更新 {update_idx:3d} | 步数 {total_steps:6d} | "
                  f"近100均 {np.mean(window):6.1f} | KL {stats['approx_kl']:.4f} | "
                  f"KL(首轮) {stats['epoch_kls'][0]:.4f} | "
                  f"KL(末轮) {stats['epoch_kls'][-1]:.4f} | "
                  f"梯度步 {total_grad_steps:6d}")

        if solved_at is None and len(all_returns) >= 100 and \
                np.mean(all_returns[-100:]) >= args.target_reward:
            solved_at = total_steps

    elapsed = time.time() - t_start
    eval_env = gym.make("CartPole-v1")
    eval_returns = evaluate(eval_env, net, args.eval_episodes, device)
    eval_env.close()

    os.makedirs(args.save_dir, exist_ok=True)
    torch.save(net.state_dict(), os.path.join(args.save_dir, f"{tag}.pt"))

    auc = curve_auc(xs, ys, args.max_steps)
    last100 = float(np.mean(all_returns[-100:])) if all_returns else 0.0
    k1 = float(np.mean(kl_first)) if kl_first else 0.0
    k2 = float(np.mean(kl_last)) if kl_last else 0.0
    result = {
        "tag": tag, "spu": spu, "epochs": epochs,
        "updates": update_idx, "grad_steps": total_grad_steps,
        "grad_per_step": total_grad_steps / max(total_steps, 1),
        "kl_first": k1, "kl_last": k2, "kl_drift": k2 / max(k1, 1e-9),
        "auc": auc, "solved_at": solved_at, "last100": last100,
        "eval_mean": float(np.mean(eval_returns)), "time": elapsed,
    }
    print(f"  [{tag:12s}] 结束：步数 {total_steps} | 梯度步 {total_grad_steps} | "
          f"AUC {auc:.1f} | 首达 "
          f"{('步数 ' + str(solved_at)) if solved_at else '未达成'} | "
          f"近100均 {last100:.1f} | 评估 {np.mean(eval_returns):.1f} | "
          f"KL 漂移 {result['kl_drift']:.2f}x | 用时 {elapsed:.1f}s")
    env.close()
    return result


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="第027章：PPO 样本效率分析（数据复用与更新频率）")
    p.add_argument("--configs", type=str,
                   default="2048:1,2048:5,2048:20,512:10,4096:10",
                   help="配置列表，格式 采样步数:epochs，逗号分隔")
    p.add_argument("--max-steps", type=int, default=50000, help="每个配置的环境步数上限")
    p.add_argument("--batch-size", type=int, default=64, help="小批量大小")
    p.add_argument("--lr", type=float, default=3e-4, help="学习率")
    p.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
    p.add_argument("--lam", type=float, default=0.95, help="GAE 的 lambda")
    p.add_argument("--clip", type=float, default=0.2, help="PPO 裁剪范围")
    p.add_argument("--ent-coef", type=float, default=0.01, help="熵正则系数")
    p.add_argument("--vf-coef", type=float, default=0.5, help="价值损失系数")
    p.add_argument("--log-every", type=int, default=10, help="每隔多少次更新打印日志")
    p.add_argument("--target-reward", type=float, default=475.0, help="提前停的滑窗平均回报")
    p.add_argument("--eval-episodes", type=int, default=10, help="每个配置的评估回合数")
    p.add_argument("--seed", type=int, default=0, help="随机种子")
    p.add_argument("--save-dir", type=str, default="runs/ch027", help="模型保存目录")
    p.add_argument("--quick", action="store_true", help="快速模式：缩小训练规模，几十秒跑完")
    return p.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.max_steps = min(args.max_steps, 12000)
        args.eval_episodes = 3

    configs = []
    for token in args.configs.split(","):
        token = token.strip()
        if not token:
            continue
        spu_s, ep_s = token.split(":")
        spu, ep = int(spu_s), int(ep_s)
        if spu < args.batch_size:
            print(f"提示：配置 {token} 的采样步数小于 batch_size，已按 batch_size 处理")
            spu = args.batch_size
        if args.quick:
            spu = max(args.batch_size, spu // 2)
            ep = min(ep, 8)
        configs.append((spu, ep))
    if not configs:
        raise SystemExit("--configs 为空，请检查参数")

    device = torch.device("cpu")
    print(f"设备: {device} | 环境: CartPole-v1 | 种子: {args.seed}")
    print(f"配置: {[f'{s}:{e}' for s, e in configs]} | 每配置 {args.max_steps} 步 | "
          f"batch={args.batch_size}")

    results = []
    for spu, ep in configs:
        print("-" * 92)
        print(f"开始配置：steps_per_update={spu}, epochs={ep} "
              f"（每次更新梯度步数约 {math.ceil(spu / args.batch_size) * ep}）")
        results.append(run_config(spu, ep, args, device))

    # 样本效率账本
    print("=" * 118)
    print(f"{'配置':<14}{'更新数':>7}{'梯度步':>9}{'梯度步/环境步':>13}"
          f"{'KL首轮':>9}{'KL末轮':>9}{'漂移':>7}{'AUC':>7}{'首达':>8}{'评估均':>8}{'用时(s)':>9}")
    for r in results:
        solved = str(r["solved_at"]) if r["solved_at"] else "未达成"
        print(f"{r['tag']:<14}{r['updates']:>7}{r['grad_steps']:>9}"
              f"{r['grad_per_step']:>13.4f}{r['kl_first']:>9.4f}{r['kl_last']:>9.4f}"
              f"{r['kl_drift']:>7.2f}{r['auc']:>7.1f}{solved:>8}"
              f"{r['eval_mean']:>8.1f}{r['time']:>9.1f}")
    print("=" * 118)
    best_auc = max(results, key=lambda r: r["auc"])
    print(f"AUC 最高（样本效率最好）：{best_auc['tag']}（{best_auc['auc']:.1f}）")
    print("注意：AUC 按环境步数归一，衡量'每步数据买到多少回报'；"
          "梯度步/环境步衡量计算成本。两条轴一起看，才能判断数据复用是否划算。")


if __name__ == "__main__":
    main()
