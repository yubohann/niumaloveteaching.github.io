"""
第022章 PPO的探索策略：动作空间噪声

在同一个任务（CartPole-v1）上对比三种"动作层"的探索方案（同种子、同预算）：
  - stochastic ：PPO 默认方案，直接用策略的 softmax 分布采样动作
  - egreedy    ：ε-贪心覆盖——以概率 ε_t 把动作替换成均匀随机动作
  - logitnoise ：往动作 logits 上加高斯噪声再采样，噪声强度 σ_t 线性退火

关键约定：无论动作是怎么"选"出来的，存进 rollout 的 logp_old 一律是
"干净策略 π_θ 对该动作的对数概率"。这样 PPO 的比率仍然以策略为基准，
外部噪声只改变行为分布（behavior policy），不破坏重要性比率的定义。

诊断量：
  - 偏离率：执行动作不等于干净策略 argmax 的比例（三臂可比）
  - 状态覆盖：把 4 维观测离散成 6^4 桶，统计训练中累计被访问过的桶数
  - 首达 400 分所需环境步数、最终滑窗均值、评估均值

运行：
    python code/ch022.py            # 三个探索臂各 8 万步（CPU 约 2-5 分钟）
    python code/ch022.py --quick    # 快速跑通（约 30-60 秒）
    python code/ch022.py --arms stochastic,egreedy
    python code/ch022.py --eps-start 0.5 --anneal-frac 0.8

预期：CartPole 上三臂通常都能达标；外部噪声臂的"偏离率"明显更高、
早期覆盖更广，但收敛速度不一定占优——这正说明动作噪声是"广撒网"，
需要配合退火才不拖后腿（第 023 章把噪声与熵正则化放在一起系统对比）。
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
# 网络：共享躯干 + 策略头 + 价值头（与第 021 章一致）
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
# 动作选择：三种探索臂
# ---------------------------------------------------------------------------
@torch.no_grad()
def sample_action(net, obs_t, arm: str, level: float, n_actions: int):
    """按指定探索臂选动作。

    返回 (action, logp, value, greedy_action)：
        logp 始终是"干净策略"对所选动作的对数概率（PPO 比率的基准）；
        greedy_action 是干净策略的 argmax，用于统计偏离率。
    """
    logits, value = net(obs_t)
    greedy = int(torch.argmax(logits, dim=-1).item())

    if arm == "logitnoise" and level > 0.0:
        # 在 logits 上加高斯噪声：等价于给 softmax 做随机"抖动"，
        # σ 越大采样越接近均匀分布
        noisy_logits = logits + level * torch.randn_like(logits)
        action = Categorical(logits=noisy_logits).sample()
    elif arm == "egreedy" and random.random() < level:
        # 以概率 ε 直接换成均匀随机动作，绕开策略分布
        action = torch.tensor([np.random.randint(n_actions)], device=logits.device)
    else:
        # 默认：按策略自身的 softmax 分布采样
        action = Categorical(logits=logits).sample()

    logp = Categorical(logits=logits).log_prob(action)
    return int(action.item()), logp.item(), value.item(), greedy


# ---------------------------------------------------------------------------
# GAE：广义优势估计（与第 001 章一致）
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
# 采样：行为分布由探索臂决定，logp_old 永远相对干净策略
# ---------------------------------------------------------------------------
def collect_rollout(env, net, arm, level, steps, obs_t, device, gamma, lam, n_actions):
    """与环境交互 steps 步，返回训练 batch、完成的回合回报、偏离率、覆盖桶集合。"""
    obs_buf, act_buf, logp_buf = [], [], []
    rew_buf, val_buf, term_buf = [], [], []
    ep_returns = []
    n_deviate = 0

    ep_ret = 0.0
    obs_after = obs_t
    for _ in range(steps):
        a, logp, v, greedy = sample_action(net, obs_t, arm, level, n_actions)
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
    cover_bins = obs_to_bins(obs_array)
    batch = {
        "obs": torch.as_tensor(obs_array, dtype=torch.float32, device=device),
        "act": torch.as_tensor(np.asarray(act_buf), dtype=torch.long, device=device),
        "logp_old": torch.as_tensor(np.asarray(logp_buf), dtype=torch.float32, device=device),
        "adv": torch.as_tensor(advantages, dtype=torch.float32, device=device),
        "ret": torch.as_tensor(returns, dtype=torch.float32, device=device),
    }
    deviate_rate = n_deviate / float(steps)
    return batch, ep_returns, obs_t, deviate_rate, cover_bins


# ---------------------------------------------------------------------------
# PPO 更新：裁剪目标 + 多轮小批量（与第 001 章一致）
# ---------------------------------------------------------------------------
def ppo_update(net, optimizer, batch, args):
    """在收集到的 batch 上做 epochs 轮小批量更新，返回平均诊断指标。"""
    obs, act = batch["obs"], batch["act"]
    logp_old, adv, ret = batch["logp_old"], batch["adv"], batch["ret"]
    N = obs.shape[0]

    # 优势函数标准化：降低梯度方差
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
             "approx_kl": 0.0, "clip_frac": 0.0, "n_updates": 0}

    for _ in range(args.epochs):
        idx = torch.randperm(N, device=obs.device)
        for start in range(0, N, args.batch_size):
            mb = idx[start:start + args.batch_size]
            logits, value = net(obs[mb])
            dist = Categorical(logits=logits)
            logp = dist.log_prob(act[mb])

            # 新旧策略的概率比：分母是采样时存下的干净策略 logp
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
# 评估：用确定性（贪婪）策略跑若干回合
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
# 单个探索臂的完整训练流程
# ---------------------------------------------------------------------------
def run_arm(arm: str, args, device):
    """训练一个探索臂，返回该臂的汇总结果字典。"""
    set_seed(args.seed)  # 每个臂用同一初始条件，公平对比

    env = gym.make("CartPole-v1")
    obs, _ = env.reset(seed=args.seed)
    n_actions = env.action_space.n

    net = ActorCritic(env.observation_space.shape[0], n_actions).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    obs_t = to_tensor(obs, device)
    all_returns = []
    dev_iter = []          # 每次 rollout 的偏离率
    cover_all = set()      # 训练中访问过的覆盖桶
    total_steps = 0
    update_idx = 0
    solved_at = None
    t_start = time.time()

    while total_steps < args.max_steps and solved_at is None:
        # 1) 计算当前噪声/ε 水平并采样
        progress = total_steps / max(1, args.max_steps)
        if arm == "egreedy":
            level = anneal_level(progress, args.eps_start, args.eps_end, args.anneal_frac)
        elif arm == "logitnoise":
            level = anneal_level(progress, args.sigma_start, args.sigma_end, args.anneal_frac)
        else:
            level = 0.0

        batch, ep_returns, obs_t, deviate_rate, cover_bins = collect_rollout(
            env, net, arm, level, args.steps_per_update, obs_t, device,
            args.gamma, args.lam, n_actions)
        total_steps += args.steps_per_update
        all_returns.extend(ep_returns)
        dev_iter.append(deviate_rate)
        cover_all |= cover_bins

        # 2) 更新
        stats = ppo_update(net, optimizer, batch, args)
        update_idx += 1

        # 3) 日志
        if update_idx % args.log_every == 0 or update_idx == 1:
            recent = all_returns[-20:] if all_returns else [0.0]
            level_name = "ε" if arm == "egreedy" else ("σ" if arm == "logitnoise" else "-")
            print(f"  [{arm:10s}] 更新 {update_idx:3d} | 步数 {total_steps:7d} | "
                  f"回合 {len(all_returns):4d} | 近20均 {np.mean(recent):6.1f} | "
                  f"偏离率 {deviate_rate:.3f} | 覆盖桶 {len(cover_all):4d} | "
                  f"KL {stats['approx_kl']:.4f} | 熵 {stats['entropy']:.3f} | "
                  f"{level_name} {level:.3f}")

        # 4) 达到目标提前停（节约预算）
        if solved_at is None and len(all_returns) >= 100 and \
                np.mean(all_returns[-100:]) >= args.target_reward:
            solved_at = total_steps

    elapsed = time.time() - t_start

    # 评估：确定性策略
    eval_env = gym.make("CartPole-v1")
    eval_returns = evaluate(eval_env, net, args.eval_episodes, device)

    # 保存该臂模型
    os.makedirs(args.save_dir, exist_ok=True)
    torch.save(net.state_dict(), os.path.join(args.save_dir, f"{arm}.pt"))

    last100 = float(np.mean(all_returns[-100:])) if all_returns else 0.0
    result = {
        "arm": arm,
        "steps": total_steps,
        "episodes": len(all_returns),
        "solved_at": solved_at,
        "last100": last100,
        "eval_mean": float(np.mean(eval_returns)),
        "deviate": float(np.mean(dev_iter)) if dev_iter else 0.0,
        "coverage": len(cover_all),
        "time": elapsed,
    }
    print(f"  [{arm:10s}] 结束：总步数 {total_steps} | 用时 {elapsed:.1f}s | "
          f"首达{args.target_reward:.0f} "
          f"{('步数 ' + str(solved_at)) if solved_at else '未达成'} | "
          f"近100均 {last100:.1f} | 评估 {np.mean(eval_returns):.1f} | "
          f"平均偏离率 {result['deviate']:.3f} | 覆盖桶 {len(cover_all)}")
    env.close()
    eval_env.close()
    return result


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="第022章：PPO 的动作空间噪声探索对比")
    p.add_argument("--arms", type=str, default="stochastic,egreedy,logitnoise",
                   help="要运行的探索臂，逗号分隔（stochastic / egreedy / logitnoise）")
    p.add_argument("--max-steps", type=int, default=80000, help="每个臂的环境步数上限")
    p.add_argument("--steps-per-update", type=int, default=2048, help="每次更新前采样的步数")
    p.add_argument("--epochs", type=int, default=10, help="每批数据的更新轮数")
    p.add_argument("--batch-size", type=int, default=64, help="小批量大小")
    p.add_argument("--lr", type=float, default=3e-4, help="学习率")
    p.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
    p.add_argument("--lam", type=float, default=0.95, help="GAE 的 lambda")
    p.add_argument("--clip", type=float, default=0.2, help="PPO 裁剪范围")
    p.add_argument("--ent-coef", type=float, default=0.01, help="熵正则系数")
    p.add_argument("--vf-coef", type=float, default=0.5, help="价值损失系数")
    p.add_argument("--eps-start", type=float, default=0.3, help="ε-贪心的初始 ε")
    p.add_argument("--eps-end", type=float, default=0.02, help="ε-贪心的最终 ε")
    p.add_argument("--sigma-start", type=float, default=1.0, help="logits 噪声的初始 σ")
    p.add_argument("--sigma-end", type=float, default=0.0, help="logits 噪声的最终 σ")
    p.add_argument("--anneal-frac", type=float, default=0.6,
                   help="噪声水平在前百分之多少的训练预算内线性退火")
    p.add_argument("--log-every", type=int, default=5, help="每隔多少次更新打印一次日志")
    p.add_argument("--target-reward", type=float, default=475.0, help="提前停的滑窗平均回报")
    p.add_argument("--eval-episodes", type=int, default=10, help="每个臂结束后的评估回合数")
    p.add_argument("--seed", type=int, default=0, help="随机种子")
    p.add_argument("--save-dir", type=str, default="runs/ch022", help="模型保存目录")
    p.add_argument("--quick", action="store_true", help="快速模式：缩小训练规模，几十秒跑完")
    return p.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.max_steps = min(args.max_steps, 12000)
        args.steps_per_update = 768
        args.epochs = 5
        args.eval_episodes = 3

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    valid = {"stochastic", "egreedy", "logitnoise"}
    arms = [a for a in arms if a in valid]
    if not arms:
        raise SystemExit("没有可运行的探索臂，请检查 --arms 参数")

    device = torch.device("cpu")  # 纯 CPU，保证任何机器可复现
    print(f"设备: {device} | 环境: CartPole-v1 | 种子: {args.seed}")
    print(f"配置: arms={arms}, max_steps={args.max_steps}, "
          f"steps_per_update={args.steps_per_update}, epochs={args.epochs}, "
          f"anneal_frac={args.anneal_frac}")

    results = []
    for arm in arms:
        print("-" * 78)
        print(f"开始训练探索臂: {arm}")
        results.append(run_arm(arm, args, device))

    # 汇总对比表
    print("=" * 78)
    print(f"{'arm':<12}{'首达目标步数':>12}{'近100均':>10}{'评估均':>10}"
          f"{'平均偏离率':>12}{'覆盖桶':>8}{'用时(s)':>10}")
    for r in results:
        solved = str(r["solved_at"]) if r["solved_at"] else "未达成"
        print(f"{r['arm']:<12}{solved:>12}{r['last100']:>10.1f}"
              f"{r['eval_mean']:>10.1f}{r['deviate']:>12.3f}"
              f"{r['coverage']:>8d}{r['time']:>10.1f}")
    print("=" * 78)
    best = max(results, key=lambda r: r["eval_mean"])
    print(f"评估均值最高的探索臂：{best['arm']}（{best['eval_mean']:.1f}）")
    print("提示：CartPole 过于简单，单次运行的臂间差异多为噪声；"
          "请关注偏离率与覆盖桶这两列的稳定差异，并多跑几个种子验证结论。")


if __name__ == "__main__":
    main()
