"""
第010章 PPO的早停与模型保存策略

训练不只是"跑到最后"，还包括两个交付层面的决策：什么时候停止、保存哪个模型。
本章在同一预算（12 万步）与同一随机种子下对比三种策略：

  1) baseline ：不做 KL 早停，保存最终模型
  2) klstop   ：每个 epoch 后若平均 KL 超过 1.5×目标 KL，立即停止本批剩余 epoch
                （Schulman 推荐的"KL 早停"），保存最终模型
  3) ckpt     ：不早停，但每 5 次更新用贪婪策略评估一次，保存评估最优的检查点

所有策略共享"100 回合滑窗 ≥ 475 即停止训练"的样本效率早停；
评估轨迹、首达步数、最终/最优评估、KL 峰值与耗时全部记录。

运行：
    python code/ch010.py            # 3 种策略，各 12 万步上限，CPU 约 4-8 分钟
    python code/ch010.py --quick    # 快速跑通（约 30-60 秒）
    python code/ch010.py --strategies baseline ckpt

预期：klstop 减少 KL 尖峰与无效更新，通常更快达标且更省时间；
ckpt 的"最优检查点"往往比最终模型高 5-40 分，尤其在训练后期出现波动时。
"""

import argparse
import copy
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
    """固定随机种子，保证三种策略从同一条随机流出发。"""
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
# 采样
# ---------------------------------------------------------------------------
def collect_rollout(env, net, steps, obs_t, device, gamma, lam):
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
# PPO 更新：支持 KL 早停（每个 epoch 汇总一次 KL）
# ---------------------------------------------------------------------------
def ppo_update(net, optimizer, batch, args, kl_stop: bool):
    """kl_stop=True 时，epoch 平均 KL 超过阈值就停止本批剩余 epoch。

    返回 (stats, kl_breaks)：kl_breaks 为本批触发早停的次数（0 或 1）。
    """
    obs, act = batch["obs"], batch["act"]
    logp_old, adv, ret = batch["logp_old"], batch["adv"], batch["ret"]
    N = obs.shape[0]
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    threshold = args.kl_stop_mult * args.target_kl
    stats = {"approx_kl": 0.0, "clip_frac": 0.0, "entropy": 0.0,
             "max_kl": 0.0, "n_updates": 0}
    kl_breaks = 0
    for epoch_i in range(args.epochs):
        epoch_kl_sum, epoch_cnt = 0.0, 0
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
                kl = (logp_old[mb] - logp).mean().item()
                stats["approx_kl"] += kl
                stats["max_kl"] = max(stats["max_kl"], abs(kl))
                stats["clip_frac"] += ((ratio - 1.0).abs() > args.clip).float().mean().item()
                stats["entropy"] += entropy.item()
                stats["n_updates"] += 1
            epoch_kl_sum += kl
            epoch_cnt += 1

        if kl_stop and epoch_i < args.epochs - 1:
            mean_epoch_kl = epoch_kl_sum / max(epoch_cnt, 1)
            if mean_epoch_kl > threshold:
                kl_breaks += 1
                break

    for k in stats:
        if k != "n_updates":
            stats[k] /= max(stats["n_updates"], 1)
    return stats, kl_breaks


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
# 训练一种策略
# ---------------------------------------------------------------------------
def train_one(strategy: str, args):
    set_seed(args.seed)
    device = torch.device("cpu")
    env = gym.make("CartPole-v1")
    reward_env = gym.make("CartPole-v1")     # 周期评估专用环境
    obs, _ = env.reset(seed=args.seed)
    net = ActorCritic(env.observation_space.shape[0], env.action_space.n)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    kl_stop = (strategy == "klstop")
    use_ckpt = (strategy == "ckpt")

    all_returns = []
    kls_max, breaks_total = [], 0
    eval_history = []          # (步数, 平均评估回报)
    best_eval, best_state, best_step = -np.inf, None, None
    total_steps, n_updates, obs_t = 0, 0, to_tensor(obs, device)
    solved_at = None
    t0 = time.time()

    print(f"\n[开始] 策略 = {strategy}"
          + ("（KL 早停开启）" if kl_stop else "")
          + ("（周期评估 + 最优保存）" if use_ckpt else ""))
    while total_steps < args.max_steps:
        batch, ep_returns, obs_t = collect_rollout(
            env, net, args.steps_per_update, obs_t, device, args.gamma, args.lam)
        total_steps += args.steps_per_update
        n_updates += 1
        all_returns.extend(ep_returns)

        stats, breaks = ppo_update(net, optimizer, batch, args, kl_stop)
        kls_max.append(stats["max_kl"])
        breaks_total += breaks

        # 周期评估：贪婪策略跑 eval_every 个回合
        if n_updates % args.eval_every == 0 or total_steps >= args.max_steps:
            ev_ret = float(np.mean(evaluate(reward_env, net, args.checkpoint_episodes, device)))
            eval_history.append((total_steps, ev_ret))
            if use_ckpt and ev_ret > best_eval:
                best_eval = ev_ret
                best_step = total_steps
                best_state = copy.deepcopy(net.state_dict())

        if solved_at is None and len(all_returns) >= 100 and np.mean(all_returns[-100:]) >= args.target_reward:
            solved_at = total_steps
            print(f"  [{strategy}] 达到 {args.target_reward}：步数 {solved_at}")
            if args.stop_when_solved:
                print(f"  [{strategy}] 触发早停，训练结束")
                break

        if n_updates % 5 == 0:
            print(f"  [{strategy:8}] 步数 {total_steps:7d} | 回合 {len(all_returns):4d} | "
                  f"近20均 {np.mean(all_returns[-20:]):7.1f} | KL峰值 {stats['max_kl']:.4f} | "
                  f"更新早停次数 {breaks_total}")

    final_eval = float(np.mean(evaluate(reward_env, net, args.eval_episodes, device)))
    tail = all_returns[-100:] if len(all_returns) >= 100 else all_returns

    os.makedirs(args.save_dir, exist_ok=True)
    final_path = os.path.join(args.save_dir, f"{strategy}_final.pt")
    torch.save(net.state_dict(), final_path)
    ckpt_path = None
    if use_ckpt and best_state is not None:
        ckpt_path = os.path.join(args.save_dir, f"{strategy}_best.pt")
        torch.save(best_state, ckpt_path)

    return {
        "strategy": strategy,
        "steps": total_steps,
        "updates": n_updates,
        "solved_at": solved_at,
        "last100": float(np.mean(tail)),
        "final_eval": final_eval,
        "best_eval": float(best_eval) if use_ckpt and best_state is not None
        else (max(v for _, v in eval_history) if eval_history else final_eval),
        "best_step": best_step,
        "kl_breaks": breaks_total,
        "mean_max_kl": float(np.mean(kls_max)),
        "eval_history": eval_history,
        "seconds": time.time() - t0,
        "final_path": final_path,
        "ckpt_path": ckpt_path,
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="第010章：PPO 早停与模型保存策略")
    p.add_argument("--strategies", type=str, nargs="+",
                   default=["baseline", "klstop", "ckpt"],
                   choices=["baseline", "klstop", "ckpt"], help="要对比的训练策略")
    p.add_argument("--max-steps", type=int, default=120000, help="每种策略的环境步数上限")
    p.add_argument("--steps-per-update", type=int, default=2048, help="每次更新前采样步数")
    p.add_argument("--epochs", type=int, default=10, help="每批数据的更新轮数上限")
    p.add_argument("--batch-size", type=int, default=64, help="minibatch 大小")
    p.add_argument("--lr", type=float, default=3e-4, help="学习率")
    p.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
    p.add_argument("--lam", type=float, default=0.95, help="GAE 的 lambda")
    p.add_argument("--clip", type=float, default=0.2, help="PPO 裁剪范围")
    p.add_argument("--ent-coef", type=float, default=0.01, help="熵正则系数")
    p.add_argument("--vf-coef", type=float, default=0.5, help="价值损失系数")
    p.add_argument("--target-kl", type=float, default=0.02, help="KL 早停的目标 KL")
    p.add_argument("--kl-stop-mult", type=float, default=1.5, help="KL 早停阈值 = 目标 KL × 该倍数")
    p.add_argument("--target-reward", type=float, default=475.0, help="样本效率早停的达标阈值")
    p.add_argument("--stop-when-solved", action="store_true", default=True,
                   help="达标后停止训练（默认开）")
    p.add_argument("--no-stop-when-solved", dest="stop_when_solved", action="store_false",
                   help="达标后继续跑满预算")
    p.add_argument("--eval-every", type=int, default=5, help="每隔多少次更新做一次周期评估")
    p.add_argument("--checkpoint-episodes", type=int, default=5, help="周期评估的回合数")
    p.add_argument("--eval-episodes", type=int, default=20, help="结束时评估回合数")
    p.add_argument("--seed", type=int, default=0, help="随机种子")
    p.add_argument("--save-dir", type=str, default="runs/ch010", help="模型与结果保存目录")
    p.add_argument("--quick", action="store_true", help="快速模式：8 千步、达标 300、关闭达标早停")
    return p.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.max_steps = 8000
        args.steps_per_update = 512
        args.epochs = 4
        args.target_reward = 300.0
        args.stop_when_solved = False
        args.eval_every = 2

    print("=" * 96)
    print(f"早停与模型保存策略 | 策略 {args.strategies} | 预算 {args.max_steps} 步 | 种子 {args.seed}")
    print(f"KL 早停阈值 = {args.kl_stop_mult} × {args.target_kl} = "
          f"{args.kl_stop_mult * args.target_kl:.4f} | 达标阈值 {args.target_reward} | "
          f"达标即停 {args.stop_when_solved}")
    print("=" * 96)

    results = []
    for s in args.strategies:
        res = train_one(s, args)
        results.append(res)
        print(f"[完成] {s}: 首达 {res['solved_at']} | 最终评估 {res['final_eval']:.1f} | "
              f"最佳评估 {res['best_eval']:.1f} | KL 早停 {res['kl_breaks']} 次 | "
              f"用时 {res['seconds']:.1f}s")

    print("\n" + "=" * 96)
    print("策略对比总结（同种子、同步数上限）")
    print("-" * 96)
    print(f"{'策略':>9} | {'总步数':>7} | {'更新数':>6} | {'首达475步数':>11} | {'近100均':>8} | "
          f"{'最终评估':>8} | {'最优评估':>8} | {'KL早停次数':>9} | {'KL峰值均':>8} | {'用时(s)':>8}")
    for r in results:
        solved = str(r["solved_at"]) if r["solved_at"] is not None else "未达标"
        print(f"{r['strategy']:>9} | {r['steps']:>7d} | {r['updates']:>6d} | {solved:>11} | "
              f"{r['last100']:>8.1f} | {r['final_eval']:>8.1f} | {r['best_eval']:>8.1f} | "
              f"{r['kl_breaks']:>9d} | {r['mean_max_kl']:>8.4f} | {r['seconds']:>8.1f}")

    print("\n评估轨迹（步数: 平均回报）")
    for r in results:
        pts = " ".join(f"{s}:{v:.0f}" for s, v in r["eval_history"])
        print(f"  {r['strategy']:>9}  {pts}")

    print("\n保存的模型：")
    for r in results:
        print(f"  {r['strategy']:>9} 最终模型 -> {r['final_path']}")
        if r["ckpt_path"]:
            print(f"  {r['strategy']:>9} 最优检查点 -> {r['ckpt_path']}（步数 {r['best_step']}）")

    os.makedirs(args.save_dir, exist_ok=True)
    path = os.path.join(args.save_dir, "early_stop_summary.txt")
    with open(path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(f"strategy={r['strategy']} steps={r['steps']} solved_at={r['solved_at']} "
                    f"final_eval={r['final_eval']:.2f} best_eval={r['best_eval']:.2f} "
                    f"kl_breaks={r['kl_breaks']} seconds={r['seconds']:.1f} "
                    f"final={r['final_path']} ckpt={r['ckpt_path']}\n")
    print(f"\n结果已写入 {path}")


if __name__ == "__main__":
    main()
