"""
第013章 PPO的循环神经网络策略

部分可观测（POMDP）任务里，当前观测不足以决定最优动作，策略必须"记住"过去。
本章自建 CueTMaze：4 步固定长度的 T 型迷宫——
  - 第 0 步在起点显示随机线索（左/右），之后观测里线索消失
  - 第 3 步到岔路口，选择与线索一致的方向得 +1，否则得 0
  - 之后线索不可见，无记忆的策略上限是 50% 成功率

对比两种策略结构（同种子、同回合预算）：
  - mlp：普通 MLP，只能看到当前观测 → 预期成功率卡在 0.5 左右
  - gru：GRUCell 循环策略，把线索压进隐状态，更新时按"整段回合"重放序列

工程点：
  - 回合等长时的定长序列批处理（带 padding/mask 的通用写法）
  - RNN 隐状态在回合开始时重置，更新时从零状态重放整段
  - 优势/比率的计算只在有效时间步上进行（mask）

运行：
    python code/ch013.py            # 2 种策略，各 2 万回合，CPU 约 1-3 分钟
    python code/ch013.py --quick    # 快速跑通（约 20-40 秒）
    python code/ch013.py --arms gru

预期：mlp 的成功率稳定在 0.5 附近（学到"总选同一边"），
gru 的成功率随训练上升，通常在几千回合内超过 0.9。
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


# ---------------------------------------------------------------------------
# 自定义环境：T 型迷宫（线索只在起点可见）
# ---------------------------------------------------------------------------
class CueTMaze(gym.Env):
    """固定 4 步：t=0 显示线索，t=1/2 是走廊，t=3 选择方向。

    观测：[t/3, 线索]；线索仅在第 0 步可见（其余为 0）。
    动作：0=左臂（对应线索 −1），1=右臂（对应线索 +1）。
    奖励：第 3 步选择正确 +1，错误 0；前 3 步的动作为空操作。
    """

    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(2)
        self.t = 0
        self.cue = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self.cue = 1.0 if self.np_random.random() < 0.5 else -1.0
        return self._obs(), {}

    def _obs(self):
        cue_visible = self.cue if self.t == 0 else 0.0
        return np.array([self.t / 3.0, cue_visible], dtype=np.float32)

    def step(self, action):
        if self.t < 3:
            self.t += 1
            return self._obs(), 0.0, False, False, {"correct": None}
        # 第 3 步：做决定
        correct = bool(int(action) == (1 if self.cue > 0 else 0))
        self.t = 4
        r = 1.0 if correct else 0.0
        return self._obs(), r, True, False, {"correct": correct}


# ---------------------------------------------------------------------------
# 两种策略网络
# ---------------------------------------------------------------------------
class MLPActorCritic(nn.Module):
    """无记忆基线：只看当前观测。"""

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
    def step(self, obs: torch.Tensor, hidden=None):
        logits, value = self.forward(obs)
        return logits, value, None

    @torch.no_grad()
    def value(self, obs: torch.Tensor, hidden=None) -> float:
        _, v = self.forward(obs)
        return float(v.item())

    def forward_seq(self, obs: torch.Tensor):
        """(B, L, D) → (B, L, act_dim), (B, L)，与 GRU 版接口对齐。"""
        B, L, D = obs.shape
        logits, values = self.forward(obs.reshape(B * L, D))
        return (logits.reshape(B, L, -1),
                values.reshape(B, L))


class GRUActorCritic(nn.Module):
    """循环策略：GRUCell 把历史压进隐状态。"""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 32):
        super().__init__()
        self.hidden_size = hidden
        self.gru = nn.GRUCell(obs_dim, hidden)
        self.actor = nn.Linear(hidden, act_dim)
        self.critic = nn.Linear(hidden, 1)

    def init_hidden(self, batch: int, device):
        return torch.zeros(batch, self.hidden_size, device=device)

    @torch.no_grad()
    def step(self, obs: torch.Tensor, hidden: torch.Tensor):
        """单步前向（采样用，无梯度）。obs: (1, D)。"""
        h = self.gru(obs, hidden)
        logits = self.actor(h)
        value = self.critic(h).squeeze(-1)
        return logits, value, h

    @torch.no_grad()
    def value(self, obs: torch.Tensor, hidden: torch.Tensor) -> float:
        h = self.gru(obs, hidden)
        return float(self.critic(h).item())

    def forward_seq(self, obs: torch.Tensor):
        """按整段序列重放，返回 (B, L, act_dim) 的 logits 与 (B, L) 的价值。"""
        B, L, D = obs.shape
        h = self.init_hidden(B, obs.device)
        logits_list, values_list = [], []
        for t in range(L):
            h = self.gru(obs[:, t], h)
            logits_list.append(self.actor(h))
            values_list.append(self.critic(h).squeeze(-1))
        return torch.stack(logits_list, dim=1), torch.stack(values_list, dim=1)


def build_net(arm: str, obs_dim: int, act_dim: int):
    return MLPActorCritic(obs_dim, act_dim) if arm == "mlp" else GRUActorCritic(obs_dim, act_dim)


def net_step(net, arm: str, obs_t, hidden):
    """统一单步接口。"""
    if arm == "gru":
        return net.step(obs_t, hidden)
    logits, value, _ = net.step(obs_t, None)
    return logits, value, None


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
# 采集 n 个完整回合，垫成定长序列批次
# ---------------------------------------------------------------------------
def collect_episodes(env, net, arm, n_episodes, device, gamma, lam):
    episodes = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        hidden = net.init_hidden(1, device) if arm == "gru" else None

        obs_l, act_l, logp_l, rew_l, val_l, term_l = [], [], [], [], [], []
        ep_raw_return = 0.0
        done = False
        while not done:
            logits, value, hidden = net_step(net, arm, obs_t, hidden)
            dist = Categorical(logits=logits)
            action = dist.sample()
            next_obs, r, terminated, truncated, _ = env.step(action.item())

            obs_l.append(obs_t.squeeze(0).cpu().numpy())
            act_l.append(action.item())
            logp_l.append(float(dist.log_prob(action).item()))
            rew_l.append(float(r))
            val_l.append(float(value.item()))
            term_l.append(float(terminated))
            ep_raw_return += float(r)

            obs_t = torch.as_tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0)
            done = terminated or truncated

        # 回合末 bootstrap（本环境全部是真实终止，last_value 实际用不到，但保留通用写法）
        if arm == "gru":
            last_value = net.value(obs_t, hidden)
        else:
            last_value = net.value(obs_t)
        adv, ret = compute_gae(rew_l, val_l, term_l, last_value, gamma, lam)
        episodes.append({
            "obs": np.asarray(obs_l, dtype=np.float32),
            "act": np.asarray(act_l, dtype=np.int64),
            "logp": np.asarray(logp_l, dtype=np.float32),
            "adv": adv,
            "ret": ret,
            "raw_return": ep_raw_return,
        })

    # padding 到最长回合，构建 (B, L, 特征) 张量与 mask
    max_len = max(len(e["obs"]) for e in episodes)
    B = len(episodes)
    obs_t = torch.zeros(B, max_len, env.observation_space.shape[0], device=device)
    act_t = torch.zeros(B, max_len, dtype=torch.long, device=device)
    logp_t = torch.zeros(B, max_len, device=device)
    adv_t = torch.zeros(B, max_len, device=device)
    ret_t = torch.zeros(B, max_len, device=device)
    mask_t = torch.zeros(B, max_len, device=device)
    for i, e in enumerate(episodes):
        L = len(e["obs"])
        obs_t[i, :L] = torch.as_tensor(e["obs"], device=device)
        act_t[i, :L] = torch.as_tensor(e["act"], device=device)
        logp_t[i, :L] = torch.as_tensor(e["logp"], device=device)
        adv_t[i, :L] = torch.as_tensor(e["adv"], device=device)
        ret_t[i, :L] = torch.as_tensor(e["ret"], device=device)
        mask_t[i, :L] = 1.0

    batch = {"obs": obs_t, "act": act_t, "logp_old": logp_t,
             "adv": adv_t, "ret": ret_t, "mask": mask_t}
    ep_returns = [float(e["raw_return"]) for e in episodes]
    return batch, ep_returns


# ---------------------------------------------------------------------------
# PPO 更新：只在有效时间步上计算损失
# ---------------------------------------------------------------------------
def ppo_update(net, arm, optimizer, batch, args):
    obs, act, mask = batch["obs"], batch["act"], batch["mask"]
    logp_old, adv, ret = batch["logp_old"], batch["adv"], batch["ret"]
    B, L, D = obs.shape

    valid = mask > 0.5
    adv_flat = adv[valid]
    adv_norm = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)
    adv = torch.zeros_like(adv)
    adv[valid] = adv_norm

    stats = {"approx_kl": 0.0, "clip_frac": 0.0, "entropy": 0.0, "n_updates": 0}
    for _ in range(args.epochs):
        idx = torch.randperm(B, device=obs.device)
        for start in range(0, B, args.episode_batch):
            mb = idx[start:start + args.episode_batch]
            logits, values = net.forward_seq(obs[mb])
            dist = Categorical(logits=logits)
            logp = dist.log_prob(act[mb])
            m = mask[mb]

            ratio = torch.exp(logp - logp_old[mb])
            surr1 = ratio * adv[mb]
            surr2 = torch.clamp(ratio, 1.0 - args.clip, 1.0 + args.clip) * adv[mb]
            policy_loss = -(torch.min(surr1, surr2) * m).sum() / m.sum().clamp(min=1.0)
            value_loss = (((values - ret[mb]) ** 2) * m).sum() / m.sum().clamp(min=1.0)
            entropy = (dist.entropy() * m).sum() / m.sum().clamp(min=1.0)
            loss = policy_loss + args.vf_coef * value_loss - args.ent_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            optimizer.step()

            with torch.no_grad():
                kl = ((logp_old[mb] - logp) * m).sum() / m.sum().clamp(min=1.0)
                clip_frac = ((((ratio - 1.0).abs() > args.clip).float()) * m).sum() / m.sum().clamp(min=1.0)
                stats["approx_kl"] += kl.item()
                stats["clip_frac"] += clip_frac.item()
                stats["entropy"] += entropy.item()
                stats["n_updates"] += 1

    for k in stats:
        if k != "n_updates":
            stats[k] /= max(stats["n_updates"], 1)
    return stats


# ---------------------------------------------------------------------------
# 评估：贪婪策略（第 3 步的动作决定成败）
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(env, net, arm, episodes: int, device):
    success = []
    for _ in range(episodes):
        obs, _ = env.reset()
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        hidden = net.init_hidden(1, device) if arm == "gru" else None
        done = False
        correct = False
        while not done:
            if arm == "gru":
                logits, _, hidden = net.step(obs_t, hidden)
            else:
                logits, _ = net(obs_t)
            action = int(torch.argmax(logits, dim=-1).item())
            next_obs, r, terminated, truncated, info = env.step(action)
            if info.get("correct") is not None:
                correct = bool(info["correct"])
            obs_t = torch.as_tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0)
            done = terminated or truncated
        success.append(1.0 if correct else 0.0)
    return float(np.mean(success))


# ---------------------------------------------------------------------------
# 训练一个 arm
# ---------------------------------------------------------------------------
def train_one(arm: str, args):
    set_seed(args.seed)
    device = torch.device("cpu")
    env = CueTMaze()
    net = build_net(arm, env.observation_space.shape[0], env.action_space.n)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    recent_success = []          # 最近回合的成功队列
    success_at = None
    episodes_done, n_updates = 0, 0
    t0 = time.time()

    print(f"\n[开始] arm={arm} | 参数量 {sum(p.numel() for p in net.parameters())}")
    while episodes_done < args.total_episodes:
        batch, ep_returns = collect_episodes(
            env, net, arm, args.episodes_per_update, device, args.gamma, args.lam)
        episodes_done += args.episodes_per_update
        n_updates += 1
        recent_success.extend(ep_returns)

        stats = ppo_update(net, arm, optimizer, batch, args)

        window = recent_success[-args.success_window:]
        if success_at is None and len(recent_success) >= args.success_window and np.mean(window) >= 0.9:
            success_at = episodes_done

        if n_updates % 5 == 0:
            print(f"  [{arm:4}] 回合 {episodes_done:6d} | 近{args.success_window}成功 "
                  f"{np.mean(window):.3f} | KL {stats['approx_kl']:.4f} | "
                  f"熵 {stats['entropy']:.3f}")

    eval_success = evaluate(CueTMaze(), net, arm, args.eval_episodes, device)
    window = recent_success[-args.success_window:]
    return {
        "arm": arm,
        "episodes": episodes_done,
        "success_window": float(np.mean(window)),
        "success_at": success_at,
        "eval_success": eval_success,
        "mean_kl": float(stats["approx_kl"]),
        "seconds": time.time() - t0,
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="第013章：PPO 循环神经网络策略")
    p.add_argument("--arms", type=str, nargs="+", default=["mlp", "gru"],
                   choices=["mlp", "gru"], help="策略结构")
    p.add_argument("--total-episodes", type=int, default=20000, help="每个 arm 的总回合数")
    p.add_argument("--episodes-per-update", type=int, default=256, help="每次更新采集的回合数")
    p.add_argument("--episode-batch", type=int, default=64, help="每次梯度更新的回合批大小")
    p.add_argument("--epochs", type=int, default=8, help="每批数据的更新轮数")
    p.add_argument("--lr", type=float, default=3e-4, help="学习率")
    p.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
    p.add_argument("--lam", type=float, default=0.95, help="GAE 的 lambda")
    p.add_argument("--clip", type=float, default=0.2, help="PPO 裁剪范围")
    p.add_argument("--ent-coef", type=float, default=0.01, help="熵正则系数")
    p.add_argument("--vf-coef", type=float, default=0.5, help="价值损失系数")
    p.add_argument("--success-window", type=int, default=500, help="成功率滑窗（回合数）")
    p.add_argument("--eval-episodes", type=int, default=200, help="评估回合数")
    p.add_argument("--seed", type=int, default=0, help="随机种子")
    p.add_argument("--save-dir", type=str, default="runs/ch013", help="模型保存目录")
    p.add_argument("--quick", action="store_true", help="快速模式：2 个 arm、各 3000 回合")
    return p.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.total_episodes = 3000
        args.episodes_per_update = 128
        args.episode_batch = 32
        args.epochs = 4
        args.success_window = 200
        args.eval_episodes = 100

    print("=" * 96)
    print(f"循环策略对比 | arms {args.arms} | 每个 {args.total_episodes} 回合 | 种子 {args.seed}")
    print("环境：CueTMaze（线索只在第 0 步可见；无记忆策略上限 50%）")
    print("=" * 96)

    results = []
    for arm in args.arms:
        res = train_one(arm, args)
        results.append(res)
        print(f"[完成] {arm}: 近{args.success_window}成功 {res['success_window']:.3f} | "
              f"评估成功 {res['eval_success']:.3f} | 首达90%回合 {res['success_at']} | "
              f"用时 {res['seconds']:.1f}s")

    print("\n" + "=" * 96)
    print("对比总结（同种子、同回合预算）")
    print("-" * 96)
    print(f"{'arm':>5} | {'总回合':>7} | {'首达90%成功回合':>13} | {'近窗口成功率':>11} | "
          f"{'评估成功率':>9} | {'平均KL':>8} | {'用时(s)':>8}")
    for r in results:
        reached = str(r["success_at"]) if r["success_at"] is not None else "未达到"
        print(f"{r['arm']:>5} | {r['episodes']:>7d} | {reached:>13} | {r['success_window']:>11.3f} | "
              f"{r['eval_success']:>9.3f} | {r['mean_kl']:>8.4f} | {r['seconds']:>8.1f}")

    print("\n观察提示：")
    print("  - mlp 看不到起点线索，最优只能压到单边，成功率≈0.5；")
    print("  - gru 把线索带进隐状态，理论上可达 1.0；")
    print("  - 训练更新时按整段回合重放序列，mask 保证 padding 不参与损失。")

    os.makedirs(args.save_dir, exist_ok=True)
    path = os.path.join(args.save_dir, "recurrent.txt")
    with open(path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(f"arm={r['arm']} episodes={r['episodes']} success_at={r['success_at']} "
                    f"success_window={r['success_window']:.3f} eval_success={r['eval_success']:.3f} "
                    f"kl={r['mean_kl']:.5f}\n")
    print(f"\n结果已写入 {path}")


if __name__ == "__main__":
    main()
