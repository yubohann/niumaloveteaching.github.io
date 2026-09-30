"""
第099章 多智能体环境：金融交易

实验内容（4 个交易者在带私有信号的噪声市场里下单，纯 numpy 内联环境）：
  - 基本面 F 服从 AR(1)：F ← 0.98F + η；市场价 p 受订单流冲击、
    向基本面靠拢、叠加噪声
  - 4 个交易者中 2 个是"知情交易者"（信号噪声小）、2 个是"噪声交易者"
    （信号噪声大）；每期从 5 档订单 {-2,-1,0,1,2} 选一档，奖励为正头寸
    在下一期的实现盈亏：r_i = a_i·(F_{t+1} − p_t) − 0.01|a_i|
  - 算法：共享网络的离散 PPO（观测含交易者类型独热，网络可服务两类角色）
  - 基线：零订单、追信号、追动量
  - 指标：两类交易者的平均盈亏、价格发现误差 |F−p|、价格波动率

运行：
    python code/ch099.py           # 完整训练（CPU 约 2-4 分钟）
    python code/ch099.py --quick   # 快速跑通（约 20-40 秒）

预期：知情交易者的学习盈亏为正、噪声交易者接近零或为负；学习后的
价格发现误差比"零订单"基线显著更低（以实跑为准）。
"""

import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# 环境：带私有信号的噪声市场
# ---------------------------------------------------------------------------
class TradingMarketEnv:
    """4 个交易者的单资产市场（每局 max_steps 期）。

    基本面：F ← 0.98F + η, η ~ N(0, 0.06)；价格：p ← p + 0.08·净订单流
    + 0.2·(F − p) + N(0, 0.02)。
    观测（6 维/交易者）：F − p、价格动量 Δp、私有信号 s_i、上期仓位/2、
    类型独热(2)。动作：5 档订单 {-2..2}。
    """

    n_agents = 4
    obs_dim = 6
    n_actions = 5
    signal_noise = (0.02, 0.02, 0.12, 0.12)     # 前两人知情
    lambda_impact = 0.08
    kappa_revert = 0.2
    price_noise = 0.02
    eta_std = 0.06
    fee = 0.01

    def __init__(self, max_steps: int = 100):
        self.max_steps = max_steps
        self.rng = np.random.default_rng(0)

    def _roll(self):
        """抽出下一期基本面增量并生成各交易者的私有信号。"""
        eta = float(self.rng.normal(0.0, self.eta_std))
        self.F_next = 0.98 * self.F + eta
        dF = self.F_next - self.F
        self.signals = np.array(
            [dF + float(self.rng.normal(0.0, s)) for s in self.signal_noise],
            dtype=np.float32)

    def reset(self, rng: np.random.Generator):
        self.rng = rng
        self.F = 0.0
        self.p = 0.0
        self.prev_p = 0.0
        self.last_order = np.zeros(4, dtype=np.float32)
        self.steps = 0
        self.tot_pnl = np.zeros(4, dtype=np.float32)
        self.mispricing = []
        self.price_changes = []
        self._roll()
        return self._obs()

    def _obs(self) -> np.ndarray:
        obs = []
        dp = self.p - self.prev_p
        for i in range(self.n_agents):
            type_onehot = np.array([1.0, 0.0]) if i < 2 else np.array([0.0, 1.0])
            o = np.concatenate([
                [self.F - self.p],
                [dp],
                [float(self.signals[i])],
                [self.last_order[i] / 2.0],
                type_onehot,
            ])
            obs.append(o.astype(np.float32))
        return np.stack(obs)

    def step(self, actions: np.ndarray):
        orders = (actions.astype(np.float32) - 2.0)       # 0..4 -> -2..2
        net = float(orders.sum())
        self.steps += 1

        # 1) 价格更新：订单冲击 + 向基本面回归 + 噪声
        self.prev_p = self.p
        self.p = (self.p + self.lambda_impact * net
                  + self.kappa_revert * (self.F - self.p)
                  + float(self.rng.normal(0.0, self.price_noise)))
        self.price_changes.append(self.p - self.prev_p)

        # 2) 实现盈亏：以当前价建仓，按下一期基本面结算（教学简化）
        rewards = orders * (self.F_next - self.prev_p) - self.fee * np.abs(orders)
        self.tot_pnl += rewards
        self.last_order = orders.copy()
        self.mispricing.append(abs(self.F - self.p))

        # 3) 基本面推进一期，并滚动出新的场景
        self.F = self.F_next
        self._roll()

        truncated = self.steps >= self.max_steps
        info = {"mispricing": float(abs(self.F - self.p)),
                "net_order": net}
        return self._obs(), rewards.astype(np.float32), False, truncated, info


# ---------------------------------------------------------------------------
# 网络：离散 Actor-Critic（类型独热让知情/噪声交易者共享网络）
# ---------------------------------------------------------------------------
class PPOActorCritic(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.pi_head = nn.Linear(hidden, n_actions)
        self.v_head = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor):
        h = self.body(obs)
        return self.pi_head(h), self.v_head(h).squeeze(-1)

    @torch.no_grad()
    def act_batch(self, obs_np: np.ndarray, device):
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
        logits, v = self.forward(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return (action.cpu().numpy().astype(np.int64),
                dist.log_prob(action).cpu().numpy().astype(np.float32),
                v.cpu().numpy().astype(np.float32))

    @torch.no_grad()
    def act_deterministic(self, obs_np: np.ndarray, device) -> np.ndarray:
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
        logits, _ = self.forward(obs)
        return torch.argmax(logits, dim=-1).cpu().numpy().astype(np.int64)


def compute_gae(rewards, values, dones, last_values, gamma, lam):
    """rewards/values/dones：(T, N)；每交易者一份奖励。"""
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


def collect_rollout(env, net, rng, episodes, device, args):
    obs_buf, act_buf, logp_buf, val_buf = [], [], [], []
    rew_buf, done_buf = [], []
    ep_stats = []
    for _ in range(episodes):
        obs = env.reset(rng)
        done = False
        while not done:
            a, logp, v = net.act_batch(obs, device)
            next_obs, rew, terminated, truncated, info = env.step(a)
            obs_buf.append(obs)
            act_buf.append(a)
            logp_buf.append(logp)
            val_buf.append(v)
            rew_buf.append(rew.astype(np.float32))
            done_buf.append(np.zeros(env.n_agents, dtype=np.float32))
            done = terminated or truncated
            obs = next_obs
        ep_stats.append({"pnl": env.tot_pnl.copy(),
                         "mispricing": float(np.mean(env.mispricing))})
    with torch.no_grad():
        _, last_v = net.forward(
            torch.as_tensor(obs, dtype=torch.float32, device=device))
        last_values = last_v.cpu().numpy().astype(np.float32)
    values = np.stack(val_buf)
    adv, ret = compute_gae(np.stack(rew_buf), values, np.stack(done_buf),
                           last_values, args.gamma, args.lam)
    T, N = adv.shape
    batch = {
        "obs": np.stack(obs_buf).reshape(T * N, -1),
        "act": np.stack(act_buf).reshape(T * N),
        "logp": np.stack(logp_buf).reshape(T * N),
        "adv": adv.reshape(T * N),
        "ret": ret.reshape(T * N),
    }
    return batch, ep_stats


def ppo_update(net, optimizer, batch, args, device):
    obs = torch.as_tensor(batch["obs"], dtype=torch.float32, device=device)
    act = torch.as_tensor(batch["act"], dtype=torch.int64, device=device)
    logp_old = torch.as_tensor(batch["logp"], dtype=torch.float32, device=device)
    adv = torch.as_tensor(batch["adv"], dtype=torch.float32, device=device)
    ret = torch.as_tensor(batch["ret"], dtype=torch.float32, device=device)
    N = obs.shape[0]
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    stats = {"kl": 0.0, "clip_frac": 0.0, "n": 0}
    for _ in range(args.epochs):
        idx = torch.randperm(N, device=device)
        for start in range(0, N, args.batch_size):
            mb = idx[start:start + args.batch_size]
            logits, v = net(obs[mb])
            dist = Categorical(logits=logits)
            logp = dist.log_prob(act[mb])
            ratio = torch.exp(logp - logp_old[mb])
            surr1 = ratio * adv[mb]
            surr2 = torch.clamp(ratio, 1.0 - args.clip, 1.0 + args.clip) * adv[mb]
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(v, ret[mb])
            entropy = dist.entropy().mean()
            loss = policy_loss + 0.5 * value_loss - args.ent_coef * entropy
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            optimizer.step()
            with torch.no_grad():
                stats["kl"] += (logp_old[mb] - logp).mean().item()
                stats["clip_frac"] += ((ratio - 1.0).abs() > args.clip).float().mean().item()
                stats["n"] += 1
    stats["kl"] /= max(stats["n"], 1)
    stats["clip_frac"] /= max(stats["n"], 1)
    return stats


# ---------------------------------------------------------------------------
# 评估：学习策略与三种脚本交易者
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(net, episodes, device, seed, mode="learned"):
    """返回：(知情者均盈亏, 噪声者均盈亏, 价格发现误差, 价格波动率)。"""
    env = TradingMarketEnv()
    rng = np.random.default_rng(seed)
    informed_pnl, noise_pnl, mis, vol = [], [], [], []
    for _ in range(episodes):
        obs = env.reset(rng)
        done = False
        while not done:
            if mode == "learned":
                a = net.act_deterministic(obs, device)
            elif mode == "zero":
                a = np.full(env.n_agents, 2, dtype=np.int64)
            elif mode == "signal":     # 追信号：信号正就买 1 手，负就卖 1 手
                a = np.zeros(env.n_agents, dtype=np.int64)
                for i in range(env.n_agents):
                    s = float(env.signals[i])
                    a[i] = 3 if s > 0.005 else (1 if s < -0.005 else 2)
            else:                      # 追动量：价格在涨就买
                a = np.zeros(env.n_agents, dtype=np.int64)
                dp = env.p - env.prev_p
                for i in range(env.n_agents):
                    a[i] = 3 if dp > 0 else (1 if dp < 0 else 2)
            obs, rew, terminated, truncated, info = env.step(a)
            done = terminated or truncated
        informed_pnl.append(float(env.tot_pnl[:2].mean()))
        noise_pnl.append(float(env.tot_pnl[2:].mean()))
        mis.append(float(np.mean(env.mispricing)))
        vol.append(float(np.std(env.price_changes)))
    return (float(np.mean(informed_pnl)), float(np.mean(noise_pnl)),
            float(np.mean(mis)), float(np.mean(vol)))


def parse_args():
    p = argparse.ArgumentParser(description="第099章：金融交易（知情交易与价格发现）")
    p.add_argument("--max-steps", type=int, default=60000, help="环境步数上限")
    p.add_argument("--episode-steps", type=int, default=100, help="每局期数")
    p.add_argument("--rollout-episodes", type=int, default=10,
                   help="每次更新采集多少局")
    p.add_argument("--epochs", type=int, default=4, help="PPO 更新轮数")
    p.add_argument("--batch-size", type=int, default=256, help="小批量大小")
    p.add_argument("--lr", type=float, default=3e-4, help="学习率")
    p.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
    p.add_argument("--lam", type=float, default=0.95, help="GAE lambda")
    p.add_argument("--clip", type=float, default=0.2, help="裁剪范围")
    p.add_argument("--ent-coef", type=float, default=0.01, help="熵正则系数")
    p.add_argument("--hidden", type=int, default=128, help="隐藏层宽度")
    p.add_argument("--eval-every", type=int, default=10000, help="评估间隔（步）")
    p.add_argument("--eval-episodes", type=int, default=10, help="评估局数")
    p.add_argument("--seed", type=int, default=0, help="随机种子")
    p.add_argument("--save-dir", type=str, default="runs/ch099", help="模型保存目录")
    p.add_argument("--quick", action="store_true", help="快速模式：缩小训练规模")
    return p.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.max_steps = min(args.max_steps, 16000)
        args.rollout_episodes = 5
        args.eval_every = 4000
        args.eval_episodes = 5
    set_seed(args.seed)

    device = torch.device("cpu")
    env = TradingMarketEnv(max_steps=args.episode_steps)
    rng = np.random.default_rng(args.seed)
    net = PPOActorCritic(env.obs_dim, env.n_actions, args.hidden).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    print(f"设备 {device} | 4 交易者（2 知情 / 2 噪声）| 每局 {args.episode_steps} 期 | "
          f"种子 {args.seed}")

    for mode, label in (("zero", "零订单基线"), ("signal", "追信号基线"),
                        ("momentum", "追动量基线")):
        inf, noi, mis, vol = evaluate(net, args.eval_episodes, device,
                                      args.seed + 111, mode=mode)
        print(f"{label}：知情者盈亏 {inf:+7.3f} | 噪声者盈亏 {noi:+7.3f} | "
              f"价格发现误差 {mis:.4f} | 波动率 {vol:.4f}")

    total_steps, update_idx = 0, 0
    all_stats = []
    t_start = time.time()

    while total_steps < args.max_steps:
        episodes = max(1, args.rollout_episodes)
        batch, ep_stats = collect_rollout(env, net, rng, episodes, device, args)
        total_steps += episodes * env.max_steps
        all_stats.extend(ep_stats)
        stats = ppo_update(net, optimizer, batch, args, device)
        update_idx += 1

        if total_steps % args.eval_every < env.max_steps * episodes:
            inf, noi, mis, vol = evaluate(net, args.eval_episodes, device,
                                          args.seed + 777)
            recent = all_stats[-5:]
            m_pnl = np.mean([s["pnl"][:2].mean() for s in recent]) if recent else float("nan")
            print(f"[更新 {update_idx:3d}] 步 {total_steps:6d} | 知情者盈亏 {inf:+7.3f} | "
                  f"噪声者盈亏 {noi:+7.3f} | 发现误差 {mis:.4f} | 波动率 {vol:.4f} | "
                  f"训练知情盈亏 {m_pnl:+7.3f} | KL {stats['kl']:.4f}")

    elapsed = time.time() - t_start
    inf, noi, mis, vol = evaluate(net, args.eval_episodes, device, args.seed + 999)
    print("-" * 74)
    print(f"训练结束 | 用时 {elapsed:.1f}s | 环境步 {total_steps} | 更新 {update_idx} 次")
    print(f"评估（{args.eval_episodes} 局，确定性策略）：知情者盈亏 {inf:+.3f} | "
          f"噪声者盈亏 {noi:+.3f} | 价格发现误差 {mis:.4f} | 波动率 {vol:.4f}")

    os.makedirs(args.save_dir, exist_ok=True)
    path = os.path.join(args.save_dir, "ippo_trading.pt")
    torch.save({"policy": net.state_dict(), "args": vars(args)}, path)
    print(f"模型已保存到 {path}")


if __name__ == "__main__":
    main()
