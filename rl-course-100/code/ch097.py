"""
第097章 多智能体环境：推荐系统

实验内容（3 家内容平台争夺 3 组用户的注意力市场，纯 numpy 内联环境）：
  - 3 家平台各属一个内容类别（科技/体育/文化），每个平台每步选择质量投入
    q ∈ {0,1,2,3}，投入越高成本越大、对用户越有吸引力
  - 3 组用户各有会漂移的类别偏好；每一步每组用户选择对自己效用最高的平台
    消费，消费又让偏好向该平台的类别漂移（习惯锁定效应）
  - 算法：共享网络的双重 DQN（类别独热进观测，网络可服务三家平台），
    逐平台 ε-贪婪
  - 指标：消费者效用、总投入成本（广告军备竞赛指标）、平台利润、
    市场集中度 HHI、消费类别熵
  - 基线：零投入、满投入、随机投入

运行：
    python code/ch097.py           # 完整训练（CPU 约 2-4 分钟）
    python code/ch097.py --quick   # 快速跑通（约 20-40 秒）

预期：学习策略的消费者效用高于零投入基线、总成本低于满投入基线——学会
"战况激烈时投入、用户锁定后收手"，市场集中度随时间上升（赢家通吃趋势）。
"""

import argparse
import collections
import copy
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# 环境：注意力市场（内容推荐竞争）
# ---------------------------------------------------------------------------
class AttentionMarket:
    """3 平台 × 3 用户组的内容注意力市场。

    观测（14 维/平台）：自己上步投入的独热(4)、上步拿到的份额(1)、
    三组用户对自己类别的偏好(3)、另两家平台上步份额(2)、时间进度(1)、
    自己的类别独热(3)。
    动作：质量投入 q ∈ {0,1,2,3}。
    奖励：本步份额 − 0.15×投入。
    """

    n_firms = 3
    n_segments = 3
    n_quality = 4
    categories = (0, 1, 2)
    obs_dim = 14
    n_actions = 4
    invest_cost = 0.15

    def __init__(self, max_steps: int = 200, pref_inertia: float = 0.9):
        self.max_steps = max_steps
        self.pref_inertia = pref_inertia
        self.rng = np.random.default_rng(0)

    def reset(self, rng: np.random.Generator):
        self.rng = rng
        # 每组用户的类别偏好：随机但归一化
        self.prefs = rng.uniform(0.2, 1.0, size=(self.n_segments, 3)).astype(np.float32)
        self.prefs /= self.prefs.sum(axis=1, keepdims=True)
        self.last_q = np.zeros(self.n_firms, dtype=np.int64)
        self.last_share = np.full(self.n_firms, 1.0 / 3.0, dtype=np.float32)
        self.steps = 0
        self.share_history = []
        return self._obs()

    def _obs(self) -> np.ndarray:
        obs = []
        for i in range(self.n_firms):
            q_onehot = np.zeros(self.n_quality, dtype=np.float32)
            q_onehot[self.last_q[i]] = 1.0
            others = [float(self.last_share[j])
                      for j in range(self.n_firms) if j != i]
            cat_onehot = np.zeros(3, dtype=np.float32)
            cat_onehot[self.categories[i]] = 1.0
            o = np.concatenate([
                q_onehot,
                [float(self.last_share[i])],
                self.prefs[:, self.categories[i]],
                np.asarray(others),
                [self.steps / self.max_steps],
                cat_onehot,
            ])
            obs.append(o.astype(np.float32))
        return np.stack(obs)

    def step(self, actions: np.ndarray):
        q = np.clip(actions.astype(np.int64), 0, self.n_quality - 1)
        shares = np.zeros(self.n_firms, dtype=np.float32)
        consumer_util = 0.0

        # 1) 每组用户选择效用最高的平台（效用 = 偏好 × 吸引力 + 噪声）
        for s in range(self.n_segments):
            utilities = np.zeros(self.n_firms, dtype=np.float64)
            for i in range(self.n_firms):
                attractiveness = 1.0 + 0.5 * float(q[i])
                utilities[i] = (float(self.prefs[s, self.categories[i]])
                                * attractiveness
                                + self.rng.uniform(0.0, 0.15))
            winner = int(np.argmax(utilities))
            shares[winner] += 1.0
            consumer_util += float(self.prefs[s, self.categories[winner]]) \
                * (1.0 + 0.5 * float(q[winner]))
            # 2) 习惯锁定：偏好向被消费的类别漂移
            target = np.zeros(3, dtype=np.float32)
            target[self.categories[winner]] = 1.0
            self.prefs[s] = (self.pref_inertia * self.prefs[s]
                             + (1.0 - self.pref_inertia) * target)
            self.prefs[s] += self.rng.uniform(0.0, 0.01, size=3).astype(np.float32)
            self.prefs[s] /= self.prefs[s].sum()

        shares /= self.n_segments
        rewards = shares - self.invest_cost * q.astype(np.float32)
        self.last_q = q.copy()
        self.last_share = shares.copy()
        self.share_history.append(shares.copy())
        self.steps += 1

        truncated = self.steps >= self.max_steps
        hhi = float(np.sum(shares ** 2))
        info = {"shares": shares.copy(), "hhi": hhi,
                "consumer_util": consumer_util / self.n_segments,
                "total_cost": float(self.invest_cost * q.sum())}
        return self._obs(), rewards, False, truncated, info


# ---------------------------------------------------------------------------
# 网络与双重 DQN（共享网络：类别独热让三家平台共用一个网络）
# ---------------------------------------------------------------------------
class QNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DoubleDQNAgent:
    def __init__(self, env: AttentionMarket, args, device):
        self.args = args
        self.device = device
        self.q = QNet(env.obs_dim, env.n_actions, args.hidden).to(device)
        self.q_target = copy.deepcopy(self.q)
        for p in self.q_target.parameters():
            p.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.q.parameters(), lr=args.lr)
        self.buffer = collections.deque(maxlen=args.buffer_capacity)
        self.updates = 0

    @torch.no_grad()
    def act(self, obs_np: np.ndarray, epsilon: float, rng: np.random.Generator):
        n = obs_np.shape[0]
        actions = rng.integers(0, self.args.n_actions, size=n).astype(np.int64)
        explore = rng.uniform(size=n) < epsilon
        if not explore.all():
            obs = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device)
            greedy = np.argmax(self.q(obs).cpu().numpy(), axis=1).astype(np.int64)
            actions[~explore] = greedy[~explore]
        return actions

    def push(self, obs, act, rew, next_obs, done):
        for i in range(obs.shape[0]):
            self.buffer.append((obs[i].copy(), int(act[i]), float(rew[i]),
                                next_obs[i].copy(), float(done)))

    def update(self):
        args = self.args
        batch = random.sample(self.buffer, args.batch_size)
        obs = torch.as_tensor(np.stack([b[0] for b in batch]),
                              dtype=torch.float32, device=self.device)
        act = torch.as_tensor([b[1] for b in batch],
                              dtype=torch.int64, device=self.device).unsqueeze(-1)
        rew = torch.as_tensor([b[2] for b in batch],
                              dtype=torch.float32, device=self.device).unsqueeze(-1)
        next_obs = torch.as_tensor(np.stack([b[3] for b in batch]),
                                   dtype=torch.float32, device=self.device)
        done = torch.as_tensor([b[4] for b in batch],
                               dtype=torch.float32, device=self.device).unsqueeze(-1)
        with torch.no_grad():
            next_actions = self.q(next_obs).argmax(dim=1, keepdim=True)   # 双重 DQN
            q_next = self.q_target(next_obs).gather(1, next_actions)
            y = rew + args.gamma * (1.0 - done) * q_next
        q = self.q(obs).gather(1, act)
        loss = F.smooth_l1_loss(q, y)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 5.0)
        self.optimizer.step()
        self.updates += 1
        if self.updates % args.target_update == 0:
            self.q_target.load_state_dict(self.q.state_dict())
        return float(loss.item())


# ---------------------------------------------------------------------------
# 评估：学习策略与三种投入基线
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(agent, episodes, device, seed, mode="learned"):
    """返回：消费者效用、总投入成本、平台总利润、HHI、消费类别熵。"""
    env = AttentionMarket()
    rng = np.random.default_rng(seed)
    utils, costs, profits, hhis, entropies = [], [], [], [], []
    for _ in range(episodes):
        obs = env.reset(rng)
        done, n = False, 0
        u_sum, c_sum, p_sum, hhi_sum = 0.0, 0.0, 0.0, 0.0
        consumed_counts = np.zeros(3)
        info = {"hhi": 0.0, "consumer_util": 0.0, "total_cost": 0.0, "shares": np.zeros(3)}
        while not done:
            if mode == "learned":
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
                a = np.argmax(agent.q(obs_t).cpu().numpy(), axis=1)
            elif mode == "zero":
                a = np.zeros(env.n_firms, dtype=np.int64)
            elif mode == "max":
                a = np.full(env.n_firms, env.n_quality - 1, dtype=np.int64)
            else:   # random
                a = rng.integers(0, env.n_actions, size=env.n_firms)
            obs, rew, terminated, truncated, info = env.step(a)
            u_sum += info["consumer_util"]
            c_sum += info["total_cost"]
            p_sum += float(rew.sum())
            hhi_sum += info["hhi"]
            consumed_counts += info["shares"] * env.n_segments
            n += 1
            done = terminated or truncated
        utils.append(u_sum / max(n, 1))
        costs.append(c_sum / max(n, 1))
        profits.append(p_sum / max(n, 1))
        hhis.append(hhi_sum / max(n, 1))
        p = consumed_counts / max(consumed_counts.sum(), 1e-9)
        entropies.append(float(-np.sum(p * np.log(p + 1e-9))))
    return (float(np.mean(utils)), float(np.mean(costs)), float(np.mean(profits)),
            float(np.mean(hhis)), float(np.mean(entropies)))


def parse_args():
    p = argparse.ArgumentParser(description="第097章：推荐系统（注意力市场竞争）")
    p.add_argument("--max-steps", type=int, default=60000, help="环境步数上限")
    p.add_argument("--warmup", type=int, default=2000, help="随机动作预热步数")
    p.add_argument("--update-every", type=int, default=2, help="每多少步一次更新")
    p.add_argument("--batch-size", type=int, default=64, help="采样批量")
    p.add_argument("--buffer-capacity", type=int, default=80000, help="回放池容量")
    p.add_argument("--lr", type=float, default=1e-3, help="学习率")
    p.add_argument("--gamma", type=float, default=0.95, help="折扣因子")
    p.add_argument("--hidden", type=int, default=128, help="隐藏层宽度")
    p.add_argument("--eps-start", type=float, default=1.0, help="初始探索率")
    p.add_argument("--eps-end", type=float, default=0.05, help="最终探索率")
    p.add_argument("--eps-decay-steps", type=int, default=25000, help="探索率衰减步数")
    p.add_argument("--target-update", type=int, default=200, help="目标网络更新间隔")
    p.add_argument("--eval-every", type=int, default=10000, help="评估间隔（步）")
    p.add_argument("--eval-episodes", type=int, default=5, help="评估回合数")
    p.add_argument("--seed", type=int, default=0, help="随机种子")
    p.add_argument("--save-dir", type=str, default="runs/ch097", help="模型保存目录")
    p.add_argument("--quick", action="store_true", help="快速模式：缩小训练规模")
    return p.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.max_steps = min(args.max_steps, 16000)
        args.eps_decay_steps = min(args.eps_decay_steps, 6000)
        args.eval_every = 4000
        args.eval_episodes = 3
    set_seed(args.seed)

    device = torch.device("cpu")
    env = AttentionMarket()
    args.n_actions = env.n_actions
    rng = np.random.default_rng(args.seed)
    agent = DoubleDQNAgent(env, args, device)

    print(f"设备 {device} | 3 平台 × 3 用户组 | 投入档位 0..{env.n_quality - 1} | "
          f"种子 {args.seed}")

    baselines = {}
    for mode, label in (("zero", "零投入基线"), ("max", "满投入基线"), ("random", "随机投入")):
        baselines[mode] = evaluate(agent, args.eval_episodes, device,
                                   args.seed + 111, mode=mode)
        u, c, pr, hhi, ent = baselines[mode]
        print(f"{label}：消费者效用 {u:.3f} | 投入成本/步 {c:.3f} | "
              f"利润/步 {pr:+.3f} | HHI {hhi:.3f} | 类别熵 {ent:.3f}")

    obs = env.reset(rng)
    ep_stats = []
    losses = []
    ep_rets = np.zeros(env.n_firms, dtype=np.float32)
    ep_len = 0
    t_start = time.time()

    for step in range(1, args.max_steps + 1):
        frac = min(1.0, step / max(args.eps_decay_steps, 1))
        epsilon = args.eps_start + frac * (args.eps_end - args.eps_start)
        if step <= args.warmup:
            actions = rng.integers(0, env.n_actions, size=env.n_firms)
        else:
            actions = agent.act(obs, epsilon, rng)

        next_obs, rew, terminated, truncated, info = env.step(actions)
        done = terminated or truncated
        if step > args.warmup:
            agent.push(obs, actions, rew, next_obs, float(terminated))
        ep_rets += rew
        ep_len += 1

        if step > args.warmup and step % args.update_every == 0 \
                and len(agent.buffer) >= args.batch_size:
            losses.append(agent.update())

        if done:
            ep_stats.append({"profit": float(ep_rets.sum()), "hhi": info["hhi"],
                             "cost": info["total_cost"], "len": ep_len})
            obs = env.reset(rng)
            ep_rets = np.zeros(env.n_firms, dtype=np.float32)
            ep_len = 0
        else:
            obs = next_obs

        if step % args.eval_every == 0:
            u, c, pr, hhi, ent = evaluate(agent, args.eval_episodes, device,
                                          args.seed + 777)
            m_loss = float(np.mean(losses[-500:])) if losses else float("nan")
            print(f"[步 {step:6d}] ε {epsilon:.2f} | 消费者效用 {u:.3f} | "
                  f"成本/步 {c:.3f} | 利润/步 {pr:+.3f} | HHI {hhi:.3f} | "
                  f"熵 {ent:.3f} | 损失 {m_loss:.4f}")

    elapsed = time.time() - t_start
    u, c, pr, hhi, ent = evaluate(agent, args.eval_episodes, device, args.seed + 999)
    print("-" * 74)
    print(f"训练结束 | 用时 {elapsed:.1f}s | 环境步 {args.max_steps} | "
          f"梯度更新 {agent.updates} 次")
    print(f"评估（{args.eval_episodes} 回合，贪婪策略）：消费者效用 {u:.3f} | "
          f"投入成本/步 {c:.3f} | 利润/步 {pr:+.3f} | HHI {hhi:.3f} | 类别熵 {ent:.3f}")
    print(f"对照零投入：效用 {baselines['zero'][0]:.3f} → {u:.3f} | "
          f"成本 {baselines['zero'][1]:.3f} → {c:.3f}")
    print(f"对照满投入：效用 {baselines['max'][0]:.3f} → {u:.3f} | "
          f"成本 {baselines['max'][1]:.3f} → {c:.3f}")

    os.makedirs(args.save_dir, exist_ok=True)
    path = os.path.join(args.save_dir, "double_dqn_recommender.pt")
    torch.save({"q": agent.q.state_dict(), "args": vars(args)}, path)
    print(f"模型已保存到 {path}")


if __name__ == "__main__":
    main()
