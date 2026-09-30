"""
第096章 多智能体环境：自动驾驶

实验内容（双车道环形公路上的协同驾驶，纯 numpy 内联环境）：
  - 6 辆车在同一条环形公路上行驶，车辆只有速度和车道两个自由度
  - 动作 9 个：纵向 {-1,0,+1} 档加速度 × 横向 {向左变道, 保持, 向右变道}
  - 奖励（每车独立）：5×速度 - 跟车过近罚款 - 变道成本 - 碰撞重罚
  - 算法：共享网络的离散 PPO（Categorical 策略 + GAE）
  - 基线：匀速 0.08 不换道
  - 指标：平均速度、碰撞次数、变道次数；评价"快"与"安全"的平衡

运行：
    python code/ch096.py           # 完整训练（CPU 约 3-5 分钟）
    python code/ch096.py --quick   # 快速跑通（约 30-60 秒）

预期：学习策略的平均速度显著高于匀速基线（典型 0.12～0.17，以实跑为准），
碰撞次数接近 0，且速度方差较小（说明学会了跟车与超车时机）。
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
# 环境：双车道环形公路
# ---------------------------------------------------------------------------
class RingRoadEnv:
    """6 辆车在周长 1.0 的环形公路上行驶（车道 0 内圈、1 外圈）。

    观测（9 维/车）：速度(1)、车道独热(2)、本车道前车间距与前车速度(2)、
    另一车道前车间距与前车速度(2)、本车道后车间距(1)、是否换道中标记(0)。
    动作（9 个）：纵向加速度 {-1,0,1} × 横向 {-1,0,1}。
    奖励：5×速度 - 2.0×跟车过近 - 4.0×碰撞 - 0.1×变道。
    """

    n_agents = 6
    obs_dim = 9
    n_actions = 9
    safe_gap = 0.03          # 跟车过近阈值（环形周长归一化为 1）
    crash_gap = 0.015        # 碰撞阈值
    gap_cap = 0.3            # 观测中间距截断
    v_min, v_max = 0.02, 0.2
    dv = 0.02

    def __init__(self, n_cars: int = 6, max_steps: int = 150):
        self.n_cars = n_cars
        self.max_steps = max_steps
        self.rng = np.random.default_rng(0)

    def reset(self, rng: np.random.Generator):
        self.rng = rng
        # 车辆沿环等距撒点，车道交替；速度统一 0.08
        self.pos = (np.arange(self.n_cars) / self.n_cars
                    + rng.uniform(-0.005, 0.005, size=self.n_cars)) % 1.0
        self.lane = (np.arange(self.n_cars) % 2).astype(np.int64)
        self.v = np.full(self.n_cars, 0.08, dtype=np.float32)
        self.steps = 0
        self.collisions = 0
        self.lane_changes = 0
        return self._obs()

    def _gap_ahead(self, i, lane):
        """返回 (环形前方最近车的间距, 该车速度)，没有车时返回 (1.0, 0)。"""
        best_gap, best_v = 1.0, 0.0
        for j in range(self.n_cars):
            if j == i or self.lane[j] != lane:
                continue
            gap = (self.pos[j] - self.pos[i]) % 1.0
            if gap < best_gap:
                best_gap, best_v = gap, float(self.v[j])
        return best_gap, best_v

    def _gap_behind(self, i, lane):
        best = 1.0
        for j in range(self.n_cars):
            if j == i or self.lane[j] != lane:
                continue
            gap = (self.pos[i] - self.pos[j]) % 1.0
            if gap < best:
                best = gap
        return best

    def _obs(self) -> np.ndarray:
        obs = []
        for i in range(self.n_cars):
            other = 1 - self.lane[i]
            fg, fv = self._gap_ahead(i, self.lane[i])
            og, ov = self._gap_ahead(i, other)
            rg = self._gap_behind(i, self.lane[i])
            lane_onehot = np.zeros(2, dtype=np.float32)
            lane_onehot[self.lane[i]] = 1.0
            o = np.concatenate([
                [self.v[i] / self.v_max],
                lane_onehot,
                [min(fg, self.gap_cap) / self.gap_cap, fv / self.v_max],
                [min(og, self.gap_cap) / self.gap_cap, ov / self.v_max],
                [min(rg, self.gap_cap) / self.gap_cap],
                [0.0],                       # 预留"是否正在变道"标记
            ])
            obs.append(o.astype(np.float32))
        return np.stack(obs)

    def step(self, actions: np.ndarray):
        actions = actions.astype(np.int64)
        accel_idx = actions // 3          # 0,1,2 -> -1,0,+1 档
        lane_pref = actions % 3 - 1       # -1,0,+1
        self.steps += 1

        # 1) 变道：目标车道在当前位置附近（±0.03）有车则婉拒
        changed = np.zeros(self.n_cars, dtype=bool)
        for i in range(self.n_cars):
            if lane_pref[i] == 0:
                continue
            target = int(np.clip(self.lane[i] + lane_pref[i], 0, 1))
            if target == self.lane[i]:
                continue
            blocked = False
            for j in range(self.n_cars):
                if j == i or self.lane[j] != target:
                    continue
                d = min((self.pos[j] - self.pos[i]) % 1.0,
                        (self.pos[i] - self.pos[j]) % 1.0)
                if d < 0.03:
                    blocked = True
                    break
            if not blocked:
                self.lane[i] = target
                changed[i] = True
        self.lane_changes += int(changed.sum())

        # 2) 纵向动力学：速度在 [v_min, v_max] 内加减，位置环形推进
        dv = (accel_idx - 1).astype(np.float32) * self.dv
        self.v = np.clip(self.v + dv, self.v_min, self.v_max)
        self.pos = (self.pos + self.v) % 1.0

        # 3) 奖励：速度收益 + 安全罚款
        rewards = 5.0 * self.v.copy()
        for i in range(self.n_cars):
            fg, _ = self._gap_ahead(i, self.lane[i])
            if fg < self.safe_gap:
                rewards[i] -= 2.0
            if fg < self.crash_gap:
                rewards[i] -= 4.0
                self.collisions += 1
            if changed[i]:
                rewards[i] -= 0.1

        truncated = self.steps >= self.max_steps
        info = {"mean_speed": float(self.v.mean()),
                "collisions": self.collisions,
                "lane_changes": self.lane_changes,
                "min_gap": min(self._gap_ahead(i, self.lane[i])[0]
                               for i in range(self.n_cars))}
        return self._obs(), rewards.astype(np.float32), False, truncated, info


# ---------------------------------------------------------------------------
# 网络：离散动作 Actor-Critic
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


def collect_rollout(env, net, rng, steps, obs, device, args):
    obs_buf, act_buf, logp_buf, val_buf = [], [], [], []
    rew_buf, done_buf = [], []
    ep_stats = []
    ep_ret, ep_len = 0.0, 0
    ep_speed, ep_col, ep_lc = 0.0, 0, 0

    for _ in range(steps):
        a, logp, v = net.act_batch(obs, device)
        next_obs, rew, terminated, truncated, info = env.step(a)
        obs_buf.append(obs)
        act_buf.append(a)
        logp_buf.append(logp)
        val_buf.append(v)
        rew_buf.append(rew.astype(np.float32))
        done_buf.append(np.zeros(env.n_agents, dtype=np.float32))
        ep_ret += float(rew.mean())
        ep_len += 1
        ep_speed += info["mean_speed"]
        ep_col += info["collisions"]
        ep_lc += info["lane_changes"]
        if terminated or truncated:
            ep_stats.append({"return": ep_ret, "mean_speed": ep_speed / max(ep_len, 1),
                             "collisions": ep_col, "lane_changes": ep_lc})
            obs = env.reset(rng)
            ep_ret, ep_len, ep_speed, ep_col, ep_lc = 0.0, 0, 0.0, 0, 0
        else:
            obs = next_obs

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
    return batch, ep_stats, obs


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
# 评估：学习策略与匀速基线
# ---------------------------------------------------------------------------
def evaluate(net, episodes, device, seed, mode="learned"):
    env = RingRoadEnv()
    rng = np.random.default_rng(seed)
    speeds, cols, lcs = [], [], []
    for _ in range(episodes):
        obs = env.reset(rng)
        done, n = False, 0
        speed_sum = 0.0
        info = {"collisions": 0, "lane_changes": 0}
        while not done:
            if mode == "learned":
                a = net.act_deterministic(obs, device)
            else:   # 匀速基线：保持车距、不换道
                a = np.full(env.n_agents, 4, dtype=np.int64)   # 4 = 保持档
            obs, rew, terminated, truncated, info = env.step(a)
            speed_sum += info["mean_speed"]
            n += 1
            done = terminated or truncated
        speeds.append(speed_sum / max(n, 1))
        cols.append(info["collisions"])
        lcs.append(info["lane_changes"])
    return (float(np.mean(speeds)), float(np.mean(cols)), float(np.mean(lcs)))


def parse_args():
    p = argparse.ArgumentParser(description="第096章：自动驾驶（环形公路协同驾驶）")
    p.add_argument("--max-steps", type=int, default=60000, help="环境步数上限")
    p.add_argument("--rollout", type=int, default=2048, help="每次更新采样步数")
    p.add_argument("--epochs", type=int, default=4, help="PPO 更新轮数")
    p.add_argument("--batch-size", type=int, default=256, help="小批量大小")
    p.add_argument("--lr", type=float, default=3e-4, help="学习率")
    p.add_argument("--gamma", type=float, default=0.95, help="折扣因子")
    p.add_argument("--lam", type=float, default=0.95, help="GAE lambda")
    p.add_argument("--clip", type=float, default=0.2, help="裁剪范围")
    p.add_argument("--ent-coef", type=float, default=0.02, help="熵正则系数")
    p.add_argument("--hidden", type=int, default=128, help="隐藏层宽度")
    p.add_argument("--eval-every", type=int, default=10000, help="评估间隔（步）")
    p.add_argument("--eval-episodes", type=int, default=5, help="评估回合数")
    p.add_argument("--seed", type=int, default=0, help="随机种子")
    p.add_argument("--save-dir", type=str, default="runs/ch096", help="模型保存目录")
    p.add_argument("--quick", action="store_true", help="快速模式：缩小训练规模")
    return p.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.max_steps = min(args.max_steps, 16000)
        args.rollout = 1024
        args.eval_every = 4000
        args.eval_episodes = 3
    set_seed(args.seed)

    device = torch.device("cpu")
    env = RingRoadEnv()
    rng = np.random.default_rng(args.seed)
    net = PPOActorCritic(env.obs_dim, env.n_actions, args.hidden).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    print(f"设备 {device} | {env.n_cars} 辆车 / 双车道环线 | 9 个离散动作 | 种子 {args.seed}")

    sp, co, lc = evaluate(net, args.eval_episodes, device, args.seed + 111, mode="fixed")
    print(f"匀速基线：平均速度 {sp:.3f} | 碰撞 {co:.1f} 次/局 | 变道 {lc:.1f} 次/局")

    obs = env.reset(rng)
    total_steps, update_idx = 0, 0
    all_stats = []
    t_start = time.time()

    while total_steps < args.max_steps:
        batch, ep_stats, obs = collect_rollout(
            env, net, rng, args.rollout, obs, device, args)
        total_steps += args.rollout
        all_stats.extend(ep_stats)
        stats = ppo_update(net, optimizer, batch, args, device)
        update_idx += 1

        if total_steps % args.eval_every < args.rollout:
            sp, co, lc = evaluate(net, args.eval_episodes, device, args.seed + 777)
            recent = all_stats[-10:]
            m_sp = np.mean([s["mean_speed"] for s in recent]) if recent else float("nan")
            print(f"[更新 {update_idx:3d}] 步 {total_steps:6d} | 评估速度 {sp:.3f} | "
                  f"碰撞 {co:4.1f} | 变道 {lc:5.1f} | 训练局速度 {m_sp:.3f} | "
                  f"KL {stats['kl']:.4f}")

    elapsed = time.time() - t_start
    sp, co, lc = evaluate(net, args.eval_episodes, device, args.seed + 999)
    print("-" * 74)
    print(f"训练结束 | 用时 {elapsed:.1f}s | 环境步 {total_steps} | 更新 {update_idx} 次")
    print(f"评估（{args.eval_episodes} 回合，确定性策略）：平均速度 {sp:.3f} | "
          f"碰撞 {co:.1f} 次/局 | 变道 {lc:.1f} 次/局")
    print(f"对照匀速基线：速度 {sp:.3f} vs 0.080 | 速度提升 "
          f"{(sp / 0.08 - 1) * 100:+.1f}%")

    os.makedirs(args.save_dir, exist_ok=True)
    path = os.path.join(args.save_dir, "ippo_ring_road.pt")
    torch.save({"policy": net.state_dict(), "args": vars(args)}, path)
    print(f"模型已保存到 {path}")


if __name__ == "__main__":
    main()
