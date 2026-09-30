"""
第100章 多智能体环境：游戏AI

实验内容（2v2 竞技场对战 + 自博弈训练，纯 numpy 内联环境）：
  - 4 名斗士在 [-1,1]² 的竞技场里移动、贴身攻击；每人 3 点生命、
    攻击后有 3 步冷却；一方全灭即分胜负
  - 动作 10 个：5 个移动方向 × {不出手, 出手}
  - 奖励：命中 +0.5/次、击杀 +2、获胜方存活者 +2、每步 -0.005 时间成本
  - 算法：共享网络的离散 PPO + 自博弈（4 名斗士共用一张策略网络，
    双方互为对手、同时进化）
  - 评估：对脚本对手（追最近敌人、进入射程就攻击）的胜率、自博弈胜率
    （理论上应稳定在 50%）、平均回合长度、命中总量
  - 本章是全课程的最后一章：小结部分附上 100 章技术地图回顾

运行：
    python code/ch100.py           # 完整训练（CPU 约 3-6 分钟）
    python code/ch100.py --quick   # 快速跑通（约 30-60 秒）

预期：对脚本对手的胜率从 ~0.5 升到 0.65～0.9；自博弈胜率稳定在 0.5 附近
（对称性检验）；平均回合长度随训练下降（打得更果断）。
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
# 环境：2v2 竞技场
# ---------------------------------------------------------------------------
class ArenaEnv:
    """2v2 竞技场对战（队伍 A：0/1 号，队伍 B：2/3 号）。

    观测（13 维/斗士）：自己位置(2)、生命/3(1)、冷却/3(1)、队友相对位置(2)、
    队友生命/3(1)、两名对手相对位置(4)、最近对手生命/3(1)、时间进度(1)。
    动作（10 个）：移动方向 {停,上,下,左,右} × {不出手,出手}。
    """

    n_agents = 4
    teams = np.array([0, 0, 1, 1], dtype=np.int64)
    teammates = (1, 0, 3, 2)
    n_actions = 10
    obs_dim = 13
    move_step = 0.09
    attack_range = 0.3
    max_hp = 3
    cooldown = 3
    time_cost = 0.005
    hit_reward = 0.5
    kill_reward = 2.0
    win_reward = 2.0

    def __init__(self, max_steps: int = 120, bound: float = 1.0):
        self.max_steps = max_steps
        self.bound = bound
        self.rng = np.random.default_rng(0)
        self.move_vecs = np.array([[0, 0], [0, -1], [0, 1], [-1, 0], [1, 0]],
                                  dtype=np.float32)

    def reset(self, rng: np.random.Generator):
        self.rng = rng
        # A 队从左半场出发、B 队从右半场出发
        self.pos = np.array([[-0.7, -0.3], [-0.7, 0.3],
                             [0.7, -0.3], [0.7, 0.3]], dtype=np.float32)
        self.pos += rng.uniform(-0.05, 0.05, size=(4, 2)).astype(np.float32)
        self.hp = np.full(4, self.max_hp, dtype=np.int64)
        self.cd = np.zeros(4, dtype=np.int64)
        self.steps = 0
        self.hits_total = 0
        self.kills = [0, 0]
        return self._obs()

    def _obs(self) -> np.ndarray:
        obs = []
        for i in range(4):
            mate = self.teammates[i]
            enemies = [j for j in range(4) if self.teams[j] != self.teams[i]]
            e0, e1 = enemies
            nearest_hp = self.hp[e0] if (self.hp[e0] <= self.hp[e1] and self.hp[e0] > 0) \
                else (self.hp[e1] if self.hp[e1] > 0 else self.hp[e0])
            o = np.concatenate([
                self.pos[i],
                [self.hp[i] / self.max_hp],
                [self.cd[i] / self.cooldown],
                self.pos[mate] - self.pos[i],
                [self.hp[mate] / self.max_hp],
                self.pos[e0] - self.pos[i],
                self.pos[e1] - self.pos[i],
                [nearest_hp / self.max_hp],
                [self.steps / self.max_steps],
            ])
            obs.append(o.astype(np.float32))
        return np.stack(obs)

    def _alive(self):
        return self.hp > 0

    def step(self, actions: np.ndarray):
        actions = actions.astype(np.int64)
        moves = actions // 2
        attacks = actions % 2
        alive = self._alive()
        rewards = np.full(4, -self.time_cost, dtype=np.float32)

        # 1) 移动（阵亡者不动）
        for i in range(4):
            if alive[i]:
                self.pos[i] = np.clip(
                    self.pos[i] + self.move_step * self.move_vecs[moves[i]],
                    -self.bound, self.bound)

        # 2) 冷却递减
        self.cd = np.maximum(self.cd - 1, 0)

        # 3) 出手结算：对射程内最近的敌人造成 1 点伤害
        for i in range(4):
            if not alive[i] or attacks[i] == 0 or self.cd[i] > 0:
                continue
            enemies = [j for j in range(4)
                       if self.teams[j] != self.teams[i] and self.hp[j] > 0]
            if not enemies:
                continue
            dists = [np.linalg.norm(self.pos[j] - self.pos[i]) for j in enemies]
            k = int(np.argmin(dists))
            if dists[k] <= self.attack_range:
                self.cd[i] = self.cooldown
                target = enemies[k]
                self.hp[target] -= 1
                self.hits_total += 1
                rewards[i] += self.hit_reward
                if self.hp[target] <= 0:
                    rewards[i] += self.kill_reward
                    self.kills[self.teams[i]] += 1

        self.steps += 1

        # 4) 胜负判定与团队奖励
        team_alive = [bool(np.any(self.hp[self.teams == t] > 0)) for t in (0, 1)]
        terminated = False
        winner = None
        if not team_alive[0] and not team_alive[1]:
            terminated = True          # 双灭算平局
        elif not team_alive[0] or not team_alive[1]:
            terminated = True
            winner = 0 if team_alive[0] else 1
            for i in range(4):
                if self.teams[i] == winner and self.hp[i] > 0:
                    rewards[i] += self.win_reward

        truncated = (not terminated) and self.steps >= self.max_steps
        info = {"winner": winner, "hp": self.hp.copy(),
                "hits": self.hits_total}
        return self._obs(), rewards, terminated, truncated, info


# ---------------------------------------------------------------------------
# 网络：离散 Actor-Critic（4 名斗士共用）
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
            done_buf.append(np.full(env.n_agents, float(terminated), dtype=np.float32))
            done = terminated or truncated
            obs = next_obs
        ep_stats.append({"winner": info["winner"], "len": env.steps,
                         "hits": info["hits"]})
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
# 评估：自博弈 / 对脚本对手 / 对随机对手
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(net, episodes, device, seed, opponent="scripted"):
    """返回：A 队胜率、平均回合长度、平均命中总量。opponent: self/scripted/random。"""
    env = ArenaEnv()
    rng = np.random.default_rng(seed)
    wins, lens, hits = 0, [], []
    for _ in range(episodes):
        obs = env.reset(rng)
        done = False
        info = {"winner": None, "hits": 0}
        while not done:
            actions = net.act_deterministic(obs, device)
            if opponent == "scripted":
                # B 队（2/3 号）用脚本：追最近敌人，进射程就出手
                # 动作编码 a = 移动×2 + 出手；移动 {0停,1上,2下,3左,4右}
                for i in (2, 3):
                    enemies = [j for j in (0, 1) if env.hp[j] > 0]
                    if not enemies:
                        actions[i] = 0
                        continue
                    dists = [np.linalg.norm(env.pos[j] - env.pos[i]) for j in enemies]
                    k = int(np.argmin(dists))
                    target = enemies[k]
                    d = env.pos[target] - env.pos[i]
                    if dists[k] <= env.attack_range:
                        actions[i] = 1                       # 停留 + 出手
                    else:
                        dx, dy = d
                        if abs(dx) >= abs(dy):
                            if dx > 0:
                                actions[i] = 9                   # 右 + 出手
                            elif dx < 0:
                                actions[i] = 7                   # 左 + 出手
                            else:
                                actions[i] = 1
                        else:
                            if dy > 0:
                                actions[i] = 5                   # 下 + 出手
                            elif dy < 0:
                                actions[i] = 3                   # 上 + 出手
                            else:
                                actions[i] = 1
            elif opponent == "random":
                actions[2:] = rng.integers(0, env.n_actions, size=2)
            obs, rew, terminated, truncated, info = env.step(actions)
            done = terminated or truncated
        winner = info["winner"]
        if winner == 0:
            wins += 1
        elif winner is None:
            wins += 0.5                          # 平局算半分
        lens.append(env.steps)
        hits.append(info["hits"])
    return wins / max(episodes, 1), float(np.mean(lens)), float(np.mean(hits))


def parse_args():
    p = argparse.ArgumentParser(description="第100章：2v2 竞技场游戏AI（自博弈）")
    p.add_argument("--max-steps", type=int, default=80000, help="环境步数上限")
    p.add_argument("--rollout-episodes", type=int, default=20, help="每次更新采集局数")
    p.add_argument("--epochs", type=int, default=4, help="PPO 更新轮数")
    p.add_argument("--batch-size", type=int, default=256, help="小批量大小")
    p.add_argument("--lr", type=float, default=3e-4, help="学习率")
    p.add_argument("--gamma", type=float, default=0.97, help="折扣因子")
    p.add_argument("--lam", type=float, default=0.95, help="GAE lambda")
    p.add_argument("--clip", type=float, default=0.2, help="裁剪范围")
    p.add_argument("--ent-coef", type=float, default=0.01, help="熵正则系数")
    p.add_argument("--hidden", type=int, default=128, help="隐藏层宽度")
    p.add_argument("--eval-every", type=int, default=10000, help="评估间隔（步）")
    p.add_argument("--eval-episodes", type=int, default=40, help="评估局数")
    p.add_argument("--seed", type=int, default=0, help="随机种子")
    p.add_argument("--save-dir", type=str, default="runs/ch100", help="模型保存目录")
    p.add_argument("--quick", action="store_true", help="快速模式：缩小训练规模")
    return p.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.max_steps = min(args.max_steps, 16000)
        args.rollout_episodes = 8
        args.eval_every = 4000
        args.eval_episodes = 20
    set_seed(args.seed)

    device = torch.device("cpu")
    env = ArenaEnv()
    rng = np.random.default_rng(args.seed)
    net = PPOActorCritic(env.obs_dim, env.n_actions, args.hidden).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    print(f"设备 {device} | 2v2 竞技场 | 自博弈（4 人共享网络）| 种子 {args.seed}")

    # 未训练基线的三种对手评估
    for opp, label in (("random", "对随机对手"), ("scripted", "对脚本对手"),
                       ("self", "自博弈")):
        wr, ln, ht = evaluate(net, args.eval_episodes, device,
                              args.seed + 111, opponent=opp)
        print(f"初始（{label}）：A 队得分率 {wr:.2f} | 平均回合 {ln:5.1f} 步 | "
              f"命中总量 {ht:4.1f}")

    total_steps, update_idx = 0, 0
    all_stats = []
    t_start = time.time()

    while total_steps < args.max_steps:
        episodes = max(1, args.rollout_episodes)
        batch, ep_stats = collect_rollout(env, net, rng, episodes, device, args)
        # 统计每局步数（采样用的局数可能因步数上限略有波动）
        total_steps += sum(s["len"] for s in ep_stats)
        all_stats.extend(ep_stats)
        stats = ppo_update(net, optimizer, batch, args, device)
        update_idx += 1

        if update_idx % max(1, (args.max_steps // args.eval_every)) == 0 \
                or total_steps >= args.max_steps:
            wr_s, ln_s, _ = evaluate(net, args.eval_episodes, device,
                                     args.seed + 777, opponent="scripted")
            wr_f, ln_f, _ = evaluate(net, args.eval_episodes, device,
                                     args.seed + 777, opponent="self")
            recent = all_stats[-20:]
            win_a = np.mean([1.0 if s["winner"] == 0 else (0.5 if s["winner"] is None else 0.0)
                             for s in recent])
            m_len = np.mean([s["len"] for s in recent])
            print(f"[更新 {update_idx:3d}] 步 {total_steps:6d} | 对脚本胜率 {wr_s:.2f} | "
                  f"自博弈胜率 {wr_f:.2f} | 回合长(脚本) {ln_s:5.1f}/(自) {ln_f:5.1f} | "
                  f"训练近20局 A得分 {win_a:.2f} 回合 {m_len:5.1f} | KL {stats['kl']:.4f}")

    elapsed = time.time() - t_start
    wr_s, ln_s, ht_s = evaluate(net, args.eval_episodes, device,
                                args.seed + 999, opponent="scripted")
    wr_f, ln_f, ht_f = evaluate(net, args.eval_episodes, device,
                                args.seed + 999, opponent="self")
    print("-" * 74)
    print(f"训练结束 | 用时 {elapsed:.1f}s | 环境步 {total_steps} | 更新 {update_idx} 次")
    print(f"评估（{args.eval_episodes} 局）：对脚本对手胜率 {wr_s:.2f}（回合 {ln_s:.1f} 步）| "
          f"自博弈胜率 {wr_f:.2f}（回合 {ln_f:.1f} 步）| 自博弈命中 {ht_f:.1f}/局")
    print(f"对称性检验：自博弈胜率接近 0.50 说明双方共享同一策略、胜负由座位与")
    print(f"随机性决定；对脚本胜率高于 0.50 的差距才是学到的真实强度。")

    os.makedirs(args.save_dir, exist_ok=True)
    path = os.path.join(args.save_dir, "ippo_arena_selfplay.pt")
    torch.save({"policy": net.state_dict(), "args": vars(args)}, path)
    print(f"模型已保存到 {path}")


if __name__ == "__main__":
    main()
