"""
第098章 多智能体环境：电网调度

实验内容（3 个区域的协同经济调度，纯 numpy 内联环境）：
  - 每小时内，各区域观测本地负荷、可再生出力与储能状态，决定：
      常规机组出力 g ∈ {0,1,2,3}（成本 0.4g + 0.05g²）
      储能动作 ∈ {放电, 待机, 充电}（效率 90%，充放有磨损费）
  - 区域之间可以通过联络线互济（总传输容量有限）；未满足负荷罚款很高，
    因此三个区域有强合作动机——奖励取"全社会总成本"的负值（共享奖励）
  - 算法：共享网络的离散 PPO（Categorical 策略 + 共享奖励的 GAE）
  - 基线：随机调度、本地区域规则的贪婪调度
  - 指标：总成本/日、未满足电量、可再生利用率、储能循环次数

运行：
    python code/ch098.py           # 完整训练（CPU 约 3-5 分钟）
    python code/ch098.py --quick   # 快速跑通（约 30-60 秒）

预期：学习策略的总成本低于两个基线（以实跑为准），未满足电量降到接近 0，
并学会在负荷高峰用储能、低谷充电的跨时段套利。
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
# 环境：3 区域协同经济调度
# ---------------------------------------------------------------------------
class GridDispatchEnv:
    """3 个区域电网的经济调度（每步 1 小时，一天 24 步为一局）。

    观测（8 维/区域）：本区负荷(1)、本区可再生出力(1)、储能/3(1)、
    时间 sin/cos(2)、另两区负荷(2)、上步机组出力/3(1)。
    动作（12 个）：机组出力 g∈{0..3} × 储能 {放电, 待机, 充电}。
    奖励（全队共享）：−(机组成本 + 磨损 + 4×未满足电量)。
    """

    n_agents = 3
    obs_dim = 8
    n_actions = 12
    battery_cap = 3.0
    battery_eff = 0.9
    wear = 0.02

    def __init__(self, max_steps: int = 24, line_capacity: float = 1.5):
        self.max_steps = max_steps
        self.line_capacity = line_capacity
        self.rng = np.random.default_rng(0)

    def reset(self, rng: np.random.Generator):
        self.rng = rng
        self.steps = 0
        self.battery = rng.uniform(0.5, 1.5, size=3).astype(np.float32)
        self.last_g = np.zeros(3, dtype=np.float32)
        self.tot_gen = 0.0
        self.tot_unmet = 0.0
        self.tot_renew = 0.0
        self.tot_curtail = 0.0
        self.battery_cycles = 0
        # 先给出第 0 小时的需求与可再生场景，观测才有内容
        self.demand = self._demand(0.0)
        self.renewable = self._renewable()
        return self._obs()

    def _demand(self, t: float) -> np.ndarray:
        """日负荷曲线：正午单峰 + 区域差异 + 噪声。"""
        base = np.array([1.5, 1.7, 1.4], dtype=np.float32)
        amp = np.array([0.8, 0.9, 0.7], dtype=np.float32)
        phase = np.array([0.0, 0.4, -0.3], dtype=np.float32)
        shape = 0.5 * (1.0 - np.cos(2.0 * np.pi * (t / 24.0)))
        d = base + amp * shape * (1.0 + 0.6 * np.sin(2.0 * np.pi * (t - 6.0) / 24.0)) \
            + phase * 0.0
        d = d + self.rng.uniform(-0.1, 0.1, size=3).astype(np.float32)
        return d.astype(np.float32)

    def _renewable(self) -> np.ndarray:
        """可再生出力：中午高、夜间低，叠加云量噪声。"""
        t = float(self.steps)
        solar = np.maximum(0.0, np.sin(np.pi * (t % 24.0 - 6.0) / 12.0))
        base = np.array([1.8, 1.4, 2.0], dtype=np.float32) * solar
        noise = self.rng.uniform(0.0, 0.6, size=3).astype(np.float32)
        return (base + noise * solar).astype(np.float32)

    def _obs(self) -> np.ndarray:
        demand = self.demand
        renew = self.renewable
        t = float(self.steps)
        obs = []
        for i in range(self.n_agents):
            others = [float(demand[j]) for j in range(3) if j != i]
            o = np.concatenate([
                [demand[i] / 3.0, renew[i] / 3.0, self.battery[i] / self.battery_cap],
                [np.sin(2 * np.pi * t / 24.0), np.cos(2 * np.pi * t / 24.0)],
                np.asarray(others) / 3.0,
                [self.last_g[i] / 3.0],
            ])
            obs.append(o.astype(np.float32))
        return np.stack(obs)

    def step(self, actions: np.ndarray):
        actions = actions.astype(np.int64)
        g = (actions // 3).astype(np.float32)              # 0..3
        batt_mode = actions % 3                            # 0=放电 1=待机 2=充电

        # 每步先模拟一个随机需求/可再生场景（内部状态）
        self.demand = self._demand(float(self.steps))
        self.renewable = self._renewable()

        self.tot_renew += float(self.renewable.sum())
        gen_cost = 0.0
        wear_cost = 0.0
        net = np.zeros(3, dtype=np.float32)

        for i in range(3):
            gen_cost += 0.4 * g[i] + 0.05 * g[i] ** 2
            supply = float(self.renewable[i]) + float(g[i])
            if batt_mode[i] == 0 and self.battery[i] >= 1.0:     # 放电
                self.battery[i] -= 1.0
                supply += self.battery_eff
                wear_cost += self.wear
                self.battery_cycles += 1
            elif batt_mode[i] == 2 and self.battery[i] <= self.battery_cap - 0.9:
                # 充电：需要 1.0 的富余电量，实际存 0.9
                if supply - float(self.demand[i]) >= 1.0:
                    supply -= 1.0
                    self.battery[i] += self.battery_eff
                    wear_cost += self.wear
                    self.battery_cycles += 1
            net[i] = supply - float(self.demand[i])

        # 区域互济：富余区支援缺额区（容量受限）
        surplus = float(np.sum(np.maximum(net, 0.0)))
        deficit = float(np.sum(np.maximum(-net, 0.0)))
        transfer = min(surplus, deficit, self.line_capacity)
        unmet = max(0.0, deficit - transfer)
        curtailed = max(0.0, surplus - transfer)

        self.tot_gen += float(g.sum())
        self.tot_unmet += unmet
        self.tot_curtail += curtailed
        self.steps += 1
        self.last_g = g.copy()

        cost = gen_cost + wear_cost + 4.0 * unmet
        reward = -cost
        truncated = self.steps >= self.max_steps
        info = {"cost": cost, "unmet": unmet, "curtailed": curtailed,
                "gen": float(g.sum()), "renew": float(self.renewable.sum())}
        return self._obs(), reward, False, truncated, info


# ---------------------------------------------------------------------------
# 网络：离散 Actor-Critic（共享奖励，参数共享）
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
    """共享奖励：rewards (T,)，values (T, N)。"""
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


def collect_rollout(env, net, rng, episodes, obs, device, args):
    """按局采集（一天 24 步为一局），共采 episodes 局。"""
    obs_buf, act_buf, logp_buf, val_buf = [], [], [], []
    rew_buf, done_buf = [], []
    ep_stats = []
    for _ in range(episodes):
        obs = env.reset(rng)
        day = {"day_cost": 0.0, "day_unmet": 0.0, "day_curtail": 0.0,
               "day_gen": 0.0, "day_renew": 0.0}
        done = False
        while not done:
            a, logp, v = net.act_batch(obs, device)
            next_obs, reward, terminated, truncated, info = env.step(a)
            obs_buf.append(obs)
            act_buf.append(a)
            logp_buf.append(logp)
            val_buf.append(v)
            rew_buf.append(float(reward))
            done_buf.append(float(terminated))
            day["day_cost"] += info["cost"]
            day["day_unmet"] += info["unmet"]
            day["day_curtail"] += info["curtailed"]
            day["day_gen"] += info["gen"]
            day["day_renew"] += info["renew"]
            done = terminated or truncated
            obs = next_obs
        ep_stats.append(day)
    with torch.no_grad():
        _, last_v = net.forward(
            torch.as_tensor(obs, dtype=torch.float32, device=device))
        last_values = last_v.cpu().numpy().astype(np.float32)
    values = np.stack(val_buf)
    adv, ret = compute_gae(np.asarray(rew_buf, dtype=np.float32), values,
                           np.asarray(done_buf, dtype=np.float32),
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
# 评估：学习策略 / 规则调度 / 随机调度
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(net, days, device, seed, mode="learned"):
    """返回：日均成本、日均未满足电量、可再生利用率、日均储能循环次数。"""
    env = GridDispatchEnv()
    rng = np.random.default_rng(seed)
    costs, unmet, renew_util, cycles = [], [], [], []
    for _ in range(days):
        obs = env.reset(rng)
        done = False
        day_cost = 0.0
        while not done:
            if mode == "learned":
                a = net.act_deterministic(obs, device)
            elif mode == "random":
                a = rng.integers(0, env.n_actions, size=env.n_agents)
            else:   # 规则：尽量用机组补足"负荷 − 可再生 − 储能放电"
                a = np.zeros(env.n_agents, dtype=np.int64)
                for i in range(env.n_agents):
                    need = (float(env.demand[i]) - float(env.renewable[i])
                            - (env.battery_eff if env.battery[i] >= 1.0 else 0.0))
                    g = int(np.clip(round(need), 0, 3))
                    if need > 0.3 and env.battery[i] >= 1.0:
                        batt = 0
                    elif need < -1.0 and env.battery[i] < env.battery_cap - 0.9:
                        batt = 2
                    else:
                        batt = 1
                    a[i] = g * 3 + batt
            obs, reward, terminated, truncated, info = env.step(a)
            day_cost += info["cost"]
            done = terminated or truncated
        costs.append(day_cost)
        unmet.append(env.tot_unmet)
        # 可再生利用率：被浪费的部分至多算到可再生总量上（截断到 [0,1]）
        wasted = min(env.tot_curtail, env.tot_renew)
        renew_util.append(1.0 - wasted / max(env.tot_renew, 1e-6))
        cycles.append(float(env.battery_cycles))
    return (float(np.mean(costs)), float(np.mean(unmet)),
            float(np.mean(renew_util)), float(np.mean(cycles)))


def parse_args():
    p = argparse.ArgumentParser(description="第098章：电网经济调度（离散 PPO）")
    p.add_argument("--max-steps", type=int, default=60000, help="环境步数上限")
    p.add_argument("--episode-steps", type=int, default=24, help="每局步数（一天）")
    p.add_argument("--rollout-episodes", type=int, default=25,
                   help="每次更新采集多少天的数据")
    p.add_argument("--epochs", type=int, default=4, help="PPO 更新轮数")
    p.add_argument("--batch-size", type=int, default=256, help="小批量大小")
    p.add_argument("--lr", type=float, default=3e-4, help="学习率")
    p.add_argument("--gamma", type=float, default=0.97, help="折扣因子")
    p.add_argument("--lam", type=float, default=0.95, help="GAE lambda")
    p.add_argument("--clip", type=float, default=0.2, help="裁剪范围")
    p.add_argument("--ent-coef", type=float, default=0.01, help="熵正则系数")
    p.add_argument("--hidden", type=int, default=128, help="隐藏层宽度")
    p.add_argument("--eval-every", type=int, default=10000, help="评估间隔（步）")
    p.add_argument("--eval-days", type=int, default=10, help="评估天数")
    p.add_argument("--seed", type=int, default=0, help="随机种子")
    p.add_argument("--save-dir", type=str, default="runs/ch098", help="模型保存目录")
    p.add_argument("--quick", action="store_true", help="快速模式：缩小训练规模")
    return p.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.max_steps = min(args.max_steps, 16000)
        args.eval_every = 4000
        args.eval_days = 5
    set_seed(args.seed)

    device = torch.device("cpu")
    env = GridDispatchEnv(max_steps=args.episode_steps)
    rng = np.random.default_rng(args.seed)
    net = PPOActorCritic(env.obs_dim, env.n_actions, args.hidden).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    print(f"设备 {device} | 3 区域电网 | 动作 12 个（机组×储能）| 种子 {args.seed}")

    for mode, label in (("random", "随机调度基线"), ("rule", "规则调度基线")):
        cost, unmet, util, cyc = evaluate(net, args.eval_days, device,
                                          args.seed + 111, mode=mode)
        print(f"{label}：日均成本 {cost:7.2f} | 日均未满足 {unmet:.3f} | "
              f"可再生利用率 {util:.2f} | 储能循环 {cyc:.1f}")

    obs = env.reset(rng)
    total_steps, update_idx = 0, 0
    all_stats = []
    t_start = time.time()

    while total_steps < args.max_steps:
        episodes = max(1, args.rollout_episodes)
        batch, ep_stats = collect_rollout(env, net, rng, episodes, obs, device, args)
        total_steps += episodes * env.max_steps
        all_stats.extend(ep_stats)
        stats = ppo_update(net, optimizer, batch, args, device)
        update_idx += 1

        if total_steps % args.eval_every < env.max_steps * episodes:
            cost, unmet, util, cyc = evaluate(net, args.eval_days, device,
                                              args.seed + 777)
            recent = all_stats[-10:]
            m_cost = np.mean([s["day_cost"] for s in recent]) if recent else float("nan")
            print(f"[更新 {update_idx:3d}] 步 {total_steps:6d} | 评估日成本 {cost:7.2f} | "
                  f"未满足 {unmet:.3f} | 利用率 {util:.2f} | 循环 {cyc:.1f} | "
                  f"训练日成本 {m_cost:7.2f} | KL {stats['kl']:.4f}")

    elapsed = time.time() - t_start
    cost, unmet, util, cyc = evaluate(net, args.eval_days, device, args.seed + 999)
    print("-" * 74)
    print(f"训练结束 | 用时 {elapsed:.1f}s | 环境步 {total_steps} | 更新 {update_idx} 次")
    print(f"评估（{args.eval_days} 天，确定性策略）：日均成本 {cost:.2f} | "
          f"日均未满足 {unmet:.3f} | 可再生利用率 {util:.2f} | 储能循环 {cyc:.1f}")

    os.makedirs(args.save_dir, exist_ok=True)
    path = os.path.join(args.save_dir, "ippo_grid_dispatch.pt")
    torch.save({"policy": net.state_dict(), "args": vars(args)}, path)
    print(f"模型已保存到 {path}")


if __name__ == "__main__":
    main()
