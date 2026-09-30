"""
第095章 多智能体环境：资源收集

实验内容（4 个采集者在 4 块可再生资源场上作业，纯 numpy 内联环境）：
  - 每块资源有存量 S，按逻辑斯蒂规律再生：S ← S + 0.08·S·(1−S/10)
  - 采集者靠近资源（<0.25）即按步采集 0.12；存量耗尽后需要漫长时间恢复
  - 两种奖励制度对比：
      individual：只有自己采到的算自己的（公地悲剧压力）
      shared    ：全员共享总采集量（可持续管理压力）
  - 算法：共享网络的独立 PPO（连续动作） + GAE
  - 指标：总采集量、后半程采集量、末态总存量、耗尽步数占比
  - 脚本基线：就近采集（选存量足够的最近资源）

运行：
    python code/ch095.py                    # 默认 individual（约 3-5 分钟）
    python code/ch095.py --regime shared    # 共享奖励制度
    python code/ch095.py --quick            # 快速跑通（约 30-60 秒）

预期：individual 制度下前期采集更快、后期资源枯竭导致总采集量反而下降；
shared 制度下采集更平稳、末态存量更高（以实跑为准）。这组对比就是"公地悲剧"
在可再生资源上的最小复现。
"""

import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_STD_MIN, LOG_STD_MAX = -20.0, 2.0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# 环境：可再生资源收集
# ---------------------------------------------------------------------------
class ResourceEnv:
    """4 个采集者与 4 块可再生资源。

    观测（16 维/采集者）：自己位置+速度(4)、最近两块资源的相对坐标与归一化
    存量(2+1)×2=6、其他三名采集者的相对位置(6)。
    动作：2 维加速度 [-1,1]。
    奖励：采集到的资源量（individual 模式只算自己，shared 模式全队总和）。
    """

    obs_dim = 16
    act_dim = 2
    n_agents = 4
    K = 10.0                      # 资源最大存量

    def __init__(self, harvest_radius: float = 0.25, harvest_rate: float = 0.12,
                 growth: float = 0.08, max_steps: int = 200,
                 accel: float = 0.12, friction: float = 0.6,
                 v_max: float = 1.0, bound: float = 1.0):
        self.harvest_radius = harvest_radius
        self.harvest_rate = harvest_rate
        self.growth = growth
        self.max_steps = max_steps
        self.accel = accel
        self.friction = friction
        self.v_max = v_max
        self.bound = bound
        self.patch_pos = np.array([[0.6, 0.6], [-0.6, 0.6],
                                   [0.6, -0.6], [-0.6, -0.6]], dtype=np.float32)

    def reset(self, rng: np.random.Generator):
        self.rng = rng
        # 资源初始存量在 5~9 之间抖动，避免所有回合千篇一律
        self.stock = rng.uniform(5.0, 9.0, size=4).astype(np.float32)
        self.pos = rng.uniform(-0.15, 0.15, size=(self.n_agents, 2)).astype(np.float32)
        self.vel = np.zeros((self.n_agents, 2), dtype=np.float32)
        self.steps = 0
        self.total_harvest = np.zeros(self.n_agents, dtype=np.float32)
        return self._obs()

    def _obs(self) -> np.ndarray:
        obs = []
        for i in range(self.n_agents):
            d = np.linalg.norm(self.patch_pos - self.pos[i], axis=1)
            order = np.argsort(d)
            patch_feats = []
            for k in order[:2]:
                patch_feats.append(self.patch_pos[k] - self.pos[i])
                patch_feats.append(np.asarray([self.stock[k] / self.K], dtype=np.float32))
            others = [self.pos[j] - self.pos[i]
                      for j in range(self.n_agents) if j != i]
            o = np.concatenate([self.pos[i], self.vel[i],
                                np.concatenate(patch_feats),
                                np.concatenate(others)])
            obs.append(o.astype(np.float32))
        return np.stack(obs)

    def step(self, actions: np.ndarray):
        a = np.clip(actions, -1.0, 1.0).astype(np.float32)
        self.vel = np.clip(self.friction * self.vel + self.accel * a,
                           -self.v_max, self.v_max)
        self.pos = np.clip(self.pos + self.vel, -self.bound, self.bound)
        self.steps += 1

        # 1) 资源按逻辑斯蒂规律再生
        self.stock = np.clip(
            self.stock + self.growth * self.stock * (1.0 - self.stock / self.K),
            0.0, self.K)

        # 2) 采集：靠近资源的采集者按顺序各取一份
        harvest = np.zeros(self.n_agents, dtype=np.float32)
        for i in range(self.n_agents):
            for j in range(len(self.patch_pos)):
                if np.linalg.norm(self.pos[i] - self.patch_pos[j]) < self.harvest_radius:
                    take = min(float(self.stock[j]), self.harvest_rate)
                    self.stock[j] -= take
                    harvest[i] += take
        self.total_harvest += harvest

        truncated = self.steps >= self.max_steps
        info = {"harvest": harvest.copy(), "stock": self.stock.copy(),
                "depleted": int(np.sum(self.stock < 0.5))}
        return self._obs(), harvest, False, truncated, info


# ---------------------------------------------------------------------------
# 网络与 PPO
# ---------------------------------------------------------------------------
class PPOActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.mu_head = nn.Linear(hidden, act_dim)
        self.v_head = nn.Linear(hidden, 1)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs: torch.Tensor):
        h = self.body(obs)
        mu = self.mu_head(h)
        std = torch.clamp(self.log_std, LOG_STD_MIN, LOG_STD_MAX).exp()
        v = self.v_head(h).squeeze(-1)
        return mu, std, v

    @torch.no_grad()
    def act_batch(self, obs_np: np.ndarray, device):
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
        mu, std, v = self.forward(obs)
        dist = Normal(mu, std)
        x = dist.sample()
        action = torch.tanh(x)
        logp = dist.log_prob(x) - torch.log(1.0 - action.pow(2) + 1e-6)
        return (action.cpu().numpy().astype(np.float32),
                logp.sum(dim=-1).cpu().numpy().astype(np.float32),
                v.cpu().numpy().astype(np.float32))

    @torch.no_grad()
    def act_deterministic(self, obs_np: np.ndarray, device) -> np.ndarray:
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
        mu, _, _ = self.forward(obs)
        return torch.tanh(mu).cpu().numpy().astype(np.float32)

    def log_prob_and_value(self, obs, act):
        mu, std, v = self.forward(obs)
        dist = Normal(mu, std)
        a = torch.clamp(act, -0.999999, 0.999999)
        x = torch.atanh(a)
        logp = dist.log_prob(x) - torch.log(1.0 - a.pow(2) + 1e-6)
        return logp.sum(dim=-1), v, dist.entropy().sum(dim=-1)


def compute_gae(rewards, values, dones, last_values, gamma, lam):
    """rewards 形状 (T, N)（每采集者一份奖励）。"""
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
    ep_harvest = np.zeros(env.n_agents, dtype=np.float32)

    for _ in range(steps):
        a, logp, v = net.act_batch(obs, device)
        next_obs, harvest, terminated, truncated, info = env.step(a)
        # 奖励制度：individual 各算各的；shared 所有人拿到全队总采集
        if args.regime == "shared":
            rew = np.full(env.n_agents, float(harvest.sum()), dtype=np.float32)
        else:
            rew = harvest.copy()
        obs_buf.append(obs)
        act_buf.append(a)
        logp_buf.append(logp)
        val_buf.append(v)
        rew_buf.append(rew)
        done_buf.append(np.zeros(env.n_agents, dtype=np.float32))
        ep_ret += float(rew.mean())
        ep_harvest += harvest
        ep_len += 1
        if terminated or truncated:
            ep_stats.append({"return": ep_ret, "harvest": float(ep_harvest.sum()),
                             "stock": float(env.stock.sum()),
                             "depleted": info["depleted"], "len": ep_len})
            obs = env.reset(rng)
            ep_ret = 0.0
            ep_harvest = np.zeros(env.n_agents, dtype=np.float32)
            ep_len = 0
        else:
            obs = next_obs

    with torch.no_grad():
        _, _, last_v = net.forward(
            torch.as_tensor(obs, dtype=torch.float32, device=device))
        last_values = last_v.cpu().numpy().astype(np.float32)
    values = np.stack(val_buf)
    adv, ret = compute_gae(np.stack(rew_buf), values, np.stack(done_buf),
                           last_values, args.gamma, args.lam)
    T, N = adv.shape
    batch = {
        "obs": np.stack(obs_buf).reshape(T * N, -1),
        "act": np.stack(act_buf).reshape(T * N, -1),
        "logp": np.stack(logp_buf).reshape(T * N),
        "adv": adv.reshape(T * N),
        "ret": ret.reshape(T * N),
    }
    return batch, ep_stats, obs


def ppo_update(net, optimizer, batch, args, device):
    obs = torch.as_tensor(batch["obs"], dtype=torch.float32, device=device)
    act = torch.as_tensor(batch["act"], dtype=torch.float32, device=device)
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
            logp, v, entropy = net.log_prob_and_value(obs[mb], act[mb])
            ratio = torch.exp(logp - logp_old[mb])
            surr1 = ratio * adv[mb]
            surr2 = torch.clamp(ratio, 1.0 - args.clip, 1.0 + args.clip) * adv[mb]
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(v, ret[mb])
            loss = policy_loss + 0.5 * value_loss - args.ent_coef * entropy.mean()
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
# 评估：学习策略与"就近采集"脚本基线
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(net, episodes, device, seed, mode="learned"):
    env = ResourceEnv()
    rng = np.random.default_rng(seed)
    total, second_half, final_stock, depleted_ratio = [], [], [], []
    for _ in range(episodes):
        obs = env.reset(rng)
        done, n = False, 0
        half_harvest, dh = 0.0, 0
        info = {"depleted": 0}
        while not done:
            if mode == "learned":
                a = net.act_deterministic(obs, device)
            else:   # 就近采集：选"最近且存量 > 1"的资源，直接朝其方向走
                a = np.zeros((env.n_agents, env.act_dim), dtype=np.float32)
                for i in range(env.n_agents):
                    d = np.linalg.norm(env.patch_pos - env.pos[i], axis=1)
                    cand = [j for j in range(4) if env.stock[j] > 1.0]
                    j = int(np.argmin(d[cand])) if cand else int(np.argmin(d))
                    vec = env.patch_pos[j] - env.pos[i]
                    a[i] = vec / (np.linalg.norm(vec) + 1e-8)
            obs, rew, terminated, truncated, info = env.step(a)
            n += 1
            if n > env.max_steps // 2:
                half_harvest += float(info["harvest"].sum())
                dh += info["depleted"]
            done = terminated or truncated
        total.append(float(env.total_harvest.sum()))
        second_half.append(half_harvest)
        final_stock.append(float(env.stock.sum()))
        depleted_ratio.append(dh / max(n // 2, 1))
    return (float(np.mean(total)), float(np.mean(second_half)),
            float(np.mean(final_stock)), float(np.mean(depleted_ratio)))


def parse_args():
    p = argparse.ArgumentParser(description="第095章：可再生资源收集（公地悲剧对比）")
    p.add_argument("--regime", type=str, default="individual",
                   choices=["individual", "shared"], help="奖励制度")
    p.add_argument("--max-steps", type=int, default=60000, help="环境步数上限")
    p.add_argument("--rollout", type=int, default=2048, help="每次更新采样步数")
    p.add_argument("--epochs", type=int, default=4, help="PPO 更新轮数")
    p.add_argument("--batch-size", type=int, default=256, help="小批量大小")
    p.add_argument("--lr", type=float, default=3e-4, help="学习率")
    p.add_argument("--gamma", type=float, default=0.97, help="折扣因子")
    p.add_argument("--lam", type=float, default=0.95, help="GAE lambda")
    p.add_argument("--clip", type=float, default=0.2, help="裁剪范围")
    p.add_argument("--ent-coef", type=float, default=0.01, help="熵正则系数")
    p.add_argument("--hidden", type=int, default=128, help="隐藏层宽度")
    p.add_argument("--eval-every", type=int, default=10000, help="评估间隔（步）")
    p.add_argument("--eval-episodes", type=int, default=5, help="评估回合数")
    p.add_argument("--seed", type=int, default=0, help="随机种子")
    p.add_argument("--save-dir", type=str, default="runs/ch095", help="模型保存目录")
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
    env = ResourceEnv()
    rng = np.random.default_rng(args.seed)
    net = PPOActorCritic(env.obs_dim, env.act_dim, args.hidden).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    print(f"设备 {device} | 4 采集者 / 4 资源 | 奖励制度 {args.regime} | 种子 {args.seed}")

    tot, half, stock, dep = evaluate(net, args.eval_episodes, device,
                                     args.seed + 111, mode="scripted")
    print(f"就近采集基线：总采集 {tot:.2f} | 后半程 {half:.2f} | "
          f"末态存量 {stock:.2f} | 耗尽比例 {dep:.2f}")

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
            tot, half, stock, dep = evaluate(net, args.eval_episodes, device,
                                             args.seed + 777)
            recent = all_stats[-10:]
            m_h = np.mean([s["harvest"] for s in recent]) if recent else float("nan")
            m_s = np.mean([s["stock"] for s in recent]) if recent else float("nan")
            print(f"[更新 {update_idx:3d}] 步 {total_steps:6d} | 总采集 {tot:6.2f} | "
                  f"后半程 {half:6.2f} | 末态存量 {stock:5.2f} | 耗尽 {dep:.2f} | "
                  f"训练局采集 {m_h:6.2f} 存量 {m_s:5.2f} | KL {stats['kl']:.4f}")

    elapsed = time.time() - t_start
    tot, half, stock, dep = evaluate(net, args.eval_episodes, device, args.seed + 999)
    print("-" * 74)
    print(f"训练结束 | 用时 {elapsed:.1f}s | 环境步 {total_steps} | 更新 {update_idx} 次")
    print(f"评估（{args.eval_episodes} 回合，确定性策略）：总采集 {tot:.2f} | "
          f"后半程采集 {half:.2f} | 末态总存量 {stock:.2f} | 耗尽步比例 {dep:.2f}")
    print(f"训练尾段（最后10局）：平均采集 "
          f"{np.mean([s['harvest'] for s in all_stats[-10:]]):.2f} | 平均末态存量 "
          f"{np.mean([s['stock'] for s in all_stats[-10:]]):.2f}")

    os.makedirs(args.save_dir, exist_ok=True)
    path = os.path.join(args.save_dir, f"ippo_resource_{args.regime}.pt")
    torch.save({"policy": net.state_dict(), "args": vars(args)}, path)
    print(f"模型已保存到 {path}")


if __name__ == "__main__":
    main()
