"""
第090章 多智能体环境：交通仿真

实验内容（2x2 路网的信号灯协同控制，环境为脚本内联的纯 numpy 实现）：
  - 4 个交叉口各有一组信号灯，动作 = 放行"南北"或"东西"相位
  - 车辆在路网上行驶：外部按概率到达、绿灯按通行能力放行、下游拥堵会
    向上游排队回溢（spillback），路口之间通过车流耦合
  - 奖励（每路口独立）：-（本口排队）- 0.25×（下游排队）- 0.5×换相惩罚
  - 算法：参数共享的独立 PPO（离散动作，Categorical 策略）
  - 对比：学习策略 vs 固定配时（每 4 步轮换）vs 贪心（每步选排队多的相位）
  - 指标：平均总排队长度、通行量（驶出路网车辆/步）、换相频率

运行：
    python code/ch090.py           # 完整训练（CPU 约 3-5 分钟）
    python code/ch090.py --quick   # 快速跑通（约 30-60 秒）

预期：学习策略的平均排队长度显著低于固定配时（典型改善 15%～40%，
以实跑为准），换相频率低于贪心基线。
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
# 环境：2x2 路网信号控制
# ---------------------------------------------------------------------------
class TrafficGridEnv:
    """2x2 交叉口路网。车道索引用 [N, E, S, W] 表示"车从哪一侧来"。

    车辆行驶规则：从小号行/列流入，向大号行/列驶出（如 W 车道向东行驶）。
    绿灯相位放出 min(排队, capacity) 辆车，流向下一路口的同名车道；
    下游车道满则回溢（车辆滞留原车道）。
    """

    n_agents = 4
    obs_dim = 11
    n_actions = 2                     # 0=南北放行, 1=东西放行
    lane_names = ["北", "东", "南", "西"]

    def __init__(self, arrival_p: float = 0.5, capacity: int = 2,
                 max_queue: int = 20, max_steps: int = 300,
                 switch_penalty: float = 0.5):
        self.arrival_p = arrival_p
        self.capacity = capacity
        self.max_queue = max_queue
        self.max_steps = max_steps
        self.switch_penalty = switch_penalty
        self.rng = np.random.default_rng(0)

    def reset(self, rng: np.random.Generator):
        self.rng = rng
        self.queue = np.zeros((2, 2, 4), dtype=np.int32)   # [行, 列, 车道]
        self.phase = np.zeros(4, dtype=np.int64)           # 当前相位
        self.since_switch = np.zeros(4, dtype=np.int64)
        self.steps = 0
        self.exited = 0
        self.switches = 0
        return self._obs()

    # ---------------- 车道与邻居关系 ----------------
    def _downstream(self, r: int, c: int, lane: int):
        """返回 (下一路口行, 列, 车道)；出口车道返回 None。"""
        if lane == 0:      # 北来车 → 向南行驶
            nr, nc = r + 1, c
        elif lane == 2:    # 南来车 → 向北行驶
            nr, nc = r - 1, c
        elif lane == 3:    # 西来车 → 向东行驶
            nr, nc = r, c + 1
        else:              # 东来车 → 向西行驶
            nr, nc = r, c - 1
        if 0 <= nr < 2 and 0 <= nc < 2:
            return nr, nc, lane
        return None

    def _obs(self) -> np.ndarray:
        obs = []
        for r in range(2):
            for c in range(2):
                i = r * 2 + c
                q = self.queue[r, c] / 10.0
                phase_onehot = np.array([1.0, 0.0]) if self.phase[i] == 0 else np.array([0.0, 1.0])
                since = min(self.since_switch[i], 10) / 10.0
                down = []
                for lane in range(4):
                    nxt = self._downstream(r, c, lane)
                    down.append(self.queue[nxt] / 10.0 if nxt else -1.0)
                o = np.concatenate([q, phase_onehot, [since], np.asarray(down)])
                obs.append(o.astype(np.float32))
        return np.stack(obs)

    def step(self, actions: np.ndarray):
        rng = self.rng
        rewards = np.zeros(4, dtype=np.float32)
        released_total = 0

        # 1) 换相处理（带惩罚与时延统计）
        for i in range(4):
            if int(actions[i]) != int(self.phase[i]):
                self.phase[i] = int(actions[i])
                self.since_switch[i] = 0
                self.switches += 1
                rewards[i] -= self.switch_penalty
            else:
                self.since_switch[i] += 1

        # 2) 绿灯放行：南北相位放 0/2 车道，东西相位放 1/3 车道
        new_queue = self.queue.copy()
        for r in range(2):
            for c in range(2):
                i = r * 2 + c
                lanes = (0, 2) if self.phase[i] == 0 else (1, 3)
                for lane in lanes:
                    movable = min(int(self.queue[r, c, lane]), self.capacity)
                    if movable == 0:
                        continue
                    nxt = self._downstream(r, c, lane)
                    new_queue[r, c, lane] -= movable
                    if nxt is None:
                        self.exited += movable               # 驶出路网
                        released_total += movable
                    else:
                        nr, nc, nl = nxt
                        space = self.max_queue - new_queue[nr, nc, nl]
                        enter = min(movable, max(space, 0))
                        new_queue[nr, nc, nl] += enter
                        new_queue[r, c, lane] += movable - enter  # 回溢：滞留
                        released_total += enter
        self.queue = new_queue

        # 3) 外部到达：每条边界进口车道按概率来车
        for r in range(2):
            for c in range(2):
                for lane in range(4):
                    is_entry = ((lane == 0 and r == 0) or (lane == 2 and r == 1)
                                or (lane == 3 and c == 0) or (lane == 1 and c == 1))
                    if is_entry and rng.random() < self.arrival_p:
                        if self.queue[r, c, lane] < self.max_queue:
                            self.queue[r, c, lane] += 1

        self.steps += 1

        # 4) 奖励：本地排队 + 下游排队（耦合项）
        for r in range(2):
            for c in range(2):
                i = r * 2 + c
                down_sum = 0
                for lane in range(4):
                    nxt = self._downstream(r, c, lane)
                    if nxt:
                        down_sum += self.queue[nxt]
                rewards[i] += -float(self.queue[r, c].sum()) - 0.25 * down_sum

        truncated = self.steps >= self.max_steps
        info = {"total_queue": int(self.queue.sum()), "exited": self.exited,
                "switches": self.switches, "released": released_total}
        return self._obs(), rewards, False, truncated, info


# ---------------------------------------------------------------------------
# 网络：离散动作的 Actor-Critic
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


# ---------------------------------------------------------------------------
# GAE（每智能体不同的奖励）
# ---------------------------------------------------------------------------
def compute_gae(rewards, values, dones, last_values, gamma, lam):
    """rewards/values/dones 形状均为 (T, N)。"""
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


def collect_rollout(env, net, rng, steps, obs, device):
    obs_buf, act_buf, logp_buf, val_buf = [], [], [], []
    rew_buf, done_buf = [], []
    queue_hist = []
    for _ in range(steps):
        a, logp, v = net.act_batch(obs, device)
        next_obs, rew, terminated, truncated, info = env.step(a)
        obs_buf.append(obs)
        act_buf.append(a)
        logp_buf.append(logp)
        val_buf.append(v)
        rew_buf.append(rew.astype(np.float32))
        done_buf.append(np.zeros(env.n_agents, dtype=np.float32))
        queue_hist.append(info["total_queue"])
        if terminated or truncated:
            obs = env.reset(rng)
        else:
            obs = next_obs
    with torch.no_grad():
        _, last_v = net.forward(
            torch.as_tensor(obs, dtype=torch.float32, device=device))
        last_values = last_v.cpu().numpy().astype(np.float32)
    values = np.stack(val_buf)
    adv, ret = compute_gae(np.stack(rew_buf), values, np.stack(done_buf),
                           last_values, 0.95, 0.95)
    T, N = adv.shape
    batch = {
        "obs": np.stack(obs_buf).reshape(T * N, -1),
        "act": np.stack(act_buf).reshape(T * N),
        "logp": np.stack(logp_buf).reshape(T * N),
        "adv": adv.reshape(T * N),
        "ret": ret.reshape(T * N),
    }
    return batch, queue_hist, obs


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
# 评估：学习策略与两类规则基线
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_policy(net, episodes, device, seed, mode="learned"):
    """mode: learned / fixed / greedy。返回平均排队、通行/步、换相/步。"""
    env = TrafficGridEnv()
    rng = np.random.default_rng(seed)
    total_q, throughput, switch_rate = [], [], []
    for _ in range(episodes):
        obs = env.reset(rng)
        done = False
        q_sum, n = 0, 0
        while not done:
            total_q_val = int(env.queue.sum())
            if mode == "learned":
                a = net.act_deterministic(obs, device)
            elif mode == "fixed":
                a = np.full(4, (n // 4) % 2, dtype=np.int64)
            else:   # greedy：每步选排队多的相位
                a = np.zeros(4, dtype=np.int64)
                for r in range(2):
                    for c in range(2):
                        i = r * 2 + c
                        ns = env.queue[r, c, 0] + env.queue[r, c, 2]
                        ew = env.queue[r, c, 1] + env.queue[r, c, 3]
                        a[i] = 0 if ns >= ew else 1
            obs, rew, terminated, truncated, info = env.step(a)
            q_sum += total_q_val
            n += 1
            done = terminated or truncated
        total_q.append(q_sum / max(n, 1))
        throughput.append(info["exited"] / max(n, 1))
        switch_rate.append(info["switches"] / max(n, 1))
    return (float(np.mean(total_q)), float(np.mean(throughput)),
            float(np.mean(switch_rate)))


def parse_args():
    p = argparse.ArgumentParser(description="第090章：交通信号协同控制")
    p.add_argument("--max-steps", type=int, default=40000, help="环境步数上限")
    p.add_argument("--rollout", type=int, default=2048, help="每次更新采样步数")
    p.add_argument("--epochs", type=int, default=4, help="PPO 更新轮数")
    p.add_argument("--batch-size", type=int, default=256, help="小批量大小")
    p.add_argument("--lr", type=float, default=3e-4, help="学习率")
    p.add_argument("--clip", type=float, default=0.2, help="裁剪范围")
    p.add_argument("--ent-coef", type=float, default=0.01, help="熵正则系数")
    p.add_argument("--hidden", type=int, default=128, help="隐藏层宽度")
    p.add_argument("--eval-every", type=int, default=8000, help="评估间隔（步）")
    p.add_argument("--eval-episodes", type=int, default=5, help="评估回合数")
    p.add_argument("--seed", type=int, default=0, help="随机种子")
    p.add_argument("--save-dir", type=str, default="runs/ch090", help="模型保存目录")
    p.add_argument("--quick", action="store_true", help="快速模式：缩小训练规模")
    return p.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.max_steps = min(args.max_steps, 12000)
        args.rollout = 1024
        args.eval_every = 4000
        args.eval_episodes = 3
    set_seed(args.seed)

    device = torch.device("cpu")
    env = TrafficGridEnv()
    rng = np.random.default_rng(args.seed)
    net = PPOActorCritic(env.obs_dim, env.n_actions, args.hidden).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    print(f"设备 {device} | 2x2 路网 | 4 个信号灯智能体 | 种子 {args.seed}")

    # 规则基线
    q_fix, tp_fix, sw_fix = run_policy(net, args.eval_episodes, device,
                                       args.seed + 111, mode="fixed")
    q_grd, tp_grd, sw_grd = run_policy(net, args.eval_episodes, device,
                                       args.seed + 111, mode="greedy")
    print(f"固定配时基线：平均排队 {q_fix:.2f} | 通行/步 {tp_fix:.2f} | 换相/步 {sw_fix:.2f}")
    print(f"贪心基线    ：平均排队 {q_grd:.2f} | 通行/步 {tp_grd:.2f} | 换相/步 {sw_grd:.2f}")

    obs = env.reset(rng)
    total_steps, update_idx = 0, 0
    queue_hist = []
    t_start = time.time()

    while total_steps < args.max_steps:
        batch, qh, obs = collect_rollout(env, net, rng, args.rollout, obs, device)
        total_steps += args.rollout
        queue_hist.extend(qh)
        stats = ppo_update(net, optimizer, batch, args, device)
        update_idx += 1

        if total_steps % args.eval_every < args.rollout:
            q_l, tp_l, sw_l = run_policy(net, args.eval_episodes, device,
                                         args.seed + 777, mode="learned")
            recent_q = float(np.mean(queue_hist[-2000:]))
            print(f"[更新 {update_idx:3d}] 步 {total_steps:6d} | 评估平均排队 {q_l:6.2f} | "
                  f"通行/步 {tp_l:.2f} | 换相/步 {sw_l:.2f} | "
                  f"训练近2000步排队 {recent_q:.2f} | KL {stats['kl']:.4f}")

    elapsed = time.time() - t_start
    q_l, tp_l, sw_l = run_policy(net, args.eval_episodes, device, args.seed + 999)
    print("-" * 74)
    print(f"训练结束 | 用时 {elapsed:.1f}s | 环境步 {total_steps} | 更新 {update_idx} 次")
    print(f"评估（{args.eval_episodes} 回合）：学习策略 排队 {q_l:.2f} / 通行 {tp_l:.2f} / "
          f"换相 {sw_l:.2f}")
    print(f"              固定配时 排队 {q_fix:.2f} / 通行 {tp_fix:.2f} / 换相 {sw_fix:.2f}")
    print(f"              贪心策略 排队 {q_grd:.2f} / 通行 {tp_grd:.2f} / 换相 {sw_grd:.2f}")
    improve = (q_fix - q_l) / max(q_fix, 1e-6) * 100.0
    print(f"相对固定配时的排队改善：{improve:+.1f}%")

    os.makedirs(args.save_dir, exist_ok=True)
    path = os.path.join(args.save_dir, "ippo_traffic.pt")
    torch.save({"policy": net.state_dict(), "args": vars(args)}, path)
    print(f"模型已保存到 {path}")


if __name__ == "__main__":
    main()
