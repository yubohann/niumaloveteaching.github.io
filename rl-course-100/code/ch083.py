"""
第083章 多智能体SAC：混合任务

实验内容（2v2 连续空间抢道具，环境为脚本内联的纯 numpy 实现）：
  - 4 个智能体分成两队（A 队：0/1 号，B 队：2/3 号），共享一个道具场
  - 混合动机：道具被捡走后立即在别处重生，队内共享回报（合作），
    两队抢同一批道具（竞争）
  - 每个智能体训练一个独立的 SAC，但评论家的输入有两种模式：
      indep：只看自己的观测与动作（完全独立）
      ctde ：评论家额外看"队友的观测与动作"（集中训练、分散执行）
  - 指标：双方累计得分差、每队场均得分、队友"撞目标率"（两人最近道具相同）

运行：
    python code/ch083.py                  # 默认 ctde，约 4-8 分钟（CPU）
    python code/ch083.py --quick          # 快速跑通，约 40-70 秒
    python code/ch083.py --mode indep     # 基线：完全独立的 SAC

预期：ctde 模式下队友撞目标率下降、净胜分分布更稳定；两种模式的差距随种子
波动，请固定 --seed 做公平对比，并只看多次运行的平均趋势。
"""

import argparse
import copy
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
    """固定随机种子，保证实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# 环境：2v2 抢道具（混合合作-竞争）
# ---------------------------------------------------------------------------
class TeamGatherEnv:
    """两队争抢会重生的道具。

    观测（15 维/智能体）：自己的位置+速度(4)、队友位置+速度(4)、
    两名对手的位置(4)、最近道具的相对坐标(2)、比分差(1)。
    动作：2 维连续加速度，限幅 [-1, 1]。
    奖励：捡到道具 = 自己 +1、队友 +1（队内合作）；每步 -0.02（时间成本）；
    队友靠太近互相挤压再 -0.05。
    """

    obs_dim = 15
    act_dim = 2
    teams = (0, 0, 1, 1)
    teammates = (1, 0, 3, 2)
    opponents = ((2, 3), (2, 3), (0, 1), (0, 1))

    def __init__(self, n_items: int = 2, pickup_radius: float = 0.09,
                 max_steps: int = 100, friction: float = 0.8,
                 accel: float = 0.12, v_max: float = 1.2, bound: float = 1.0):
        self.n_items = n_items
        self.pickup_radius = pickup_radius
        self.max_steps = max_steps
        self.friction = friction
        self.accel = accel
        self.v_max = v_max
        self.bound = bound
        self.rng = np.random.default_rng(0)

    def reset(self, rng: np.random.Generator):
        """A 队从左半场出发，B 队从右半场出发；道具随机撒在场地中部。"""
        self.rng = rng
        self.pos = rng.uniform(-0.5, 0.5, size=(4, 2)).astype(np.float32)
        self.pos[:2, 0] = rng.uniform(-0.7, -0.3, size=2)
        self.pos[2:, 0] = rng.uniform(0.3, 0.7, size=2)
        self.vel = np.zeros((4, 2), dtype=np.float32)
        self.items = rng.uniform(-0.7, 0.7, size=(self.n_items, 2)).astype(np.float32)
        self.steps = 0
        self.score = np.zeros(2, dtype=np.float32)
        return self._obs()

    def _nearest_item(self, p: np.ndarray):
        d = np.linalg.norm(self.items - p, axis=1)
        k = int(np.argmin(d))
        return k, float(d[k])

    def _obs(self) -> np.ndarray:
        obs = []
        for i in range(4):
            mate = self.teammates[i]
            opp0, opp1 = self.opponents[i]
            k, _ = self._nearest_item(self.pos[i])
            rel = self.items[k] - self.pos[i]
            my_team = self.teams[i]
            diff = self.score[my_team] - self.score[1 - my_team]
            o = np.concatenate([
                self.pos[i], self.vel[i],           # 自己
                self.pos[mate], self.vel[mate],     # 队友
                self.pos[opp0], self.pos[opp1],     # 两名对手的位置
                rel,                                # 最近道具的相对坐标
                np.array([diff]),                   # 当前比分差
            ])
            obs.append(o)
        return np.stack(obs).astype(np.float32)

    def step(self, actions: np.ndarray):
        """动作 shape=(4,2)。返回 obs, rewards, terminated, truncated, info。"""
        a = np.clip(actions, -1.0, 1.0).astype(np.float32)
        self.vel = np.clip(self.friction * self.vel + self.accel * a,
                           -self.v_max, self.v_max)
        self.pos = np.clip(self.pos + self.vel, -self.bound, self.bound)
        self.steps += 1

        rewards = np.full(4, -0.02, dtype=np.float32)   # 时间成本
        collected_by_team = [0, 0]
        for i in range(4):
            k, d = self._nearest_item(self.pos[i])
            if d <= self.pickup_radius:
                team = self.teams[i]
                self.score[team] += 1.0
                collected_by_team[team] += 1
                rewards[i] += 1.0                       # 捡到者
                rewards[self.teammates[i]] += 1.0       # 队内共享：合作信号
                # 道具换位重生
                self.items[k] = self.rng.uniform(-0.7, 0.7, size=2).astype(np.float32)
        # 队友挤在一起的轻微惩罚：迫使队友分工而不是粘连
        for i in range(4):
            if np.linalg.norm(self.pos[i] - self.pos[self.teammates[i]]) < 0.08:
                rewards[i] -= 0.05

        truncated = self.steps >= self.max_steps
        info = {"score": self.score.copy(), "collected": collected_by_team,
                "overlap": self.team_overlap()}
        return self._obs(), rewards, False, truncated, info

    def team_overlap(self):
        """1 表示该队两人盯上了同一个道具（重复用工），0 表示分工。"""
        res = []
        for t in (0, 1):
            members = [i for i in range(4) if self.teams[i] == t]
            k0, _ = self._nearest_item(self.pos[members[0]])
            k1, _ = self._nearest_item(self.pos[members[1]])
            res.append(1.0 if k0 == k1 else 0.0)
        return res


# ---------------------------------------------------------------------------
# 网络
# ---------------------------------------------------------------------------
class SquashedGaussianActor(nn.Module):
    """SAC 连续策略：高斯 + tanh 压缩，含对数概率修正。"""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden, act_dim)
        self.log_std_head = nn.Linear(hidden, act_dim)

    def forward(self, obs: torch.Tensor, deterministic: bool = False):
        h = self.net(obs)
        mu = self.mu_head(h)
        log_std = torch.clamp(self.log_std_head(h), LOG_STD_MIN, LOG_STD_MAX)
        dist = Normal(mu, log_std.exp())
        x = mu if deterministic else dist.rsample()
        action = torch.tanh(x)
        log_prob = dist.log_prob(x) - torch.log(1.0 - action.pow(2) + 1e-6)
        return action, log_prob.sum(dim=-1, keepdim=True)


class QNet(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# 单个智能体的 SAC
# ---------------------------------------------------------------------------
class SACAgent:
    def __init__(self, agent_id: int, env: TeamGatherEnv, args, device):
        self.agent_id = agent_id
        self.args = args
        self.device = device
        self.act_dim = env.act_dim
        # ctde 模式下评论家多接收队友的观测+动作（仅训练时可用）
        extra = (env.obs_dim + env.act_dim) if args.mode == "ctde" else 0
        critic_in = env.obs_dim + env.act_dim + extra

        self.actor = SquashedGaussianActor(env.obs_dim, env.act_dim, args.hidden).to(device)
        self.q1 = QNet(critic_in, args.hidden).to(device)
        self.q2 = QNet(critic_in, args.hidden).to(device)
        self.q1_target = copy.deepcopy(self.q1)
        self.q2_target = copy.deepcopy(self.q2)
        for net in (self.q1_target, self.q2_target):
            for p in net.parameters():
                p.requires_grad_(False)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=args.lr)
        self.critic_opt = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=args.lr)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=args.lr)
        self.target_entropy = -float(env.act_dim)

        self.buffer = []
        self.pending_idx = None   # 用于回填"下一状态的队友动作"

    # ---------------- 交互 ----------------
    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        action, _ = self.actor(obs_t, deterministic)
        return action.squeeze(0).cpu().numpy().astype(np.float32)

    def store(self, obs, act, rew, next_obs, terminated,
              mate_obs, mate_act, next_mate_obs, next_mate_act):
        """存一条转移；调用前应先调用 patch_next_mate_act 回填上一条。"""
        entry = {
            "obs": obs, "act": act, "rew": rew, "next_obs": next_obs,
            "done": float(terminated),
            "mate_obs": mate_obs, "mate_act": mate_act,
            "next_mate_obs": next_mate_obs, "next_mate_act": next_mate_act,
        }
        self.buffer.append(entry)
        if len(self.buffer) > self.args.buffer_size:
            del self.buffer[: len(self.buffer) // 2]
            self.pending_idx = None
        self.pending_idx = None if terminated else len(self.buffer) - 1

    def patch_next_mate_act(self, mate_act):
        """把本步队友动作写入上一条转移的 next_mate_act（同一回合内才有效）。"""
        if self.pending_idx is not None:
            self.buffer[self.pending_idx]["next_mate_act"] = mate_act

    def clear_pending(self):
        """回合结束（含时间截断）时调用：断开跨回合的动作回填链。"""
        self.pending_idx = None

    # ---------------- 评论家输入 ----------------
    def _q_input(self, obs, act, mate_obs, mate_act):
        if self.args.mode == "ctde":
            return torch.cat([obs, act, mate_obs, mate_act], dim=-1)
        return torch.cat([obs, act], dim=-1)

    # ---------------- 训练 ----------------
    def sample_batch(self, size: int):
        idx = np.random.randint(0, len(self.buffer), size=size)
        entries = [self.buffer[i] for i in idx]

        def stack(key):
            return torch.as_tensor(np.stack([e[key] for e in entries]),
                                   dtype=torch.float32, device=self.device)

        zero_a = np.zeros(self.act_dim, dtype=np.float32)
        next_mate_act = torch.as_tensor(
            np.stack([e["next_mate_act"] if e["next_mate_act"] is not None else zero_a
                      for e in entries]),
            dtype=torch.float32, device=self.device)
        return (stack("obs"), stack("act"), stack("rew").unsqueeze(-1),
                stack("next_obs"), stack("done").unsqueeze(-1),
                stack("mate_obs"), stack("mate_act"),
                stack("next_mate_obs"), next_mate_act)

    def update(self, batch):
        (obs, act, rew, next_obs, done,
         mate_obs, mate_act, next_mate_obs, next_mate_act) = batch
        alpha = self.log_alpha.exp().detach()

        with torch.no_grad():
            next_a, next_logp = self.actor(next_obs)
            q_in = self._q_input(next_obs, next_a, next_mate_obs, next_mate_act)
            q1_next = self.q1_target(q_in)
            q2_next = self.q2_target(q_in)
            y = rew + self.args.gamma * (1.0 - done) * (
                torch.min(q1_next, q2_next) - alpha * next_logp)

        q_in = self._q_input(obs, act, mate_obs, mate_act)
        critic_loss = F.mse_loss(self.q1(q_in), y) + F.mse_loss(self.q2(q_in), y)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        new_a, logp = self.actor(obs)
        q_in = self._q_input(obs, new_a, mate_obs, mate_act)
        actor_loss = (alpha * logp - torch.min(self.q1(q_in), self.q2(q_in))).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        with torch.no_grad():
            for net, tnet in ((self.q1, self.q1_target), (self.q2, self.q2_target)):
                for p, tp in zip(net.parameters(), tnet.parameters()):
                    tp.data.mul_(1.0 - self.args.tau).add_(self.args.tau * p.data)
        return float(critic_loss.item()), float(actor_loss.item())


# ---------------------------------------------------------------------------
# 评估
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(env, agents, episodes, rng):
    diffs, scores, overlaps = [], [], []
    for _ in range(episodes):
        obs = env.reset(rng)
        done = False
        info = {"score": np.zeros(2, dtype=np.float32), "overlap": [0.0, 0.0]}
        overlap_acc = np.zeros(2)
        n = 0
        while not done:
            actions = np.stack([agents[i].act(obs[i], deterministic=True)
                                for i in range(len(agents))])
            obs, rew, terminated, truncated, info = env.step(actions)
            overlap_acc += np.asarray(info["overlap"])
            n += 1
            done = terminated or truncated
        diffs.append(float(info["score"][0] - info["score"][1]))
        scores.append(info["score"].copy())
        overlaps.append(overlap_acc / max(n, 1))
    scores = np.stack(scores)
    overlaps = np.stack(overlaps)
    return diffs, scores, overlaps


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="第083章：多智能体SAC 混合任务")
    p.add_argument("--mode", type=str, default="ctde", choices=["indep", "ctde"],
                   help="评论家是否接收队友信息")
    p.add_argument("--max-steps", type=int, default=40000, help="环境联合步数上限")
    p.add_argument("--warmup", type=int, default=2000, help="随机动作预热步数")
    p.add_argument("--update-every", type=int, default=2, help="每多少步训练一次")
    p.add_argument("--batch-size", type=int, default=128, help="采样批量")
    p.add_argument("--buffer-size", type=int, default=100000, help="回放缓冲上限")
    p.add_argument("--lr", type=float, default=3e-4, help="学习率")
    p.add_argument("--gamma", type=float, default=0.95, help="折扣因子")
    p.add_argument("--tau", type=float, default=0.01, help="软更新系数")
    p.add_argument("--hidden", type=int, default=128, help="隐藏层宽度")
    p.add_argument("--log-every", type=int, default=2500, help="日志间隔（步）")
    p.add_argument("--eval-episodes", type=int, default=20, help="最终评估回合数")
    p.add_argument("--seed", type=int, default=0, help="随机种子")
    p.add_argument("--save-dir", type=str, default="runs/ch083", help="模型保存目录")
    p.add_argument("--quick", action="store_true", help="快速模式：缩小训练规模")
    return p.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.max_steps = min(args.max_steps, 8000)
        args.warmup = min(args.warmup, 500)
        args.log_every = 1000
        args.eval_episodes = 10
    set_seed(args.seed)

    device = torch.device("cpu")
    env = TeamGatherEnv()
    rng = np.random.default_rng(args.seed)
    agents = [SACAgent(i, env, args, device) for i in range(4)]

    print(f"设备 {device} | 模式 {args.mode} | 种子 {args.seed}")
    print(f"配置：max_steps={args.max_steps}, warmup={args.warmup}, "
          f"batch={args.batch_size}, lr={args.lr}, gamma={args.gamma}")

    obs = env.reset(rng)
    team_scores = []       # 每回合 (A 队得分, B 队得分)
    overlap_hist = []      # 每回合两队撞目标率
    t_start = time.time()

    for step in range(1, args.max_steps + 1):
        # 1) 选动作
        if step <= args.warmup:
            actions = rng.uniform(-1.0, 1.0, size=(4, env.act_dim)).astype(np.float32)
        else:
            actions = np.stack([agents[i].act(obs[i]) for i in range(4)])

        # 2) 先回填"上一状态的下一队友动作"（同回合内有效）
        if step > args.warmup:
            for i in range(4):
                agents[i].patch_next_mate_act(actions[env.teammates[i]])

        next_obs, rew, terminated, truncated, info = env.step(actions)
        done = terminated or truncated

        # 3) 各智能体存转移（teammate 视角随智能体不同）
        if step > args.warmup:
            for i in range(4):
                j = env.teammates[i]
                agents[i].store(obs[i], actions[i], rew[i], next_obs[i],
                                float(terminated), obs[j], actions[j], next_obs[j],
                                None)     # next_mate_act 留待下一步回填
                if done:
                    agents[i].clear_pending()

        # 4) 训练
        if step > args.warmup and step % args.update_every == 0:
            for agent in agents:
                if len(agent.buffer) >= args.batch_size:
                    agent.update(agent.sample_batch(args.batch_size))

        # 5) 回合结束
        if done:
            team_scores.append(info["score"].copy())
            overlap_hist.append(np.asarray(info["overlap"]).copy())
            obs = env.reset(rng)
        else:
            obs = next_obs

        # 6) 日志
        if step % args.log_every == 0 and team_scores:
            recent = team_scores[-20:]
            a_mean = float(np.mean([s[0] for s in recent]))
            b_mean = float(np.mean([s[1] for s in recent]))
            ov = np.stack(overlap_hist[-20:]).mean(axis=0)
            print(f"[步 {step:6d}] 回合 {len(team_scores):5d} | "
                  f"A 队近20均 {a_mean:5.2f} | B 队近20均 {b_mean:5.2f} | "
                  f"撞目标率 A {ov[0]:.2f} / B {ov[1]:.2f}")

    elapsed = time.time() - t_start

    # 7) 评估
    eval_rng = np.random.default_rng(args.seed + 12345)
    diffs, scores, overlaps = evaluate(env, agents, args.eval_episodes, eval_rng)
    print("-" * 74)
    print(f"训练结束 | 用时 {elapsed:.1f}s | 联合步数 {args.max_steps} | "
          f"回合 {len(team_scores)}")
    print(f"评估（{args.eval_episodes} 回合）：A 队场均 {scores[:, 0].mean():.2f} | "
          f"B 队场均 {scores[:, 1].mean():.2f} | "
          f"净胜分 {np.mean(diffs):+.2f}（标准差 {np.std(diffs):.2f}）")
    print(f"撞目标率（越低越好）：A 队 {overlaps[:, 0].mean():.2f} | "
          f"B 队 {overlaps[:, 1].mean():.2f}")

    os.makedirs(args.save_dir, exist_ok=True)
    ckpt = {f"agent{i}_actor": agents[i].actor.state_dict() for i in range(4)}
    ckpt["args"] = vars(args)
    path = os.path.join(args.save_dir, "masac_mixed_task.pt")
    torch.save(ckpt, path)
    print(f"模型已保存到 {path}")


if __name__ == "__main__":
    main()
