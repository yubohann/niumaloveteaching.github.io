"""
第082章 多智能体SAC：对手建模

实验内容（连续空间追逐-逃逸，环境为脚本内联的纯 numpy 实现）：
  - 2 个智能体：追者（pursuer）与逃者（evader），零和博弈
  - 每个智能体各训练一个独立的 SAC（双 Q + 目标网络 + 自动温度系数）
  - 关键改动：评论家（critic）的输入中加入"对手动作"。对手动作有三种来源：
      off  ：不使用对手信息（独立 SAC 基线）
      pred ：用对手模型（监督学习）从对手观测预测其动作（本章主角）
      true ：直接使用对手真实动作（类似 MADDPG 的评论家，训练期上界）
  - 对手模型只做监督学习（MSE 回归），它的梯度不会回传给评论家与策略

运行：
    python code/ch082.py                    # 默认 pred：约 4-8 分钟（CPU）
    python code/ch082.py --quick            # 快速跑通：约 40-70 秒
    python code/ch082.py --opp-model off    # 基线：不做对手建模
    python code/ch082.py --opp-model true   # 上界：评论家看到对手真实动作

预期：pred 的追获率（tag rate）通常高于 off；true 是理论上界。收敛速度与最终
指标随随机种子波动，请以实跑为准，并固定 --seed 做公平对比。
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


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    """固定随机种子，保证实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# 环境：连续空间追逐-逃逸（纯 numpy）
# ---------------------------------------------------------------------------
class PursuitEvaderEnv:
    """2 个智能体的追逐-逃逸环境。

    状态：每个智能体观测 [自己位置(2), 自己速度(2), 对手位置(2), 对手速度(2)]。
    动作：2 维连续加速度，取值 [-1, 1]，经惯性积分后作用于位置。
    奖励：追者每步 -距离，逃者每步 +距离（零和）；追者贴上逃者时 +10/-10 并结束。
    """

    obs_dim = 8
    act_dim = 2

    def __init__(self, tag_radius: float = 0.2, max_steps: int = 80,
                 friction: float = 0.8, accel: float = 0.12,
                 v_max: float = 1.2, bound: float = 1.2):
        self.tag_radius = tag_radius
        self.max_steps = max_steps
        self.friction = friction
        self.accel = accel
        self.v_max = v_max
        self.bound = bound

    def reset(self, rng: np.random.Generator):
        """随机初始化双方位置：追者在左、逃者在右，速度从零开始。"""
        self.pos = np.stack([
            np.array([-0.8, 0.0]) + rng.uniform(-0.1, 0.1, size=2),
            np.array([0.8, 0.0]) + rng.uniform(-0.1, 0.1, size=2),
        ]).astype(np.float32)
        self.vel = np.zeros((2, 2), dtype=np.float32)
        self.steps = 0
        return self._obs()

    def _obs(self) -> np.ndarray:
        """拼装两个智能体的观测（互为对手视角）。"""
        p0 = np.concatenate([self.pos[0], self.vel[0], self.pos[1], self.vel[1]])
        p1 = np.concatenate([self.pos[1], self.vel[1], self.pos[0], self.vel[0]])
        return np.stack([p0, p1]).astype(np.float32)

    def step(self, actions: np.ndarray):
        """动作 shape=(2,2)。返回 obs, rewards, terminated, truncated, info。"""
        a = np.clip(actions, -1.0, 1.0).astype(np.float32)
        # 一阶惯性动力学：速度按摩擦衰减，再叠加加速度
        self.vel = np.clip(self.friction * self.vel + self.accel * a,
                           -self.v_max, self.v_max)
        self.pos = np.clip(self.pos + self.vel, -self.bound, self.bound)
        self.steps += 1

        dist = float(np.linalg.norm(self.pos[0] - self.pos[1]))
        rewards = np.array([-dist, dist], dtype=np.float32)  # 零和塑形奖励
        terminated = False
        if dist <= self.tag_radius:                          # 追者成功贴住逃者
            rewards[0] += 10.0
            rewards[1] -= 10.0
            terminated = True
        truncated = (not terminated) and self.steps >= self.max_steps
        return self._obs(), rewards, terminated, truncated, {"dist": dist}


# ---------------------------------------------------------------------------
# 网络
# ---------------------------------------------------------------------------
class SquashedGaussianActor(nn.Module):
    """SAC 的连续策略：高斯采样 + tanh 压缩到 [-1, 1]，并做对数概率修正。"""

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
        x = mu if deterministic else dist.rsample()      # 重参数化采样
        action = torch.tanh(x)
        # tanh 变换后的对数概率修正项
        log_prob = dist.log_prob(x) - torch.log(1.0 - action.pow(2) + 1e-6)
        return action, log_prob.sum(dim=-1, keepdim=True)


class QNet(nn.Module):
    """动作价值网络 Q(s, a)（含对手动作时输入维度会变大）。"""

    def __init__(self, in_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class OpponentModel(nn.Module):
    """对手模型：输入对手观测，输出对对手动作的预测（有界于 [-1,1]）。"""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, act_dim), nn.Tanh(),
        )

    def forward(self, opp_obs: torch.Tensor) -> torch.Tensor:
        return self.net(opp_obs)


# ---------------------------------------------------------------------------
# 单个智能体的 SAC（自带回放缓冲与对手模型）
# ---------------------------------------------------------------------------
class SACAgent:
    def __init__(self, name: str, env: PursuitEvaderEnv, args, device):
        self.name = name
        self.args = args
        self.device = device
        self.act_dim = env.act_dim
        # 评论家输入维度：自己的 obs+act，视模式加入对手动作
        extra = 0 if args.opp_model == "off" else env.act_dim
        critic_in = env.obs_dim + env.act_dim + extra

        self.actor = SquashedGaussianActor(env.obs_dim, env.act_dim, args.hidden).to(device)
        self.q1 = QNet(critic_in, args.hidden).to(device)
        self.q2 = QNet(critic_in, args.hidden).to(device)
        self.q1_target = copy.deepcopy(self.q1)
        self.q2_target = copy.deepcopy(self.q2)
        for net in (self.q1_target, self.q2_target):
            for p in net.parameters():
                p.requires_grad_(False)
        self.opp_model = OpponentModel(env.obs_dim, env.act_dim, args.hidden).to(device)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=args.lr)
        self.critic_opt = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=args.lr)
        self.opp_opt = torch.optim.Adam(self.opp_model.parameters(), lr=args.opp_lr)

        # 自动温度：log_alpha 可学习，目标熵为 -动作维度（SAC 论文的常用取值）
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=args.lr)
        self.target_entropy = -float(env.act_dim)

        self.buffer = []        # 每条为 dict：自己的转移 + 对手信息
        self.opp_data = []      # (对手观测, 对手动作) 监督数据
        self.pending_idx = None  # 上一条转移在 buffer 中的下标，用于补 next_opp_act

    # ---------------- 交互 ----------------
    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        action, _ = self.actor(obs_t, deterministic)
        return action.squeeze(0).cpu().numpy().astype(np.float32)

    def store(self, obs, act, rew, next_obs, terminated, opp_obs, next_opp_obs, opp_act):
        """存一条转移；并把上一条转移的 next_opp_act 补齐（回合内相邻）。"""
        entry = {
            "obs": obs, "act": act, "rew": rew, "next_obs": next_obs,
            "done": float(terminated),          # 只有真实终止才切断 bootstrap
            "opp_obs": opp_obs, "next_opp_obs": next_opp_obs,
            "opp_act": opp_act, "next_opp_act": None,
        }
        self.buffer.append(entry)
        if len(self.buffer) > self.args.buffer_size:
            del self.buffer[: len(self.buffer) // 2]
            self.pending_idx = None      # 下标失效，放弃补齐（仅影响极少数边界样本）
        if self.pending_idx is not None:
            # 上一条转移的"下一状态的对手动作"正是本步采样到的对手动作
            self.buffer[self.pending_idx]["next_opp_act"] = opp_act
        # 回合在真实终止处断开，后续动作不属于同一条轨迹
        self.pending_idx = None if terminated else len(self.buffer) - 1
        self.opp_data.append((opp_obs, opp_act))
        if len(self.opp_data) > self.args.buffer_size:
            del self.opp_data[: len(self.opp_data) // 2]

    # ---------------- 评论家输入 ----------------
    def _critic_input(self, obs, act, opp_obs, opp_act, next_opp_act=None):
        """按模式拼装评论家输入。

        off  : [obs, act]
        pred : [obs, act, opponent_model(obs_opp).detach()]
        true : [obs, act, 对手真实动作]
        """
        if self.args.opp_model == "off":
            return torch.cat([obs, act], dim=-1)
        if self.args.opp_model == "true":
            # 目标 Q 处没有未来动作时用零向量占位（bootstrap 边界，占比极小）
            oat = opp_act if next_opp_act is None else next_opp_act
            return torch.cat([obs, act, oat], dim=-1)
        pred = self.opp_model(opp_obs).detach()   # 预测动作不参与梯度回传
        return torch.cat([obs, act, pred], dim=-1)

    # ---------------- 训练 ----------------
    def sample_batch(self, size: int):
        idx = np.random.randint(0, len(self.buffer), size=size)
        entries = [self.buffer[i] for i in idx]

        def stack(key):
            return torch.as_tensor(np.stack([e[key] for e in entries]),
                                   dtype=torch.float32, device=self.device)

        zero = np.zeros(self.act_dim, dtype=np.float32)
        next_opp_act = torch.as_tensor(
            np.stack([e["next_opp_act"] if e["next_opp_act"] is not None else zero
                      for e in entries]),
            dtype=torch.float32, device=self.device)
        return (stack("obs"), stack("act"), stack("rew").unsqueeze(-1),
                stack("next_obs"), stack("done").unsqueeze(-1),
                stack("opp_obs"), stack("next_opp_obs"), stack("opp_act"),
                next_opp_act)

    def update(self, batch):
        """一次 SAC 更新：双 Q 损失 -> 策略损失 -> 温度损失 -> 软更新。"""
        (obs, act, rew, next_obs, done,
         opp_obs, next_opp_obs, opp_act, next_opp_act) = batch
        alpha = self.log_alpha.exp().detach()

        # 1) 双 Q 的 Bellman 回归目标（目标动作由当前策略采样）
        with torch.no_grad():
            next_a, next_logp = self.actor(next_obs)
            q_in = self._critic_input(next_obs, next_a, next_opp_obs,
                                      opp_act, next_opp_act)
            q1_next = self.q1_target(q_in)
            q2_next = self.q2_target(q_in)
            y = rew + self.args.gamma * (1.0 - done) * (
                torch.min(q1_next, q2_next) - alpha * next_logp)

        q_in = self._critic_input(obs, act, opp_obs, opp_act)
        critic_loss = F.mse_loss(self.q1(q_in), y) + F.mse_loss(self.q2(q_in), y)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # 2) 策略损失：最大化 min(Q) - alpha * logp
        new_a, logp = self.actor(obs)
        q_in = self._critic_input(obs, new_a, opp_obs, opp_act)
        actor_loss = (alpha * logp - torch.min(self.q1(q_in), self.q2(q_in))).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # 3) 温度自适应：让策略熵逼近目标熵
        alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        # 4) 目标网络软更新
        with torch.no_grad():
            for net, tnet in ((self.q1, self.q1_target), (self.q2, self.q2_target)):
                for p, tp in zip(net.parameters(), tnet.parameters()):
                    tp.data.mul_(1.0 - self.args.tau).add_(self.args.tau * p.data)

        return float(critic_loss.item()), float(actor_loss.item())

    def train_opp_model(self):
        """对手模型的监督学习：用历史 (对手观测, 对手动作) 做一步 MSE 回归。"""
        if len(self.opp_data) < self.args.opp_batch:
            return None
        idx = np.random.randint(0, len(self.opp_data), size=self.args.opp_batch)
        data = [self.opp_data[i] for i in idx]
        obs = torch.as_tensor(np.stack([d[0] for d in data]),
                              dtype=torch.float32, device=self.device)
        act = torch.as_tensor(np.stack([d[1] for d in data]),
                              dtype=torch.float32, device=self.device)
        loss = F.mse_loss(self.opp_model(obs), act)
        self.opp_opt.zero_grad()
        loss.backward()
        self.opp_opt.step()
        return float(loss.item())


# ---------------------------------------------------------------------------
# 评估（确定性策略）
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(env, agents, episodes, rng):
    tags, total_ret, total_steps, dists = 0, [0.0, 0.0], [], []
    for _ in range(episodes):
        obs = env.reset(rng)
        done = False
        ep_ret = [0.0, 0.0]
        n = 0
        info = {"dist": 0.0}
        while not done:
            actions = np.stack([agents[i].act(obs[i], deterministic=True)
                                for i in range(len(agents))])
            obs, rew, terminated, truncated, info = env.step(actions)
            ep_ret[0] += rew[0]
            ep_ret[1] += rew[1]
            n += 1
            done = terminated or truncated
            if terminated:
                tags += 1
        total_ret[0] += ep_ret[0]
        total_ret[1] += ep_ret[1]
        total_steps.append(n)
        dists.append(info["dist"])
    return (tags / episodes, total_ret[0] / episodes, total_ret[1] / episodes,
            float(np.mean(total_steps)), float(np.mean(dists)))


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="第082章：多智能体SAC 对手建模")
    p.add_argument("--max-steps", type=int, default=40000, help="环境联合步数上限")
    p.add_argument("--warmup", type=int, default=2000, help="随机动作预热步数")
    p.add_argument("--update-every", type=int, default=2, help="每多少步做一次训练")
    p.add_argument("--batch-size", type=int, default=128, help="采样批量")
    p.add_argument("--buffer-size", type=int, default=100000, help="回放缓冲上限")
    p.add_argument("--lr", type=float, default=3e-4, help="学习率")
    p.add_argument("--opp-lr", type=float, default=1e-3, help="对手模型学习率")
    p.add_argument("--gamma", type=float, default=0.95, help="折扣因子")
    p.add_argument("--tau", type=float, default=0.01, help="目标网络软更新系数")
    p.add_argument("--hidden", type=int, default=128, help="隐藏层宽度")
    p.add_argument("--opp-model", type=str, default="pred",
                   choices=["off", "pred", "true"], help="对手信息使用方式")
    p.add_argument("--opp-train-every", type=int, default=250,
                   help="每多少步监督训练一次对手模型")
    p.add_argument("--opp-batch", type=int, default=256, help="对手模型批量")
    p.add_argument("--opp-inner-steps", type=int, default=2,
                   help="每次对手模型训练的梯度步数")
    p.add_argument("--log-every", type=int, default=2000, help="日志间隔（步）")
    p.add_argument("--eval-episodes", type=int, default=20, help="最终评估回合数")
    p.add_argument("--seed", type=int, default=0, help="随机种子")
    p.add_argument("--save-dir", type=str, default="runs/ch082", help="模型保存目录")
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
    env = PursuitEvaderEnv()
    rng = np.random.default_rng(args.seed)
    agents = [SACAgent("pursuer", env, args, device),
              SACAgent("evader", env, args, device)]

    print(f"设备 {device} | 对手建模模式 {args.opp_model} | 种子 {args.seed}")
    print(f"配置：max_steps={args.max_steps}, warmup={args.warmup}, "
          f"batch={args.batch_size}, lr={args.lr}, gamma={args.gamma}")

    obs = env.reset(rng)
    ep_returns = [[], []]      # 每个智能体最近若干回合的回报
    cur_ret = [0.0, 0.0]
    ep_count = 0
    tag_count = 0
    t_start = time.time()

    for step in range(1, args.max_steps + 1):
        # 1) 选择动作：预热期随机探索，随后由两个策略各自采样
        if step <= args.warmup:
            actions = rng.uniform(-1.0, 1.0, size=(2, env.act_dim)).astype(np.float32)
        else:
            actions = np.stack([agents[i].act(obs[i]) for i in range(2)])

        next_obs, rew, terminated, truncated, info = env.step(actions)
        done = terminated or truncated

        # 2) 两个智能体各自存转移：自己的经验 + 对手的观测/动作
        for i in range(2):
            j = 1 - i
            if step > args.warmup:
                agents[i].store(obs[i], actions[i], rew[i], next_obs[i],
                                float(terminated), obs[j], next_obs[j], actions[j])
            cur_ret[i] += float(rew[i])

        # 3) 训练：每 update_every 步做一次 SAC 更新（两个智能体各自更新）
        if step > args.warmup and step % args.update_every == 0:
            for agent in agents:
                if len(agent.buffer) >= args.batch_size:
                    agent.update(agent.sample_batch(args.batch_size))

        # 4) 对手模型的监督训练
        if step > args.warmup and args.opp_model == "pred" \
                and step % args.opp_train_every == 0:
            for agent in agents:
                for _ in range(args.opp_inner_steps):
                    agent.train_opp_model()

        # 5) 回合结束：记录回报，重置
        if done:
            ep_count += 1
            if terminated:
                tag_count += 1
            for i in range(2):
                ep_returns[i].append(cur_ret[i])
                if len(ep_returns[i]) > 200:
                    del ep_returns[i][:100]
            cur_ret = [0.0, 0.0]
            obs = env.reset(rng)
        else:
            obs = next_obs

        # 6) 日志
        if step % args.log_every == 0 and ep_count > 0:
            recent = 20
            r0 = float(np.mean(ep_returns[0][-recent:])) if ep_returns[0] else float("nan")
            r1 = float(np.mean(ep_returns[1][-recent:])) if ep_returns[1] else float("nan")
            rate = tag_count / max(ep_count, 1)
            opp_mse = float("nan")
            if args.opp_model == "pred" and agents[0].opp_data:
                with torch.no_grad():
                    d = agents[0].opp_data[-512:]
                    o = torch.as_tensor(np.stack([x[0] for x in d]), dtype=torch.float32)
                    a = torch.as_tensor(np.stack([x[1] for x in d]), dtype=torch.float32)
                    opp_mse = float(F.mse_loss(agents[0].opp_model(o), a).item())
            print(f"[步 {step:6d}] 回合 {ep_count:5d} | 追者近20均 {r0:8.2f} | "
                  f"逃者近20均 {r1:8.2f} | 贴获率 {rate:.2f} | "
                  f"对手模型MSE {opp_mse:.4f}")

    elapsed = time.time() - t_start

    # 7) 评估（确定性策略）
    eval_rng = np.random.default_rng(args.seed + 12345)
    tag_rate, p_ret, e_ret, mean_len, mean_dist = evaluate(
        env, agents, args.eval_episodes, eval_rng)

    print("-" * 74)
    print(f"训练结束 | 用时 {elapsed:.1f}s | 联合步数 {args.max_steps} | "
          f"总回合 {ep_count}")
    print(f"评估（{args.eval_episodes} 回合，确定性策略）：贴获率 {tag_rate:.2f} | "
          f"追者回报 {p_ret:.2f} | 逃者回报 {e_ret:.2f} | "
          f"平均步长 {mean_len:.1f} | 结束距离 {mean_dist:.3f}")

    os.makedirs(args.save_dir, exist_ok=True)
    ckpt = {
        "pursuer_actor": agents[0].actor.state_dict(),
        "evader_actor": agents[1].actor.state_dict(),
        "pursuer_opp_model": agents[0].opp_model.state_dict(),
        "evader_opp_model": agents[1].opp_model.state_dict(),
        "args": vars(args),
    }
    path = os.path.join(args.save_dir, "masac_opponent_model.pt")
    torch.save(ckpt, path)
    print(f"模型已保存到 {path}")


if __name__ == "__main__":
    main()
