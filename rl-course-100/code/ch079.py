"""
第079章 多智能体SAC：通信

在"感知受限"（队友超出 0.4 半径看不见）的合作导航上，给共享 SAC 策略加一条
通信通道：每个智能体输出动作的同时广播 2 维"意图消息"（预测自己下一步位移），
队友消息拼进自己的策略输入；消息头用自监督 MSE 训练。
对照：
  - off：无通信
  - on ：有通信（消息延迟一步送达，与第 071 章 PPO 版一致）

运行：
    python code/ch079.py           # 完整对照（CPU 约 5-10 分钟）
    python code/ch079.py --quick   # 快速跑通（约 40-80 秒）
    python code/ch079.py --comm on

预期（以实跑为准）：有通信版本的碰撞更少、末距更低；感知半径越小，通信收益越大。
"""

import argparse
import itertools
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ===========================================================================
# 环境：带感知半径的合作导航（与第 071 章同构）
# ===========================================================================
class CooperativeNavEnv:
    def __init__(self, n_agents=3, episode_len=50, damping=0.9, accel=0.2,
                 collision_dist=0.1, collision_penalty=0.3, seed=0, sense_radius=3.0):
        self.n_agents = n_agents
        self.episode_len = episode_len
        self.damping = damping
        self.accel = accel
        self.collision_dist = collision_dist
        self.collision_penalty = collision_penalty
        self.sense_radius = sense_radius
        self.obs_dim = 4 + 2 * n_agents + 2 * (n_agents - 1)
        self.act_dim = 2
        self._rng = np.random.RandomState(seed)
        self._perms = list(itertools.permutations(range(n_agents)))
        self.pos = np.zeros((n_agents, 2), np.float32)
        self.vel = np.zeros((n_agents, 2), np.float32)
        self.landmarks = np.zeros((n_agents, 2), np.float32)
        self.t = 0

    def reset(self):
        self.pos = self._rng.uniform(0.15, 0.85, (self.n_agents, 2)).astype(np.float32)
        self.vel = np.zeros((self.n_agents, 2), np.float32)
        self.landmarks = self._rng.uniform(0.10, 0.90, (self.n_agents, 2)).astype(np.float32)
        self.t = 0
        return self._obs()

    def _obs(self):
        obs = np.zeros((self.n_agents, self.obs_dim), np.float32)
        for i in range(self.n_agents):
            rel_land = (self.landmarks - self.pos[i]).ravel()
            rel_others = []
            for j in range(self.n_agents):
                if j == i:
                    continue
                d = self.pos[j] - self.pos[i]
                if float(np.linalg.norm(d)) > self.sense_radius:
                    d = np.zeros(2, np.float32)    # 超出感知半径：隐身
                rel_others.append(d)
            others = np.concatenate(rel_others) if rel_others else np.zeros(0, np.float32)
            obs[i] = np.concatenate([self.pos[i], self.vel[i], rel_land, others])
        return obs

    def _min_matching_distance(self) -> float:
        best = np.inf
        for perm in self._perms:
            total = 0.0
            for i, j in enumerate(perm):
                total += float(np.linalg.norm(self.pos[i] - self.landmarks[j]))
            if total < best:
                best = total
        return best / self.n_agents

    def step(self, actions):
        actions = np.clip(np.asarray(actions, dtype=np.float32), -1.0, 1.0)
        self.vel = self.damping * self.vel + self.accel * actions
        self.pos = np.clip(self.pos + self.vel, 0.0, 1.0)
        self.t += 1
        mean_dist = self._min_matching_distance()
        per_reward = np.full(self.n_agents, -mean_dist, dtype=np.float32)
        collisions = 0
        for i in range(self.n_agents):
            for j in range(i + 1, self.n_agents):
                if np.linalg.norm(self.pos[i] - self.pos[j]) < self.collision_dist:
                    per_reward[i] -= self.collision_penalty
                    per_reward[j] -= self.collision_penalty
                    collisions += 1
        done = self.t >= self.episode_len
        info = {"mean_dist": mean_dist, "collisions": collisions}
        return self._obs(), per_reward, float(per_reward.mean()), done, info


# ===========================================================================
# 带消息头的共享策略
# ===========================================================================
class CommPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, n_agents, hidden=64,
                 use_comm=True, msg_dim=2, msg_scale=0.4):
        super().__init__()
        self.use_comm = use_comm
        self.msg_dim = msg_dim
        self.msg_scale = msg_scale
        in_dim = obs_dim + (n_agents - 1) * msg_dim if use_comm else obs_dim
        self.in_dim = in_dim
        self.body = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(),
                                  nn.Linear(hidden, hidden), nn.ReLU())
        self.mean = nn.Linear(hidden, act_dim)
        self.log_std = nn.Linear(hidden, act_dim)
        if use_comm:
            self.msg_head = nn.Linear(hidden, msg_dim)

    def forward(self, x):
        h = self.body(x)
        mean = self.mean(h)
        log_std = torch.clamp(self.log_std(h), -4.0, 0.5)
        msg = torch.tanh(self.msg_head(h)) * self.msg_scale if self.use_comm else None
        return mean, log_std, msg

    def sample(self, x):
        mean, log_std, msg = self.forward(x)
        dist = Normal(mean, log_std.exp())
        z = dist.rsample()
        action = torch.tanh(z)
        logp = dist.log_prob(z) - torch.log(1.0 - action.pow(2) + 1e-6)
        return action, logp.sum(-1, keepdim=True), msg

    @torch.no_grad()
    def mean_action(self, x):
        mean, _, msg = self.forward(x)
        return torch.tanh(mean), msg


class TwinQ(nn.Module):
    """局部双 Q：输入策略输入 x（观测 + 消息）与自己的动作。"""

    def __init__(self, in_dim, act_dim, hidden=64):
        super().__init__()
        def make():
            return nn.Sequential(nn.Linear(in_dim + act_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1))
        self.q1, self.q2 = make(), make()

    def forward(self, x, act):
        inp = torch.cat([x, act], dim=-1)
        return self.q1(inp), self.q2(inp)


def build_policy_input(obs, messages, use_comm, n_agents):
    if not use_comm:
        return obs.astype(np.float32)
    rows = []
    for i in range(n_agents):
        parts = [obs[i]]
        for j in range(n_agents):
            if j != i:
                parts.append(messages[j])
        rows.append(np.concatenate(parts))
    return np.asarray(rows, dtype=np.float32)


# ===========================================================================
# 学习器：共享策略 + 双 Q + 消息辅助损失
# ===========================================================================
class CommLearner:
    def __init__(self, obs_dim, act_dim, n_agents, args, device, use_comm):
        self.device = device
        self.use_comm = use_comm
        self.policy = CommPolicy(obs_dim, act_dim, n_agents, args.hidden,
                                 use_comm, args.msg_dim).to(device)
        self.q = TwinQ(self.policy.in_dim, act_dim, args.hidden).to(device)
        self.q_target = TwinQ(self.policy.in_dim, act_dim, args.hidden).to(device)
        self.q_target.load_state_dict(self.q.state_dict())
        self.policy_opt = torch.optim.Adam(self.policy.parameters(), lr=args.lr)
        self.q_opt = torch.optim.Adam(self.q.parameters(), lr=args.lr)
        self.log_alpha = torch.tensor(np.log(args.alpha), requires_grad=True, device=device)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=args.lr)
        self.target_entropy = -float(act_dim)
        self.gamma, self.tau = args.gamma, args.tau

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def update(self, batch, args):
        """batch 里的张量已按 (B·N, ·) 展平。"""
        x = batch["x"]
        act = batch["act"]
        rew = batch["rew"]
        next_x = batch["next_x"]
        done = batch["done"]
        with torch.no_grad():
            next_act, next_logp, _ = self.policy.sample(next_x)
            q1t, q2t = self.q_target(next_x, next_act)
            target = rew.unsqueeze(-1) + self.gamma * (1.0 - done).unsqueeze(-1) * (
                torch.min(q1t, q2t) - self.alpha * next_logp)
        q1, q2 = self.q(x, act)
        q_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.q_opt.zero_grad()
        q_loss.backward()
        self.q_opt.step()

        new_act, logp, msg = self.policy.sample(x)
        q1n, q2n = self.q(x, new_act)
        policy_loss = (self.alpha.detach() * logp - torch.min(q1n, q2n)).mean()
        aux_loss = torch.tensor(0.0)
        if self.use_comm:
            aux_loss = F.mse_loss(msg, batch["disp"])
            policy_loss = policy_loss + args.comm_coef * aux_loss
        self.policy_opt.zero_grad()
        policy_loss.backward()
        self.policy_opt.step()

        alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        with torch.no_grad():
            for p, p_t in zip(self.q.parameters(), self.q_target.parameters()):
                p_t.data.mul_(1.0 - self.tau).add_(self.tau * p.data)
        return {"q_loss": q_loss.item(), "policy_loss": policy_loss.item(),
                "aux_loss": float(aux_loss.item()), "alpha": float(self.alpha.item())}


# ===========================================================================
# 采样与评估
# ===========================================================================
def collect(env, learner, steps, obs, messages, device):
    """采集 steps 个环境步；返回堆叠后的缓冲字典与消息状态。"""
    n = env.n_agents
    xs, acts, rews, next_xs, disps, dones = [], [], [], [], [], []
    for _ in range(steps):
        x = build_policy_input(obs, messages, learner.use_comm, n)
        x_t = torch.as_tensor(x, dtype=torch.float32, device=device)
        with torch.no_grad():
            action_t, _, msg_t = learner.policy.sample(x_t)
        actions = action_t.cpu().numpy()
        msgs = None if msg_t is None else msg_t.cpu().numpy()
        next_obs, per_rew, team_rew, done, info = env.step(actions)
        # 下一步的策略输入 = 下一观测 + 本步广播的消息
        next_messages = messages if msgs is None else msgs
        next_x = build_policy_input(next_obs, next_messages, learner.use_comm, n)
        xs.append(x)
        acts.append(actions)
        rews.append(per_rew)
        next_xs.append(next_x)
        disps.append(next_obs[:, :2] - obs[:, :2])   # 自身位移（消息监督目标）
        dones.append(float(done))
        obs = next_obs
        messages = np.zeros((n, learner.policy.msg_dim), np.float32) if msgs is None else msgs
        if done:
            obs = env.reset()
            messages = np.zeros((n, learner.policy.msg_dim), np.float32)
    return {
        "x": np.asarray(xs, np.float32),
        "act": np.asarray(acts, np.float32),
        "rew": np.asarray(rews, np.float32),
        "next_x": np.asarray(next_xs, np.float32),
        "disp": np.asarray(disps, np.float32),
        "done": np.asarray(dones, np.float32),
    }, obs, messages


def flatten_batch(buf, device):
    """把 (B, N, ·) 缓冲展平成 (B·N, ·) 张量。"""
    B, N = buf["x"].shape[0], buf["x"].shape[1]
    return {
        "x": torch.as_tensor(buf["x"].reshape(B * N, -1), device=device),
        "act": torch.as_tensor(buf["act"].reshape(B * N, -1), device=device),
        "rew": torch.as_tensor(buf["rew"].reshape(B * N), device=device),
        "next_x": torch.as_tensor(buf["next_x"].reshape(B * N, -1), device=device),
        "disp": torch.as_tensor(buf["disp"].reshape(B * N, -1), device=device),
        "done": torch.as_tensor(np.repeat(buf["done"], N), device=device),
    }


@torch.no_grad()
def evaluate(env, learner, episodes, device):
    rets, dists, colls = [], [], []
    n = env.n_agents
    for _ in range(episodes):
        obs = env.reset()
        messages = np.zeros((n, learner.policy.msg_dim), np.float32)
        done = False
        ep_ret = 0.0
        info = {"mean_dist": 0.0, "collisions": 0}
        while not done:
            x = build_policy_input(obs, messages, learner.use_comm, n)
            x_t = torch.as_tensor(x, dtype=torch.float32, device=device)
            actions, msg = learner.policy.mean_action(x_t)
            actions = actions.cpu().numpy()
            if msg is not None:
                messages = msg.cpu().numpy()
            obs, _, team_r, done, info = env.step(actions)
            ep_ret += team_r
        rets.append(ep_ret)
        dists.append(info["mean_dist"])
        colls.append(info["collisions"])
    return float(np.mean(rets)), float(np.mean(dists)), float(np.mean(colls))


# ===========================================================================
# 训练
# ===========================================================================
def run_training(use_comm, args, device):
    set_seed(args.seed)
    env = CooperativeNavEnv(args.n_agents, args.episode_len, seed=args.seed,
                            sense_radius=args.sense_radius)
    eval_env = CooperativeNavEnv(args.n_agents, args.episode_len, seed=args.seed + 1,
                                 sense_radius=args.sense_radius)
    learner = CommLearner(env.obs_dim, env.act_dim, env.n_agents, args, device, use_comm)
    label = "COMM-ON" if use_comm else "COMM-OFF"
    n_params = sum(p.numel() for p in learner.policy.parameters())
    print(f"[{label}] 策略参数量 {n_params} | 感知半径 {args.sense_radius} | "
          f"输入维度 {learner.policy.in_dim}")

    obs = env.reset()
    messages = np.zeros((env.n_agents, learner.policy.msg_dim), np.float32)
    total_steps = 0
    curve = []
    stats = {"aux_loss": 0.0, "alpha": args.alpha}
    t0 = time.time()

    # 预热：随机策略填池（消息此时保持为零，因为还没有策略在广播）
    replay = {k: [] for k in ["x", "act", "rew", "next_x", "disp", "done"]}
    while total_steps < args.start_steps:
        actions = np.random.uniform(-1.0, 1.0, (env.n_agents, env.act_dim)).astype(np.float32)
        x = build_policy_input(obs, messages, use_comm, env.n_agents)
        next_obs, per_rew, team_rew, done, _ = env.step(actions)
        zero_messages = np.zeros((env.n_agents, learner.policy.msg_dim), np.float32)
        replay["x"].append(x)
        replay["act"].append(actions)
        replay["rew"].append(per_rew)
        replay["disp"].append(next_obs[:, :2] - obs[:, :2])
        replay["done"].append(float(done))
        replay["next_x"].append(build_policy_input(next_obs, zero_messages,
                                                   use_comm, env.n_agents))
        obs = next_obs
        if done:
            obs = env.reset()
        total_steps += 1

    next_eval = args.start_steps + args.eval_every
    while total_steps < args.max_steps:
        buf, obs, messages = collect(env, learner, args.steps_per_update, obs, messages, device)
        for k in replay:
            replay[k].extend(list(buf[k]))
        total_steps += args.steps_per_update
        # 只保留最近 buffer_size 条
        for k in replay:
            if len(replay[k]) > args.buffer_size:
                replay[k] = replay[k][-args.buffer_size:]

        # 采样小批量做多次更新
        B = len(replay["done"])
        if B > args.batch_size:
            for _ in range(args.updates_per_round):
                idx = np.random.randint(0, B, size=args.batch_size)
                sub = {k: np.asarray(v, np.float32)[idx] for k, v in replay.items()}
                stats = learner.update(flatten_batch(sub, device), args)

        if total_steps >= next_eval:
            ret, dist, coll = evaluate(eval_env, learner, args.eval_episodes, device)
            curve.append((total_steps, ret))
            aux_str = f"消息MSE {stats['aux_loss']:.4f}" if use_comm else "无通信"
            print(f"[{label} 步 {total_steps:6d}] 评估回报 {ret:7.2f} | 平均末距 {dist:.3f} | "
                  f"碰撞 {coll:.2f}/回合 | {aux_str} | α {stats['alpha']:.3f} | "
                  f"用时 {time.time() - t0:5.1f}s")
            next_eval += args.eval_every

    elapsed = time.time() - t0
    ret, dist, coll = evaluate(eval_env, learner, args.eval_episodes * 2, device)
    print(f"[{label}] 完成：用时 {elapsed:.1f}s | 最终评估回报 {ret:.2f} | "
          f"末距 {dist:.3f} | 碰撞 {coll:.2f}")
    return learner, curve, elapsed, (ret, dist, coll)


def parse_args():
    p = argparse.ArgumentParser(description="第079章：多智能体 SAC 的通信")
    p.add_argument("--comm", type=str, default="both", choices=["both", "on", "off"])
    p.add_argument("--n-agents", type=int, default=3, help="智能体数量")
    p.add_argument("--episode-len", type=int, default=50, help="回合长度")
    p.add_argument("--sense-radius", type=float, default=0.4, help="队友可见半径")
    p.add_argument("--msg-dim", type=int, default=2, help="消息维度")
    p.add_argument("--comm-coef", type=float, default=1.0, help="消息自监督损失权重")
    p.add_argument("--max-steps", type=int, default=20000, help="环境步预算")
    p.add_argument("--steps-per-update", type=int, default=1000, help="每轮采样步数")
    p.add_argument("--updates-per-round", type=int, default=500, help="每轮采样后的更新次数")
    p.add_argument("--batch-size", type=int, default=128, help="每次更新的采样批量")
    p.add_argument("--buffer-size", type=int, default=50000, help="回放池容量（环境步）")
    p.add_argument("--start-steps", type=int, default=1000, help="随机预热步数")
    p.add_argument("--hidden", type=int, default=64, help="隐藏层宽度")
    p.add_argument("--lr", type=float, default=3e-4, help="学习率")
    p.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
    p.add_argument("--tau", type=float, default=0.005, help="软更新系数")
    p.add_argument("--alpha", type=float, default=0.2, help="温度初值")
    p.add_argument("--eval-every", type=int, default=4000, help="评估间隔（环境步）")
    p.add_argument("--eval-episodes", type=int, default=5, help="评估回合数")
    p.add_argument("--seed", type=int, default=0, help="随机种子")
    p.add_argument("--save-dir", type=str, default="runs/ch079", help="保存目录")
    p.add_argument("--quick", action="store_true", help="快速模式：5000 步/配置")
    return p.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.max_steps = min(args.max_steps, 5000)
        args.steps_per_update = 500
        args.updates_per_round = 200
        args.eval_every = 1500
        args.eval_episodes = 3
        args.start_steps = 300

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device} | 环境: 感知受限的合作导航 | 种子 {args.seed} | "
          f"预算 {args.max_steps} 环境步/配置")

    results = {}
    if args.comm in ("off", "both"):
        results["COMM-OFF"] = run_training(False, args, device)
    if args.comm in ("on", "both"):
        results["COMM-ON"] = run_training(True, args, device)

    print("=" * 76)
    for name, (learner, curve, dt, final) in results.items():
        best = max(r for _, r in curve) if curve else float("nan")
        print(f"[{name:9s}] 用时 {dt:6.1f}s | 最佳评估 {best:7.2f} | "
              f"最终（回报/末距/碰撞） {final[0]:7.2f} / {final[1]:.3f} / {final[2]:.2f}")
    if len(results) == 2:
        print("提示：评估重点看碰撞与末距；把 --sense-radius 调大可以验证"
              "'感知缺口越大，通信越有价值'。")

    os.makedirs(args.save_dir, exist_ok=True)
    for name, (learner, curve, _, _) in results.items():
        torch.save(learner.policy.state_dict(),
                   os.path.join(args.save_dir, f"{name.lower().replace('-', '_')}_policy.pt"))
        path = os.path.join(args.save_dir, f"{name.lower().replace('-', '_')}_curve.txt")
        with open(path, "w", encoding="utf-8") as f:
            for step, ret in curve:
                f.write(f"{step}\t{ret:.3f}\n")
    print(f"模型与曲线已保存到 {args.save_dir}")


if __name__ == "__main__":
    main()
