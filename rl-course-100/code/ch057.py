"""
第057章 SAC的模仿学习初始化

用脚本专家（能量泵 + PD 稳定控制器）生成一批 Pendulum 示范数据，对比三条
训练路线（统一训练预算）：

  - scratch   : 空回放池，标准 SAC
  - prefill   : 回放池预填 6000 条专家转移，纯 off-policy 学习
  - prefill+bc: 预填 + 行为克隆损失（策略均值动作对齐专家动作，系数 bc_coef）

记录：1000/2000 步处的评估回报（早期学习速度）、达到 −400 的步数、最终评估。

运行：
    python code/ch057.py           # 专家采集 + 三种模式（CPU 约 3-4 分钟）
    python code/ch057.py --quick   # 快速跑通（约 40 秒）

预期：prefill 组在前 1000~2000 步明显领先 scratch；prefill+bc 的领先更大、
早期曲线更平滑，但 BC 系数过大时最终性能会被"模仿"拖住。
"""
import argparse
import math
import os
import random
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_STD_MIN, LOG_STD_MAX = -20.0, 2.0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_pendulum(seed: int):
    env = gym.make("Pendulum-v1")
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env


# ---------------------------------------------------------------------------
# 脚本专家：能量泵把摆杆甩起来，靠近直立后切换 PD 稳定
# ---------------------------------------------------------------------------
def expert_action(obs) -> np.ndarray:
    """启发式控制器。

    能量 E = 0.5 * θ̇² + 15 * cos θ（系数 15 与 gymnasium 简化动力学的 3g/2 项对应；
    直立静止时 E = 15，自然下垂时 E = −15）。
    远离直立时用能量泵（力矩与角速度同向，把能量推向 15）；靠近直立后切换
    PD 稳定。θ 由观测 [cos θ, sin θ] 反推。
    """
    cos_t, sin_t, thd = float(obs[0]), float(obs[1]), float(obs[2])
    theta = math.atan2(sin_t, cos_t)
    energy = 0.5 * thd * thd + 15.0 * math.cos(theta)
    if abs(theta) < 0.4:
        u = -6.0 * theta - 1.5 * thd       # PD 稳定：增益 6 略高于静态所需的 5
    else:
        u = 0.015 * thd * (15.0 - energy)  # 能量泵：沿速度方向注入/耗散能量
    return np.array([np.clip(u, -2.0, 2.0)], dtype=np.float32)


def collect_expert_data(episodes: int, noise: float, seed: int):
    """采集示范数据（带高斯噪声增加状态覆盖），返回 (obs, act, rew, next_obs, done) 数组。"""
    env = make_pendulum(seed)
    obs_buf, act_buf, rew_buf, next_buf, done_buf = [], [], [], [], []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            a = expert_action(obs) + np.random.normal(0.0, noise, size=1).astype(np.float32)
            a = np.clip(a, -2.0, 2.0).astype(np.float32)
            next_obs, rew, terminated, truncated, _ = env.step(a)
            obs_buf.append(np.asarray(obs, dtype=np.float32))
            act_buf.append(a)
            rew_buf.append(rew)
            next_buf.append(np.asarray(next_obs, dtype=np.float32))
            done_buf.append(float(terminated))
            obs = next_obs
            done = terminated or truncated
    return (np.asarray(obs_buf, dtype=np.float32), np.asarray(act_buf, dtype=np.float32),
            np.asarray(rew_buf, dtype=np.float32).reshape(-1, 1),
            np.asarray(next_buf, dtype=np.float32),
            np.asarray(done_buf, dtype=np.float32).reshape(-1, 1))


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, act_dim: int):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rew = np.zeros((capacity, 1), dtype=np.float32)
        self.done = np.zeros((capacity, 1), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(self, obs, act, rew, next_obs, done) -> None:
        i = self.ptr
        self.obs[i] = obs
        self.act[i] = act
        self.rew[i] = rew
        self.next_obs[i] = next_obs
        self.done[i] = done
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def add_batch(self, obs, act, rew, next_obs, done) -> None:
        """批量写入（用于预填专家数据）。"""
        for i in range(len(obs)):
            self.add(obs[i], act[i], rew[i], next_obs[i], done[i])

    def sample(self, batch_size: int) -> dict:
        idx = np.random.randint(0, self.size, size=batch_size)
        return {
            "obs": torch.as_tensor(self.obs[idx]),
            "act": torch.as_tensor(self.act[idx]),
            "rew": torch.as_tensor(self.rew[idx]),
            "next_obs": torch.as_tensor(self.next_obs[idx]),
            "done": torch.as_tensor(self.done[idx]),
        }


class QNet(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, act], dim=-1)).squeeze(-1)


class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256,
                 act_scale: float = 1.0):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mean_layer = nn.Linear(hidden, act_dim)
        self.log_std_layer = nn.Linear(hidden, act_dim)
        self.register_buffer("act_scale", torch.tensor(float(act_scale)))

    def forward(self, obs: torch.Tensor):
        h = self.body(obs)
        mean = self.mean_layer(h)
        log_std = torch.clamp(self.log_std_layer(h), LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, obs: torch.Tensor):
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        eps = torch.randn_like(mean)
        u = mean + std * eps
        a = torch.tanh(u)
        log_prob = -0.5 * (eps ** 2 + 2.0 * log_std + np.log(2.0 * np.pi))
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        log_prob = log_prob - torch.log(1.0 - a ** 2 + 1e-6).sum(dim=-1, keepdim=True)
        action = a * self.act_scale
        log_prob = log_prob - torch.log(self.act_scale).sum()
        return action, log_prob

    def mean_action(self, obs: torch.Tensor) -> torch.Tensor:
        mean, _ = self.forward(obs)
        return torch.tanh(mean) * self.act_scale


def soft_update(net: nn.Module, target: nn.Module, tau: float) -> None:
    for tp, p in zip(target.parameters(), net.parameters()):
        tp.data.mul_(1.0 - tau).add_(tau * p.data)


class SAC:
    def __init__(self, obs_dim: int, act_dim: int, action_scale: float, args):
        self.policy = GaussianPolicy(obs_dim, act_dim, args.hidden, action_scale)
        self.q1 = QNet(obs_dim, act_dim, args.hidden)
        self.q2 = QNet(obs_dim, act_dim, args.hidden)
        self.q1_target = QNet(obs_dim, act_dim, args.hidden)
        self.q2_target = QNet(obs_dim, act_dim, args.hidden)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        for p in self.q1_target.parameters():
            p.requires_grad_(False)
        for p in self.q2_target.parameters():
            p.requires_grad_(False)
        self.log_alpha = torch.tensor(np.log(args.alpha_init), requires_grad=True)
        self.target_entropy = -float(act_dim)
        self.opt_q = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=args.lr)
        self.opt_policy = torch.optim.Adam(self.policy.parameters(), lr=args.lr)
        self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=args.lr)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp().clamp(1e-4, 10.0)

    @torch.no_grad()
    def act(self, obs, deterministic: bool = False) -> np.ndarray:
        obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
        if deterministic:
            return self.policy.mean_action(obs_t).squeeze(0).numpy()
        action, _ = self.policy.sample(obs_t)
        return action.squeeze(0).numpy()

    def update(self, batch: dict, args, bc_batch: dict = None) -> dict:
        obs, act = batch["obs"], batch["act"]
        rew, next_obs, done = batch["rew"], batch["next_obs"], batch["done"]
        rew, done = rew.squeeze(-1), done.squeeze(-1)      # 与 Q 网络的 (B,) 输出对齐

        with torch.no_grad():
            next_act, next_logp = self.policy.sample(next_obs)
            next_logp = next_logp.squeeze(-1)
            q_next = torch.minimum(self.q1_target(next_obs, next_act),
                                   self.q2_target(next_obs, next_act))
            backup = rew + args.gamma * (1.0 - done) * (q_next - self.alpha.detach() * next_logp)

        q1 = self.q1(obs, act)
        q2 = self.q2(obs, act)
        q_loss = F.mse_loss(q1, backup) + F.mse_loss(q2, backup)
        self.opt_q.zero_grad()
        q_loss.backward()
        self.opt_q.step()

        new_act, logp = self.policy.sample(obs)
        logp = logp.squeeze(-1)                            # 同上：统一为 (B,)
        q_pi = torch.minimum(self.q1(obs, new_act), self.q2(obs, new_act))
        policy_loss = (self.alpha.detach() * logp - q_pi).mean()

        bc_loss = torch.tensor(0.0)
        if bc_batch is not None and args.bc_coef > 0.0:
            # 行为克隆：策略均值动作与专家动作的均方误差
            mean_action = self.policy.mean_action(bc_batch["obs"])
            bc_loss = F.mse_loss(mean_action, bc_batch["act"])
            policy_loss = policy_loss + args.bc_coef * bc_loss

        self.opt_policy.zero_grad()
        policy_loss.backward()
        self.opt_policy.step()

        alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()
        self.opt_alpha.zero_grad()
        alpha_loss.backward()
        self.opt_alpha.step()

        with torch.no_grad():
            soft_update(self.q1, self.q1_target, args.tau)
            soft_update(self.q2, self.q2_target, args.tau)

        return {"q_loss": q_loss.item(), "policy_loss": policy_loss.item(),
                "bc_loss": float(bc_loss.item()), "alpha": float(self.alpha.item())}


@torch.no_grad()
def evaluate(agent: SAC, env, episodes: int) -> float:
    returns = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            action = agent.act(obs, deterministic=True)
            obs, rew, terminated, truncated, _ = env.step(action)
            ep_ret += rew
            done = terminated or truncated
        returns.append(ep_ret)
    return float(np.mean(returns))


# ---------------------------------------------------------------------------
# 训练一种模式
# ---------------------------------------------------------------------------
def train_mode(mode: str, expert_data, args):
    set_seed(args.seed)  # 三种模式同种子，控制变量
    env = make_pendulum(args.seed)
    eval_env = make_pendulum(args.seed + 100)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    action_scale = float(env.action_space.high[0])
    agent = SAC(obs_dim, act_dim, action_scale, args)
    buffer = ReplayBuffer(args.buffer_size, obs_dim, act_dim)
    expert_buf = None
    if mode in ("prefill", "prefill+bc"):
        buffer.add_batch(*expert_data)  # 预填专家数据
    if mode == "prefill+bc":
        # 单独的专家池：行为克隆只从示范数据里抽样，不受在线数据稀释
        expert_buf = ReplayBuffer(len(expert_data[0]) + 10, obs_dim, act_dim)
        expert_buf.add_batch(*expert_data)

    obs, _ = env.reset()
    ep_ret = 0.0
    ep_returns = []
    eval_points = {}
    solved_at = None
    t_start = time.time()
    print(f"\n===== 模式 {mode}：开始训练（回放池初始 {buffer.size} 条） =====")

    for step in range(1, args.max_steps + 1):
        if step <= args.start_steps:
            action = env.action_space.sample()
        else:
            action = agent.act(obs, deterministic=False)
        next_obs, rew, terminated, truncated, _ = env.step(action)
        buffer.add(obs, action, rew, next_obs, float(terminated))
        obs = next_obs
        ep_ret += rew
        if terminated or truncated:
            ep_returns.append(ep_ret)
            ep_ret = 0.0
            obs, _ = env.reset()

        if step % args.update_every == 0 and buffer.size >= 64:
            bc_batch = None
            if mode == "prefill+bc" and args.bc_coef > 0:
                # 从专家池抽一小批做模仿损失
                bc_batch = expert_buf.sample(args.bc_batch_size)
            agent.update(buffer.sample(args.batch_size), args, bc_batch=bc_batch)

        if step in args.eval_steps:
            eval_ret = evaluate(agent, eval_env, args.eval_episodes)
            eval_points[step] = eval_ret
            recent20 = np.mean(ep_returns[-20:]) if ep_returns else float("nan")
            print(f"[{mode}] 步数 {step:6d} | 近20回合回报 {recent20:8.1f} | 评估 {eval_ret:8.1f}")
            if solved_at is None and eval_ret >= args.threshold:
                solved_at = step
                print(f"[{mode}] 首次达到评估回报 {args.threshold}（步数 {step}）")

    elapsed = time.time() - t_start
    final_eval = evaluate(agent, eval_env, args.eval_episodes)
    print(f"[{mode}] 结束：最终评估 {final_eval:.1f} | 达标步数 "
          f"{solved_at if solved_at is not None else -1} | 用时 {elapsed:.1f}s")
    return {"mode": mode, "final_eval": final_eval, "solved_at": solved_at,
            "eval_points": eval_points, "elapsed": elapsed}


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="第057章：SAC 的模仿学习初始化")
    p.add_argument("--modes", type=str, default="scratch,prefill,prefill+bc",
                   help="逗号分隔：scratch / prefill / prefill+bc")
    p.add_argument("--expert-episodes", type=int, default=30, help="专家示范回合数")
    p.add_argument("--expert-noise", type=float, default=0.5, help="专家动作的高斯噪声标准差")
    p.add_argument("--bc-coef", type=float, default=0.5, help="行为克隆损失系数")
    p.add_argument("--bc-batch-size", type=int, default=64, help="行为克隆批大小")
    p.add_argument("--max-steps", type=int, default=8000, help="每种模式的训练步数")
    p.add_argument("--start-steps", type=int, default=500, help="训练前纯随机探索步数")
    p.add_argument("--update-every", type=int, default=1, help="每多少环境步做一次更新")
    p.add_argument("--eval-steps", type=str, default="1000,2000,4000,8000",
                   help="评估点（逗号分隔的步数）")
    p.add_argument("--eval-episodes", type=int, default=5, help="评估回合数")
    p.add_argument("--threshold", type=float, default=-400.0, help="达标阈值")
    p.add_argument("--buffer-size", type=int, default=100000, help="回放池容量")
    p.add_argument("--batch-size", type=int, default=256, help="批大小")
    p.add_argument("--hidden", type=int, default=256, help="隐藏层宽度")
    p.add_argument("--lr", type=float, default=3e-4, help="学习率")
    p.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
    p.add_argument("--tau", type=float, default=0.005, help="目标网络软更新系数")
    p.add_argument("--alpha-init", type=float, default=0.2, help="温度系数初值")
    p.add_argument("--seed", type=int, default=0, help="随机种子")
    p.add_argument("--save-dir", type=str, default="runs/ch057", help="输出目录")
    p.add_argument("--quick", action="store_true", help="快速模式：缩短训练")
    return p.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.expert_episodes = min(args.expert_episodes, 8)
        args.max_steps = min(args.max_steps, 2000)
        args.eval_steps = "500,1000,2000"
        args.start_steps = min(args.start_steps, 200)
        args.eval_episodes = 3

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    eval_steps = set(int(x) for x in args.eval_steps.split(",") if x.strip())
    args.eval_steps = eval_steps

    print(f"种子: {args.seed} | 模式: {modes} | 评估点: {sorted(eval_steps)}")
    print(f"配置: max_steps={args.max_steps}, bc_coef={args.bc_coef}, "
          f"专家回合数={args.expert_episodes}")

    # 1) 采集专家数据（带噪声）
    set_seed(args.seed + 5000)
    expert_data = collect_expert_data(args.expert_episodes, args.expert_noise,
                                      args.seed + 5000)
    print(f"专家数据：{len(expert_data[0])} 条转移（{args.expert_episodes} 个回合）")

    # 顺便评估专家本身的水平（加噪声版本）
    expert_returns = []
    for _ in range(3):
        env = make_pendulum(args.seed + 5555)
        obs, _ = env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            a = expert_action(obs) + np.random.normal(0.0, args.expert_noise, size=1)
            a = np.clip(a, -2.0, 2.0).astype(np.float32)
            obs, rew, terminated, truncated, _ = env.step(a)
            ep_ret += rew
            done = terminated or truncated
        expert_returns.append(ep_ret)
    print(f"脚本专家（带噪声）平均回报：{np.mean(expert_returns):.1f}")

    # 2) 三种模式依次训练
    results = [train_mode(m, expert_data, args) for m in modes]

    # 3) 汇总
    print("\n" + "=" * 96)
    print("模仿学习初始化对比（评估点回报与达标步数）")
    header = f"{'模式':<12}"
    for s in sorted(eval_steps):
        header += f"{'步' + str(s):>10}"
    header += f"{'最终评估':>10}{'达标步数':>10}"
    print(header)
    for r in results:
        line = f"{r['mode']:<12}"
        for s in sorted(eval_steps):
            v = r["eval_points"].get(s, float("nan"))
            line += f"{v:>10.1f}"
        solved = r["solved_at"] if r["solved_at"] is not None else -1
        line += f"{r['final_eval']:>10.1f}{solved:>10}"
        print(line)
    print("=" * 96)
    print("结论提示：prefill 让 SAC 从第一轮更新就有'像样'的样本来学，早期评估")
    print("显著领先；prefill+bc 进一步把策略均值动作拉向专家，早期优势更大。")
    print("bc_coef 过大会把策略钉在专家水平——专家本身不完美时最终性能受限。")

    os.makedirs(args.save_dir, exist_ok=True)
    print(f"输出目录：{args.save_dir}")


if __name__ == "__main__":
    main()
