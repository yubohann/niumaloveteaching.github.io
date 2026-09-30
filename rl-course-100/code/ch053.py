"""
第053章 SAC的课程学习

课程学习（Curriculum Learning）先让智能体在"简单版本"的任务上学到手感，
再逐步过渡到目标任务。本章把 MountainCarContinuous 的物理引擎重写为可调
"引擎功率"的轻量环境 PowerMountainCar：

    velocity += action * power - 0.0025 * cos(3 * position)

对比两种训练方式（统一在真实 MountainCarContinuous-v0 上评估）：

  - direct     : 全程用目标功率 0.0015 训练
  - curriculum : 功率按 0.006 -> 0.004 -> 0.0025 -> 0.0015 分四段递减，
                 每段占 1/4 预算，其余机制不变

记录：各阶段末在目标环境上的成功率、首个成功回合、最终评估回报与成功率。

运行：
    python code/ch053.py           # 两种模式依次训练（CPU 约 5-8 分钟）
    python code/ch053.py --quick   # 快速跑通（约 1 分钟）

预期：curriculum 在前几个阶段就学会"来回摆动"的动作模式，切换到目标
功率后能更快出现成功；direct 的首次成功更晚，两万步内常常学不到爬坡。
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


# ---------------------------------------------------------------------------
# 可调功率的轻量山地车环境（动力学与 MountainCarContinuous 一致，功率可调）
# ---------------------------------------------------------------------------
class PowerMountainCar(gym.Env):
    """actor: 连续力 [-1,1]；目标：position >= 0.45 且 velocity >= 0。"""

    metadata = {"render_modes": []}

    def __init__(self, power: float = 0.0015, max_speed: float = 0.07):
        super().__init__()
        self.power = power
        self.max_speed = max_speed
        self.min_position, self.max_position = -1.2, 0.6
        self.goal_position = 0.45
        low = np.array([self.min_position, -max_speed], dtype=np.float32)
        high = np.array([self.max_position, max_speed], dtype=np.float32)
        self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.rng = np.random.default_rng(0)
        self.state = np.array([-0.5, 0.0], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.state = np.array([self.rng.uniform(-0.6, -0.4), 0.0], dtype=np.float32)
        return self.state.copy(), {}

    def step(self, action):
        position, velocity = float(self.state[0]), float(self.state[1])
        force = float(np.clip(np.asarray(action, dtype=np.float32)[0], -1.0, 1.0))
        velocity += force * self.power - 0.0025 * math.cos(3 * position)
        velocity = float(np.clip(velocity, -self.max_speed, self.max_speed))
        position += velocity
        position = float(np.clip(position, self.min_position, self.max_position))
        if position == self.min_position and velocity < 0:
            velocity = 0.0
        terminated = bool(position >= self.goal_position and velocity >= 0.0)
        reward = (100.0 if terminated else 0.0) - 0.1 * force ** 2
        self.state = np.array([position, velocity], dtype=np.float32)
        return self.state.copy(), float(reward), terminated, False, {}


def make_stage_env(power: float, seed: int):
    env = PowerMountainCar(power=power)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env


def make_target_env(seed: int):
    """真实目标环境：gymnasium 的 MountainCarContinuous-v0。"""
    env = gym.make("MountainCarContinuous-v0")
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env


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

    def update(self, batch: dict, args) -> dict:
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

        return {"q_loss": q_loss.item(), "alpha": float(self.alpha.item())}


@torch.no_grad()
def evaluate(agent: SAC, env, episodes: int):
    """在目标环境上评估：返回 (平均回报, 成功率, 平均最大位置)。"""
    returns, successes, max_positions = [], [], []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_ret = 0.0
        max_pos = -1.2
        success = False
        while not done:
            action = agent.act(obs, deterministic=True)
            obs, rew, terminated, truncated, _ = env.step(action)
            ep_ret += rew
            max_pos = max(max_pos, float(env.unwrapped.state[0]))
            success = success or terminated
            done = terminated or truncated
        returns.append(ep_ret)
        successes.append(float(success))
        max_positions.append(max_pos)
    return float(np.mean(returns)), float(np.mean(successes)), float(np.mean(max_positions))


# ---------------------------------------------------------------------------
# 训练一种模式
# ---------------------------------------------------------------------------
def train_mode(mode: str, args):
    set_seed(args.seed)  # 两种模式同种子，控制变量
    powers = []
    if mode == "curriculum":
        powers = [float(x) for x in args.powers.split(",") if x.strip()]
    else:
        powers = [args.powers.strip().split(",")[-1]]  # 全程用目标功率

    # 每个阶段的环境（训练用可调功率版本）与目标环境（评估用真实环境）
    stage_seed = [args.seed + 13 * i for i in range(len(powers))]
    train_env = make_stage_env(float(powers[0]), stage_seed[0])
    eval_env = make_target_env(args.seed + 100)
    obs_dim = train_env.observation_space.shape[0]
    act_dim = train_env.action_space.shape[0]
    action_scale = float(train_env.action_space.high[0])
    agent = SAC(obs_dim, act_dim, action_scale, args)
    buffer = ReplayBuffer(args.buffer_size, obs_dim, act_dim)

    obs, _ = train_env.reset(seed=stage_seed[0])
    ep_ret = 0.0
    episodes = 0
    successes = []
    first_success_ep = None
    t_start = time.time()
    print(f"\n===== 模式 {mode}：功率序列 {powers} =====")

    stage_len = args.max_steps // len(powers)
    for step in range(1, args.max_steps + 1):
        # 课程切换：到达阶段边界时更换训练环境（保留 agent 与回放池）
        stage_idx = min((step - 1) // stage_len, len(powers) - 1)
        if mode == "curriculum" and step > 1 and (step - 1) % stage_len == 0:
            train_env = make_stage_env(float(powers[stage_idx]), stage_seed[stage_idx])
            obs, _ = train_env.reset(seed=stage_seed[stage_idx])
            ep_ret = 0.0
            print(f"[{mode}] 步数 {step}：进入阶段 {stage_idx + 1}/{len(powers)}，"
                  f"power={powers[stage_idx]}")

        if step <= args.start_steps:
            action = train_env.action_space.sample()
        else:
            action = agent.act(obs, deterministic=False)
        next_obs, rew, terminated, truncated, _ = train_env.step(action)
        buffer.add(obs, action, rew, next_obs, float(terminated))
        obs = next_obs
        ep_ret += rew
        if terminated or truncated:
            episodes += 1
            successes.append(float(terminated))
            if terminated and first_success_ep is None:
                first_success_ep = episodes
                print(f"[{mode}] 第 {episodes} 个训练回合首次成功（步数 {step}）")
            ep_ret = 0.0
            obs, _ = train_env.reset()

        if step % args.update_every == 0 and buffer.size >= min(args.start_steps, 1000):
            agent.update(buffer.sample(args.batch_size), args)

        if step % args.eval_every == 0:
            eval_ret, eval_succ, eval_pos = evaluate(agent, eval_env, args.eval_episodes)
            recent = successes[-50:] if successes else []
            print(f"[{mode}] 步数 {step:6d} | 训练回合 {episodes:4d} | "
                  f"近50成功 {np.mean(recent):.2f} | 评估 {eval_ret:8.1f} | "
                  f"评估成功率 {eval_succ:.2f} | 最大位置 {eval_pos:+.3f}")

    elapsed = time.time() - t_start
    eval_ret, eval_succ, eval_pos = evaluate(agent, eval_env, args.eval_episodes)
    final_succ = float(np.mean(successes[-100:])) if successes else 0.0
    print(f"[{mode}] 结束：最终评估 {eval_ret:.1f} | 评估成功率 {eval_succ:.2f} | "
          f"训练末100回合成功率 {final_succ:.2f} | 最大位置 {eval_pos:+.3f} | 用时 {elapsed:.1f}s")
    return {"mode": mode, "final_eval": eval_ret, "eval_succ": eval_succ,
            "train_succ": final_succ, "eval_pos": eval_pos,
            "first_success_ep": first_success_ep, "episodes": episodes,
            "elapsed": elapsed}


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="第053章：SAC 的课程学习")
    p.add_argument("--modes", type=str, default="direct,curriculum",
                   help="逗号分隔：direct / curriculum")
    p.add_argument("--powers", type=str, default="0.006,0.004,0.0025,0.0015",
                   help="课程各阶段的引擎功率序列；最后一个应是目标值 0.0015")
    p.add_argument("--max-steps", type=int, default=30000, help="每种模式的环境步数上限")
    p.add_argument("--start-steps", type=int, default=2000, help="训练前纯随机探索步数")
    p.add_argument("--update-every", type=int, default=1, help="每多少环境步做一次更新")
    p.add_argument("--eval-every", type=int, default=5000, help="评估间隔")
    p.add_argument("--eval-episodes", type=int, default=5, help="评估回合数")
    p.add_argument("--buffer-size", type=int, default=100000, help="回放池容量")
    p.add_argument("--batch-size", type=int, default=256, help="批大小")
    p.add_argument("--hidden", type=int, default=256, help="隐藏层宽度")
    p.add_argument("--lr", type=float, default=3e-4, help="学习率")
    p.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
    p.add_argument("--tau", type=float, default=0.005, help="目标网络软更新系数")
    p.add_argument("--alpha-init", type=float, default=0.2, help="温度系数初值")
    p.add_argument("--seed", type=int, default=0, help="随机种子")
    p.add_argument("--save-dir", type=str, default="runs/ch053", help="输出目录")
    p.add_argument("--quick", action="store_true", help="快速模式：缩短训练")
    return p.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.max_steps = min(args.max_steps, 4000)
        args.eval_every = 1000
        args.start_steps = min(args.start_steps, 400)
        args.eval_episodes = 3

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    print(f"种子: {args.seed} | 模式: {modes} | 功率序列: {args.powers}")
    print(f"配置: max_steps={args.max_steps}, batch={args.batch_size}, lr={args.lr}")

    results = [train_mode(m, args) for m in modes]

    print("\n" + "=" * 102)
    print("课程学习对比（全部在真实 MountainCarContinuous-v0 上评估）")
    print(f"{'模式':<12}{'最终评估':>10}{'评估成功率':>12}{'训练成功率':>12}"
          f"{'首次成功回合':>14}{'训练回合数':>12}{'用时(s)':>9}")
    for r in results:
        first = r["first_success_ep"] if r["first_success_ep"] is not None else -1
        print(f"{r['mode']:<12}{r['final_eval']:>10.1f}{r['eval_succ']:>12.2f}"
              f"{r['train_succ']:>12.2f}{first:>14}{r['episodes']:>12}{r['elapsed']:>9.1f}")
    print("=" * 102)
    print("结论提示：课程组先从大功率里学会'来回摆动蓄能'的动作模式，进入目标")
    print("功率后这个技能依然有效；直接训练组要等随机探索撞出第一条成功轨迹。")
    print("注意：课程环境的动力学形式与目标一致，只有功率不同，这是可持续的。")

    os.makedirs(args.save_dir, exist_ok=True)
    print(f"输出目录：{args.save_dir}")


if __name__ == "__main__":
    main()
