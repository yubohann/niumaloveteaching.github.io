"""
第050章 SAC的自动超参数搜索

手调 SAC 超参数（学习率、批大小、软更新系数、温度初值）既费时又难复现。
本章实现一个小型随机搜索（Random Search）流程：

  1. 在离散网格上随机采样 --trials 组配置（第 1 组固定为默认配置做基准）
  2. 每组用短预算（--trial-steps）训练，取中途与结束时的评估回报作为得分
  3. 按得分排序，选出最好的一组，用更长预算（--final-steps）重训
  4. 把最优配置与最终得分写入 runs/ch050/best_config.json

搜索空间：lr × batch × tau × alpha_init（共 24 种组合）。

运行：
    python code/ch050.py           # 6 组试跑 + 最优重训（CPU 约 4-6 分钟）
    python code/ch050.py --quick   # 快速跑通（约 1 分钟）

预期：短预算下各组差距未必稳定（评估方差大），但学习率与批大小的组合会
显示出明显的两极分化；重训得分通常高于试跑得分。
"""
import argparse
import copy
import json
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


def make_env(env_id: str, seed: int):
    env = gym.make(env_id)
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
    def __init__(self, obs_dim: int, act_dim: int, action_scale: float, cfg: dict):
        self.policy = GaussianPolicy(obs_dim, act_dim, cfg["hidden"], action_scale)
        self.q1 = QNet(obs_dim, act_dim, cfg["hidden"])
        self.q2 = QNet(obs_dim, act_dim, cfg["hidden"])
        self.q1_target = QNet(obs_dim, act_dim, cfg["hidden"])
        self.q2_target = QNet(obs_dim, act_dim, cfg["hidden"])
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        for p in self.q1_target.parameters():
            p.requires_grad_(False)
        for p in self.q2_target.parameters():
            p.requires_grad_(False)
        self.cfg = cfg
        self.log_alpha = torch.tensor(np.log(cfg["alpha_init"]), requires_grad=True)
        self.target_entropy = -float(act_dim)
        self.opt_q = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=cfg["lr"])
        self.opt_policy = torch.optim.Adam(self.policy.parameters(), lr=cfg["lr"])
        self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=cfg["lr"])

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

    def update(self, batch: dict, gamma: float, tau: float) -> dict:
        obs, act = batch["obs"], batch["act"]
        rew, next_obs, done = batch["rew"], batch["next_obs"], batch["done"]

        with torch.no_grad():
            next_act, next_logp = self.policy.sample(next_obs)
            q_next = torch.minimum(self.q1_target(next_obs, next_act),
                                   self.q2_target(next_obs, next_act))
            backup = rew + gamma * (1.0 - done) * (q_next - self.alpha.detach() * next_logp)

        q1 = self.q1(obs, act)
        q2 = self.q2(obs, act)
        q_loss = F.mse_loss(q1, backup) + F.mse_loss(q2, backup)
        self.opt_q.zero_grad()
        q_loss.backward()
        self.opt_q.step()

        new_act, logp = self.policy.sample(obs)
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
            soft_update(self.q1, self.q1_target, tau)
            soft_update(self.q2, self.q2_target, tau)

        return {"q_loss": q_loss.item(), "alpha": float(self.alpha.item())}


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
# 用一组配置训练并返回得分（供随机搜索调用）
# ---------------------------------------------------------------------------
def run_config(cfg: dict, steps: int, args, tag: str, verbose: bool = False):
    """训练一个配置；每隔一半预算评估一次，返回 (得分, 最好评估)。"""
    set_seed(args.seed)  # 所有试跑同种子，减少配置间噪声
    env = make_env(args.env, args.seed)
    eval_env = make_env(args.env, args.seed + 100)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    action_scale = float(env.action_space.high[0])
    agent = SAC(obs_dim, act_dim, action_scale, cfg)
    buffer = ReplayBuffer(args.buffer_size, obs_dim, act_dim)

    obs, _ = env.reset(seed=args.seed)
    ep_ret = 0.0
    ep_returns = []
    evals = []
    start_steps = min(args.start_steps, steps // 5)
    t_start = time.time()

    for step in range(1, steps + 1):
        if step <= start_steps:
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

        if buffer.size >= start_steps and step % args.update_every == 0:
            agent.update(buffer.sample(cfg["batch"]), args.gamma, cfg["tau"])

        if step % max(steps // 2, 1) == 0:
            evals.append(evaluate(agent, eval_env, args.eval_episodes))
            if verbose:
                print(f"    [{tag}] 步数 {step} | 评估 {evals[-1]:.1f}")

    score = float(np.mean(evals))
    best = max(evals) if evals else score
    elapsed = time.time() - t_start
    if verbose:
        print(f"    [{tag}] 结束：均分 {score:.1f} | 最好 {best:.1f} | 用时 {elapsed:.1f}s")
    return score, best, elapsed


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="第050章：SAC 的自动超参数搜索")
    p.add_argument("--env", type=str, default="Pendulum-v1", help="环境 ID")
    p.add_argument("--trials", type=int, default=6, help="随机搜索的试跑组数（含默认配置）")
    p.add_argument("--trial-steps", type=int, default=5000, help="每组试跑的环境步数")
    p.add_argument("--final-steps", type=int, default=15000, help="最优配置重训的步数")
    p.add_argument("--start-steps", type=int, default=1000, help="训练前纯随机探索步数")
    p.add_argument("--update-every", type=int, default=1, help="每多少环境步做一次更新")
    p.add_argument("--eval-episodes", type=int, default=5, help="评估回合数")
    p.add_argument("--buffer-size", type=int, default=100000, help="回放池容量")
    p.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
    p.add_argument("--hidden", type=int, default=256, help="隐藏层宽度")
    p.add_argument("--seed", type=int, default=0, help="随机种子")
    p.add_argument("--save-dir", type=str, default="runs/ch050", help="输出目录")
    p.add_argument("--quick", action="store_true", help="快速模式：缩短试跑与重训")
    return p.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.trials = min(args.trials, 4)
        args.trial_steps = 1200
        args.final_steps = 3000
        args.start_steps = min(args.start_steps, 300)
        args.eval_episodes = 3

    # 搜索空间：离散网格上做随机采样（组合数 3*3*2*2 = 24）
    space = {
        "lr": [1e-4, 3e-4, 1e-3],
        "batch": [64, 128, 256],
        "tau": [0.005, 0.01],
        "alpha_init": [0.1, 0.2],
    }
    default_cfg = {"lr": 3e-4, "batch": 256, "tau": 0.005, "alpha_init": 0.2,
                   "hidden": args.hidden}
    rng = random.Random(args.seed)

    # 采样不重复的配置；第一组固定为默认配置
    seen = set()
    configs = [default_cfg]
    seen.add((default_cfg["lr"], default_cfg["batch"], default_cfg["tau"],
              default_cfg["alpha_init"]))
    while len(configs) < args.trials:
        cfg = {
            "lr": rng.choice(space["lr"]),
            "batch": rng.choice(space["batch"]),
            "tau": rng.choice(space["tau"]),
            "alpha_init": rng.choice(space["alpha_init"]),
            "hidden": args.hidden,
        }
        key = (cfg["lr"], cfg["batch"], cfg["tau"], cfg["alpha_init"])
        if key not in seen:
            seen.add(key)
            configs.append(cfg)

    print(f"环境: {args.env} | 种子: {args.seed}")
    print(f"搜索空间: lr={space['lr']}, batch={space['batch']}, "
          f"tau={space['tau']}, alpha_init={space['alpha_init']}")
    print(f"试跑组数 {len(configs)}，每组 {args.trial_steps} 步；"
          f"最优配置重训 {args.final_steps} 步\n")

    results = []
    for i, cfg in enumerate(configs):
        tag = "默认" if i == 0 else f"试跑{i}"
        print(f"[随机搜索 {i + 1}/{len(configs)}] {tag}: lr={cfg['lr']}, "
              f"batch={cfg['batch']}, tau={cfg['tau']}, alpha={cfg['alpha_init']}")
        score, best, elapsed = run_config(cfg, args.trial_steps, args, tag, verbose=True)
        results.append({"cfg": cfg, "score": score, "best": best,
                        "elapsed": elapsed, "tag": tag})

    results.sort(key=lambda r: r["score"], reverse=True)
    print("\n" + "=" * 92)
    print("随机搜索排名（得分 = 试跑中途与结束两次评估的均值）")
    print(f"{'名次':>4}{'lr':>8}{'batch':>7}{'tau':>7}{'alpha':>7}{'均分':>9}{'最好':>9}")
    for rank, r in enumerate(results, 1):
        c = r["cfg"]
        print(f"{rank:>4}{c['lr']:>8.0e}{c['batch']:>7}{c['tau']:>7.3f}"
              f"{c['alpha_init']:>7.2f}{r['score']:>9.1f}{r['best']:>9.1f}")
    print("=" * 92)

    best_cfg = results[0]["cfg"]
    print(f"\n[重训] 最优配置：{best_cfg}")
    final_score, final_best, final_elapsed = run_config(
        best_cfg, args.final_steps, args, "最终", verbose=True)
    print(f"[重训] 结果：均分 {final_score:.1f}（试跑同配置为 {results[0]['score']:.1f}）| "
          f"最好 {final_best:.1f} | 用时 {final_elapsed:.1f}s")

    os.makedirs(args.save_dir, exist_ok=True)
    out = {
        "best_config": best_cfg,
        "trial_score": results[0]["score"],
        "final_score": final_score,
        "final_best": final_best,
        "trials": [{"cfg": r["cfg"], "score": r["score"]} for r in results],
    }
    path = os.path.join(args.save_dir, "best_config.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"最优配置与试跑记录已保存到 {path}")
    print("说明：短预算试跑方差较大，排名靠前的配置建议用 --trial-steps 提高预算复验。")


if __name__ == "__main__":
    main()
