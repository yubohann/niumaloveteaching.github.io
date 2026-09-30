"""
第036章 SAC的连续动作重参数化

连续动作 SAC 的策略梯度有两种写法：
  pathwise（重参数化）：a = tanh(mu + sigma * xi)，xi 是外部噪声，
      梯度从 Q(s,a) 一路回传到 mu、sigma——信息量最大，方差最小；
  score（得分函数 / 似然比）：a 先固定采样出来，策略只通过
      grad log pi(a|s) 这个方向更新，Q 的数值只是一个权重。
本章把两种估计器做成同一个策略损失函数里的开关，在 Pendulum-v1 上
对比它们的评估回报、策略梯度范数与"相邻两次策略梯度的余弦相似度"
（衡量梯度方向的一致性，余弦越低说明噪声越大）。

运行：
    python code/ch036.py           # 两种估计器依次训练（CPU 约 5-8 分钟）
    python code/ch036.py --quick   # 快速跑通（约 1 分钟，只验证流程）
    python code/ch036.py --estimators score --max-steps 30000

预期：pathwise 学习明显更快、最终回报更好；score 的梯度方向一致
      性差、学习慢，用 15000 步通常只能学到部分策略。
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


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_env(env_id: str, seed: int):
    env = gym.make(env_id)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env


# ---------------------------------------------------------------------------
# 经验回放池
# ---------------------------------------------------------------------------
class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, act_dim: int):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rew = np.zeros(capacity, dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)
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


# ---------------------------------------------------------------------------
# 网络
# ---------------------------------------------------------------------------
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
    """tanh 压缩高斯策略：同时支持重参数化采样与普通采样。"""

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

    def distribution(self, obs: torch.Tensor):
        h = self.body(obs)
        mean = self.mean_layer(h)
        log_std = torch.clamp(self.log_std_layer(h), LOG_STD_MIN, LOG_STD_MAX)
        return torch.distributions.Normal(mean, log_std.exp())

    def sample(self, obs: torch.Tensor, deterministic: bool = False,
               reparameterize: bool = True):
        """reparameterize=True 时梯度可穿过采样；False 时采样值被 detach。"""
        dist = self.distribution(obs)
        if deterministic:
            return torch.tanh(dist.mean) * self.act_scale, None, dist
        x = dist.rsample() if reparameterize else dist.sample()
        y = torch.tanh(x)
        action = y * self.act_scale
        log_prob = dist.log_prob(x) - torch.log(1.0 - y.pow(2) + 1e-6)
        return action, log_prob.sum(dim=-1), dist


# ---------------------------------------------------------------------------
# SAC 智能体：策略损失由 estimator 决定
# ---------------------------------------------------------------------------
class SACAgent:
    def __init__(self, obs_dim: int, act_dim: int, act_scale: float,
                 estimator: str, alpha: float = 0.2, target_entropy: float = -1.0,
                 lr: float = 3e-4, gamma: float = 0.99, tau: float = 0.005,
                 hidden: int = 256):
        assert estimator in ("pathwise", "score"), f"未知估计器: {estimator}"
        self.estimator = estimator
        self.alpha = float(alpha)
        self.target_entropy = float(target_entropy)
        self.gamma = gamma
        self.tau = tau
        self.prev_grad = None            # 上一次策略梯度（用于余弦相似度）

        self.q1 = QNet(obs_dim, act_dim, hidden)
        self.q2 = QNet(obs_dim, act_dim, hidden)
        self.q1_target = QNet(obs_dim, act_dim, hidden)
        self.q2_target = QNet(obs_dim, act_dim, hidden)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        for p in self.q1_target.parameters():
            p.requires_grad_(False)
        for p in self.q2_target.parameters():
            p.requires_grad_(False)
        self.policy = GaussianPolicy(obs_dim, act_dim, hidden, act_scale)

        self.q_optimizer = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.log_alpha = torch.tensor(math.log(alpha), requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)

    @property
    def current_alpha(self) -> float:
        return float(self.log_alpha.exp().item())

    @torch.no_grad()
    def act(self, obs, deterministic: bool = False) -> np.ndarray:
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        action, _, _ = self.policy.sample(obs_t, deterministic=deterministic)
        return action.squeeze(0).numpy()

    def update_critic(self, batch: dict) -> dict:
        obs, act, rew = batch["obs"], batch["act"], batch["rew"]
        next_obs, done = batch["next_obs"], batch["done"]

        with torch.no_grad():
            # 评论家目标里的动作始终用重参数化采样（与估计器实验无关）
            next_act, next_logp, _ = self.policy.sample(next_obs)
            q_next = torch.min(self.q1_target(next_obs, next_act),
                               self.q2_target(next_obs, next_act))
            target = rew + self.gamma * (1.0 - done) * (
                q_next - self.current_alpha * next_logp)

        q1_pred = self.q1(obs, act)
        q2_pred = self.q2(obs, act)
        q_loss = F.mse_loss(q1_pred, target) + F.mse_loss(q2_pred, target)

        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        with torch.no_grad():
            for p, tp in zip(self.q1.parameters(), self.q1_target.parameters()):
                tp.mul_(1.0 - self.tau).add_(self.tau * p)
            for p, tp in zip(self.q2.parameters(), self.q2_target.parameters()):
                tp.mul_(1.0 - self.tau).add_(self.tau * p)

        return {"q_loss": float(q_loss.item())}

    def update_actor(self, batch: dict) -> dict:
        """策略 + 温度系数更新；本章的实验开关在这里。"""
        obs = batch["obs"]

        if self.estimator == "pathwise":
            # 重参数化：动作带梯度，Q 的梯度直接回传到 mu / sigma
            action, logp, _ = self.policy.sample(obs, reparameterize=True)
            q_act = torch.min(self.q1(obs, action), self.q2(obs, action))
            policy_loss = (self.current_alpha * logp - q_act).mean()
        else:
            # 得分函数：先用 no_grad 采出同一个动作 x，Q 与 log pi 都用它
            with torch.no_grad():
                dist_det = self.policy.distribution(obs)
                x = dist_det.sample()
                y = torch.tanh(x)
                action = y * self.policy.act_scale
                q_act = torch.min(self.q1(obs, action), self.q2(obs, action))
            # 重新构建分布并计算 log pi(x)：x 已 detach，梯度只走分布参数
            dist = self.policy.distribution(obs)
            logp = (dist.log_prob(x) - torch.log(1.0 - y.pow(2) + 1e-6)).sum(dim=-1)
            adv = (q_act - self.current_alpha * logp.detach()).detach()
            policy_loss = -(adv * logp).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()

        # 诊断：策略梯度的范数与相邻梯度的余弦相似度
        grad_vec = torch.cat([
            p.grad.reshape(-1) for p in self.policy.parameters()
            if p.grad is not None
        ])
        grad_norm = float(grad_vec.norm().item())
        grad_cos = float("nan")
        if self.prev_grad is not None:
            # 维度一致（网络结构没变），可以直接做余弦
            grad_cos = float(F.cosine_similarity(
                grad_vec.unsqueeze(0), self.prev_grad.unsqueeze(0)).item())
        self.prev_grad = grad_vec.detach().clone()

        self.policy_optimizer.step()

        alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        return {
            "policy_loss": float(policy_loss.item()),
            "grad_norm": grad_norm,
            "grad_cos": grad_cos,
            "alpha": self.current_alpha,
        }


# ---------------------------------------------------------------------------
# 评估
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(env, agent: SACAgent, episodes: int) -> float:
    returns = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            action = agent.act(obs, deterministic=True)
            obs, r, terminated, truncated, _ = env.step(action)
            ep_ret += r
            done = terminated or truncated
        returns.append(ep_ret)
    return float(np.mean(returns))


# ---------------------------------------------------------------------------
# 单个估计器的训练
# ---------------------------------------------------------------------------
def train_estimator(estimator: str, args) -> dict:
    set_seed(args.seed)
    env = make_env("Pendulum-v1", args.seed)
    eval_env = make_env("Pendulum-v1", args.seed + 500)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_scale = float(env.action_space.high[0])

    agent = SACAgent(obs_dim, act_dim, act_scale, estimator=estimator,
                     alpha=args.alpha, target_entropy=args.target_entropy,
                     lr=args.lr, gamma=args.gamma, tau=args.tau,
                     hidden=args.hidden)
    buf = ReplayBuffer(args.buffer_size, obs_dim, act_dim)

    history = {"steps": [], "eval_return": [], "grad_norm": [], "grad_cos": []}
    ep_returns = []
    # 滚动统计：最近 200 次策略更新的梯度范数与余弦
    grad_norm_window = []
    grad_cos_window = []
    obs, _ = env.reset(seed=args.seed)
    ep_ret = 0.0
    eval_ret = float("nan")
    t_start = time.time()

    print(f"[{estimator}] 开始训练 | Pendulum-v1 | 种子 {args.seed}")

    for step in range(1, args.max_steps + 1):
        if step <= args.start_steps:
            action = env.action_space.sample()
        else:
            action = agent.act(obs, deterministic=False)

        next_obs, rew, terminated, truncated, _ = env.step(action)
        buf.add(obs, action, rew, next_obs, float(terminated))
        obs = next_obs
        ep_ret += rew

        if terminated or truncated:
            ep_returns.append(ep_ret)
            ep_ret = 0.0
            obs, _ = env.reset()

        if buf.size >= args.batch_size:
            agent.update_critic(buf.sample(args.batch_size))
            actor_stats = agent.update_actor(buf.sample(args.batch_size))
            grad_norm_window.append(actor_stats["grad_norm"])
            if not math.isnan(actor_stats["grad_cos"]):
                grad_cos_window.append(actor_stats["grad_cos"])
            grad_norm_window = grad_norm_window[-200:]
            grad_cos_window = grad_cos_window[-200:]

        if step % args.eval_every == 0 or step == args.max_steps:
            eval_ret = evaluate(eval_env, agent, args.eval_episodes)
            recent = float(np.mean(ep_returns[-20:])) if ep_returns else float("nan")
            gn = float(np.mean(grad_norm_window)) if grad_norm_window else float("nan")
            gc = float(np.mean(grad_cos_window)) if grad_cos_window else float("nan")
            history["steps"].append(step)
            history["eval_return"].append(eval_ret)
            history["grad_norm"].append(gn)
            history["grad_cos"].append(gc)
            print(f"[{estimator}] 步数 {step:6d} | 近20均值 {recent:9.1f} | "
                  f"评估 {eval_ret:9.1f} | 梯度范数 {gn:7.3f} | "
                  f"梯度余弦 {gc:+.3f} | alpha {agent.current_alpha:.3f}")

    elapsed = time.time() - t_start
    print(f"[{estimator}] 训练结束 | 用时 {elapsed:.1f}s | 最终评估 {eval_ret:.1f} | "
          f"平均梯度余弦 {np.mean(grad_cos_window):+.3f}")

    os.makedirs(args.save_dir, exist_ok=True)
    ckpt = os.path.join(args.save_dir, f"sac_{estimator}.pt")
    torch.save({"policy": agent.policy.state_dict(),
                "q1": agent.q1.state_dict(),
                "estimator": estimator}, ckpt)
    print(f"[{estimator}] 模型已保存到 {ckpt}")

    return {"estimator": estimator, "final_eval": eval_ret, "elapsed": elapsed,
            "mean_grad_cos": float(np.mean(grad_cos_window)) if grad_cos_window else float("nan"),
            "history": history}


# ---------------------------------------------------------------------------
# 可选绘图
# ---------------------------------------------------------------------------
def maybe_plot(results, save_dir: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("未安装 matplotlib，跳过绘图")
        return
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for res in results:
        hist = res["history"]
        axes[0].plot(hist["steps"], hist["eval_return"], marker="o",
                     label=res["estimator"])
        axes[1].plot(hist["steps"], hist["grad_cos"], marker="o",
                     label=res["estimator"])
    axes[0].set_xlabel("环境步数")
    axes[0].set_ylabel("确定性评估回报")
    axes[0].set_title("评估回报")
    axes[0].legend()
    axes[1].set_xlabel("环境步数")
    axes[1].set_ylabel("相邻梯度余弦均值")
    axes[1].set_title("梯度方向一致性")
    axes[1].legend()
    fig.tight_layout()
    out = os.path.join(save_dir, "ch036_reparam.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"曲线已保存到 {out}")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="第036章：SAC 的连续动作重参数化")
    p.add_argument("--estimators", type=str, default="pathwise,score",
                   help="要对比的策略梯度估计器")
    p.add_argument("--alpha", type=float, default=0.2, help="alpha 初值")
    p.add_argument("--target-entropy", type=float, default=-1.0, help="目标熵")
    p.add_argument("--max-steps", type=int, default=15000, help="每种估计器的步数上限")
    p.add_argument("--start-steps", type=int, default=1000, help="纯随机探索步数")
    p.add_argument("--buffer-size", type=int, default=100000, help="回放池容量")
    p.add_argument("--batch-size", type=int, default=256, help="批量大小")
    p.add_argument("--hidden", type=int, default=256, help="隐藏层宽度")
    p.add_argument("--lr", type=float, default=3e-4, help="学习率")
    p.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
    p.add_argument("--tau", type=float, default=0.005, help="目标网络软更新系数")
    p.add_argument("--eval-every", type=int, default=3000, help="评估与日志间隔")
    p.add_argument("--eval-episodes", type=int, default=3, help="每次评估的回合数")
    p.add_argument("--seed", type=int, default=0, help="随机种子")
    p.add_argument("--save-dir", type=str, default="runs/ch036", help="模型保存目录")
    p.add_argument("--quick", action="store_true", help="快速模式：几十秒跑完")
    return p.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.max_steps = min(args.max_steps, 2500)
        args.start_steps = min(args.start_steps, 500)
        args.eval_every = 1200
        args.eval_episodes = 2

    estimators = [e.strip() for e in args.estimators.split(",") if e.strip()]
    for e in estimators:
        if e not in ("pathwise", "score"):
            raise SystemExit(f"未知估计器 {e}，可选：pathwise / score")

    results = [train_estimator(e, args) for e in estimators]

    print("=" * 74)
    print("梯度估计器对比汇总（Pendulum-v1，数值随机种子波动，以实跑为准）")
    for res in results:
        print(f"  {res['estimator']:>8s}: 最终评估 {res['final_eval']:9.1f} | "
              f"平均梯度余弦 {res['mean_grad_cos']:+.3f}")

    maybe_plot(results, args.save_dir)


if __name__ == "__main__":
    main()
