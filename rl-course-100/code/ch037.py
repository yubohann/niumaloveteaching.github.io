"""
第037章 SAC的离散动作扩展

连续 SAC 靠重参数化把梯度穿过动作采样。离散分类分布没有轻量的
重参数化方案，于是离散版 SAC（Christodoulou 2019 一文的思路）换成：
  1) 策略输出每个离散动作的 logits，用 Categorical 采样；
  2) Q 网络只吃状态、输出"每个动作"的价值向量；
  3) 评论家目标不再用"下一个动作"，而是对下一个状态的所有动作求期望：
       V(s') = sum_a pi(a|s') * ( min_j Q_j(s',a) - alpha * log pi(a|s') )
     这一步把 Bellman 目标变成平滑的期望，比直接用采样动作的方差小；
  4) 温度系数的目标熵改为 0.98 * log(动作数)。

本章在 CartPole-v1 上对比两种评论家目标：
    expectation: 对全部动作求期望（离散 SAC 的标准做法）
    sample     : 直接采样下一个动作（连续 SAC 直觉照搬过来）
观察评估回报、策略熵、|Q| 幅度三者的差异。

运行：
    python code/ch037.py           # 两种目标依次训练（CPU 约 5-8 分钟）
    python code/ch037.py --quick   # 快速跑通（约 1-2 分钟）
    python code/ch037.py --targets expectation --max-steps 40000

预期：CartPole 较容易，两种目标都能接近满分 500；期望目标通常
      收敛更快、|Q| 曲线更平滑，采样目标的 |Q| 波动更大。
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

ALL_TARGETS = ["expectation", "sample"]


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
# 经验回放池（动作是离散整数，单独用 int64 数组存）
# ---------------------------------------------------------------------------
class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act = np.zeros(capacity, dtype=np.int64)
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
        # CartPole 到 500 步是 truncated，仍然 bootstrap；done 只给真实终止
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
# 网络：Q 网络只吃状态，输出每个动作的价值
# ---------------------------------------------------------------------------
class QNetDiscrete(nn.Module):
    """Q(s, ·)：输入状态，输出向量 (B, n_actions)。"""

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class CategoricalPolicy(nn.Module):
    """离散策略：输出 logits，采样/确定性动作都在类别分布上完成。"""

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


# ---------------------------------------------------------------------------
# 离散 SAC 智能体
# ---------------------------------------------------------------------------
class DiscreteSACAgent:
    def __init__(self, obs_dim: int, n_actions: int, target_mode: str,
                 alpha: float = 0.2, target_entropy: float = -1.0,
                 lr: float = 3e-4, gamma: float = 0.99, tau: float = 0.005,
                 hidden: int = 256):
        assert target_mode in ALL_TARGETS
        self.target_mode = target_mode
        self.n_actions = n_actions
        self.alpha = float(alpha)
        # 离散版目标熵：0.98 * log(动作数)，传负值表示"用默认公式"
        if target_entropy < 0:
            self.target_entropy = 0.98 * math.log(n_actions)
        else:
            self.target_entropy = float(target_entropy)
        self.gamma = gamma
        self.tau = tau

        self.q1 = QNetDiscrete(obs_dim, n_actions, hidden)
        self.q2 = QNetDiscrete(obs_dim, n_actions, hidden)
        self.q1_target = QNetDiscrete(obs_dim, n_actions, hidden)
        self.q2_target = QNetDiscrete(obs_dim, n_actions, hidden)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        for p in self.q1_target.parameters():
            p.requires_grad_(False)
        for p in self.q2_target.parameters():
            p.requires_grad_(False)
        self.policy = CategoricalPolicy(obs_dim, n_actions, hidden)

        self.q_optimizer = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.log_alpha = torch.tensor(math.log(alpha), requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)

    @property
    def current_alpha(self) -> float:
        return float(self.log_alpha.exp().item())

    @torch.no_grad()
    def act(self, obs, deterministic: bool = False) -> int:
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        logits = self.policy(obs_t)
        if deterministic:
            return int(torch.argmax(logits, dim=-1).item())
        probs = torch.softmax(logits, dim=-1)
        return int(torch.multinomial(probs, num_samples=1).item())

    def update(self, batch: dict) -> dict:
        obs, act, rew = batch["obs"], batch["act"], batch["rew"]
        next_obs, done = batch["next_obs"], batch["done"]

        # ---------------- 1) 评论家：对下一个状态的策略求价值 ----------------
        with torch.no_grad():
            next_logits = self.policy(next_obs)
            next_log_probs = F.log_softmax(next_logits, dim=-1)
            next_probs = next_log_probs.exp()
            q_next_min = torch.min(self.q1_target(next_obs),
                                   self.q2_target(next_obs))     # (B, A)
            if self.target_mode == "expectation":
                # 期望目标：V(s') = sum_a pi(a|s') * (Q_min - alpha * log pi)
                v_next = (next_probs * (q_next_min -
                                        self.current_alpha * next_log_probs)).sum(dim=-1)
            else:
                # 采样目标：从下一个策略里抽一个动作，用它的 Q 值
                dist = torch.distributions.Categorical(probs=next_probs)
                a_next = dist.sample()
                q_next_a = q_next_min.gather(1, a_next.unsqueeze(1)).squeeze(1)
                logp_next_a = next_log_probs.gather(
                    1, a_next.unsqueeze(1)).squeeze(1)
                v_next = q_next_a - self.current_alpha * logp_next_a
            target = rew + self.gamma * (1.0 - done) * v_next

        q1_all = self.q1(obs)
        q2_all = self.q2(obs)
        q1_pred = q1_all.gather(1, act.unsqueeze(1)).squeeze(1)
        q2_pred = q2_all.gather(1, act.unsqueeze(1)).squeeze(1)
        q_loss = F.mse_loss(q1_pred, target) + F.mse_loss(q2_pred, target)

        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # ---------------- 2) 演员：对全部动作求期望的策略损失 ----------------
        logits = self.policy(obs)
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        with torch.no_grad():
            q_min = torch.min(self.q1(obs), self.q2(obs))          # (B, A)
        # E_{a~pi}[ alpha * log pi(a|s) - Q(s,a) ]
        policy_loss = (probs * (self.current_alpha * log_probs - q_min)).sum(dim=-1).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # ---------------- 3) 温度系数：熵目标为 0.98*log(|A|) ----------------
        with torch.no_grad():
            log_probs_new = F.log_softmax(self.policy(obs), dim=-1)
            probs_new = log_probs_new.exp()
            entropy = -(probs_new * log_probs_new).sum(dim=-1)      # (B,)
        alpha_loss = (self.log_alpha * (entropy - self.target_entropy)).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # ---------------- 4) 目标网络软更新 ----------------
        with torch.no_grad():
            for p, tp in zip(self.q1.parameters(), self.q1_target.parameters()):
                tp.mul_(1.0 - self.tau).add_(self.tau * p)
            for p, tp in zip(self.q2.parameters(), self.q2_target.parameters()):
                tp.mul_(1.0 - self.tau).add_(self.tau * p)

        return {
            "q_loss": float(q_loss.item()),
            "policy_loss": float(policy_loss.item()),
            "alpha_loss": float(alpha_loss.item()),
            "mean_abs_q": float(q1_all.abs().mean().item()),
            "entropy": float(entropy.mean().item()),
            "alpha": self.current_alpha,
        }


# ---------------------------------------------------------------------------
# 评估：确定性动作 = argmax logits
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(env, agent: DiscreteSACAgent, episodes: int) -> float:
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
# 单个目标模式的训练
# ---------------------------------------------------------------------------
def train_target(mode: str, args) -> dict:
    set_seed(args.seed)
    env = make_env("CartPole-v1", args.seed)
    eval_env = make_env("CartPole-v1", args.seed + 500)

    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    agent = DiscreteSACAgent(obs_dim, n_actions, target_mode=mode,
                             alpha=args.alpha, target_entropy=args.target_entropy,
                             lr=args.lr, gamma=args.gamma, tau=args.tau,
                             hidden=args.hidden)
    buf = ReplayBuffer(args.buffer_size, obs_dim)

    history = {"steps": [], "eval_return": [], "mean_abs_q": [], "entropy": []}
    ep_returns = []
    obs, _ = env.reset(seed=args.seed)
    ep_ret = 0.0
    last_stats = {"mean_abs_q": 0.0, "entropy": 0.0, "q_loss": 0.0}
    eval_ret = float("nan")
    t_start = time.time()

    print(f"[{mode}] 开始训练 | CartPole-v1 | 种子 {args.seed} | "
          f"目标熵 {agent.target_entropy:.3f}")

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
            last_stats = agent.update(buf.sample(args.batch_size))

        if step % args.eval_every == 0 or step == args.max_steps:
            eval_ret = evaluate(eval_env, agent, args.eval_episodes)
            recent = float(np.mean(ep_returns[-20:])) if ep_returns else float("nan")
            history["steps"].append(step)
            history["eval_return"].append(eval_ret)
            history["mean_abs_q"].append(last_stats["mean_abs_q"])
            history["entropy"].append(last_stats["entropy"])
            print(f"[{mode}] 步数 {step:6d} | 近20均值 {recent:7.1f} | "
                  f"评估 {eval_ret:7.1f} | 熵 {last_stats['entropy']:.3f} | "
                  f"alpha {agent.current_alpha:.3f} | |Q| {last_stats['mean_abs_q']:7.1f}")

    elapsed = time.time() - t_start
    print(f"[{mode}] 训练结束 | 用时 {elapsed:.1f}s | 最终评估 {eval_ret:.1f} | "
          f"最后20回合均值 {np.mean(ep_returns[-20:]):.1f}")

    os.makedirs(args.save_dir, exist_ok=True)
    ckpt = os.path.join(args.save_dir, f"sac_discrete_{mode}.pt")
    torch.save({"policy": agent.policy.state_dict(),
                "q1": agent.q1.state_dict(),
                "q2": agent.q2.state_dict(),
                "target_mode": mode}, ckpt)
    print(f"[{mode}] 模型已保存到 {ckpt}")

    return {"mode": mode, "final_eval": eval_ret, "elapsed": elapsed,
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
                     label=res["mode"])
        axes[1].plot(hist["steps"], hist["mean_abs_q"], marker="o",
                     label=res["mode"])
    axes[0].set_xlabel("环境步数")
    axes[0].set_ylabel("确定性评估回报")
    axes[0].set_title("评估回报")
    axes[0].legend()
    axes[1].set_xlabel("环境步数")
    axes[1].set_ylabel("|Q| 平均值")
    axes[1].set_title("价值幅度（平滑性对比）")
    axes[1].legend()
    fig.tight_layout()
    out = os.path.join(save_dir, "ch037_discrete_sac.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"曲线已保存到 {out}")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="第037章：SAC 的离散动作扩展")
    p.add_argument("--targets", type=str, default="expectation,sample",
                   help="评论家目标模式：expectation=全动作期望，sample=采样动作")
    p.add_argument("--alpha", type=float, default=0.2, help="alpha 初值")
    p.add_argument("--target-entropy", type=float, default=-1.0,
                   help="目标熵；负值表示使用 0.98*log(动作数)")
    p.add_argument("--max-steps", type=int, default=30000, help="每种模式的步数上限")
    p.add_argument("--start-steps", type=int, default=1000, help="纯随机探索步数")
    p.add_argument("--buffer-size", type=int, default=200000, help="回放池容量")
    p.add_argument("--batch-size", type=int, default=256, help="批量大小")
    p.add_argument("--hidden", type=int, default=256, help="隐藏层宽度")
    p.add_argument("--lr", type=float, default=3e-4, help="学习率")
    p.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
    p.add_argument("--tau", type=float, default=0.005, help="目标网络软更新系数")
    p.add_argument("--eval-every", type=int, default=5000, help="评估与日志间隔")
    p.add_argument("--eval-episodes", type=int, default=5, help="每次评估的回合数")
    p.add_argument("--seed", type=int, default=0, help="随机种子")
    p.add_argument("--save-dir", type=str, default="runs/ch037", help="模型保存目录")
    p.add_argument("--quick", action="store_true", help="快速模式：1-2 分钟跑完")
    return p.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.max_steps = min(args.max_steps, 6000)
        args.start_steps = min(args.start_steps, 500)
        args.eval_every = 2000
        args.eval_episodes = 3

    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    for t in targets:
        if t not in ALL_TARGETS:
            raise SystemExit(f"未知目标模式 {t}，可选：{ALL_TARGETS}")

    results = [train_target(t, args) for t in targets]

    print("=" * 74)
    print("离散 SAC 目标模式对比汇总（CartPole-v1，以实跑为准）")
    for res in results:
        hist = res["history"]
        print(f"  {res['mode']:>11s}: 最终评估 {res['final_eval']:6.1f} | "
              f"熵轨迹 {[round(e, 2) for e in hist['entropy']]} | "
              f"|Q| 末值 {hist['mean_abs_q'][-1]:.1f}")

    maybe_plot(results, args.save_dir)


if __name__ == "__main__":
    main()
