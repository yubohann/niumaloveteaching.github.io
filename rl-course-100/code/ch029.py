"""
第029章 PPO的自动超参数搜索

把"调参"变成一次可复现的搜索实验：随机搜索 + 逐次减半（successive
halving），并用第 028 章的稳定性思路做提前终止。

搜索空间（随机采样，固定 GAE 相关参数）：
    lr        : 对数均匀 [1e-4, 3e-3]
    clip      : {0.1, 0.2, 0.3}
    ent_coef  : {0.0, 0.01, 0.03}
    epochs    : {5, 10}

逐次减半：初始 N 个试验跑最小预算 rung0，按评估分数保留前 50%，
晋级者继续训练到 rung1，再保留前 50% 到 rung2。淘汰者不再消耗预算。
提前终止：若某次更新平均 KL 连续两次超过 --prune-kl，试验立即标记
"提前终止"（不参与晋级），避免把预算烧在明显不稳定的配置上。

搜索结束后，用最优配置与默认配置各重训一份到 --final-steps，
对比"搜出来的"与"默认的"最终表现——这是搜索是否值得的直接答案。

运行：
    python code/ch029.py            # 8 试验 x 3 档预算 + 两次终训（CPU 约 4-8 分钟）
    python code/ch029.py --quick    # 快速跑通（约 40-80 秒）
    python code/ch029.py --trials 12 --rungs 4000,12000,30000
    python code/ch029.py --no-prune

预期：搜索通常能找到与默认配置相当或略好的组合（CartPole 上默认值
本就接近甜蜜区），但更重要的是过程：淘汰赛把大部分预算集中在少数
有希望的配置上，搜索总成本与"直接盲训一个配置"处于同一量级。
数值以实跑为准。
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
from torch.distributions import Categorical


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    """固定随机种子，保证实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_tensor(obs, device):
    """把 gym 返回的 numpy 观测转成 (1, obs_dim) 的 float32 张量。"""
    return torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)


def sample_config(rng: np.random.RandomState) -> dict:
    """在搜索空间里随机采样一组超参数。"""
    lr = float(10 ** rng.uniform(math.log10(1e-4), math.log10(3e-3)))
    return {
        "lr": lr,
        "clip": float(rng.choice([0.1, 0.2, 0.3])),
        "ent_coef": float(rng.choice([0.0, 0.01, 0.03])),
        "epochs": int(rng.choice([5, 10])),
    }


# ---------------------------------------------------------------------------
# 网络
# ---------------------------------------------------------------------------
class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 64):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.actor = nn.Linear(hidden, act_dim)   # 输出离散动作的 logits
        self.critic = nn.Linear(hidden, 1)        # 输出状态价值 V(s)

    def forward(self, obs: torch.Tensor):
        h = self.body(obs)
        return self.actor(h), self.critic(h).squeeze(-1)

    @torch.no_grad()
    def act(self, obs: torch.Tensor):
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action).item(), value.item()

    @torch.no_grad()
    def value(self, obs: torch.Tensor) -> float:
        _, v = self.forward(obs)
        return v.item()


# ---------------------------------------------------------------------------
# GAE
# ---------------------------------------------------------------------------
def compute_gae(rewards, values, terminateds, last_value, gamma: float, lam: float):
    """从后向前递推 GAE，返回 (advantages, returns)。"""
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(T)):
        next_value = last_value if t == T - 1 else values[t + 1]
        non_terminal = 1.0 - terminateds[t]
        delta = rewards[t] + gamma * next_value * non_terminal - values[t]
        last_gae = delta + gamma * lam * non_terminal * last_gae
        advantages[t] = last_gae
    returns = advantages + np.asarray(values, dtype=np.float32)
    return advantages, returns


# ---------------------------------------------------------------------------
# 采样
# ---------------------------------------------------------------------------
def collect_rollout(env, net, steps, obs_t, device, gamma, lam):
    """与环境交互 steps 步，返回训练 batch、完成的回合回报、最新观测。"""
    obs_buf, act_buf, logp_buf = [], [], []
    rew_buf, val_buf, term_buf = [], [], []
    ep_returns = []

    ep_ret = 0.0
    obs_after = obs_t
    for _ in range(steps):
        a, logp, v = net.act(obs_t)
        next_obs, r, terminated, truncated, _ = env.step(a)

        obs_buf.append(obs_t.squeeze(0).cpu().numpy())
        act_buf.append(a)
        logp_buf.append(logp)
        rew_buf.append(r)
        val_buf.append(v)
        term_buf.append(float(terminated))

        ep_ret += r
        obs_t = to_tensor(next_obs, device)
        obs_after = obs_t
        if terminated or truncated:
            ep_returns.append(ep_ret)
            ep_ret = 0.0
            reset_obs, _ = env.reset()
            obs_t = to_tensor(reset_obs, device)

    last_value = net.value(obs_after)
    advantages, returns = compute_gae(rew_buf, val_buf, term_buf, last_value, gamma, lam)
    batch = {
        "obs": torch.as_tensor(np.asarray(obs_buf), dtype=torch.float32, device=device),
        "act": torch.as_tensor(np.asarray(act_buf), dtype=torch.long, device=device),
        "logp_old": torch.as_tensor(np.asarray(logp_buf), dtype=torch.float32, device=device),
        "adv": torch.as_tensor(advantages, dtype=torch.float32, device=device),
        "ret": torch.as_tensor(returns, dtype=torch.float32, device=device),
    }
    return batch, ep_returns, obs_t


# ---------------------------------------------------------------------------
# PPO 更新（超参数由 per-trial 的 local 命名空间提供）
# ---------------------------------------------------------------------------
def ppo_update(net, optimizer, batch, local):
    """做 local.epochs 轮小批量更新，返回平均诊断指标。"""
    obs, act = batch["obs"], batch["act"]
    logp_old, adv, ret = batch["logp_old"], batch["adv"], batch["ret"]
    N = obs.shape[0]
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    sum_kl = sum_clip = sum_entropy = 0.0
    n_mb = 0

    for _ in range(local.epochs):
        idx = torch.randperm(N, device=obs.device)
        for start in range(0, N, local.batch_size):
            mb = idx[start:start + local.batch_size]
            logits, value = net(obs[mb])
            dist = Categorical(logits=logits)
            logp = dist.log_prob(act[mb])

            ratio = torch.exp(logp - logp_old[mb])
            surr1 = ratio * adv[mb]
            surr2 = torch.clamp(ratio, 1.0 - local.clip, 1.0 + local.clip) * adv[mb]
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(value, ret[mb])
            entropy = dist.entropy().mean()
            loss = policy_loss + local.vf_coef * value_loss - local.ent_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            optimizer.step()

            with torch.no_grad():
                sum_kl += (logp_old[mb] - logp).mean().item()
                sum_clip += ((ratio - 1.0).abs() > local.clip).float().mean().item()
                sum_entropy += entropy.item()
            n_mb += 1

    d = max(n_mb, 1)
    return {"approx_kl": sum_kl / d, "clip_frac": sum_clip / d, "entropy": sum_entropy / d}


# ---------------------------------------------------------------------------
# 评估
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(env, net, episodes: int, device):
    returns = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            logits, _ = net(to_tensor(obs, device))
            action = int(torch.argmax(logits, dim=-1).item())
            obs, r, terminated, truncated, _ = env.step(action)
            ep_ret += r
            done = terminated or truncated
        returns.append(ep_ret)
    return returns


# ---------------------------------------------------------------------------
# 试验：一个配置的完整生命周期（可分段增量训练）
# ---------------------------------------------------------------------------
class Trial:
    """持有一组超参数及其训练状态，可被逐次减半晋级复用。"""

    def __init__(self, tid: int, config: dict, args, device):
        self.tid = tid
        self.config = config
        self.args = args
        self.device = device
        # 每个试验独立种子：同一搜索种子下可复现
        set_seed(args.seed + 1000 * (tid + 1))

        self.env = gym.make("CartPole-v1")
        obs, _ = self.env.reset(seed=args.seed + tid)
        self.net = ActorCritic(self.env.observation_space.shape[0],
                               self.env.action_space.n).to(device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=config["lr"])
        self.obs_t = to_tensor(obs, device)
        self.returns = []
        self.steps = 0
        self.update_idx = 0
        self.solved_at = None
        self.pruned = False          # 提前终止标记
        self.kl_high_streak = 0
        self.trained_at_rung = {}    # rung -> 该轮结束时的步数

    def local_args(self):
        """把试验超参数覆盖到参数命名空间上，供 ppo_update 使用。"""
        local = argparse.Namespace(**vars(self.args))
        local.lr = self.config["lr"]
        local.clip = self.config["clip"]
        local.ent_coef = self.config["ent_coef"]
        local.epochs = self.config["epochs"]
        return local

    def train_until(self, target_steps: int):
        """训练到 target_steps（增量式：已训练过的不重跑）。"""
        local = self.local_args()
        while self.steps < target_steps and not self.pruned and self.solved_at is None:
            batch, ep_returns, self.obs_t = collect_rollout(
                self.env, self.net, self.args.steps_per_update, self.obs_t,
                self.device, self.args.gamma, self.args.lam)
            self.steps += self.args.steps_per_update
            self.returns.extend(ep_returns)

            stats = ppo_update(self.net, self.optimizer, batch, local)
            self.update_idx += 1

            # 提前终止：KL 连续两次超限，不再浪费预算
            if stats["approx_kl"] > self.args.prune_kl:
                self.kl_high_streak += 1
            else:
                self.kl_high_streak = 0
            if self.args.prune and self.kl_high_streak >= 2:
                self.pruned = True
                print(f"    [T{self.tid:02d}] 提前终止：KL 连续超 "
                      f"{self.args.prune_kl}（步数 {self.steps}）")

            if len(self.returns) >= 100 and \
                    np.mean(self.returns[-100:]) >= self.args.target_reward:
                self.solved_at = self.steps

    def evaluate(self, episodes: int) -> float:
        eval_env = gym.make("CartPole-v1")
        rets = evaluate(eval_env, self.net, episodes, self.device)
        eval_env.close()
        return float(np.mean(rets))

    def recent_mean(self) -> float:
        return float(np.mean(self.returns[-100:])) if self.returns else 0.0

    def close(self):
        self.env.close()


# ---------------------------------------------------------------------------
# 搜索主流程
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="第029章：PPO 的自动超参数搜索（随机搜索 + 逐次减半）")
    p.add_argument("--trials", type=int, default=8, help="初始试验数量")
    p.add_argument("--rungs", type=str, default="4000,12000,30000",
                   help="逐次减半的累计步数台阶，逗号分隔（递增）")
    p.add_argument("--keep-frac", type=float, default=0.5, help="每个 rung 保留晋级者的比例")
    p.add_argument("--final-steps", type=int, default=30000,
                   help="搜索结束后对最优配置与默认配置的终训步数")
    p.add_argument("--prune", action="store_true", default=True,
                   help="启用 KL 稳定性提前终止（默认开启）")
    p.add_argument("--no-prune", dest="prune", action="store_false",
                   help="关闭提前终止")
    p.add_argument("--prune-kl", type=float, default=0.08, help="触发提前终止的平均 KL")
    p.add_argument("--steps-per-update", type=int, default=2048, help="每次更新前采样的步数")
    p.add_argument("--batch-size", type=int, default=64, help="小批量大小")
    p.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
    p.add_argument("--lam", type=float, default=0.95, help="GAE 的 lambda")
    p.add_argument("--vf-coef", type=float, default=0.5, help="价值损失系数")
    p.add_argument("--eval-episodes", type=int, default=5, help="每个 rung 的评估回合数")
    p.add_argument("--target-reward", type=float, default=475.0, help="视为达标的滑窗平均回报")
    p.add_argument("--seed", type=int, default=0, help="随机种子")
    p.add_argument("--save-dir", type=str, default="runs/ch029", help="模型保存目录")
    p.add_argument("--quick", action="store_true", help="快速模式：缩小搜索规模，几十秒跑完")
    return p.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.trials = min(args.trials, 4)
        args.rungs = "1500,4000"
        args.final_steps = 4000
        args.steps_per_update = 768
        args.eval_episodes = 3

    rungs = [int(x) for x in args.rungs.split(",") if x.strip()]
    rungs = sorted(r for r in rungs if r > 0)
    if args.trials < 2 or not rungs:
        raise SystemExit("--trials 或 --rungs 非法，请检查参数")

    device = torch.device("cpu")
    print(f"设备: {device} | 环境: CartPole-v1 | 搜索种子: {args.seed}")
    print(f"搜索配置: trials={args.trials}, rungs={rungs}, keep_frac={args.keep_frac}, "
          f"prune={'on' if args.prune else 'off'}（KL>{args.prune_kl} 连续两次）")

    rng = np.random.RandomState(args.seed)
    configs = [sample_config(rng) for _ in range(args.trials)]
    print("\n初始试验配置：")
    for i, c in enumerate(configs):
        print(f"  T{i:02d}: lr={c['lr']:.2e} | clip={c['clip']:.1f} | "
              f"ent={c['ent_coef']:.2f} | epochs={c['epochs']}")

    t_start = time.time()
    trials = [Trial(i, c, args, device) for i, c in enumerate(configs)]
    alive = list(trials)
    total_search_steps = 0

    for rung in rungs:
        print("\n" + "-" * 96)
        print(f"Rung: 训练到累积 {rung} 步（存活 {len(alive)} 个试验）")
        for t in alive:
            before = t.steps
            t.train_until(rung)
            total_search_steps += t.steps - before

        # 评估并排名（提前终止的试验直接排在最后）
        scored = []
        for t in alive:
            score = t.evaluate(args.eval_episodes)
            scored.append((t, score))
        scored.sort(key=lambda x: (not x[0].pruned, x[1]), reverse=True)

        print(f"Rung 结果（按评估均分排序）：")
        print(f"  {'试验':<6}{'lr':>9}{'clip':>6}{'ent':>6}{'ep':>4}"
              f"{'步数':>8}{'近100均':>9}{'评估均':>8}  状态")
        for t, score in scored:
            if t.pruned:
                state = "提前终止"
            elif t.solved_at is not None:
                state = f"已达标({t.solved_at})"
            else:
                state = ""
            print(f"  T{t.tid:<5}{t.config['lr']:>9.2e}{t.config['clip']:>6.1f}"
                  f"{t.config['ent_coef']:>6.2f}{t.config['epochs']:>4}"
                  f"{t.steps:>8}{t.recent_mean():>9.1f}{score:>8.1f}  {state}")

        # 逐次减半：保留前 keep_frac（至少 1 个）
        if rung != rungs[-1]:
            n_keep = max(1, int(len(scored) * args.keep_frac))
            alive = [t for t, _ in scored[:n_keep]]
            eliminated = [t.tid for t, _ in scored[n_keep:]]
            print(f"  晋级：{['T%02d' % t.tid for t in alive]}"
                  f" | 淘汰：{['T%02d' % i for i in eliminated]}")
            for t, _ in scored[n_keep:]:
                t.close()

    # 选出搜索最优（未提前终止里分数最高的）
    final_scores = {}
    for t in alive:
        final_scores[t.tid] = t.evaluate(args.eval_episodes)
    best = max(alive, key=lambda t: final_scores[t.tid])
    best_cfg = dict(best.config)
    print("\n" + "=" * 96)
    print(f"搜索最优配置：T{best.tid} -> lr={best_cfg['lr']:.2e}, clip={best_cfg['clip']:.1f}, "
          f"ent={best_cfg['ent_coef']:.2f}, epochs={best_cfg['epochs']} "
          f"（终评 {final_scores[best.tid]:.1f}）")
    print(f"搜索阶段总环境步数：{total_search_steps}")
    for t in alive:
        t.close()

    # 终训：搜出来的最优配置 vs 默认配置（各自从零训练）
    print("-" * 96)
    print(f"终训对比（各 {args.final_steps} 步，独立种子）：")
    default_cfg = {"lr": 3e-4, "clip": 0.2, "ent_coef": 0.01, "epochs": 10}
    finals = []
    for name, cfg in (("搜索最优", best_cfg), ("默认配置", default_cfg)):
        t = Trial(tid=90 if name == "搜索最优" else 91, config=cfg, args=args, device=device)
        t.train_until(args.final_steps)
        score = t.evaluate(args.eval_episodes)
        finals.append((name, cfg, t, score))
        t.close()

    print(f"  {'配置':<8}{'lr':>10}{'clip':>6}{'ent':>6}{'ep':>4}"
          f"{'步数':>8}{'近100均':>9}{'评估均':>8}{'首达':>8}")
    for name, cfg, t, score in finals:
        solved = str(t.solved_at) if t.solved_at else "未达成"
        print(f"  {name:<8}{cfg['lr']:>10.2e}{cfg['clip']:>6.1f}{cfg['ent_coef']:>6.2f}"
              f"{cfg['epochs']:>4}{t.steps:>8}{t.recent_mean():>9.1f}"
              f"{score:>8.1f}{solved:>8}")

    os.makedirs(args.save_dir, exist_ok=True)
    torch.save(best.net.state_dict(), os.path.join(args.save_dir, "best_trial.pt"))
    elapsed = time.time() - t_start
    print("=" * 96)
    print(f"总用时 {elapsed:.1f}s。搜索的价值不在'一定赢过默认'：CartPole 的默认超参"
          f"已接近甜蜜区，逐次减半证明的是搜索成本可以控制在一次盲训的量级。"
          f"换到更敏感的任务（或更大搜索空间）时，同样的流程才会拉开差距。")


if __name__ == "__main__":
    main()
