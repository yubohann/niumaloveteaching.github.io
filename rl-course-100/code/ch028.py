"""
第028章 PPO的稳定性指标监控

给 PPO 装一套"健康检查仪表盘"：每次更新计算 6 个诊断量并给出
健康 / 警告 / 危险三档状态，然后跑三个配置（稳健 / 激进 / 鲁莽），
验证一件事——诊断指标能否在回报崩掉之前先亮红灯。

诊断量（每次更新）：
  - approx_kl        ：新旧策略平均 KL
  - clip_frac        ：被裁剪样本比例
  - entropy          ：策略熵
  - explained_var    ：价值函数解释方差 1 - Var(ret-V)/Var(ret)
  - grad_norm        ：梯度总范数（clip_grad_norm_ 的返回值）
  - drawdown         ：近100回合滑窗相对历史最高的回撤（回报崩没崩）

阈值（见文件顶部的 THRESHOLDS）分两档：警告 / 危险。
每次更新打印状态标签与触发的原因；训练结束后输出稳定性报告：
各类状态的次数、首次警告步数、首次危险步数、首次性能下跌步数，
以及"预警提前量 = 性能下跌步数 - 首次危险（或警告）步数"。

运行：
    python code/ch028.py            # 三个配置各 4 万步（CPU 约 2-4 分钟）
    python code/ch028.py --quick    # 快速跑通（约 30-60 秒）
    python code/ch028.py --arms good,aggressive
    python code/ch028.py --arms reckless --max-steps 80000

预期：good 臂绝大多数更新是"健康"，最多出现少量警告；aggressive /
reckless 臂会先出现 KL、裁剪比例、梯度范数与解释方差的警告，
随后才出现回报回撤——预警提前量通常有几千到上万步。以实跑为准。
"""

import argparse
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
# 健康检查阈值：(警告阈值, 危险阈值)
# ---------------------------------------------------------------------------
THRESHOLDS = {
    "kl_warn": 0.02, "kl_danger": 0.05,        # 平均 KL
    "clip_warn": 0.25, "clip_danger": 0.50,    # 裁剪比例
    "ev_warn": 0.30, "ev_danger": 0.00,        # 解释方差（低于阈值报警）
    "grad_warn": 2.0, "grad_danger": 5.0,      # 梯度总范数
    "dd_warn": 50.0, "dd_danger": 150.0,       # 滑窗回撤（分）
}
STATUS_NAME = {0: "健康", 1: "警告", 2: "危险"}


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


def explained_variance(values, returns) -> float:
    """1 - Var(ret - V) / Var(ret)，衡量价值函数对回报的解释程度。"""
    v = np.asarray(values, dtype=np.float64)
    r = np.asarray(returns, dtype=np.float64)
    var_r = float(np.var(r))
    if var_r < 1e-8:
        return 1.0
    return 1.0 - float(np.var(r - v)) / var_r


def classify(metrics, update_idx, warmup=5):
    """根据诊断量给出（状态等级, 触发原因列表）。warmup 内不检查 EV。"""
    sev = 0
    reasons = []

    def flag(level, msg):
        nonlocal sev
        if level > sev:
            sev = level
        reasons.append(msg)

    if metrics["approx_kl"] > THRESHOLDS["kl_danger"]:
        flag(2, "KL")
    elif metrics["approx_kl"] > THRESHOLDS["kl_warn"]:
        flag(1, "KL")

    if metrics["clip_frac"] > THRESHOLDS["clip_danger"]:
        flag(2, "裁剪")
    elif metrics["clip_frac"] > THRESHOLDS["clip_warn"]:
        flag(1, "裁剪")

    if metrics["grad_norm"] > THRESHOLDS["grad_danger"]:
        flag(2, "梯度")
    elif metrics["grad_norm"] > THRESHOLDS["grad_warn"]:
        flag(1, "梯度")

    if update_idx > warmup:                     # 早期价值网络本来就不准
        if metrics["explained_var"] < THRESHOLDS["ev_danger"]:
            flag(2, "解释方差")
        elif metrics["explained_var"] < THRESHOLDS["ev_warn"]:
            flag(1, "解释方差")

    if metrics["drawdown"] > THRESHOLDS["dd_danger"]:
        flag(2, "回撤")
    elif metrics["drawdown"] > THRESHOLDS["dd_warn"]:
        flag(1, "回撤")

    return sev, reasons


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
# PPO 更新：额外记录梯度总范数
# ---------------------------------------------------------------------------
def ppo_update(net, optimizer, batch, args):
    """做 epochs 轮小批量更新，返回含梯度范数的诊断统计。"""
    obs, act = batch["obs"], batch["act"]
    logp_old, adv, ret = batch["logp_old"], batch["adv"], batch["ret"]
    N = obs.shape[0]
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    sum_kl = sum_clip = sum_entropy = sum_vloss = sum_gnorm = 0.0
    n_mb = 0

    for _ in range(args.epochs):
        idx = torch.randperm(N, device=obs.device)
        for start in range(0, N, args.batch_size):
            mb = idx[start:start + args.batch_size]
            logits, value = net(obs[mb])
            dist = Categorical(logits=logits)
            logp = dist.log_prob(act[mb])

            ratio = torch.exp(logp - logp_old[mb])
            surr1 = ratio * adv[mb]
            surr2 = torch.clamp(ratio, 1.0 - args.clip, 1.0 + args.clip) * adv[mb]
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(value, ret[mb])
            entropy = dist.entropy().mean()
            loss = policy_loss + args.vf_coef * value_loss - args.ent_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            # clip_grad_norm_ 返回裁剪前的梯度总范数，正好当作健康指标
            gnorm = nn.utils.clip_grad_norm_(net.parameters(), 0.5).item()
            optimizer.step()

            with torch.no_grad():
                approx_kl = (logp_old[mb] - logp).mean().item()
                clip_frac = ((ratio - 1.0).abs() > args.clip).float().mean().item()
            sum_kl += approx_kl
            sum_clip += clip_frac
            sum_entropy += entropy.item()
            sum_vloss += value_loss.item()
            sum_gnorm += gnorm
            n_mb += 1

    d = max(n_mb, 1)
    return {
        "approx_kl": sum_kl / d,
        "clip_frac": sum_clip / d,
        "entropy": sum_entropy / d,
        "value_loss": sum_vloss / d,
        "grad_norm": sum_gnorm / d,
    }


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
# 单臂：带健康监控的训练
# ---------------------------------------------------------------------------
def run_arm(tag: str, lr: float, clip: float, args, device):
    """训练一个配置；每个更新做一次健康检查，返回稳定性报告。"""
    set_seed(args.seed)

    env = gym.make("CartPole-v1")
    obs, _ = env.reset(seed=args.seed)
    net = ActorCritic(env.observation_space.shape[0], env.action_space.n).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    obs_t = to_tensor(obs, device)
    all_returns = []
    total_steps = 0
    update_idx = 0
    solved_at = None
    best_window = 0.0
    first_warn_step = None
    first_danger_step = None
    first_drop_step = None
    max_drawdown = 0.0
    max_kl = 0.0
    ev_values = []
    status_count = {0: 0, 1: 0, 2: 0}
    t_start = time.time()

    local = argparse.Namespace(**vars(args))   # 每臂独立的学习率与裁剪半径
    local.lr = lr
    local.clip = clip

    while total_steps < args.max_steps and solved_at is None:
        batch, ep_returns, obs_t = collect_rollout(
            env, net, args.steps_per_update, obs_t, device, args.gamma, args.lam)
        total_steps += args.steps_per_update
        all_returns.extend(ep_returns)

        # 更新前的价值解释方差（用当前网络对这批数据估值）
        with torch.no_grad():
            _, v_pred = net(batch["obs"])
        ev = explained_variance(v_pred.cpu().numpy(), batch["ret"].cpu().numpy())

        stats = ppo_update(net, optimizer, batch, local)
        update_idx += 1

        window = float(np.mean(all_returns[-100:])) if all_returns else 0.0
        best_window = max(best_window, window)
        drawdown = max(0.0, best_window - window)
        max_drawdown = max(max_drawdown, drawdown)
        max_kl = max(max_kl, stats["approx_kl"])
        ev_values.append(ev)

        metrics = {
            "approx_kl": stats["approx_kl"], "clip_frac": stats["clip_frac"],
            "explained_var": ev, "grad_norm": stats["grad_norm"],
            "drawdown": drawdown,
        }
        sev, reasons = classify(metrics, update_idx, warmup=args.warmup)
        status_count[sev] += 1

        if sev >= 1 and first_warn_step is None:
            first_warn_step = total_steps
        if sev >= 2 and first_danger_step is None:
            first_danger_step = total_steps
        # 性能下跌：滑窗相对历史最高回撤超过 100 分
        if first_drop_step is None and drawdown > 100.0 and len(all_returns) >= 100:
            first_drop_step = total_steps

        reason_txt = ",".join(reasons) if reasons else "-"
        print(f"  [{tag:10s}] 更新 {update_idx:3d} | 步数 {total_steps:6d} | "
              f"近100均 {window:6.1f} | KL {stats['approx_kl']:.4f} | "
              f"裁剪 {stats['clip_frac']:.3f} | EV {ev:+.2f} | "
              f"梯度 {stats['grad_norm']:5.2f} | 回撤 {drawdown:5.1f} | "
              f"状态 {STATUS_NAME[sev]}({reason_txt})")

        if solved_at is None and len(all_returns) >= 100 and \
                np.mean(all_returns[-100:]) >= args.target_reward:
            solved_at = total_steps

    elapsed = time.time() - t_start
    eval_env = gym.make("CartPole-v1")
    eval_returns = evaluate(eval_env, net, args.eval_episodes, device)
    eval_env.close()

    os.makedirs(args.save_dir, exist_ok=True)
    torch.save(net.state_dict(), os.path.join(args.save_dir, f"{tag}.pt"))

    # 预警提前量：性能首跌 - 首次危险（若无危险则用首次警告）
    alert_step = first_danger_step if first_danger_step is not None else first_warn_step
    lead = None
    if alert_step is not None and first_drop_step is not None:
        lead = first_drop_step - alert_step

    result = {
        "tag": tag, "lr": lr, "clip": clip,
        "ok": status_count[0], "warn": status_count[1], "danger": status_count[2],
        "first_warn": first_warn_step, "first_danger": first_danger_step,
        "first_drop": first_drop_step, "lead": lead,
        "max_kl": max_kl, "ev_mean": float(np.mean(ev_values)) if ev_values else 0.0,
        "max_drawdown": max_drawdown, "solved_at": solved_at,
        "last100": float(np.mean(all_returns[-100:])) if all_returns else 0.0,
        "eval_mean": float(np.mean(eval_returns)), "time": elapsed,
    }
    lead_txt = str(lead) if lead is not None else "-"
    print(f"  [{tag:10s}] 结束：健康 {result['ok']} / 警告 {result['warn']} / "
          f"危险 {result['danger']} | 最大KL {max_kl:.4f} | "
          f"最大回撤 {max_drawdown:.0f} | 预警提前量 {lead_txt} 步 | "
          f"首达 {('步数 ' + str(solved_at)) if solved_at else '未达成'} | "
          f"评估 {np.mean(eval_returns):.1f} | 用时 {elapsed:.1f}s")
    env.close()
    return result


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
ARM_PRESETS = {
    # 名称: (学习率, 裁剪半径)
    "good": (3e-4, 0.2),
    "aggressive": (1e-3, 0.5),
    "reckless": (5e-3, 0.5),
}


def parse_args():
    p = argparse.ArgumentParser(description="第028章：PPO 稳定性指标监控")
    p.add_argument("--arms", type=str, default="good,aggressive,reckless",
                   help="要运行的配置，逗号分隔（good / aggressive / reckless）")
    p.add_argument("--max-steps", type=int, default=40000, help="每个配置的环境步数上限")
    p.add_argument("--steps-per-update", type=int, default=2048, help="每次更新前采样的步数")
    p.add_argument("--epochs", type=int, default=10, help="每批数据的更新轮数")
    p.add_argument("--batch-size", type=int, default=64, help="小批量大小")
    p.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
    p.add_argument("--lam", type=float, default=0.95, help="GAE 的 lambda")
    p.add_argument("--ent-coef", type=float, default=0.01, help="熵正则系数")
    p.add_argument("--vf-coef", type=float, default=0.5, help="价值损失系数")
    p.add_argument("--warmup", type=int, default=5, help="前多少次更新不检查解释方差")
    p.add_argument("--target-reward", type=float, default=475.0, help="提前停的滑窗平均回报")
    p.add_argument("--eval-episodes", type=int, default=10, help="每个配置的评估回合数")
    p.add_argument("--seed", type=int, default=0, help="随机种子")
    p.add_argument("--save-dir", type=str, default="runs/ch028", help="模型保存目录")
    p.add_argument("--quick", action="store_true", help="快速模式：缩小训练规模，几十秒跑完")
    return p.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.max_steps = min(args.max_steps, 10000)
        args.steps_per_update = 768
        args.epochs = 5
        args.eval_episodes = 3

    arm_names = [a.strip() for a in args.arms.split(",") if a.strip()]
    arm_names = [a for a in arm_names if a in ARM_PRESETS]
    if not arm_names:
        raise SystemExit("--arms 为空或非法，请检查参数")

    device = torch.device("cpu")
    print(f"设备: {device} | 环境: CartPole-v1 | 种子: {args.seed}")
    print(f"配置: {arm_names} | 每臂 {args.max_steps} 步 | "
          f"阈值 KL>{THRESHOLDS['kl_warn']:.2f} 警告 / >{THRESHOLDS['kl_danger']:.2f} 危险，"
          f"裁剪>{THRESHOLDS['clip_warn']:.2f} / >{THRESHOLDS['clip_danger']:.2f}，"
          f"EV<{THRESHOLDS['ev_warn']:.2f} / <{THRESHOLDS['ev_danger']:.2f}")

    results = []
    for name in arm_names:
        lr, clip = ARM_PRESETS[name]
        print("-" * 100)
        print(f"开始配置：{name}（lr={lr}, clip={clip}）")
        results.append(run_arm(name, lr, clip, args, device))

    print("=" * 116)
    print(f"{'配置':<12}{'健康':>6}{'警告':>6}{'危险':>6}{'首次警告步':>11}"
          f"{'首次危险步':>11}{'性能首跌步':>11}{'预警提前量':>11}"
          f"{'最大KL':>9}{'EV均值':>8}{'最大回撤':>9}{'评估均':>8}")
    for r in results:
        fw = str(r["first_warn"]) if r["first_warn"] else "-"
        fd = str(r["first_danger"]) if r["first_danger"] else "-"
        dp = str(r["first_drop"]) if r["first_drop"] else "-"
        ld = str(r["lead"]) if r["lead"] is not None else "-"
        print(f"{r['tag']:<12}{r['ok']:>6}{r['warn']:>6}{r['danger']:>6}"
              f"{fw:>11}{fd:>11}{dp:>11}{ld:>11}"
              f"{r['max_kl']:>9.4f}{r['ev_mean']:>8.2f}{r['max_drawdown']:>9.0f}"
              f"{r['eval_mean']:>8.1f}")
    print("=" * 116)
    print("读表要点：预警提前量为正且足够大，说明诊断指标确实先于回报恶化亮灯；"
          "若某次崩溃没有任何预警，请检查阈值是否过松，或问题是否发生在监控范围之外"
          "（例如采样分布突变、环境异常）。")


if __name__ == "__main__":
    main()
