"""
第026章 PPO的信任域约束可视化

把"裁剪就是一种信任域"从直觉变成可看的证据。对比四种更新策略：
  - clip=0.1 / 0.2 / 0.5 ：不同裁剪半径的 PPO
  - klstop              ：clip=0.2 + 按目标 KL 早停（KL 超限就提前结束该批的
                          epochs，近似原文的自适应 KL 信任域）

每个臂记录逐次更新的诊断量：近似 KL、裁剪比例、比率均值/标准差、
超越裁剪上下界的比例、实际用掉的 epochs；最后一次更新的比率分布被
完整采样，用于画文本直方图。

产出（默认纯文本，不依赖 matplotlib）：
  1) 裁剪目标的数值表：显示 |ratio| 越界后 min 项变成常数（梯度为零）
  2) 每个臂的 KL 轨迹火花线（Unicode 方块字符，终端不支持时自动降级）
  3) 每个臂最后一次更新的比率分布文本直方图
  4) 四臂汇总表：信任域半径与稳定性/效率的权衡
加 --plot 时若安装了 matplotlib，额外保存 PNG 到 --save-dir（Agg 后端）。

运行：
    python code/ch026.py            # 4 臂各 5 万步（CPU 约 2-4 分钟）
    python code/ch026.py --quick    # 快速跑通（约 30-60 秒）
    python code/ch026.py --arms 0.05,0.2            # 只跑两档裁剪
    python code/ch026.py --target-kl 0.01           # 更严的 KL 早停阈值

预期：裁剪半径越小，平均 KL 与裁剪比例越低、比率分布越集中在 1 附近，
但首达 475 的步数通常更多；klstop 臂平均只用 5～8 个 epochs 就把 KL 压在
目标附近，是一种"按需分配计算量"的信任域。数值以实跑为准。
"""

import argparse
import os
import random
import sys
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


# ---------------------------------------------------------------------------
# 文本可视化：火花线与直方图（自动适配终端编码）
# ---------------------------------------------------------------------------
def _spark_chars():
    """返回 8 级火花线字符；终端无法编码方块字符时降级为 ASCII。"""
    blocks = "▁▂▃▄▅▆▇█"
    try:
        (sys.stdout.encoding or "utf-8")
        blocks.encode(sys.stdout.encoding or "utf-8")
        return blocks
    except (UnicodeEncodeError, LookupError):
        return ".,-~+=*#"


def ascii_sparkline(values, width=64, label="") -> str:
    """把一串数压缩成 width 个字符的火花线（按最大最小值线性映射）。"""
    if not values:
        return f"{label}(无数据)"
    v = np.asarray(values, dtype=np.float64)
    if len(v) > width:                      # 等宽重采样到 width 个点
        idx = np.linspace(0, len(v) - 1, width).astype(np.int64)
        v = v[idx]
    lo, hi = float(v.min()), float(v.max())
    chars = _spark_chars()
    span = max(hi - lo, 1e-12)
    line = "".join(chars[min(len(chars) - 1, int((x - lo) / span * (len(chars) - 1) + 0.5))]
                   for x in v)
    return f"{label}min={lo:.4f} max={hi:.4f}\n    |{line}|"


def ascii_hist(values, lo=0.3, hi=1.7, bins=28, mark=(0.8, 1.2), title=""):
    """把一批比率画成文本直方图；mark 给出要标注的裁剪边界。"""
    if values is None or len(values) == 0:
        return f"{title}(无比率样本)"
    v = np.asarray(values, dtype=np.float64)
    v = np.clip(v, lo, hi)
    counts, edges = np.histogram(v, bins=bins, range=(lo, hi))
    scale = max(1, counts.max())
    lines = [title + f"（样本 {len(v)}，{lo:.2f}~{hi:.2f}，每行一个区间）"]
    mark_bins = [int((m - lo) / (hi - lo) * bins) for m in mark if lo <= m <= hi]
    for i, c in enumerate(counts):
        bar = "#" * int(round(c / scale * 48))
        flag = "  <-- 裁剪边界" if i in mark_bins else ""
        lines.append(f"  {edges[i]:.2f} |{bar:<48}{flag}")
    return "\n".join(lines)


def print_surrogate_table(clip: float):
    """打印裁剪目标的数值表，直观展示"越界即常数"的机制。"""
    print(f"\n裁剪目标 L = min(r*A, clip(r, 1-{clip}, 1+{clip})*A) 的数值表：")
    print(f"{'ratio r':>9}{'A=+1: 未裁剪':>14}{'A=+1: 裁剪后':>14}"
          f"{'A=+1: min':>11}{'A=-1: min':>11}")
    low, high = 1.0 - clip, 1.0 + clip
    for r in np.arange(0.5, 1.51, 0.1):
        r = round(float(r), 2)
        surr_r = r
        clipped_r = min(max(r, low), high)   # 把比率夹在 [1-c, 1+c]
        a_pos = min(surr_r, clipped_r)
        a_neg = -max(surr_r, clipped_r)
        note = "  <-梯度0" if (r > 1.0 + clip) else ""
        print(f"{r:>9.2f}{surr_r:>14.2f}{clipped_r:>14.2f}{a_pos:>11.2f}{a_neg:>11.2f}{note}")
    print("解读：A>0 时 r 超过上界后 min 被钉在裁剪值上（斜率 0）；"
          "A<0 时镜像成立。这就是'越界即停'的信任域。")


# ---------------------------------------------------------------------------
# 网络：共享躯干 + 策略头 + 价值头
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
# PPO 更新：带比率统计与可选的 KL 早停
# ---------------------------------------------------------------------------
def ppo_update(net, optimizer, batch, args, clip: float, target_kl, ratio_sink):
    """做多轮小批量更新；返回诊断统计。

    参数：
        clip       : 本臂的裁剪半径
        target_kl  : 若不为 None，则在每个 epoch 后检查平均 KL，超限提前停
        ratio_sink : 若为 list，则把本次更新的比率样本追加进去（用于直方图）
    """
    obs, act = batch["obs"], batch["act"]
    logp_old, adv, ret = batch["logp_old"], batch["adv"], batch["ret"]
    N = obs.shape[0]
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    sum_kl = sum_clip = sum_entropy = sum_vloss = 0.0
    sum_r_mean = sum_r_std = sum_r_low = sum_r_high = 0.0
    n_mb = 0
    epochs_used = 0
    stopped_by_kl = 0.0

    for _ in range(args.epochs):
        epoch_kl = 0.0
        epoch_mb = 0
        idx = torch.randperm(N, device=obs.device)
        for start in range(0, N, args.batch_size):
            mb = idx[start:start + args.batch_size]
            logits, value = net(obs[mb])
            dist = Categorical(logits=logits)
            logp = dist.log_prob(act[mb])

            ratio = torch.exp(logp - logp_old[mb])
            surr1 = ratio * adv[mb]
            surr2 = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * adv[mb]
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(value, ret[mb])
            entropy = dist.entropy().mean()
            loss = policy_loss + args.vf_coef * value_loss - args.ent_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            optimizer.step()

            with torch.no_grad():
                approx_kl = (logp_old[mb] - logp).mean().item()
                r = ratio.detach()
                sum_r_mean += r.mean().item()
                sum_r_std += r.std().item() if r.numel() > 1 else 0.0
                sum_r_low += (r < 1.0 - clip).float().mean().item()
                sum_r_high += (r > 1.0 + clip).float().mean().item()
                if ratio_sink is not None and len(ratio_sink) < 6000:
                    ratio_sink.extend(r.cpu().numpy().tolist())
            sum_kl += approx_kl
            sum_clip += ((ratio - 1.0).abs() > clip).float().mean().item()
            sum_entropy += entropy.item()
            sum_vloss += value_loss.item()
            n_mb += 1
            epoch_kl += approx_kl
            epoch_mb += 1

        epochs_used += 1
        epoch_kl /= max(epoch_mb, 1)
        if target_kl is not None and epoch_kl > target_kl:
            stopped_by_kl = 1.0
            break

    d = max(n_mb, 1)
    return {
        "approx_kl": sum_kl / d,
        "clip_frac": sum_clip / d,
        "entropy": sum_entropy / d,
        "value_loss": sum_vloss / d,
        "ratio_mean": sum_r_mean / d,
        "ratio_std": sum_r_std / d,
        "ratio_low": sum_r_low / d,
        "ratio_high": sum_r_high / d,
        "epochs_used": epochs_used,
        "stopped_by_kl": stopped_by_kl,
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
# 单臂训练
# ---------------------------------------------------------------------------
def run_arm(tag: str, clip: float, target_kl, args, device):
    """训练一个信任域配置，返回结果与诊断序列。"""
    set_seed(args.seed)

    env = gym.make("CartPole-v1")
    obs, _ = env.reset(seed=args.seed)
    net = ActorCritic(env.observation_space.shape[0], env.action_space.n).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    obs_t = to_tensor(obs, device)
    all_returns = []
    kl_series, clip_series, epoch_series = [], [], []
    last_ratios = []
    total_steps = 0
    update_idx = 0
    solved_at = None
    t_start = time.time()

    while total_steps < args.max_steps and solved_at is None:
        batch, ep_returns, obs_t = collect_rollout(
            env, net, args.steps_per_update, obs_t, device, args.gamma, args.lam)
        total_steps += args.steps_per_update
        all_returns.extend(ep_returns)

        sink = []
        stats = ppo_update(net, optimizer, batch, args, clip, target_kl, sink)
        last_ratios = sink
        kl_series.append(stats["approx_kl"])
        clip_series.append(stats["clip_frac"])
        epoch_series.append(stats["epochs_used"])
        update_idx += 1

        if update_idx % args.log_every == 0 or update_idx == 1:
            recent = all_returns[-20:] if all_returns else [0.0]
            print(f"  [{tag:9s}] 更新 {update_idx:3d} | 步数 {total_steps:6d} | "
                  f"近20均 {np.mean(recent):6.1f} | KL {stats['approx_kl']:.4f} | "
                  f"裁剪比例 {stats['clip_frac']:.3f} | 比率σ {stats['ratio_std']:.3f} | "
                  f"epochs {stats['epochs_used']:2d} | 熵 {stats['entropy']:.3f}")

        if solved_at is None and len(all_returns) >= 100 and \
                np.mean(all_returns[-100:]) >= args.target_reward:
            solved_at = total_steps

    elapsed = time.time() - t_start
    eval_env = gym.make("CartPole-v1")
    eval_returns = evaluate(eval_env, net, args.eval_episodes, device)
    eval_env.close()

    os.makedirs(args.save_dir, exist_ok=True)
    torch.save(net.state_dict(), os.path.join(args.save_dir, f"{tag}.pt"))

    # 文本可视化：KL 轨迹 + 最后一次更新的比率分布
    print(f"\n  [{tag}] KL 轨迹（每次更新一个点）：")
    print("    " + ascii_sparkline(kl_series, width=min(len(kl_series), 72), label="KL "))
    print(f"  [{tag}] 最后一次更新的策略比率分布：")
    print(ascii_hist(last_ratios, lo=0.3, hi=1.7, bins=28,
                     mark=(1.0 - clip, 1.0 + clip), title=f"  比率直方图 {tag} "))

    if args.plot:
        save_plots(tag, kl_series, clip_series, last_ratios, clip, args)

    last100 = float(np.mean(all_returns[-100:])) if all_returns else 0.0
    result = {
        "tag": tag, "clip": clip, "target_kl": target_kl,
        "solved_at": solved_at, "last100": last100,
        "eval_mean": float(np.mean(eval_returns)),
        "kl_mean": float(np.mean(kl_series)), "kl_max": float(np.max(kl_series)),
        "clip_mean": float(np.mean(clip_series)),
        "epochs_mean": float(np.mean(epoch_series)),
        "time": elapsed,
        "kl_series": kl_series,
    }
    print(f"  [{tag}] 结束：步数 {total_steps} | 用时 {elapsed:.1f}s | 首达 "
          f"{('步数 ' + str(solved_at)) if solved_at else '未达成'} | "
          f"近100均 {last100:.1f} | 评估 {np.mean(eval_returns):.1f} | "
          f"平均KL {result['kl_mean']:.4f} | 最大KL {result['kl_max']:.4f} | "
          f"平均 epochs {result['epochs_mean']:.1f}\n")
    env.close()
    return result


# ---------------------------------------------------------------------------
# 可选：matplotlib 保存 PNG（惰性导入，Agg 后端，不弹窗）
# ---------------------------------------------------------------------------
def save_plots(tag, kl_series, clip_series, ratios, clip, args):
    """若安装了 matplotlib，则保存 KL 曲线、裁剪比例曲线与比率直方图。"""
    try:
        import matplotlib
        matplotlib.use("Agg")                       # 不弹窗
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"  [{tag}] 未安装 matplotlib，跳过 --plot 出图（文本图表仍然可用）")
        return
    os.makedirs(args.save_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    axes[0].plot(kl_series, color="tab:blue")
    axes[0].set_title(f"{tag} approx KL")
    axes[0].set_xlabel("update")
    axes[1].plot(clip_series, color="tab:orange")
    axes[1].set_title(f"{tag} clip fraction")
    axes[1].set_xlabel("update")
    axes[2].hist(ratios, bins=60, range=(0.3, 1.7), color="tab:green")
    axes[2].axvline(1.0 - clip, color="red", linestyle="--")
    axes[2].axvline(1.0 + clip, color="red", linestyle="--")
    axes[2].set_title(f"{tag} ratio hist")
    fig.tight_layout()
    path = os.path.join(args.save_dir, f"{tag}_trust_region.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  [{tag}] 图像已保存到 {path}")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="第026章：PPO 信任域约束可视化")
    p.add_argument("--arms", type=str, default="0.1,0.2,0.5,klstop",
                   help="要对比的臂：裁剪半径列表（逗号分隔），可用 klstop 表示 KL 早停臂")
    p.add_argument("--clip", type=float, default=0.2, help="klstop 臂使用的裁剪半径")
    p.add_argument("--target-kl", type=float, default=0.02,
                   help="klstop 臂的 KL 早停阈值（超过即停止本批 epochs）")
    p.add_argument("--max-steps", type=int, default=50000, help="每个臂的环境步数上限")
    p.add_argument("--steps-per-update", type=int, default=2048, help="每次更新前采样的步数")
    p.add_argument("--epochs", type=int, default=10, help="每批数据的最大更新轮数")
    p.add_argument("--batch-size", type=int, default=64, help="小批量大小")
    p.add_argument("--lr", type=float, default=3e-4, help="学习率")
    p.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
    p.add_argument("--lam", type=float, default=0.95, help="GAE 的 lambda")
    p.add_argument("--ent-coef", type=float, default=0.01, help="熵正则系数")
    p.add_argument("--vf-coef", type=float, default=0.5, help="价值损失系数")
    p.add_argument("--log-every", type=int, default=5, help="每隔多少次更新打印日志")
    p.add_argument("--target-reward", type=float, default=475.0, help="提前停的滑窗平均回报")
    p.add_argument("--eval-episodes", type=int, default=10, help="每个臂的评估回合数")
    p.add_argument("--plot", action="store_true", help="尝试用 matplotlib 保存 PNG（可选）")
    p.add_argument("--seed", type=int, default=0, help="随机种子")
    p.add_argument("--save-dir", type=str, default="runs/ch026", help="模型与图像保存目录")
    p.add_argument("--quick", action="store_true", help="快速模式：缩小训练规模，几十秒跑完")
    return p.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.max_steps = min(args.max_steps, 12000)
        args.steps_per_update = 768
        args.epochs = 5
        args.eval_episodes = 3

    arm_specs = []
    for token in args.arms.split(","):
        token = token.strip()
        if not token:
            continue
        if token == "klstop":
            arm_specs.append((f"klstop{args.target_kl:.2f}", args.clip, args.target_kl))
        else:
            c = float(token)
            arm_specs.append((f"clip{c:.2f}", c, None))
    if not arm_specs:
        raise SystemExit("--arms 为空，请检查参数")

    device = torch.device("cpu")
    print(f"设备: {device} | 环境: CartPole-v1 | 种子: {args.seed}")
    print(f"配置: arms={[a[0] for a in arm_specs]}, max_steps={args.max_steps}, "
          f"epochs<={args.epochs}, target_kl={args.target_kl}")

    print_surrogate_table(args.clip)

    results = []
    for tag, clip, target_kl in arm_specs:
        print("-" * 82)
        print(f"开始训练：{tag}（clip={clip}, KL早停={'无' if target_kl is None else target_kl}）")
        results.append(run_arm(tag, clip, target_kl, args, device))

    print("=" * 104)
    print(f"{'arm':<12}{'平均KL':>9}{'最大KL':>9}{'平均裁剪比例':>13}"
          f"{'比率σ':>8}{'平均epochs':>11}{'首达目标':>10}{'近100均':>9}{'评估均':>9}")
    for r in results:
        solved = str(r["solved_at"]) if r["solved_at"] else "未达成"
        print(f"{r['tag']:<12}{r['kl_mean']:>9.4f}{r['kl_max']:>9.4f}"
              f"{r['clip_mean']:>13.3f}{'-':>8}{r['epochs_mean']:>11.1f}"
              f"{solved:>10}{r['last100']:>9.1f}{r['eval_mean']:>9.1f}")
    print("=" * 104)
    print("读图提示：裁剪半径越大，比率分布越宽、平均 KL 越高；klstop 臂用平均 "
          "epochs 一列显示它实际“烧掉”了多少计算量——这才是信任域调度的直接成本。")
    print("注意：单次运行的臂间排序有随机波动，结论请以多组种子重复实验为准。")


if __name__ == "__main__":
    main()
