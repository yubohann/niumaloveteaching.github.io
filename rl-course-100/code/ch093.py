"""
第093章 多智能体环境：拍卖竞价

实验内容（3 个投标者的一价/二价密封拍卖，机制为脚本内联实现）：
  - 每轮 3 个投标者各自收到独立私有价值 v ~ U[0,1]，同时提交报价
  - 一价拍卖：最高价者获胜并支付自己的报价；二价拍卖：支付第二高价
  - 算法：共享策略网络的 REINFORCE（策略梯度 + 移动平均基线 + 熵正则），
    3 个投标者用同一张网络，21 档离散报价
  - 与博弈论基准对比：
      一价拍卖的对称贝叶斯纳什均衡 b*(v) = (n-1)/n · v = 2v/3
      二价拍卖的占优策略是真实报价 b*(v) = v
  - 指标：卖家收入、配置效率、投标者的期望效用（对比均衡效用 1/12）、
    学到的"报价-价值"曲线与均衡线的平均偏差

运行：
    python code/ch093.py                  # 一价拍卖训练（约 1-2 分钟 CPU）
    python code/ch093.py --mechanism second   # 二价拍卖
    python code/ch093.py --quick          # 快速跑通（约 15-30 秒）

预期：一价拍卖的学到的报价曲线逐渐贴近 2v/3；二价拍卖逐渐贴近 b=v。
玩家收入随训练收敛到 0.5 附近（收入等价定理），效率接近 1。
"""

import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# 拍卖机制（无状态转移的"环境"）
# ---------------------------------------------------------------------------
class Auction:
    """密封拍卖机制：给定收益机制（first/second），返回每个投标者的效用。"""

    n_bidders = 3

    def __init__(self, mechanism: str = "first"):
        assert mechanism in ("first", "second")
        self.mechanism = mechanism

    def sample_values(self, rng: np.random.Generator, rounds: int) -> np.ndarray:
        """每轮 3 个独立私有价值。"""
        return rng.uniform(0.0, 1.0, size=(rounds, self.n_bidders)).astype(np.float32)

    def payoff(self, values: np.ndarray, bids: np.ndarray,
               rng: np.random.Generator) -> np.ndarray:
        """计算每个投标者的效用：v−支付 若获胜，否则 0。"""
        rounds, n = bids.shape
        payoff = np.zeros_like(bids, dtype=np.float32)
        for t in range(rounds):
            top = bids[t].max()
            winners = np.where(np.isclose(bids[t], top))[0]
            winner = int(rng.choice(winners))              # 并列随机裁决
            if self.mechanism == "first":
                payment = bids[t, winner]
            else:                                           # 二价：支付第二高价
                others = np.delete(bids[t], winner)
                payment = float(others.max()) if len(others) else 0.0
            payoff[t, winner] = values[t, winner] - payment
        return payoff


# ---------------------------------------------------------------------------
# 策略网络：从私有价值到报价分布
# ---------------------------------------------------------------------------
class BidPolicy(nn.Module):
    def __init__(self, n_bids: int = 21, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, n_bids),
        )
        self.bid_grid = torch.linspace(0.0, 1.0, n_bids)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.net(values.unsqueeze(-1))


def bid_values(bid_grid: np.ndarray, actions: np.ndarray) -> np.ndarray:
    return bid_grid[actions]


# ---------------------------------------------------------------------------
# 训练：REINFORCE + 移动平均基线
# ---------------------------------------------------------------------------
def train(args, device):
    auction = Auction(args.mechanism)
    rng = np.random.default_rng(args.seed)
    net = BidPolicy(args.n_bids, args.hidden).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    bid_grid = net.bid_grid.numpy()

    baseline = 0.0
    history = []
    t_start = time.time()

    total_rounds = args.iterations * args.rounds_per_update
    for it in range(1, args.iterations + 1):
        # 1) 采样：每轮 3 个价值，全部用共享策略报价
        values = auction.sample_values(rng, args.rounds_per_update)      # (R, 3)
        v_flat = torch.as_tensor(values.reshape(-1), dtype=torch.float32, device=device)
        with torch.no_grad():
            logits = net(v_flat)
            dist = Categorical(logits=logits)
            actions = dist.sample()                                     # (3R,)
        bids = bid_values(bid_grid, actions.cpu().numpy()).reshape(
            args.rounds_per_update, auction.n_bidders)
        rewards = auction.payoff(values, bids, rng).reshape(-1)         # (3R,)

        # 2) 基线更新（指数移动平均，作为 REINFORCE 的方差缩减）
        batch_mean = float(rewards.mean())
        baseline = 0.95 * baseline + 0.05 * batch_mean

        # 3) 单遍策略梯度（分小批量）
        adv = torch.as_tensor(rewards - baseline, dtype=torch.float32, device=device)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)   # 标准化不改变期望方向
        n = v_flat.shape[0]
        perm = torch.randperm(n, device=device)
        stats = {"policy_loss": 0.0, "entropy": 0.0, "steps": 0}
        for start in range(0, n, args.batch_size):
            mb = perm[start:start + args.batch_size]
            logits = net(v_flat[mb])
            dist = Categorical(logits=logits)
            logp = dist.log_prob(actions[mb])
            entropy = dist.entropy().mean()
            loss = -(logp * adv[mb]).mean() - args.ent_coef * entropy
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            optimizer.step()
            stats["policy_loss"] += float(loss.item())
            stats["entropy"] += float(entropy.item())
            stats["steps"] += 1

        # 4) 日志：对策略的报价曲线做诊断
        with torch.no_grad():
            probe_v = torch.linspace(0.05, 0.95, 19, device=device)
            probe_logits = net(probe_v)
            probe_bids = bid_values(
                bid_grid, torch.argmax(probe_logits, dim=-1).cpu().numpy())
            probe_v_np = probe_v.cpu().numpy()
            if args.mechanism == "first":
                target = 2.0 / 3.0 * probe_v_np
            else:
                target = probe_v_np
            gap = float(np.mean(np.abs(probe_bids - target)))
        revenue, efficiency, util = quick_stats(auction, net, bid_grid,
                                                1000, rng, device)
        history.append((revenue, efficiency, util))
        if it % max(1, args.iterations // 15) == 0 or it == args.iterations:
            print(f"[迭代 {it:3d}] 回报均值 {batch_mean:.4f} | 基线 {baseline:.4f} | "
                  f"收入 {revenue:.4f} | 效率 {efficiency:.3f} | 效用 {util:.4f} | "
                  f"报价-均衡偏差 {gap:.4f} | 熵 {stats['entropy'] / max(stats['steps'], 1):.3f}")

    elapsed = time.time() - t_start
    print(f"训练结束 | 用时 {elapsed:.1f}s | 共 {total_rounds} 轮拍卖 | "
          f"机制 {args.mechanism}")
    return net, elapsed


@torch.no_grad()
def quick_stats(auction, net, bid_grid, rounds, rng, device):
    """快速统计：卖家收入、配置效率、投标者平均效用（贪婪策略）。"""
    values = auction.sample_values(rng, rounds)
    v_flat = torch.as_tensor(values.reshape(-1), dtype=torch.float32, device=device)
    actions = torch.argmax(net(v_flat), dim=-1).cpu().numpy()
    bids = bid_values(bid_grid, actions).reshape(rounds, auction.n_bidders)
    payoff = auction.payoff(values, bids, rng)
    winners = np.argmax(bids, axis=1)
    efficiency = float(np.mean(
        [values[t, winners[t]] >= values[t].max() - 1e-8 for t in range(rounds)]))
    if auction.mechanism == "first":
        revenue = float(np.mean([bids[t, winners[t]] for t in range(rounds)]))
    else:
        revenue = float(np.mean([np.sort(bids[t])[-2] for t in range(rounds)]))
    return revenue, efficiency, float(payoff.mean())


# ---------------------------------------------------------------------------
# 基准策略与均衡效用
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_strategy(auction, bid_fn, rounds, rng, device):
    """bid_fn: numpy 函数 (values) -> bids，返回收入/效率/效用。"""
    values = auction.sample_values(rng, rounds)
    bids = bid_fn(values)
    payoff = auction.payoff(values, bids, rng)
    winners = np.argmax(bids, axis=1)
    efficiency = float(np.mean(
        [values[t, winners[t]] >= values[t].max() - 1e-8 for t in range(rounds)]))
    if auction.mechanism == "first":
        revenue = float(np.mean([bids[t, winners[t]] for t in range(rounds)]))
    else:
        revenue = float(np.mean([np.sort(bids[t])[-2] for t in range(rounds)]))
    return revenue, efficiency, float(payoff.mean())


def equilibrium_bid_fn(mechanism):
    if mechanism == "first":
        return lambda v: (2.0 / 3.0) * v
    return lambda v: v


def truthful_bid_fn():
    return lambda v: v


@torch.no_grad()
def evaluate_greedy(auction, net, bid_grid, rounds, rng, device):
    def bid_fn(values):
        v_flat = torch.as_tensor(values.reshape(-1), dtype=torch.float32, device=device)
        actions = torch.argmax(net(v_flat), dim=-1).cpu().numpy()
        return bid_values(bid_grid, actions).reshape(values.shape)
    return evaluate_strategy(auction, bid_fn, rounds, rng, device)


def parse_args():
    p = argparse.ArgumentParser(description="第093章：拍卖竞价（策略梯度 vs 博弈论均衡）")
    p.add_argument("--mechanism", type=str, default="first",
                   choices=["first", "second"], help="一价 / 二价密封拍卖")
    p.add_argument("--iterations", type=int, default=60, help="策略梯度迭代次数")
    p.add_argument("--rounds-per-update", type=int, default=2048,
                   help="每次迭代采样多少轮拍卖（每轮 3 张样本）")
    p.add_argument("--batch-size", type=int, default=256, help="小批量大小")
    p.add_argument("--lr", type=float, default=1e-3, help="学习率")
    p.add_argument("--ent-coef", type=float, default=0.01, help="熵正则系数")
    p.add_argument("--hidden", type=int, default=64, help="隐藏层宽度")
    p.add_argument("--n-bids", type=int, default=21, help="报价档位数（0~1 均分）")
    p.add_argument("--eval-rounds", type=int, default=5000, help="最终评估的拍卖轮数")
    p.add_argument("--seed", type=int, default=0, help="随机种子")
    p.add_argument("--save-dir", type=str, default="runs/ch093", help="模型保存目录")
    p.add_argument("--quick", action="store_true", help="快速模式：缩小训练规模")
    return p.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.iterations = min(args.iterations, 15)
        args.rounds_per_update = 512
        args.eval_rounds = 2000
    set_seed(args.seed)
    device = torch.device("cpu")

    auction = Auction(args.mechanism)
    print(f"设备 {device} | {args.n_bids} 档报价 | 3 投标者 | 机制："
          f"{'一价' if args.mechanism == 'first' else '二价'}密封拍卖 | 种子 {args.seed}")

    # 基准先算：均衡、真实报价、随机
    rng0 = np.random.default_rng(args.seed + 1)
    burn_in = 20000
    _ = auction.sample_values(rng0, burn_in)   # 预热随机流，避免与训练重叠
    rev_eq, eff_eq, util_eq = evaluate_strategy(
        auction, equilibrium_bid_fn(args.mechanism), args.eval_rounds, rng0, device)
    rev_tr, eff_tr, util_tr = evaluate_strategy(
        auction, truthful_bid_fn(), args.eval_rounds, rng0, device)
    rev_rd, eff_rd, util_rd = evaluate_strategy(
        auction, lambda v: rng0.uniform(0.0, 1.0, size=v.shape).astype(np.float32),
        args.eval_rounds, rng0, device)
    print(f"基准（同批随机流）：均衡收入 {rev_eq:.4f} / 效率 {eff_eq:.3f} / 效用 {util_eq:.4f}")
    print(f"                    真实报价收入 {rev_tr:.4f} / 效率 {eff_tr:.3f} / 效用 {util_tr:.4f}")
    print(f"                    随机报价收入 {rev_rd:.4f} / 效率 {eff_rd:.3f} / 效用 {util_rd:.4f}")

    net, elapsed = train(args, device)
    bid_grid = net.bid_grid.numpy()

    # 最终评估（换成独立随机流）
    eval_rng = np.random.default_rng(args.seed + 12345)
    rev_l, eff_l, util_l = evaluate_greedy(auction, net, bid_grid,
                                           args.eval_rounds, eval_rng, device)

    # 报价曲线诊断（每 0.1 一个价值区间）
    with torch.no_grad():
        edges = np.linspace(0, 1, 11)
        curve = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            v = torch.tensor([(lo + hi) / 2], device=device, dtype=torch.float32)
            logits = net(v)
            a = int(torch.argmax(logits, dim=-1).item())
            curve.append((lo, hi, float(bid_grid[a])))
    print("-" * 74)
    print(f"学到的报价曲线（价值区间 → 报价）：" +
          " ".join(f"[{lo:.1f},{hi:.1f})→{b:.2f}" for lo, hi, b in curve[:5]))
    print("   " + " ".join(f"[{lo:.1f},{hi:.1f})→{b:.2f}" for lo, hi, b in curve[5:]))
    target_name = "2v/3" if args.mechanism == "first" else "v"
    print(f"评估（{args.eval_rounds} 轮）：收入 {rev_l:.4f} | 效率 {eff_l:.3f} | "
          f"平均效用 {util_l:.4f}")
    print(f"对照均衡（{target_name}）：收入 {rev_eq:.4f} | 效率 {eff_eq:.3f} | "
          f"效用 {util_eq:.4f}")
    print(f"收入差距（学习−均衡）：{rev_l - rev_eq:+.4f} | "
          f"效率差距：{eff_l - eff_eq:+.3f} | 效用差距：{util_l - util_eq:+.4f}")

    os.makedirs(args.save_dir, exist_ok=True)
    path = os.path.join(args.save_dir, f"reinforce_auction_{args.mechanism}.pt")
    torch.save({"policy": net.state_dict(), "args": vars(args)}, path)
    print(f"模型已保存到 {path}")


if __name__ == "__main__":
    main()
