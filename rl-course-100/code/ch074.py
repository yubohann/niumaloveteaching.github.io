"""
第074章 多智能体PPO：对手建模

单学习者（追捕者）对抗脚本化的逃跑者。本章对比两种追捕者：
  - plain：只用 [自身位置/速度, 对手相对位置/速度] 8 维观测
  - model：额外训练一个"对手下一步位移"预测头，把预测拼进策略与价值输入
训练对手固定为"逃跑"策略；评估时换三种对手（随机 / 逃跑 / 折线），
用来检验对手建模能否提升对新对手的泛化。

运行：
    python code/ch074.py           # 完整对照（CPU 约 3-6 分钟）
    python code/ch074.py --quick   # 快速跑通（约 30-60 秒）
    python code/ch074.py --use-model on

预期（以实跑为准）：两种追捕者对"逃跑"对手的捕获率接近；对训练中没见过的
"折线"对手，带对手建模的版本通常捕获率更高（预测位移提供了提前量）。
"""

import argparse
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
# 环境：追捕者 vs 脚本对手（单位方格）
# ===========================================================================
class PursuitEnv:
    """追捕者（学习者）与逃跑者（脚本策略）在 1×1 方格内。

    追捕者观测（8 维）：自身位置(2) + 自身速度(2) + 对手相对位置(2) + 对手相对速度(2)
    动作：连续 2 维力 [-1,1]；动力学与导航环境一致（阻尼 0.9 + 加速度 0.2）
    奖励：每步 -距离；捕获（距离 < 0.1）立即结束并给 +5
    对手策略：random（随机游走）/ flee（远离追捕者）/ zigzag（垂直于视线折返）
    """

    def __init__(self, opponent="flee", episode_len=50, catch_dist=0.1, seed=0):
        self.opponent = opponent
        self.episode_len = episode_len
        self.catch_dist = catch_dist
        self.obs_dim = 8
        self.act_dim = 2
        self.damping, self.accel = 0.9, 0.2
        self._rng = np.random.RandomState(seed)
        self.chaser_p = np.zeros(2, np.float32)
        self.chaser_v = np.zeros(2, np.float32)
        self.runner_p = np.zeros(2, np.float32)
        self.runner_v = np.zeros(2, np.float32)
        self.t = 0
        self.zigzag_sign = 1.0

    def reset(self):
        self.chaser_p = self._rng.uniform(0.15, 0.45, 2).astype(np.float32)
        self.runner_p = self._rng.uniform(0.55, 0.85, 2).astype(np.float32)
        self.chaser_v = np.zeros(2, np.float32)
        self.runner_v = np.zeros(2, np.float32)
        self.t = 0
        self.zigzag_sign = 1.0 if self._rng.random() < 0.5 else -1.0
        return self._chaser_obs()

    def _chaser_obs(self):
        return np.concatenate([self.chaser_p, self.chaser_v,
                               self.runner_p - self.chaser_p,
                               self.runner_v - self.chaser_v]).astype(np.float32)

    def _runner_action(self):
        rng = self._rng
        if self.opponent == "random":
            return rng.uniform(-1.0, 1.0, 2).astype(np.float32)
        if self.opponent == "flee":
            d = self.runner_p - self.chaser_p
            norm = np.linalg.norm(d) + 1e-6
            action = d / norm + rng.normal(0.0, 0.15, 2)
            return np.clip(action, -1.0, 1.0).astype(np.float32)
        # zigzag：沿垂直于"追捕者视线"的方向移动，每隔一段时间换向
        d = self.runner_p - self.chaser_p
        norm = np.linalg.norm(d) + 1e-6
        perp = np.array([-d[1], d[0]], np.float32) / norm
        if int(self.t) % 8 == 0:
            self.zigzag_sign *= -1.0
        action = self.zigzag_sign * perp + 0.2 * (d / norm) + rng.normal(0.0, 0.1, 2)
        return np.clip(action, -1.0, 1.0).astype(np.float32)

    def step(self, chaser_action):
        chaser_action = np.clip(np.asarray(chaser_action, np.float32), -1.0, 1.0)
        runner_action = self._runner_action()
        self.chaser_v = self.damping * self.chaser_v + self.accel * chaser_action
        self.runner_v = self.damping * self.runner_v + self.accel * runner_action
        self.chaser_p = np.clip(self.chaser_p + self.chaser_v, 0.0, 1.0)
        self.runner_p = np.clip(self.runner_p + self.runner_v, 0.0, 1.0)
        self.t += 1

        dist = float(np.linalg.norm(self.runner_p - self.chaser_p))
        caught = dist < self.catch_dist
        reward = -dist + (5.0 if caught else 0.0)
        done = caught or self.t >= self.episode_len
        info = {"dist": dist, "caught": bool(caught)}
        return self._chaser_obs(), reward, done, info


# ===========================================================================
# 网络：可选对手位移预测头
# ===========================================================================
class ChaserNet(nn.Module):
    """追捕者网络；use_model=True 时增加位移预测头，并把预测拼进策略/价值输入。"""

    def __init__(self, obs_dim, act_dim, hidden=64, use_model=True):
        super().__init__()
        self.use_model = use_model
        self.body = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(),
                                  nn.Linear(hidden, hidden), nn.Tanh())
        head_in = hidden + (2 if use_model else 0)
        self.actor = nn.Sequential(nn.Linear(head_in, hidden), nn.Tanh(),
                                   nn.Linear(hidden, act_dim))
        self.critic = nn.Sequential(nn.Linear(head_in, hidden), nn.Tanh(),
                                    nn.Linear(hidden, 1))
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.7))
        if use_model:
            self.pred_head = nn.Linear(hidden, 2)

    def forward(self, obs):
        h = self.body(obs)
        if self.use_model:
            pred = torch.tanh(self.pred_head(h)) * 0.4      # 预测对手下一步位移
            feat = torch.cat([h, pred], dim=-1)
        else:
            pred = None
            feat = h
        mean = self.actor(feat)
        log_std = torch.clamp(self.log_std, -4.0, 0.5).expand_as(mean)
        value = self.critic(feat).squeeze(-1)
        return mean, log_std, value, pred

    @torch.no_grad()
    def act(self, obs_t):
        mean, log_std, value, pred = self.forward(obs_t)
        dist = Normal(mean, log_std.exp())
        action = torch.clamp(dist.sample(), -1.0, 1.0)
        return (action.squeeze(0).cpu().numpy(),
                float(dist.log_prob(action).sum(-1).item()),
                float(value.item()),
                None if pred is None else pred.squeeze(0).cpu().numpy())

    @torch.no_grad()
    def value(self, obs_t) -> float:
        _, _, v, _ = self.forward(obs_t)
        return float(v.item())

    @torch.no_grad()
    def mean_action(self, obs_t):
        mean, _, _, _ = self.forward(obs_t)
        return torch.clamp(mean, -1.0, 1.0)


def opponent_displacement(obs, next_obs):
    """从相邻观测推算对手的真实位移：Δ(自身位置) + Δ(相对位置)。"""
    d_own = next_obs[:, 0:2] - obs[:, 0:2]
    d_rel = next_obs[:, 4:6] - obs[:, 4:6]
    return d_own + d_rel


def compute_gae(rewards, values, cuts, last_value, gamma, lam):
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(T)):
        next_value = last_value if t == T - 1 else values[t + 1]
        non_terminal = 1.0 - cuts[t]
        delta = rewards[t] + gamma * next_value * non_terminal - values[t]
        last_gae = delta + gamma * lam * non_terminal * last_gae
        adv[t] = last_gae
    return adv, adv + np.asarray(values, dtype=np.float32)


# ===========================================================================
# 采样与更新
# ===========================================================================
def collect(env, net, steps, obs, args, device):
    obs_buf, act_buf, logp_buf, val_buf, rew_buf, cut_buf = [], [], [], [], [], []
    next_obs_buf = []
    ep_rets, catches = [], []
    ep_ret = 0.0
    for _ in range(steps):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        action, logp, value, _ = net.act(obs_t)
        next_obs, reward, done, info = env.step(action)
        obs_buf.append(obs)
        act_buf.append(action)
        logp_buf.append(logp)
        val_buf.append(value)
        rew_buf.append(reward)
        cut_buf.append(float(done))
        next_obs_buf.append(next_obs)
        ep_ret += reward
        obs = next_obs
        if done:
            ep_rets.append(ep_ret)
            catches.append(1.0 if info["caught"] else 0.0)
            ep_ret = 0.0
            obs = env.reset()

    last_value = net.value(torch.as_tensor(obs, dtype=torch.float32,
                                           device=device).unsqueeze(0))
    adv, ret = compute_gae(rew_buf, val_buf, cut_buf, last_value, args.gamma, args.lam)
    batch = {
        "obs": torch.as_tensor(np.asarray(obs_buf), dtype=torch.float32, device=device),
        "act": torch.as_tensor(np.asarray(act_buf), dtype=torch.float32, device=device),
        "logp_old": torch.as_tensor(np.asarray(logp_buf), dtype=torch.float32, device=device),
        "adv": torch.as_tensor(adv, device=device),
        "ret": torch.as_tensor(ret, device=device),
    }
    if net.use_model:
        disp_target = opponent_displacement(np.asarray(obs_buf), np.asarray(next_obs_buf))
        batch["disp"] = torch.as_tensor(disp_target, dtype=torch.float32, device=device)
    return batch, ep_rets, catches, obs


def update(net, optim, batch, args) -> dict:
    obs, act = batch["obs"], batch["act"]
    logp_old, adv, ret = batch["logp_old"], batch["adv"], batch["ret"]
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    stats = {"policy_loss": 0.0, "value_loss": 0.0, "aux_loss": 0.0,
             "approx_kl": 0.0, "clip_frac": 0.0, "n": 0}
    N = obs.shape[0]
    for _ in range(args.epochs):
        idx = torch.randperm(N, device=obs.device)
        for start in range(0, N, args.batch_size):
            mb = idx[start:start + args.batch_size]
            mean, log_std, value, pred = net(obs[mb])
            dist = Normal(mean, log_std.exp())
            logp = dist.log_prob(act[mb]).sum(-1)
            ratio = torch.exp(logp - logp_old[mb])
            surr1 = ratio * adv[mb]
            surr2 = torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * adv[mb]
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(value, ret[mb])
            loss = policy_loss + args.vf_coef * value_loss
            if net.use_model:
                aux_loss = F.mse_loss(pred, batch["disp"][mb])
                loss = loss + args.model_coef * aux_loss
            else:
                aux_loss = torch.tensor(0.0)
            optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            optim.step()
            with torch.no_grad():
                stats["policy_loss"] += policy_loss.item()
                stats["value_loss"] += value_loss.item()
                stats["aux_loss"] += float(aux_loss.item())
                stats["approx_kl"] += (logp_old[mb] - logp).mean().item()
                stats["clip_frac"] += ((ratio - 1.0).abs() > args.clip).float().mean().item()
                stats["n"] += 1
    for k in ["policy_loss", "value_loss", "aux_loss", "approx_kl", "clip_frac"]:
        stats[k] /= max(stats["n"], 1)
    return stats


@torch.no_grad()
def evaluate_opponent(env, net, opponent, episodes, device):
    """在指定对手策略下评估：返回捕获率、平均捕获步数、平均末距。"""
    env.opponent = opponent
    catches, steps_list, dists = [], [], []
    for _ in range(episodes):
        obs = env.reset()
        done = False
        steps = 0
        info = {"dist": 1.0, "caught": False}
        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            action = net.mean_action(obs_t).squeeze(0).cpu().numpy()
            obs, _, done, info = env.step(action)
            steps += 1
        catches.append(1.0 if info["caught"] else 0.0)
        steps_list.append(steps if info["caught"] else env.episode_len)
        dists.append(info["dist"])
    return float(np.mean(catches)), float(np.mean(steps_list)), float(np.mean(dists))


def run_variant(use_model, args, device):
    set_seed(args.seed)
    env = PursuitEnv(opponent=args.opponent, seed=args.seed)
    eval_env = PursuitEnv(opponent=args.opponent, seed=args.seed + 1)
    net = ChaserNet(env.obs_dim, env.act_dim, args.hidden, use_model).to(device)
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    label = "MODEL" if use_model else "PLAIN"

    obs = env.reset()
    total_steps, update_idx = 0, 0
    all_rets, all_catches = [], []
    t0 = time.time()
    next_eval = args.eval_every

    print(f"[{label}] 参数量 {sum(p.numel() for p in net.parameters())} | "
          f"训练对手 {args.opponent}")
    while total_steps < args.max_steps:
        batch, ep_rets, catches, obs = collect(env, net, args.steps_per_update, obs,
                                               args, device)
        total_steps += args.steps_per_update
        stats = update(net, optim, batch, args)
        all_rets.extend(ep_rets)
        all_catches.extend(catches)
        update_idx += 1
        recent_r = np.mean(all_rets[-30:]) if all_rets else float("nan")
        recent_c = np.mean(all_catches[-30:]) if all_catches else float("nan")
        aux_str = f"对手预测MSE {stats['aux_loss']:.4f}" if use_model else "无对手模型"
        print(f"[{label} 更新 {update_idx:3d}] 环境步 {total_steps:6d} | "
              f"近30回合回报 {recent_r:7.2f} | 捕获率 {recent_c:.2f} | KL {stats['approx_kl']:.4f} | "
              f"裁剪 {stats['clip_frac']:.3f} | {aux_str} | 用时 {time.time() - t0:5.1f}s")
        if total_steps >= next_eval:
            ev = evaluate_opponent(eval_env, net, args.opponent, args.eval_episodes, device)
            print(f"    [评估 vs {args.opponent}] 捕获率 {ev[0]:.2f} | "
                  f"平均捕获步数 {ev[1]:.1f} | 末距 {ev[2]:.3f}")
            next_eval += args.eval_every

    # 跨对手泛化评估
    results = {}
    for opp in ["random", "flee", "zigzag"]:
        results[opp] = evaluate_opponent(eval_env, net, opp, args.eval_episodes * 3, device)
        print(f"[{label} 泛化评估 vs {opp:6s}] 捕获率 {results[opp][0]:.2f} | "
              f"平均捕获步数 {results[opp][1]:.1f} | 末距 {results[opp][2]:.3f}")
    eval_env.opponent = args.opponent
    return net, results, time.time() - t0


def parse_args():
    p = argparse.ArgumentParser(description="第074章：多智能体 PPO 的对手建模")
    p.add_argument("--use-model", type=str, default="both", choices=["both", "off", "on"])
    p.add_argument("--opponent", type=str, default="flee", choices=["random", "flee", "zigzag"],
                   help="训练时的对手策略")
    p.add_argument("--max-steps", type=int, default=60000, help="每种变体的环境步预算")
    p.add_argument("--steps-per-update", type=int, default=2000, help="每次更新前的采样步数")
    p.add_argument("--eval-every", type=int, default=15000, help="训练期评估间隔")
    p.add_argument("--epochs", type=int, default=8, help="每批数据的更新轮数")
    p.add_argument("--batch-size", type=int, default=128, help="小批量大小")
    p.add_argument("--hidden", type=int, default=64, help="隐藏层宽度")
    p.add_argument("--lr", type=float, default=3e-4, help="学习率")
    p.add_argument("--gamma", type=float, default=0.95, help="折扣因子")
    p.add_argument("--lam", type=float, default=0.95, help="GAE lambda")
    p.add_argument("--clip", type=float, default=0.2, help="PPO 裁剪范围")
    p.add_argument("--vf-coef", type=float, default=0.5, help="价值损失系数")
    p.add_argument("--model-coef", type=float, default=1.0, help="对手预测损失权重")
    p.add_argument("--eval-episodes", type=int, default=10, help="每次评估回合数")
    p.add_argument("--seed", type=int, default=0, help="随机种子")
    p.add_argument("--save-dir", type=str, default="runs/ch074", help="模型保存目录")
    p.add_argument("--quick", action="store_true", help="快速模式：15000 步/变体")
    return p.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.max_steps = min(args.max_steps, 15000)
        args.steps_per_update = 1000
        args.eval_every = 5000
        args.epochs = 5
        args.eval_episodes = 4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device} | 环境: 追捕 vs 脚本对手 | 种子 {args.seed} | "
          f"每变体预算 {args.max_steps} 环境步")

    results = {}
    if args.use_model in ("off", "both"):
        net_plain, res_plain, dt_plain = run_variant(False, args, device)
        results["PLAIN"] = (res_plain, dt_plain)
    if args.use_model in ("on", "both"):
        net_model, res_model, dt_model = run_variant(True, args, device)
        results["MODEL"] = (res_model, dt_model)

    print("=" * 76)
    for name, (res, dt) in results.items():
        line = " | ".join(f"{opp}:{res[opp][0]:.2f}" for opp in ["random", "flee", "zigzag"])
        print(f"[{name:5s}] 用时 {dt:6.1f}s | 捕获率（随机/逃跑/折线） {line}")
    if len(results) == 2 and args.opponent in results["PLAIN"][0]:
        print("提示：重点比较训练对手与未见过对手之间的捕获率差距（泛化缺口），"
              "以及对手预测 MSE 的下降速度。")

    os.makedirs(args.save_dir, exist_ok=True)
    if "PLAIN" in results:
        torch.save(net_plain.state_dict(), os.path.join(args.save_dir, "chaser_plain.pt"))
    if "MODEL" in results:
        torch.save(net_model.state_dict(), os.path.join(args.save_dir, "chaser_model.pt"))
    print(f"模型已保存到 {args.save_dir}")


if __name__ == "__main__":
    main()
