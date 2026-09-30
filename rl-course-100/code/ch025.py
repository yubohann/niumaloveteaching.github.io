"""
第025章 PPO的异步优势更新

承接第 024 章的同步多环境采样，去掉"同步屏障"：
  - sync  ：主线程轮流让 N 个环境各采一段（chunk），全部到齐后更新一次
  - async ：N 个采集线程各自持有一份策略快照，采完一段立刻把
            "已算好 GAE 的数据包"（含版本号）投进有界队列；
            learner 从队列持续取数据、凑够一批就更新，不等待任何线程

教学简化说明：线程版只模拟真实的 actor-learner 数据流（快照、队列、
背压、陈旧度、丢弃），并不提供真正的多机并行；受 Python GIL 与锁竞争
影响，async 的吞吐未必高于 sync——这正是本章要暴露的工程边界。

关键机制：
  1) 参数快照：worker 开始采集时锁定当前参数版本，采完的数据天然带版本
  2) 陈旧度 staleness = 消费时的参数版本 - 数据包版本
  3) 有界队列 + max_staleness 丢弃：把"用旧数据训练"控制在一定范围内
  4) 优势在 worker 端计算（异步优势更新），learner 只做网络更新

运行：
    python code/ch025.py            # sync vs async，各 8 万步（CPU 约 2-5 分钟）
    python code/ch025.py --quick    # 快速跑通（约 30-60 秒）
    python code/ch025.py --arms async --workers 4 --chunk-steps 128
    python code/ch025.py --arms sync,async --max-staleness 0   # 完全不许用旧数据

预期：async 的平均陈旧度通常在 0.2～1.5 之间；两臂学习曲线相当，
吞吐差距取决于机器与 chunk 大小；把 --max-staleness 设成 0 会频繁丢弃，
能看到"新鲜度"与"吞吐"的直接冲突。
"""

import argparse
import os
import queue
import random
import threading
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


# ---------------------------------------------------------------------------
# GAE：单条时间序列的广义优势估计
# ---------------------------------------------------------------------------
def compute_gae(rewards, values, terminateds, last_value, gamma: float, lam: float):
    """从后向前递推 GAE，返回 (advantages, returns)。"""
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(T)):
        next_value = last_value if t == T - 1 else values[t + 1]
        non_terminal = 1.0 - terminateds[t]       # 真实终止时价值为 0
        delta = rewards[t] + gamma * next_value * non_terminal - values[t]
        last_gae = delta + gamma * lam * non_terminal * last_gae
        advantages[t] = last_gae
    returns = advantages + np.asarray(values, dtype=np.float32)
    return advantages, returns


# ---------------------------------------------------------------------------
# 单个环境的采集状态机：async 的每个 worker 与 sync 的每个环境各持一个
# ---------------------------------------------------------------------------
class SingleEnv:
    """持有一个环境 + 当前观测 + 累计回报，collect() 采一段 chunk 并算好 GAE。"""

    def __init__(self, env_id: str, seed: int):
        self.env = gym.make(env_id)
        self.obs, _ = self.env.reset(seed=seed)
        self.ep_ret = 0.0
        self.obs_dim = self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.n

    @torch.no_grad()
    def collect(self, net, chunk_steps: int, device, gamma: float, lam: float):
        """用给定网络采集 chunk_steps 步，返回数据包（numpy）与完成的回合回报。"""
        obs_buf, act_buf, logp_buf = [], [], []
        rew_buf, val_buf, term_buf = [], [], []
        finished = []

        for _ in range(chunk_steps):
            obs_t = torch.as_tensor(self.obs, dtype=torch.float32, device=device).unsqueeze(0)
            logits, value = net(obs_t)
            dist = Categorical(logits=logits)
            act = dist.sample()
            logp = dist.log_prob(act)

            obs_buf.append(self.obs)
            act_buf.append(int(act.item()))
            logp_buf.append(float(logp.item()))
            val_buf.append(float(value.item()))

            next_obs, r, terminated, truncated, _ = self.env.step(int(act.item()))
            rew_buf.append(float(r))
            term_buf.append(float(terminated))     # 只有真实终止切断 bootstrap
            self.ep_ret += r

            self.obs = next_obs
            if terminated or truncated:
                finished.append(self.ep_ret)
                self.ep_ret = 0.0
                reset_obs, _ = self.env.reset()
                self.obs = reset_obs

        # 段尾 bootstrap：用同一个快照网络估值
        obs_last = torch.as_tensor(self.obs, dtype=torch.float32, device=device).unsqueeze(0)
        _, value_last = net(obs_last)
        advantages, returns = compute_gae(
            rew_buf, val_buf, term_buf, float(value_last.item()), gamma, lam)

        item = {
            "obs": np.asarray(obs_buf, dtype=np.float32),
            "act": np.asarray(act_buf, dtype=np.int64),
            "logp_old": np.asarray(logp_buf, dtype=np.float32),
            "adv": advantages,
            "ret": returns,
            "n": chunk_steps,
        }
        return item, finished

    def close(self):
        self.env.close()


# ---------------------------------------------------------------------------
# PPO 更新
# ---------------------------------------------------------------------------
def ppo_update(net, optimizer, batch, args):
    """在拼好的 batch 上做 epochs 轮小批量更新，返回平均诊断指标。"""
    obs, act = batch["obs"], batch["act"]
    logp_old, adv, ret = batch["logp_old"], batch["adv"], batch["ret"]
    N = obs.shape[0]

    adv = (adv - adv.mean()) / (adv.std() + 1e-8)  # 优势标准化

    stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
             "approx_kl": 0.0, "clip_frac": 0.0, "n_updates": 0}

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
            nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            optimizer.step()

            with torch.no_grad():
                approx_kl = (logp_old[mb] - logp).mean().item()
                clip_frac = ((ratio - 1.0).abs() > args.clip).float().mean().item()
            stats["policy_loss"] += policy_loss.item()
            stats["value_loss"] += value_loss.item()
            stats["entropy"] += entropy.item()
            stats["approx_kl"] += approx_kl
            stats["clip_frac"] += clip_frac
            stats["n_updates"] += 1

    for k in stats:
        if k != "n_updates":
            stats[k] /= max(stats["n_updates"], 1)
    return stats


# ---------------------------------------------------------------------------
# 学习器：持有全局网络与参数版本号，供 worker 取快照
# ---------------------------------------------------------------------------
class Learner:
    def __init__(self, obs_dim, act_dim, args, device):
        self.args = args
        self.device = device
        self.net = ActorCritic(obs_dim, act_dim).to(device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=args.lr)
        self.version = 0
        self.lock = threading.Lock()   # 保护参数快照与更新不互相踩踏

    def publish(self):
        """返回 (当前版本号, 参数快照)；worker 拿到后立刻释放锁去采集。"""
        with self.lock:
            snapshot = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
            return self.version, snapshot

    def apply(self, items):
        """把若干个数据包拼成一批，做一次 PPO 更新，版本号 +1。"""
        obs = np.concatenate([it["obs"] for it in items], axis=0)
        act = np.concatenate([it["act"] for it in items], axis=0)
        logp_old = np.concatenate([it["logp_old"] for it in items], axis=0)
        adv = np.concatenate([it["adv"] for it in items], axis=0)
        ret = np.concatenate([it["ret"] for it in items], axis=0)

        d = self.device
        batch = {
            "obs": torch.as_tensor(obs, dtype=torch.float32, device=d),
            "act": torch.as_tensor(act, dtype=torch.long, device=d),
            "logp_old": torch.as_tensor(logp_old, dtype=torch.float32, device=d),
            "adv": torch.as_tensor(adv, dtype=torch.float32, device=d),
            "ret": torch.as_tensor(ret, dtype=torch.float32, device=d),
        }
        with self.lock:                    # 更新期间 worker 无法取快照，保证一致性
            stats = ppo_update(self.net, self.optimizer, batch, self.args)
            self.version += 1
        return stats


# ---------------------------------------------------------------------------
# 异步采集线程
# ---------------------------------------------------------------------------
class AsyncWorker(threading.Thread):
    def __init__(self, wid, env_id, seed, learner, q, state, counter_lock,
                 counter, stop_event, args, device):
        super().__init__(daemon=True, name=f"worker-{wid}")
        self.wid = wid
        self.learner = learner
        self.q = q
        self.state = state
        self.counter_lock = counter_lock
        self.counter = counter
        self.stop_event = stop_event
        self.args = args
        self.device = device
        self.local_net = None

    def run(self):
        while not self.stop_event.is_set():
            # 1) 取一份"当前最新参数"的快照，本次采集全程使用它
            version, snapshot = self.learner.publish()
            if self.local_net is None:
                self.local_net = ActorCritic(
                    self.state.obs_dim, self.state.act_dim).to(self.device)
            self.local_net.load_state_dict(snapshot)

            # 2) 采集一段 chunk，并在本地算 GAE（异步优势更新）
            item, finished = self.state.collect(
                self.local_net, self.args.chunk_steps, self.device,
                self.args.gamma, self.args.lam)
            item["version"] = version
            item["put_time"] = time.time()
            with self.counter_lock:
                self.counter["ep_returns"].extend(finished)

            # 3) 投递到有界队列；队列满则阻塞重试（背压）
            while not self.stop_event.is_set():
                try:
                    self.q.put(item, timeout=0.2)
                    break
                except queue.Full:
                    continue

            # 4) 上报步数；达到预算就通知全体停下
            with self.counter_lock:
                self.counter["steps"] += self.args.chunk_steps
                if self.counter["steps"] >= self.args.max_steps:
                    self.stop_event.set()
        self.state.close()


# ---------------------------------------------------------------------------
# 评估：贪婪策略
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(env, net, episodes: int, device):
    returns = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            logits, _ = net(obs_t)
            action = int(torch.argmax(logits, dim=-1).item())
            obs, r, terminated, truncated, _ = env.step(action)
            ep_ret += r
            done = terminated or truncated
        returns.append(ep_ret)
    return returns


def log_line(tag, upd, steps, returns, stats, extra=""):
    """统一格式打印一行训练日志。"""
    recent = returns[-20:] if returns else [0.0]
    print(f"  [{tag}] 更新 {upd:3d} | 步数 {steps:7d} | 回合 {len(returns):4d} | "
          f"近20均 {np.mean(recent):6.1f} | KL {stats['approx_kl']:.4f} | "
          f"熵 {stats['entropy']:.3f}{extra}")


# ---------------------------------------------------------------------------
# sync 臂：主线程轮流采集，全部到齐后更新（第 024 章数据流的对照）
# ---------------------------------------------------------------------------
def run_sync(args, device):
    set_seed(args.seed)
    states = [SingleEnv("CartPole-v1", args.seed + i) for i in range(args.workers)]
    learner = Learner(states[0].obs_dim, states[0].act_dim, args, device)

    all_returns = []
    total_steps = 0
    update_idx = 0
    solved_at = None
    t_start = time.time()

    while total_steps < args.max_steps and solved_at is None:
        items = []
        for st in states:                      # 同步屏障：等所有环境都采完
            item, finished = st.collect(
                learner.net, args.chunk_steps, device, args.gamma, args.lam)
            items.append(item)
            all_returns.extend(finished)
        total_steps += args.chunk_steps * args.workers

        stats = learner.apply(items)
        update_idx += 1
        if update_idx % args.log_every == 0 or update_idx == 1:
            log_line("sync ", update_idx, total_steps, all_returns, stats)
        if solved_at is None and len(all_returns) >= 100 and \
                np.mean(all_returns[-100:]) >= args.target_reward:
            solved_at = total_steps

    elapsed = time.time() - t_start
    for st in states:
        st.close()
    return finish_arm("sync", learner, all_returns, total_steps, elapsed,
                      solved_at, [], 0, 0.0, args, device, updates=update_idx)


# ---------------------------------------------------------------------------
# async 臂：N 个采集线程 + 有界队列 + learner 持续消费
# ---------------------------------------------------------------------------
def run_async(args, device):
    set_seed(args.seed)
    env0 = gym.make("CartPole-v1")
    obs_dim = env0.observation_space.shape[0]
    act_dim = env0.action_space.n
    env0.close()

    learner = Learner(obs_dim, act_dim, args, device)
    q = queue.Queue(maxsize=args.queue_size)
    counter_lock = threading.Lock()
    counter = {"steps": 0, "ep_returns": []}
    stop_event = threading.Event()

    workers = []
    for i in range(args.workers):
        st = SingleEnv("CartPole-v1", args.seed + i)
        w = AsyncWorker(i, "CartPole-v1", args.seed + i, learner, q,
                        st, counter_lock, counter, stop_event, args, device)
        workers.append(w)

    all_returns = []
    pending = []
    pending_samples = 0
    staleness_log = []
    queue_delay_log = []
    dropped = 0
    update_idx = 0
    solved_at = None
    t_start = time.time()

    for w in workers:
        w.start()

    while True:
        stopping = stop_event.is_set()
        if stopping and q.empty():
            # 收尾：剩余样本若够半批就再来一次更新，然后退出
            if pending_samples >= args.chunk_steps * 2:
                stats = learner.apply(pending)
                update_idx += 1
                log_line("async", update_idx, counter["steps"],
                         counter["ep_returns"], stats)
            break

        try:
            item = q.get(timeout=0.2)
        except queue.Empty:
            continue

        version_now = learner.version
        staleness = version_now - item["version"]
        if staleness > args.max_staleness:
            dropped += 1                     # 数据太旧：丢弃，保护训练质量
            continue

        queue_delay_log.append(time.time() - item["put_time"])
        staleness_log.append(staleness)
        pending.append(item)
        pending_samples += item["n"]

        if pending_samples >= args.steps_per_update:
            stats = learner.apply(pending)
            update_idx += 1
            with counter_lock:
                all_returns = list(counter["ep_returns"])
            pending = []
            pending_samples = 0
            if update_idx % args.log_every == 0 or update_idx == 1:
                mean_st = float(np.mean(staleness_log[-20:])) if staleness_log else 0.0
                log_line("async", update_idx, counter["steps"], all_returns, stats,
                         extra=f" | 陈旧度 {mean_st:.2f} | 丢弃 {dropped}")
            if solved_at is None and len(all_returns) >= 100 and \
                    np.mean(all_returns[-100:]) >= args.target_reward:
                solved_at = counter["steps"]

    stop_event.set()
    for w in workers:
        w.join(timeout=5.0)
    elapsed = time.time() - t_start
    with counter_lock:
        all_returns = list(counter["ep_returns"])
    return finish_arm("async", learner, all_returns, counter["steps"], elapsed,
                      solved_at, staleness_log, dropped,
                      float(np.mean(queue_delay_log)) if queue_delay_log else 0.0,
                      args, device, updates=update_idx)


# ---------------------------------------------------------------------------
# 收尾：评估 + 保存 + 汇总
# ---------------------------------------------------------------------------
def finish_arm(tag, learner, all_returns, total_steps, elapsed, solved_at,
               staleness_log, dropped, queue_delay, args, device, updates=0):
    eval_env = gym.make("CartPole-v1")
    eval_returns = evaluate(eval_env, learner.net, args.eval_episodes, device)
    eval_env.close()

    os.makedirs(args.save_dir, exist_ok=True)
    torch.save(learner.net.state_dict(), os.path.join(args.save_dir, f"{tag}.pt"))

    last100 = float(np.mean(all_returns[-100:])) if all_returns else 0.0
    result = {
        "tag": tag,
        "total_steps": total_steps,
        "time": elapsed,
        "steps_per_sec": total_steps / max(elapsed, 1e-6),
        "updates": updates,
        "staleness": float(np.mean(staleness_log)) if staleness_log else 0.0,
        "dropped": dropped,
        "queue_delay": queue_delay,
        "solved_at": solved_at,
        "last100": last100,
        "eval_mean": float(np.mean(eval_returns)),
    }
    print(f"  [{tag}] 结束：总步数 {total_steps} | 用时 {elapsed:.1f}s | "
          f"吞吐 {result['steps_per_sec']:.0f} 步/秒 | 首达 "
          f"{('步数 ' + str(solved_at)) if solved_at else '未达成'} | "
          f"近100均 {last100:.1f} | 评估 {np.mean(eval_returns):.1f} | "
          f"平均陈旧度 {result['staleness']:.2f} | 丢弃 {dropped}")
    return result


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="第025章：PPO 的异步优势更新（教学简化）")
    p.add_argument("--arms", type=str, default="sync,async",
                   help="要运行的臂：sync / async，逗号分隔")
    p.add_argument("--workers", type=int, default=8, help="采集环境/线程数")
    p.add_argument("--chunk-steps", type=int, default=256, help="每个环境每次采集的步数")
    p.add_argument("--steps-per-update", type=int, default=2048,
                   help="async 学习者每次更新前凑齐的样本数")
    p.add_argument("--queue-size", type=int, default=4, help="异步数据包队列容量（背压）")
    p.add_argument("--max-staleness", type=int, default=2,
                   help="允许的最大陈旧度（超过即丢弃数据包）")
    p.add_argument("--max-steps", type=int, default=80000, help="每个臂的环境步数上限")
    p.add_argument("--epochs", type=int, default=10, help="每批数据的更新轮数")
    p.add_argument("--batch-size", type=int, default=64, help="小批量大小")
    p.add_argument("--lr", type=float, default=3e-4, help="学习率")
    p.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
    p.add_argument("--lam", type=float, default=0.95, help="GAE 的 lambda")
    p.add_argument("--clip", type=float, default=0.2, help="PPO 裁剪范围")
    p.add_argument("--ent-coef", type=float, default=0.01, help="熵正则系数")
    p.add_argument("--vf-coef", type=float, default=0.5, help="价值损失系数")
    p.add_argument("--log-every", type=int, default=5, help="每隔多少次更新打印日志")
    p.add_argument("--target-reward", type=float, default=475.0, help="提前停的滑窗平均回报")
    p.add_argument("--eval-episodes", type=int, default=10, help="每个臂的评估回合数")
    p.add_argument("--seed", type=int, default=0, help="随机种子")
    p.add_argument("--save-dir", type=str, default="runs/ch025", help="模型保存目录")
    p.add_argument("--quick", action="store_true", help="快速模式：缩小训练规模，几十秒跑完")
    return p.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.max_steps = min(args.max_steps, 12000)
        args.workers = 4
        args.chunk_steps = 128
        args.steps_per_update = 1024
        args.epochs = 5
        args.eval_episodes = 3

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    arms = [a for a in arms if a in ("sync", "async")]
    if not arms:
        raise SystemExit("--arms 为空或非法，请检查参数")

    device = torch.device("cpu")
    print(f"设备: {device} | 环境: CartPole-v1 | 种子: {args.seed}")
    print(f"配置: arms={arms}, workers={args.workers}, chunk={args.chunk_steps}, "
          f"max_steps={args.max_steps}, queue={args.queue_size}, "
          f"max_staleness={args.max_staleness}")

    results = []
    for arm in arms:
        print("-" * 80)
        print(f"开始训练：{arm}")
        if arm == "sync":
            results.append(run_sync(args, device))
        else:
            results.append(run_async(args, device))

    print("=" * 98)
    print(f"{'arm':<8}{'总步数':>10}{'用时(s)':>10}{'步/秒':>9}{'更新数':>8}"
          f"{'平均陈旧度':>11}{'丢弃':>7}{'队列等待(s)':>12}{'首达目标':>10}{'评估均':>9}")
    for r in results:
        solved = str(r["solved_at"]) if r["solved_at"] else "未达成"
        print(f"{r['tag']:<8}{r['total_steps']:>10}{r['time']:>10.1f}"
              f"{r['steps_per_sec']:>9.0f}{r['updates']:>8}{r['staleness']:>11.2f}"
              f"{r['dropped']:>7}{r['queue_delay']:>12.3f}{solved:>10}"
              f"{r['eval_mean']:>9.1f}")
    print("=" * 98)
    print("提示：线程版受 GIL 与参数锁影响，async 吞吐未必超过 sync；"
          "真正的收益来自多进程/多机部署。观察 --max-staleness 与丢弃数、"
          "陈旧度的联动，那才是异步系统要权衡的核心。")


if __name__ == "__main__":
    main()
