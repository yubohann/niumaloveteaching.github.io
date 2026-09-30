"""
第094章 多智能体环境：社交博弈

实验内容（4 人重复公共品博弈，纯 numpy + 表格 Q 学习）：
  - 每轮 4 个玩家同时选择贡献 c ∈ {0,1,2,3}；总贡献乘以系数 1.6 后
    四人均分：u_i = 1.6·Σc / 4 − c_i
  - 一次性博弈里"贡献 0"是占优策略，但全员贡献 3 时社会福利最高——
    这是经典的搭便车困境；重复博弈里玩家可以用"条件合作"互相回应
  - 观测（记忆 1 轮）：自己上轮贡献的独热(4)、他人上轮贡献均值(1)、
    他人上轮贡献最小值(1)、回合进度(1)
  - 算法：四个玩家独立维护 Q 表（离散状态 640 个），ε-贪婪在线 Q 学习
  - 基线：纳什（永远 0）、社会最优（永远 3）、条件合作（跟随他人均值）、随机
  - 指标：平均贡献、合作率（贡献=3 的比例）、社会福利、首尾 50 轮对比

运行：
    python code/ch094.py           # 完整训练（CPU 约 1-3 分钟）
    python code/ch094.py --quick   # 快速跑通（约 15-30 秒）

预期：学习到的策略介于纳什与条件合作之间——重复博弈下合作水平显著高于
全 0，但通常低于社会最优，且存在"终局崩塌"（最后几十轮贡献下降）。
"""

import argparse
import os
import random
import time

import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


# ---------------------------------------------------------------------------
# 环境：重复公共品博弈
# ---------------------------------------------------------------------------
class PublicGoodsGame:
    """4 人重复公共品博弈（每局 max_rounds 轮）。

    收益：u_i = multiplier · Σc / 4 − c_i（multiplier=1.6，边际回报 0.4 < 1）。
    观测：自己上轮贡献独热(4)、他人上轮贡献均值、他人上轮贡献最小值、
    回合进度 = 当前轮 / 总轮数。
    """

    n_agents = 4
    n_actions = 4                     # 贡献水平 0,1,2,3
    obs_dim = 7

    def __init__(self, multiplier: float = 1.6, max_rounds: int = 300):
        self.multiplier = multiplier
        self.max_rounds = max_rounds
        self.rng = np.random.default_rng(0)

    def reset(self, rng: np.random.Generator):
        self.rng = rng
        self.last = np.zeros(self.n_agents, dtype=np.int64)
        self.round = 0
        self.contrib_history = []
        return self._obs()

    def _obs(self) -> np.ndarray:
        obs = []
        for i in range(self.n_agents):
            others = np.delete(self.last, i)
            onehot = np.zeros(self.n_actions, dtype=np.float32)
            onehot[self.last[i]] = 1.0
            o = np.concatenate([
                onehot,
                [float(others.mean()), float(others.min())],
                [self.round / self.max_rounds],
            ])
            obs.append(o.astype(np.float32))
        return np.stack(obs)

    def step(self, actions: np.ndarray):
        contrib = np.clip(actions.astype(np.int64), 0, self.n_actions - 1)
        total = float(contrib.sum())
        payoff = self.multiplier * total / self.n_agents - contrib
        self.last = contrib.copy()
        self.round += 1
        self.contrib_history.append(contrib.copy())
        truncated = self.round >= self.max_rounds
        info = {"contrib": contrib.copy(), "total": total,
                "welfare": float(payoff.sum())}
        return self._obs(), payoff.astype(np.float32), False, truncated, info


# ---------------------------------------------------------------------------
# 表格 Q 学习：每个玩家一张 Q 表
# ---------------------------------------------------------------------------
N_OWN, N_MEAN, N_MIN, N_ROUND = 4, 4, 4, 10
N_STATES = N_OWN * N_MEAN * N_MIN * N_ROUND      # 640


def state_index(obs_row: np.ndarray) -> int:
    """把观测压成一个离散状态：贡献独热→4 档，均值/最小值→4 档，进度→10 档。"""
    own = int(np.argmax(obs_row[0:N_OWN]))
    mean_b = min(int(obs_row[N_OWN]), N_MEAN - 1)
    min_b = min(int(obs_row[N_OWN + 1]), N_MIN - 1)
    round_b = min(int(obs_row[N_OWN + 2] * N_ROUND), N_ROUND - 1)
    return ((own * N_MEAN + mean_b) * N_MIN + min_b) * N_ROUND + round_b


class TabularQLearner:
    """单个玩家的在线 Q 学习（无回放，表格直接更新）。"""

    def __init__(self, n_actions: int, args):
        self.q = np.zeros((N_STATES, n_actions), dtype=np.float32)
        self.args = args

    def act(self, state: int, epsilon: float, rng: np.random.Generator) -> int:
        if rng.uniform() < epsilon:
            return int(rng.integers(0, self.args.n_actions))
        return int(np.argmax(self.q[state]))

    def greedy(self, state: int) -> int:
        return int(np.argmax(self.q[state]))

    def update(self, s: int, a: int, r: float, s2: int, done: bool) -> None:
        target = r if done else r + self.args.gamma * float(self.q[s2].max())
        self.q[s, a] += self.args.lr * (target - self.q[s, a])


# ---------------------------------------------------------------------------
# 评估：学习策略与脚本基线
# ---------------------------------------------------------------------------
def scripted_action(name: str, env: PublicGoodsGame, i: int,
                    rng: np.random.Generator) -> int:
    if name == "nash":
        return 0
    if name == "social":
        return 3
    if name == "random":
        return int(rng.integers(0, env.n_actions))
    if name == "conditional":       # 跟随他人上轮平均贡献（四舍五入），首轮给 3
        others = np.delete(env.last, i)
        return int(np.clip(np.round(others.mean()), 0, 3))
    raise ValueError(name)


def evaluate(learners, env_cls, episodes, seed, mode="learned"):
    """返回：平均贡献、合作率、社会福利、首/尾 50 轮平均贡献。"""
    rng = np.random.default_rng(seed)
    env = env_cls()
    contribs, coop, welfare, early, late = [], [], [], [], []
    for _ in range(episodes):
        obs = env.reset(rng)
        done = False
        ep_contrib, ep_welfare = [], []
        while not done:
            if mode == "learned":
                states = [state_index(obs[i]) for i in range(env.n_agents)]
                actions = np.array([learners[i].greedy(states[i])
                                    for i in range(env.n_agents)])
            else:
                actions = np.array([scripted_action(mode, env, i, rng)
                                    for i in range(env.n_agents)])
            obs, rew, terminated, truncated, info = env.step(actions)
            ep_contrib.append(info["contrib"].mean())
            ep_welfare.append(info["welfare"])
            done = terminated or truncated
        ep_contrib = np.asarray(ep_contrib)
        contribs.append(float(ep_contrib.mean()))
        coop.append(float((np.asarray(env.contrib_history) == 3).mean()))
        welfare.append(float(np.mean(ep_welfare)))
        early.append(float(ep_contrib[:50].mean()))
        late.append(float(ep_contrib[-50:].mean()))
    return (float(np.mean(contribs)), float(np.mean(coop)),
            float(np.mean(welfare)), float(np.mean(early)), float(np.mean(late)))


def parse_args():
    p = argparse.ArgumentParser(description="第094章：社交博弈（公共品与条件合作）")
    p.add_argument("--max-rounds", type=int, default=150000, help="总博弈轮数（训练）")
    p.add_argument("--episode-rounds", type=int, default=300, help="每局多少轮")
    p.add_argument("--n-actions", type=int, default=4, help="可选贡献档位数（0..n-1）")
    p.add_argument("--lr", type=float, default=0.1, help="Q 学习步长")
    p.add_argument("--gamma", type=float, default=0.95, help="折扣因子")
    p.add_argument("--eps-start", type=float, default=1.0, help="初始探索率")
    p.add_argument("--eps-end", type=float, default=0.05, help="最终探索率")
    p.add_argument("--eps-decay", type=int, default=60000, help="探索率衰减轮数")
    p.add_argument("--eval-every", type=int, default=30000, help="评估间隔（轮）")
    p.add_argument("--eval-episodes", type=int, default=20, help="评估局数")
    p.add_argument("--seed", type=int, default=0, help="随机种子")
    p.add_argument("--save-dir", type=str, default="runs/ch094", help="模型保存目录")
    p.add_argument("--quick", action="store_true", help="快速模式：缩小训练规模")
    return p.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.max_rounds = min(args.max_rounds, 30000)
        args.eps_decay = min(args.eps_decay, 12000)
        args.eval_every = 10000
        args.eval_episodes = 10
    set_seed(args.seed)

    env_cls = PublicGoodsGame
    rng = np.random.default_rng(args.seed)
    env = env_cls(multiplier=1.6, max_rounds=args.episode_rounds)
    learners = [TabularQLearner(args.n_actions, args)
                for _ in range(env.n_agents)]

    print(f"4 人公共品博弈 | 贡献档位 0..{args.n_actions - 1} | "
          f"乘数 1.6 | 每局 {args.episode_rounds} 轮 | 种子 {args.seed}")

    # 脚本基线
    for name, label in (("nash", "纳什(全0)"), ("social", "社会最优(全3)"),
                        ("conditional", "条件合作"), ("random", "随机")):
        mc, mc2, wf, early, late = evaluate(learners, env_cls,
                                            args.eval_episodes, args.seed + 555,
                                            mode=name)
        print(f"{label:12s}：平均贡献 {mc:4.2f} | 合作率 {mc2:.2f} | "
              f"社会福利 {wf:6.2f} | 首50轮 {early:4.2f} / 尾50轮 {late:4.2f}")

    obs = env.reset(rng)
    done = False
    ep_count = 0
    t_start = time.time()
    last_eval_round = 0

    for rnd in range(1, args.max_rounds + 1):
        frac = min(1.0, rnd / max(args.eps_decay, 1))
        epsilon = args.eps_start + frac * (args.eps_end - args.eps_start)
        states = [state_index(obs[i]) for i in range(env.n_agents)]
        actions = np.array([learners[i].act(states[i], epsilon, rng)
                            for i in range(env.n_agents)])

        next_obs, rew, terminated, truncated, info = env.step(actions)
        done = terminated or truncated
        next_states = [state_index(next_obs[i]) for i in range(env.n_agents)]
        if not done:
            for i in range(env.n_agents):
                learners[i].update(states[i], int(actions[i]), float(rew[i]),
                                   next_states[i], done)

        if done:
            ep_count += 1
            obs = env.reset(rng)
        else:
            obs = next_obs

        if rnd - last_eval_round >= args.eval_every:
            last_eval_round = rnd
            mc, mc2, wf, early, late = evaluate(learners, env_cls,
                                                args.eval_episodes,
                                                args.seed + 777, mode="learned")
            print(f"[轮 {rnd:7d}] ε {epsilon:.2f} | 评估平均贡献 {mc:4.2f} | "
                  f"合作率 {mc2:.2f} | 福利 {wf:6.2f} | 首50轮 {early:4.2f} / "
                  f"尾50轮 {late:4.2f}")

    elapsed = time.time() - t_start
    mc, mc2, wf, early, late = evaluate(learners, env_cls, args.eval_episodes,
                                        args.seed + 999, mode="learned")
    print("-" * 74)
    print(f"训练结束 | 用时 {elapsed:.1f}s | 总博弈 {args.max_rounds} 轮 | "
          f"完成局数 {ep_count}")
    print(f"评估（{args.eval_episodes} 局，贪婪策略，无探索）：平均贡献 {mc:.2f} | "
          f"合作率 {mc2:.2f} | 社会福利 {wf:.2f} | 首50轮 {early:.2f} / 尾50轮 {late:.2f}")
    print(f"对照：全 0 福利 0.0 | 全 3 福利 7.2/轮 | 学习策略 {wf:.2f}/轮")

    os.makedirs(args.save_dir, exist_ok=True)
    path = os.path.join(args.save_dir, "tabular_q_public_goods.npz")
    np.savez(path, **{f"q{i}": learners[i].q for i in range(env.n_agents)},
             args=np.array([vars(args)], dtype=object))
    print(f"Q 表已保存到 {path}")


if __name__ == "__main__":
    main()
