<div align="center">

# niumaloveteaching · 机器人学习课程库

**8 门课程 · 800 讲 · 独立 HTML 页面 · 重原理、重推导、重注释**

[![课程](https://img.shields.io/badge/课程-8门-2563eb)](https://github.com/yubohann)
[![讲次](https://img.shields.io/badge/讲次-800-0ea5e9)](https://github.com/yubohann)
[![形式](https://img.shields.io/badge/形式-HTML-1e3a8a)](https://github.com/yubohann)
[![版权](https://img.shields.io/badge/版权-保留所有权利-red)](LICENSE)

</div>

---

## 这是什么

一套面向机器人学习与具身智能方向的自学课程库：**8 门课程、800 讲**。每讲是一个独立 HTML 页面，双击即可阅读，无需构建与运行环境。

每讲按统一结构展开：**学习目标 → 背景与动机 → 核心原理 → 实现思路与设计 → 关键代码与解读 → 常见坑 → 练习与思考 → 小结**。推导不跳步、数值有算例、结论有对比、参数有依据、失效有边界。

## 课程目录

| # | 课程 | 主题 | 篇幅 | 入口 |
|---|------|------|------|------|
| 01 | 深度强化学习课程 | PPO、SAC 与多智能体系统 | 5 篇 100 讲 | [进入](rl-course-100/index.html) |
| 02 | 多智能体强化学习课程 | CTDE 框架与协同算法 | 5 篇 100 讲 | [进入](ctde-marl-100/index.html) |
| 03 | 分布式强化学习课程 | Ray RLlib 训练与调优 | 5 篇 100 讲 | [进入](ray-rl-100/index.html) |
| 04 | 多机器人协同学习课程 | ROS2 通信与系统集成 | 5 篇 100 讲 | [进入](ros2-multirobot-comm-100/index.html) |
| 05 | 机械臂规划学习课程 | ROS2 MoveIt2 运动规划与操作 | 5 篇 100 讲 | [进入](moveit2-arm-100/index.html) |
| 06 | 机器人柔顺控制课程 | 阻抗与导纳控制算法 | 5 篇 100 讲 | [进入](compliance-control-100/index.html) |
| 07 | 机器人仿真学习课程 | Isaac Sim 强化学习训练 | 5 篇 100 讲 | [进入](isaac-sim-rl-100/index.html) |
| 08 | 具身智能学习课程 | VLA 与世界模型联合训练 | 5 篇 100 讲 | [进入](vla-worldmodel-100/index.html) |

## 每讲包含什么

- **完整推导**：公式给出符号说明、适用条件与推导链条，不跳步；
- **数值算例**：关键结论配可复算的数字（Python 先验算后再写入正文）；
- **对比分析**：方案取舍用对照表呈现，给出选择判据；
- **参数依据**：每个关键超参给出默认值、作用与取值理由；
- **边界与失效**：讲清方法什么时候不成立、失效模式如何识别；
- **分级练习**：每讲 4–5 道练习，含 ★★★ 挑战题与折叠提示；
- **常见坑**：每讲 4–5 条实践中最容易踩的坑与修复方式。

## 快速开始

### 方式一：打开门户

下载或克隆本仓库后，双击根目录的 `index.html`，从门户进入任意课程；每门课程的 `index.html` 是带搜索的 100 讲目录。

### 方式二：本地静态服务器

```bash
python -m http.server 8000
# 浏览器打开 http://localhost:8000
```

### 方式三：随代码动手跑（第 01 门课）

```bash
pip install -r rl-course-100/requirements.txt
python rl-course-100/code/ch001.py --quick   # 逐章脚本，纯 CPU 可跑
```

## 目录结构

```
.
├── index.html                  # 门户主页（8 门课程入口 + 分类筛选）
├── rl-course-100/              # 课程 01
│   ├── index.html              # 课程主页（100 讲目录 + 搜索）
│   ├── chapters/ch001–ch100.html
│   ├── assets/style.css
│   ├── code/ch001–ch100.py     # 完整可运行脚本（附录，选读）
│   └── requirements.txt
├── ctde-marl-100/              # 课程 02（结构同上，无 code/）
├── ray-rl-100/                 # 课程 03
├── ros2-multirobot-comm-100/   # 课程 04
├── moveit2-arm-100/            # 课程 05
├── compliance-control-100/     # 课程 06
├── isaac-sim-rl-100/           # 课程 07
├── vla-worldmodel-100/         # 课程 08
├── README.md
├── LICENSE
└── robots.txt                  # 反爬虫声明
```

## 部署到 GitHub Pages

本仓库所有页面均使用相对链接，可直接作为静态站点发布：

1. 仓库 `Settings → Pages → Source` 选择 `Deploy from a branch`，分支选 `main`、目录选 `/ (root)`；
2. 等待 1–2 分钟，访问 `https://yubohann.github.io/niumaloveteaching.github.io/` 即可（若仓库改名为 `yubohann.github.io`，则访问 `https://yubohann.github.io/`）。

## 版权与声明

- 本课程库为作者原创整理，页面内嵌有**作者署名水印**；去除、遮挡或篡改水印均属侵权。
- 允许非商业性的个人学习、研究与教学引用（请注明来源）；**禁止抓取、转载、镜像、再分发与 AI 训练使用**，详情见 [LICENSE](LICENSE)。
- 仓库附带 `robots.txt`，已声明拒绝主流 AI 爬虫抓取。

## 联系

GitHub: [@yubohann](https://github.com/yubohann)

---

<div align="center">

如果这套课程对你有帮助，欢迎 Star 支持。

</div>
