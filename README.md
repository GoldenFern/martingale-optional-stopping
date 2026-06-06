# 鞅过程可选停止定理的数值模拟

2025--2026 春季学期 · 上海交通大学随机过程（MATH4704）课程大作业

## 选题原因

我在学习鞅过程的时候发了这样一条QQ动态：

> 一个公平的掷硬币过程，正面+1反面-1。
> 从0开始，累计得分到1分所需要的期望掷硬币次数是多少？
> 如果制定这样的一个策略：累计赢到1分就立刻收手。然后玩很多局掷硬币游戏，看起来似乎就能"稳赚不赔"。除了这个策略，还有一个更经典的策略：每次下注赌资翻倍，直到赢为止。这样，赢一次就可以收回所有赌资。
> 这种看起来可以赚的策略都违背了时停定理对于有界性的要求。而违背的后果就是我之前说的那个问题的答案是正无穷。
> 头一次感觉无穷就在我身边。鞅过程真有意思。

正是这个原因让我决定系统地对可选停止定理（Optional Stopping Theorem, OST）进行数值模拟研究。本项目的论文选取了三个代表性随机过程模型（Moran 种群遗传过程、Cramér-Lundberg 保险破产过程、赌徒破产过程），通过 Monte Carlo 数值实验展示了 OST 的完整理论谱系。

## 项目架构

```
Martingale/
├── paper/                     # LaTeX 论文工程
│   ├── main.tex               # 主文件（入口）
│   ├── setup.tex              # 文档类设置与宏包加载
│   ├── nomenclature.tex       # 符号表
│   ├── references.bib         # BibLaTeX 参考文献库
│   ├── sections/              # 各章节 tex 源文件
│   │   ├── ch01_intro.tex     # 第1章：绪论
│   │   ├── ch02_moran.tex     # 第2章：Moran 模型
│   │   ├── ch03_insurance.tex # 第3章：保险破产模型
│   │   ├── ch04_ruin.tex      # 第4章：赌徒破产模型
│   │   ├── ch05_secretary.tex # [未纳入论文] 秘书问题
│   │   ├── ch06_option.tex    # [未纳入论文] 美式期权
│   │   └── ch07_conclusion.tex# 第5章：总结与展望
│   └── figures/               # 论文插图（PDF）
├── src/                       # Python 数值实验代码
│   ├── core/                  # 公共核心层
│   │   ├── processes.py       # 鞅过程抽象基类及具体实现
│   │   ├── stopping_times.py  # 停时抽象基类及常用子类
│   │   ├── simulation.py      # 通用 Monte Carlo 模拟引擎
│   │   ├── batching.py        # 批次估计
│   │   └── visualization.py   # 统一绘图风格
│   ├── ch02_moran/            # 第2章：Moran 模型实验
│   │   ├── moran_model.py     # 模型定义
│   │   ├── experiment.py      # 实验脚本
│   │   └── plot.py            # 绘图脚本
│   ├── ch03_insurance/        # 第3章：保险破产模型实验
│   │   ├── surplus_model.py   # 模型定义
│   │   ├── experiment.py      # 实验脚本
│   │   └── plot.py            # 绘图脚本
│   ├── ch04_ruin/             # 第4章：赌徒破产模型实验
│   │   ├── experiment.py      # 实验脚本
│   │   └── plot.py            # 绘图脚本
│   ├── ch05_secretary/        # [额外探索] 秘书问题实验
│   │   ├── secretary_model.py # 模型定义
│   │   ├── experiment.py      # 实验脚本
│   │   └── plot.py            # 绘图脚本
│   └── ch06_option/           # [额外探索] 美式期权实验
│       ├── lsm_pricer.py      # Longstaff-Schwartz 算法
│       ├── experiment.py      # 实验脚本
│       └── plot.py            # 绘图脚本
├── tests/                     # 单元测试
│   ├── test_processes.py
│   ├── test_stopping_times.py
│   ├── test_simulation.py
│   ├── test_batching.py
│   ├── test_moran.py
│   └── test_insurance.py
├── output/                    # 实验输出
│   ├── data/                  # 模拟数据（CSV/NPY）
│   └── figures/               # 生成的图表
└── docs/                      # 参考资料（已 gitignore）
```

### 架构设计原则

- **核心层复用**：`src/core/` 提供可复用的基础组件（过程基类、停时基类、模拟引擎），各章实验层仅需编写模型定义与参数配置，无需重复实现模拟逻辑。
- **模块分工统一**：每章实验层遵循相同的模块分工——模型定义文件负责构造该章特定的随机过程与停时，实验脚本调用核心引擎执行批量模拟并保存数据，绘图脚本从数据文件生成论文插图。
- **数据与代码分离**：实验输出（数据与图表）统一存放于 `output/` 目录，便于复现与版本管理。

## 环境依赖

- Python 3.13
- NumPy 2.1.3
- SciPy 1.15.3
- Matplotlib 3.10.0

安装：`pip install -r requirements.txt`

## 编译论文

### 依赖

- TeX 发行版（TeX Live 2024+ 推荐），需包含 XeLaTeX 和 BibLaTeX
- 上海交通大学学位论文模板 `sjtuthesis`（本项目借用了该模板的文档类与样式，但本文为课程大作业而非学位论文）

### 编译步骤

```bash
cd paper
xelatex main.tex
biber main
xelatex main.tex
xelatex main.tex
```

或者使用 `latexmk`：

```bash
cd paper
latexmk -xelatex main.tex
```

## 运行数值实验

### 环境要求

- Python 3.13+
- NumPy, SciPy, Matplotlib

### 运行方式

```bash
# 运行某一章的实验
python src/ch02_moran/experiment.py
python src/ch03_insurance/experiment.py
python src/ch04_ruin/experiment.py

# 运行测试
pytest tests/
```

## 关于第5章（秘书问题）和第6章（美式期权）

本仓库中保留了两章额外的探索性代码：

- `src/ch05_secretary/` —— 秘书问题的 Monte Carlo 实验与 Snell 包络可视化
- `src/ch06_option/` —— 美式期权的 Longstaff-Schwartz 算法实现
- `paper/sections/ch05_secretary.tex` 和 `paper/sections/ch06_option.tex` —— 对应的 LaTeX 源文件


这两章的主题（最优停时问题）与论文的核心主题（可选停止定理的成立条件与失效机制）存在较明显的偏离——OST 关注的是"给定停时下期望是否守恒"，而秘书问题和美式期权关注的是"如何主动选择停时以最大化收益"。

此外，课程大作业写七章篇幅过长，我目前也没有足够的时间对这两章的全部内容进行严格把关。**`src/ch05_secretary/` 和 `src/ch06_option/` 中的代码我尚未系统核查过**，可能存在AI幻觉。如需使用这部分代码，请仔细验证其正确性。目前这部分代码留作未来进一步探索的机会。

## 致谢

感谢授课老师张登老师和课程助教的教学与指导。

## AI使用说明

本项目由 DeepSeek V4（Claude Code 环境）、GPT 5.4（Codex 环境）以及 Gemini 3.5 Flash（普通问答）协助完成。