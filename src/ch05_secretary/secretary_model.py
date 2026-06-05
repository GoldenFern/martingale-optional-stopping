"""秘书问题 / 最优停时"""
import numpy as np


def simulate_secretary(N: int, r: int) -> bool:
    """模拟一次秘书问题。

    前 r-1 个候选人仅观察（建立 benchmark），
    之后录用第一个超过 benchmark 的候选人。
    返回是否选到了最优（全局第 1 名）。
    """
    # 随机排列的绝对排名
    ranks = np.random.permutation(N) + 1  # 1=最优, N=最差
    benchmark = np.min(ranks[:r])  # 前 r 个中最好的排名（越小越好）

    for i in range(r, N):
        if ranks[i] < benchmark:  # 比之前所有人都好
            return ranks[i] == 1
    return False  # 没选到


def simulate_secretary_batch(N: int, r: int, n_trials: int) -> float:
    """批量模拟，返回成功率。"""
    successes = 0
    for _ in range(n_trials):
        if simulate_secretary(N, r):
            successes += 1
    return successes / n_trials


def snell_envelope(N: int) -> tuple[np.ndarray, np.ndarray]:
    """计算 Snell 包络：最优值函数与即时 payoff。

    设当前为第 n 个（0-indexed），且它是历史最优（record）。
    - 即时 payoff g(n) = (n+1)/N  （这个 record 是全局最优的概率）
    - 最优值函数 V(n) = max(g(n), C(n))，其中 C(n) 为跳过后的最优期望成功率。

    利用逆向递推：
    - 在 n = N-1 时（最后一个），必须选：V(N-1) = 1
    - 递推：如果跳过第 n 个，则等下一个 record。
      C(n) = sum_{k=n+1}^{N-1} P(下一个 record 出现在 k) * V(k)
      而 P(第 k 个是下一个 record) = (n+1)/(k*(k+1))
    """
    V = np.zeros(N)
    g = np.array([(n + 1) / N for n in range(N)])
    V[N - 1] = 1.0  # 最后一个必须选

    for n in range(N - 2, -1, -1):
        # continuation value: 跳过第 n 个后的期望
        cont = 0.0
        for k in range(n + 1, N):
            # P(下一个 record 在 k) = (n+1) / (k*(k+1))
            prob_next_record_at_k = (n + 1) / (k * (k + 1))
            cont += prob_next_record_at_k * V[k]
        V[n] = max(g[n], cont)

    return V, g


def theoretical_success_prob(r: int, N: int) -> float:
    """策略 '前 r-1 个跳过' 的成功概率精确公式。

    P(success) = ((r-1)/N) * sum_{k=r}^{N} 1/(k-1)
    """
    if r <= 1:
        return 1.0 / N
    s = sum(1.0 / (k - 1) for k in range(r, N + 1))
    return ((r - 1) / N) * s


def simulate_noise_control(N: int, r: int, n_trials=2000) -> float:
    """对照组：用零均值正态噪声替代相对排名，验证同一策略的效果。"""
    successes = 0
    for _ in range(n_trials):
        noise = np.random.randn(N)
        benchmark = np.max(noise[:r])
        for i in range(r, N):
            if noise[i] > benchmark:
                if noise[i] == np.max(noise):
                    successes += 1
                break
    return successes / n_trials
