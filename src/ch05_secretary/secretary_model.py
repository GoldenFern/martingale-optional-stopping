"""秘书问题 / 最优停时"""
import numpy as np


def simulate_secretary(num_candidates: int, observation_phase: int) -> bool:
    """模拟一次秘书问题。

    前 r-1 个候选人仅观察（建立 benchmark），
    之后录用第一个超过 benchmark 的候选人。
    返回是否选到了最优（全局第 1 名）。
    """
    # 随机排列的绝对排名
    ranks = np.random.permutation(num_candidates) + 1  # 1=最优, N=最差
    benchmark_rank = np.min(ranks[:observation_phase])  # 前 r 个中最好的排名（越小越好）

    for candidate_idx in range(observation_phase, num_candidates):
        if ranks[candidate_idx] < benchmark_rank:  # 比之前所有人都好
            return ranks[candidate_idx] == 1
    return False  # 没选到


def simulate_secretary_batch(num_candidates: int, observation_phase: int, num_trials: int) -> float:
    """批量模拟，返回成功率。"""
    successes = 0
    for _ in range(num_trials):
        if simulate_secretary(num_candidates, observation_phase):
            successes += 1
    return successes / num_trials


def snell_envelope(num_candidates: int) -> tuple[np.ndarray, np.ndarray]:
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
    value_function = np.zeros(num_candidates)
    immediate_payoff = np.array([(n + 1) / num_candidates for n in range(num_candidates)])
    value_function[num_candidates - 1] = 1.0  # 最后一个必须选

    for candidate_idx in range(num_candidates - 2, -1, -1):
        # continuation value: 跳过第 n 个后的期望
        continuation_value = 0.0
        for next_record_pos in range(candidate_idx + 1, num_candidates):
            # P(下一个 record 在 k) = (n+1) / (k*(k+1))
            record_probability = (candidate_idx + 1) / (next_record_pos * (next_record_pos + 1))
            continuation_value += record_probability * value_function[next_record_pos]
        value_function[candidate_idx] = max(immediate_payoff[candidate_idx], continuation_value)

    return value_function, immediate_payoff


def theoretical_success_prob(observation_phase: int, num_candidates: int) -> float:
    """策略 '前 r-1 个跳过' 的成功概率精确公式。

    P(success) = ((r-1)/N) * sum_{k=r}^{N} 1/(k-1)
    """
    if observation_phase <= 1:
        return 1.0 / num_candidates
    running_sum = sum(1.0 / (k - 1) for k in range(observation_phase, num_candidates + 1))
    return ((observation_phase - 1) / num_candidates) * running_sum


def simulate_noise_control(num_candidates: int, observation_phase: int, num_trials=2000) -> float:
    """对照组：用零均值正态噪声替代相对排名，验证同一策略的效果。"""
    successes = 0
    for _ in range(num_trials):
        noise_scores = np.random.randn(num_candidates)
        observation_best = np.max(noise_scores[:observation_phase])
        for candidate_idx in range(observation_phase, num_candidates):
            if noise_scores[candidate_idx] > observation_best:
                if noise_scores[candidate_idx] == np.max(noise_scores):
                    successes += 1
                break
    return successes / num_trials
