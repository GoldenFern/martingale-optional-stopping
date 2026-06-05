"""Moran 模型 —— 有限种群遗传漂变"""
import numpy as np
from scipy.linalg import solve


class MoranProcess:
    """haploid Moran 模型。

    N 个个体，每个携带 A 或 a 等位基因。
    每代：随机选一个个体繁殖 → 随机选一个个体被替换。
    X_n = 第 n 代 A 等位基因的数量，{0,...,N}，0 和 N 为吸收态。
    X_n 本身是鞅：E[X_{n+1} | X_n] = X_n。
    """

    def __init__(self, pop_size: int):
        self.pop_size = pop_size
        self._allele_count = 0

    def reset(self, initial_allele_count: int):
        self._allele_count = initial_allele_count

    @property
    def state(self) -> int:
        return self._allele_count

    def step(self) -> int:
        """执行一代，返回新状态。"""
        current_count = self._allele_count
        pop_size = self.pop_size
        reproduce_A = np.random.random() < current_count / pop_size
        kill_A = np.random.random() < current_count / pop_size
        if reproduce_A and not kill_A:
            self._allele_count = current_count + 1
        elif not reproduce_A and kill_A:
            self._allele_count = current_count - 1
        # else: 不变
        return self._allele_count

    def simulate_path(self, max_gen: int = 100000) -> np.ndarray:
        """模拟直到吸收或 max_gen 代。返回包含初始值的路径。"""
        path = [self._allele_count]
        for _ in range(max_gen):
            if self._allele_count == 0 or self._allele_count == self.pop_size:
                break
            path.append(self.step())
        return np.array(path)


def expected_tau_moran(pop_size: int) -> np.ndarray:
    """利用出生-死亡链递推关系精确求解 E[τ | X_0 = i]，返回形状 (N+1,)。"""
    # 边界: E[τ_0] = 0, E[τ_N] = 0
    # 内部: E[τ_i] = 1 + p_up*E[τ_{i+1}] + p_down*E[τ_{i-1}] + p_stay*E[τ_i]
    # 其中 p_up = p_down = i(N-i)/N^2, p_stay = 1 - 2i(N-i)/N^2
    # 化简为三对角线性方程组
    num_interior = pop_size - 1  # 内部状态数
    coefficient_matrix = np.zeros((num_interior, num_interior))
    rhs = -np.ones(num_interior)

    for row_idx in range(num_interior):
        state_value = row_idx + 1  # 实际状态
        transition_prob = state_value * (pop_size - state_value) / (pop_size * pop_size)
        if state_value > 1:
            coefficient_matrix[row_idx, row_idx - 1] = transition_prob          # i-1
        coefficient_matrix[row_idx, row_idx] = -2 * transition_prob              # i (stay term cancels out)
        if state_value < pop_size - 1:
            coefficient_matrix[row_idx, row_idx + 1] = transition_prob          # i+1

    # 处理 i=1 和 i=N-1 的边界贡献（来自吸收态 τ_0=τ_N=0）
    # i=1: -2p*τ_1 + p*τ_2 = -1 (无 τ_0 项)
    # i=N-1: p*τ_{N-2} - 2p*τ_{N-1} = -1 (无 τ_N 项)
    # 已经自动处理正确

    interior_tau = solve(coefficient_matrix, rhs)
    expected_tau = np.zeros(pop_size + 1)
    expected_tau[1:pop_size] = interior_tau
    return expected_tau
