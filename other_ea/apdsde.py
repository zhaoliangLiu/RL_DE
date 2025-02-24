import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from cec2017.functions import all_functions
import config
def apdsde(bounds, dim, func_id, pop_size=50, max_eval=10000):
    # 参数初始化
    NP_init = pop_size
    NP_end = 4
    a = 1.4
    e = 0.5
    p = 0.11
    H = 6  # 历史存储器大小
    memory_size = H
    CR_memory = np.ones(memory_size) * 0.5
    F_memory = np.ones(memory_size) * 0.5
    memory_pos = 0

    # 初始化种群
    pop = np.random.uniform(bounds[0], bounds[1], (NP_init, dim))
    # 修改：使用2维输入调用适应度函数，并提取返回的标量
    fitness = np.array(config.FUNC_MAP[func_id](pop)) - func_id * 100
    eval_count = NP_init

    # 历史记录初始化
    history_best_fit = [np.min(fitness)]
    best_idx = np.argmin(fitness)
    best, best_fit = pop[best_idx].copy(), fitness[best_idx]

    # 外部存档A
    A = []
    max_A_size = int(2.6 * NP_init)

    # 参数存储器
    M_CR = np.ones(H) * 0.5
    M_F = np.ones(H) * 0.5

    while eval_count < max_eval:
        NP = len(pop)
        F_weights = []
        CR_weights = []

        # 生成控制参数
        CR = np.clip(np.random.normal(M_CR[memory_pos], 0.1, NP), 0, 1)
        F = np.clip(np.random.standard_cauchy(NP) * 0.1 + M_F[memory_pos], 0, 1)

        # 计算SP_G
        SP = 1 / (1 + np.exp(1 - (eval_count/max_eval)**2))

        # 变异操作
        pbest_size = max(int(NP * p), 1)
        pbest_indices = np.argsort(fitness)[:pbest_size]

        all_indices = np.arange(NP)
        X_Amean = np.zeros((NP, dim))

        # 计算Amean
        if len(A) > 0:
            m = min(int(e * len(A)), len(A))
            weights = (np.log(m + 0.5) - np.log(np.arange(1, m+1))) 
            weights /= weights.sum()
            A_indices = np.random.choice(len(A), m, replace=False)
            X_A = np.array([A[i] for i in A_indices])
            X_Amean = np.dot(weights, X_A)

        # 变异策略选择
        strategy_mask = np.random.rand(NP) < SP
        r1 = np.random.randint(0, NP, NP)
        if len(A) > 0:
            r2_orig = np.random.randint(0, NP + len(A), NP)
            mask = r2_orig < NP
            # For indices coming from archive A, generate separate indices
            r2_A = np.random.randint(0, len(A), NP)
            X_r2 = np.empty((NP, dim))
            X_r2[mask] = pop[r2_orig[mask]]
            X_r2[~mask] = np.array(A)[r2_A[~mask]]
        else:
            r2 = np.random.randint(0, NP, NP)
            X_r2 = pop[r2]

        # DE/current-to-pBest-w/1
        F_w_pbest = (0.7 + (eval_count/max_eval) * (a - 0.7)) * F
        pbest_idx = pbest_indices[np.random.randint(0, pbest_size, NP)]
        pbest = pop[pbest_idx]
        mutant_pbest = pop + F_w_pbest[:, None] * (pbest - pop) + F[:, None] * (pop[r1] - X_r2)

        # DE/current-to-Amean-w/1
        F_w_Amean = (0.7 + (1 - eval_count/max_eval) * (a - 0.7)) * F
        mutant_Amean = pop + F_w_Amean[:, None] * (X_Amean - pop) + F[:, None] * (pop[r1] - X_r2)
        
        # 合并变异结果
        mutant = np.where(strategy_mask[:, None], mutant_pbest, mutant_Amean)
        
        # 交叉操作
        crossover = np.random.rand(NP, dim) < CR[:, None]
        j_rand = np.random.randint(0, dim, NP)  # 每个个体随机选一个维度
        for i in range(NP):
            crossover[i, j_rand[i]] = True  # 至少继承一个分量
        trial = np.where(crossover, mutant, pop)
        
        # 边界处理
        trial = np.clip(trial, bounds[0], bounds[1])
        
        # 修改：确保适应度函数输入为2维数组
        trial_fitness = np.array(config.FUNC_MAP[func_id](trial)) - func_id * 100
        eval_count += NP
        
        # 选择操作
        improved = trial_fitness < fitness
        pop[improved] = trial[improved]
        fitness[improved] = trial_fitness[improved]
        
        # 更新存档A
        A.extend(pop[~improved].tolist())
        if len(A) > max_A_size:
            del A[:len(A)-max_A_size]
        
        # 记录历史最优
        current_best_idx = np.argmin(fitness)
        if fitness[current_best_idx] < best_fit:
            best = pop[current_best_idx].copy()
            best_fit = fitness[current_best_idx]
        history_best_fit.extend([best_fit]*NP)
        
        # 更新参数存储器
        if improved.any():
            S_CR = CR[improved]
            S_F = F[improved]
            diff_pop = pop[improved] - trial[improved]
            numerator = np.sum(diff_pop * (pop[improved] - trial[improved]), axis=1)
            denominator = np.linalg.norm(pop[improved], axis=1) * np.linalg.norm(trial[improved], axis=1)
            w_m = np.abs(numerator / (denominator + 1e-30))
            
            if len(S_CR) > 0:
                M_CR[memory_pos] = np.sum(w_m * S_CR**2) / (np.sum(w_m * S_CR) + 1e-30)
            if len(S_F) > 0: 
                M_F[memory_pos] = np.sum(w_m * S_F**2) / (np.sum(w_m * S_F) + 1e-30)
            
            memory_pos = (memory_pos + 1) % H
        
        # 非线性种群大小调整
        NP_new = round(NP_init - (NP_init - NP_end) * 
                         (eval_count/max_eval)**(1 - (eval_count/max_eval)**2))
        NP_new = max(NP_new, NP_end)
        
        if NP_new < NP:
            indices = np.argpartition(fitness, NP_new)[:NP_new]
            pop = pop[indices]
            fitness = fitness[indices]
    
    history_best_fit = history_best_fit[:max_eval]  # 保证长度一致
    return best, best_fit, np.array(history_best_fit)

# 示例用法
if __name__ == "__main__":
    func_id = 10
    best_sol, best_fit, hist = apdsde(
        bounds=[-100, 100],
        dim=10,  # 强制适应度函数输入维度为2
        func_id=func_id,
        pop_size=50,
        max_eval=10000
    )
    print(f"最优解: {best_sol}")
    print(f"最优适应度: {best_fit}")
