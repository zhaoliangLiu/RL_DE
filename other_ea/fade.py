import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
import config

def evaluate_individual(func_id, ind):
    """辅助函数：评估单个个体"""
    return float(config.FUNC_MAP[func_id](ind.reshape(1, -1)) - func_id * 100) 

def evaluate_population(func_id, pop):
    """向量化评估种群"""
    return np.array([float(config.FUNC_MAP[func_id](p.reshape(1, -1))) for p in pop]) - func_id * 100

def safe_reshape(pop, fitness, ss, dim):
    """保证pop的行数为ss的倍数，不足时进行填充"""
    if len(pop) % ss != 0:
        padding_size = (ss - (len(pop) % ss)) % ss
        best_idx = np.argmin(fitness)
        padding = np.tile(pop[best_idx], (padding_size, 1))
        padding_fitness = np.tile(fitness[best_idx], padding_size)
        pop = np.vstack([pop, padding])
        fitness = np.concatenate([fitness, padding_fitness])
    swarms = pop.reshape(-1, ss, dim)
    fitness_swarms = fitness.reshape(-1, ss)
    return pop, fitness, swarms, fitness_swarms

def fade_optimizer(func_id, bounds, dim, pop_size=50, max_eval=10000):
    # 参数初始化
    ss = 3  # 每个swarm的大小
    ns = pop_size // ss  # 初始swarm数量
    ps_ini = ns * ss
    ps_min = ps_ini // 3
    ps_max = ps_ini
    max_g_imp = 2
    max_g_stag = 2
    p_c = 0.9
    p_m = 0.1

    # 初始化种群
    pop = np.random.uniform(bounds[0], bounds[1], (ps_ini, dim))
    fitness = evaluate_population(func_id, pop)
    evaluations = ps_ini
    best_idx = np.argmin(fitness)
    best, best_fitness = pop[best_idx].copy(), fitness[best_idx]
    # 初始化历史记录
    current_best_fitness = best_fitness
    history = np.full(max_eval, np.inf)
    for i in range(len(pop)):
        history[i] = current_best_fitness

    # 初始分组（使用安全重塑）
    pop, fitness, swarms, fitness_swarms = safe_reshape(pop, fitness, ss, dim)
    # 精英存档
    A_E = swarms[:, 0, :].copy()

    SG_imp, SG_stag, prev_best = 0, 0, best_fitness

    while evaluations < max_eval:
        # 安全重塑种群，确保行数为ss的倍数
        pop, fitness, swarms, fitness_swarms = safe_reshape(pop, fitness, ss, dim)
        
        # 批量生成所有swarm的变异向量
        trials = np.empty_like(pop)
        
        for s_idx in range(swarms.shape[0]):
            swarm = swarms[s_idx]
            swarm_fit = fitness_swarms[s_idx]
            
            # 对每个swarm进行排序，获取精英个体和劣质个体
            sorted_indices = np.argsort(swarm_fit)
            elite_idx = sorted_indices[0]
            inferior_idx = sorted_indices[-1]
            
            # 进行变异和交叉
            for local_idx in range(ss):
                current = swarm[local_idx]
                
                if local_idx == elite_idx:
                    r1, r2 = np.random.choice(len(A_E), 2, replace=False)
                    mutant = current + 0.5 * (best - current + A_E[r1] - A_E[r2])
                elif local_idx == inferior_idx:
                    candidates = np.random.choice(len(pop), 5, replace=False)
                    mutant = pop[candidates[0]] + 1.0 * (pop[candidates[1]] - pop[candidates[2]] + pop[candidates[3]] - pop[candidates[4]])
                else:
                    local_best = swarm[elite_idx]
                    r1, r2 = np.random.choice(len(pop), 2, replace=False)
                    mutant = current + 0.8 * (local_best - current + pop[r1] - pop[r2])
                
                cross_points = np.random.rand(dim) <= [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1][:dim]
                cross_points[np.random.randint(dim)] = True
                trials[s_idx * ss + local_idx] = np.where(cross_points, mutant, current)

        trials = np.clip(trials, bounds[0], bounds[1])
        trial_fitness = evaluate_population(func_id, trials)
        
        # 更新历史记录
        for i, fit in enumerate(trial_fitness):
            if evaluations + i >= max_eval:
                break
            current_best_fitness = min(current_best_fitness, fit)
            history[evaluations + i] = current_best_fitness
        
        improvements = trial_fitness < fitness
        pop[improvements] = trials[improvements]
        fitness[improvements] = trial_fitness[improvements]
        
        min_idx = np.argmin(trial_fitness)
        if trial_fitness[min_idx] < best_fitness:
            best = trials[min_idx].copy()
            best_fitness = trial_fitness[min_idx]
        
        evaluations += len(trials)
        
        if best_fitness < prev_best:
            SG_imp += 1
            SG_stag = 0
        else:
            SG_stag += 1
            SG_imp = 0
        prev_best = best_fitness
        
        if SG_imp >= max_g_imp and len(pop) > ps_min:
            remove_num = min(6, len(pop)-ps_min)
            sorted_idx = np.argsort(fitness)[::-1][:remove_num]
            pop = np.delete(pop, sorted_idx, axis=0)
            fitness = np.delete(fitness, sorted_idx)
            SG_imp = 0
            pop, fitness, swarms, fitness_swarms = safe_reshape(pop, fitness, ss, dim)
            A_E = swarms[:, 0, :].copy()
        
        if SG_stag >= max_g_stag and len(pop) < ps_max:
            sigma = np.std(A_E, axis=0)
            new_ind = []
            for _ in range(3):
                p1, p2 = A_E[np.random.choice(len(A_E), 2, replace=False)]
                cross_mask = np.random.rand(dim) < p_c
                child = np.where(cross_mask, p1, p2)
                mutate_mask = np.random.rand(dim) < p_m
                child = np.where(mutate_mask, child + sigma*np.random.randn(dim), child)
                new_ind.append(child)
            new_ind = np.array(new_ind)
            new_fit = []
            for ind in new_ind:
                if evaluations < max_eval:
                    fit = evaluate_individual(func_id, ind)
                    current_best_fitness = min(current_best_fitness, fit)
                    history[evaluations] = current_best_fitness
                    new_fit.append(fit)
                    evaluations += 1
                else:
                    new_fit.append(np.inf)
            new_fit = np.array(new_fit)
            if len(pop) + len(new_ind) <= ps_max:
                pop = np.vstack([pop, new_ind])
                fitness = np.concatenate([fitness, new_fit])
            else:
                replace_idx = np.random.choice(len(pop), len(new_ind), replace=False)
                pop[replace_idx] = new_ind
                fitness[replace_idx] = new_fit
            SG_stag = 0
            pop, fitness, swarms, fitness_swarms = safe_reshape(pop, fitness, ss, dim)
            A_E = swarms[:, 0, :].copy()
        
        # 动态重新分组
        pop, fitness, swarms, fitness_swarms = safe_reshape(pop, fitness, ss, dim)
        A_E = swarms[:, 0, :].copy()

    history = history[:evaluations]
    return best, best_fitness, history

if __name__ == "__main__":
    func_id = 1
    best_sol, best_fit, hist = fade_optimizer(
        bounds=[-100, 100],
        dim=10,
        func_id=func_id,
        pop_size=50,
        max_eval=100000
    )
    print(f"最优解: {best_sol}")
    print(f"最优适应度: {best_fit}")
