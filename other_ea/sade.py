import numpy as np
# 加入父路径
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from cec2017.functions import all_functions
import config 
def sade(dim, func_id, bounds, max_eval, pop_size=50):
    # 参数初始化
    LP = 50                # 学习周期
    K = 4                  # 策略数
    F_mean, F_std = 0.5, 0.3
    CR_mean, CR_std = 0.5, 0.1
    epsilon = 0.01          # 防止除零
    
    # 修正bounds处理
    if isinstance(bounds[0], (list, tuple)):
        # 如果bounds是[[min, max]]格式
        lower = np.full(dim, bounds[0][0])
        upper = np.full(dim, bounds[0][1])
    else:
        # 如果bounds是[min, max]格式
        lower = np.full(dim, bounds[0])
        upper = np.full(dim, bounds[1])
    
    # 初始化种群
    population = np.random.uniform(lower, upper, (pop_size, dim))
    fitness_values = config.FUNC_MAP[func_id](population) - func_id * 100
    current_eval = pop_size
    best_idx = int(np.argmin(fitness_values))
    best_solution = population[best_idx].copy()
    best_fitness = fitness_values[best_idx]
    history = np.full(max_eval, best_fitness)
    
    # 策略组件初始化
    strategy_probs = np.ones(K) / K
    success_memory = np.zeros((LP, K))
    failure_memory = np.zeros((LP, K))
    memory_ptr = 0
    
    while current_eval < max_eval:
        # 1. 策略选择
        strategy_idx = np.random.choice(K, pop_size, p=strategy_probs)
        
        # 2. 参数生成
        F = np.clip(np.random.normal(F_mean, F_std, pop_size), 0.1, 1.0)
        CR = np.clip(np.random.normal(CR_mean, CR_std, pop_size), 0.0, 1.0)
        
        # 3. 生成试验向量
        trial = np.zeros_like(population)
        for i in range(pop_size):
            # 生成排除当前索引的候选
            candidates = np.delete(np.arange(pop_size), i)
            
            # 根据策略生成变异向量
            if strategy_idx[i] == 0:   # DE/rand/1/bin
                r1, r2, r3 = np.random.choice(candidates, 3, replace=False)
                mutant = population[r1] + F[i] * (population[r2] - population[r3])
                
            elif strategy_idx[i] == 1: # DE/rand-to-best/2/bin
                r1, r2, r3, r4 = np.random.choice(candidates, 4, replace=False)
                mutant = population[i] + F[i]*(best_solution - population[i]) + \
                         F[i]*(population[r1] - population[r2]) + \
                         F[i]*(population[r3] - population[r4])
                         
            elif strategy_idx[i] == 2: # DE/rand/2/bin
                r1, r2, r3, r4, r5 = np.random.choice(candidates, 5, replace=False)
                mutant = population[r1] + F[i]*(population[r2] - population[r3]) + \
                         F[i]*(population[r4] - population[r5])
                         
            else:                     # DE/current-to-rand/1
                r1, r2, r3 = np.random.choice(candidates, 3, replace=False)
                K_val = 0.5
                mutant = population[i] + K_val*(population[r1] - population[i]) + \
                         F[i]*(population[r2] - population[r3])
            
            # 二项式交叉（策略3除外）
            if strategy_idx[i] != 3:
                cross_mask = np.random.rand(dim) < CR[i]
                j_rand = np.random.randint(dim)
                cross_mask[j_rand] = True
                trial[i] = np.where(cross_mask, mutant, population[i])
            else:
                trial[i] = mutant
            
            # 边界处理
            trial[i] = np.clip(trial[i], lower, upper)
        
        # 4. 评估及更新
        trial_fitness = config.FUNC_MAP[func_id](trial) - func_id * 100
        current_eval += pop_size
        
        # 更新历史记录
        improved = trial_fitness < best_fitness
        if np.any(improved):
            best_idx = np.argmin(trial_fitness)
            best_solution = trial[best_idx].copy()
            best_fitness = trial_fitness[best_idx]
        eval_step = min(pop_size, max_eval - (current_eval - pop_size))
        history[current_eval-pop_size:current_eval] = best_fitness
        
        # 5. 更新成功/失败记忆
        success = trial_fitness < fitness_values
        for k in range(K):
            mask = (strategy_idx == k)
            ns = np.sum(success[mask])
            nf = np.sum(~success[mask])
            if memory_ptr < LP:
                success_memory[memory_ptr, k] = ns
                failure_memory[memory_ptr, k] = nf
            else:
                success_memory[:-1, k] = success_memory[1:, k]
                failure_memory[:-1, k] = failure_memory[1:, k]
                success_memory[-1, k] = ns
                failure_memory[-1, k] = nf
        memory_ptr = (memory_ptr + 1) % LP
        
        # 6. 更新策略概率
        if current_eval >= LP*pop_size:
            S = (np.sum(success_memory, axis=0) + epsilon) / \
                (np.sum(success_memory, axis=0) + np.sum(failure_memory, axis=0) + epsilon)
            strategy_probs = S / S.sum()
        
        # 7. 种群更新
        replace = success
        population[replace] = trial[replace]
        fitness_values[replace] = trial_fitness[replace]
    
    return best_solution, best_fitness, history

if __name__ == "__main__":
    dim = 10
    bounds = [-100, 100]  # 可以直接传入这种格式
    pop_size = 50
    max_eval = 10000
    func_id = 9
    best, fitness, history = sade(
        dim=dim, 
        func_id=func_id, 
        bounds=bounds,  # 直接传入bounds
        max_eval=max_eval, 
        pop_size=pop_size
    )
    print(f"Best solution: {best}")
    print(f"Fitness value: {fitness}")
    print(f"History: {history}")