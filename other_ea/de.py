import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import config

def evaluate_individual(func_id, ind):
    """评估单个个体"""
    return float(config.FUNC_MAP[func_id](ind.reshape(1, -1))) - func_id * 100

def de_optimizer(bounds, dim, func_id, pop_size=50, max_eval=10000):
    """基础DE算法 (DE/rand/1/bin)
    
    Args:
        bounds: 搜索范围 [-100, 100]
        dim: 维度
        func: 目标函数
        pop_size: 种群大小
        max_eval: 最大评估次数
    
    Returns:
        best: 最优解
        best_fitness: 最优适应度值
        history: 优化历史记录
    """
    # DE参数
    F = 0.5  # 缩放因子
    CR = 0.9  # 交叉率
    
    # 初始化
    pop = np.random.uniform(bounds[0], bounds[1], (pop_size, dim))
    fitness = config.FUNC_MAP[func_id](pop) - func_id * 100
    best = pop[np.argmin(fitness)].copy()
    best_fitness = np.min(fitness)
    
    # 初始化历史记录，记录每次评估后的最优值
    current_best = best_fitness
    history = np.full(max_eval, np.inf)
    history[:pop_size] = [min(current_best, fit) for fit in fitness]
    current_best = history[pop_size-1]
    evaluations = pop_size
    
    # 主循环
    while evaluations < max_eval:
        for i in range(pop_size):
            if evaluations >= max_eval:
                break
                
            # 随机选择三个不同的个体
            r1, r2, r3 = np.random.choice(
                [x for x in range(pop_size) if x != i], 
                3, 
                replace=False
            )
            
            # 差分变异
            mutant = pop[r1] + F * (pop[r2] - pop[r3])
            mutant = np.clip(mutant, bounds[0], bounds[1])
            
            # 二项式交叉
            cross_points = np.random.rand(dim) <= CR
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
            trial = np.where(cross_points, mutant, pop[i])
            
            # 选择
            trial_fitness = evaluate_individual(func_id, trial)
            evaluations += 1
            
            # 更新当前最优值和历史记录
            if evaluations <= max_eval:
                current_best = min(current_best, trial_fitness)
                history[evaluations-1] = current_best
            
            # 更新个体
            if trial_fitness < fitness[i]:
                pop[i] = trial
                fitness[i] = trial_fitness
                if trial_fitness < best_fitness:
                    best = trial.copy()
                    best_fitness = trial_fitness
    
    # 裁剪历史记录
    history = history[:evaluations]
    return best, best_fitness, history

if __name__ == "__main__":
    func_id = 10
    best_sol, best_fit, hist = de_optimizer(
        bounds=[-100, 100],
        dim=10,
        func_id=func_id,
        pop_size=50,
        max_eval=10000
    )
    print(f"最优解: {best_sol}")
    print(f"最优适应度: {best_fit}")