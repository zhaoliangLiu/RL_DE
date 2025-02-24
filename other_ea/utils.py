import numpy as np
from other_ea.de import de_optimizer
from other_ea.sade import sade
from other_ea.fade import fade_optimizer
from other_ea.apdsde import apdsde
from other_ea.rlpde import rlpde_optimizer
# 加入父路径
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import matplotlib.pyplot as plt
import config
def run_optimizer(algorithm, func_id, bounds, dim, max_eval=10000, pop_size=50):
    """统一的优化器接口
    
    Args:
        algorithm: 算法名称 ('de', 'sade', 'fade')
        func: 目标函数
        bounds: 搜索范围 [-100, 100]
        dim: 维度
        max_eval: 最大评估次数
        pop_size: 种群大小
        
    Returns:
        dict: 包含优化结果的字典
    """
    
    # 运行选择的算法
    if algorithm.lower() == 'de':
        best_solution, best_fitness, history = de_optimizer(
            bounds=bounds,
            dim=dim,
            func_id=func_id,
            pop_size=pop_size,
            max_eval=max_eval
        )
    elif algorithm.lower() == 'sade':
        best_solution, best_fitness, history = sade(
            dim=dim,
            func_id=func_id,
            bounds=bounds,
            max_eval=max_eval,
            pop_size=pop_size
        )
    elif algorithm.lower() == 'apdsde':
        best_solution, best_fitness, history = apdsde(
            dim=dim,
            func_id=func_id,
            bounds=bounds,
            max_eval=max_eval,
            pop_size=pop_size
        )
        
    elif algorithm.lower() == 'fade':
        # 调整种群大小使其为3的倍数
        if pop_size % 3 != 0:
            adjusted_pop_size = pop_size - (pop_size % 3)
            print(f"Adjusted pop_size for fade from {pop_size} to {adjusted_pop_size} so that it is divisible by 3.")
        else:
            adjusted_pop_size = pop_size
        best_solution, best_fitness, history = fade_optimizer(
            bounds=bounds,
            dim=dim,
            func_id=func_id,
            pop_size=adjusted_pop_size,
            max_eval=max_eval
        )
    elif algorithm.lower() == 'rlpde':
        best_solution, best_fitness, history = rlpde_optimizer(
            bounds=bounds,
            dim=dim,
            func_id=func_id,
            pop_size=pop_size,
            max_eval=max_eval
        )
    
    # 返回统一格式的结果
    return {
        'algorithm': algorithm,
        'best_solution': best_solution,
        'best_fitness': best_fitness,
        'history': history,
        'evaluations': len(history)
    }

if __name__ == "__main__": 
    
    # 测试参数
    func_id = 10
    test_bounds = [-100, 100]
    test_dim = 10
    test_max_eval = 100000
    test_pop_size = 50
    
    # 测试所有算法
    for alg in config.ALGORITHMS:
        print(f"\nTesting {alg.upper()}:")
        result = run_optimizer(
            algorithm=alg,
            func_id=func_id,
            bounds=test_bounds,
            dim=test_dim,
            max_eval=test_max_eval,
            pop_size=test_pop_size
        )
        # 取log变换
        plt.plot(np.log(result['history']), label=alg)
        
        print(f"name:{alg}  Best fitness: {result['best_fitness']}") 
    
    plt.legend()
    plt.xlabel("Evaluations")
    plt.ylabel("Fitness")
    plt.title(f"Function {func_id} Convergence")
    plt.savefig("other_ea/compare.png")