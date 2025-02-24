import numpy as np
# 加入父路径
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import config
from env import PSO_Proportional_Env
from stable_baselines3 import PPO

def rlpde_optimizer(bounds, dim, func_id, pop_size=50, max_eval=10000):
    """
    与 fade_optimizer 接口一致：
    输入参数：
      - bounds: 搜索空间边界（例如：[-100, 100]）
      - dim: 问题维度
      - func: 适应度函数（此处忽略，由环境选择对应函数）
      - pop_size: 种群大小
      - max_eval: 最大评估次数（这里使用环境的最大迭代次数）

    输出：
      - best_solution: 求得的最优解
      - best_fitness:  最优适应度（扣除了偏置等）
      - history:       每一次评估后的全局最优历史（数组）
    """
    # 创建环境
    env = PSO_Proportional_Env(
        dim=dim,
        population_size=pop_size,
        max_iter=config.MAX_ITERATIONS, 
        num_function=1,
        start_function_id=func_id
    )
    # 加载训练好的 PPO 模型（模型名称在配置文件中）
    model = PPO.load(config.MODEL_PATH)
    
    state = env.reset()
    global_best = float('inf')
    best_solution = None
    history = []

    # 环境的 step 返回信息中 info["fitness"] 为当代内每次评估后的全局最优值数组
    while True:
        action, _ = model.predict(state, deterministic=True)
        state, reward, done, info = env.step(action, is_test=True)
        # 遍历该代所有评估的全局最优值，逐步保存历史记录
        for fitness_val in info["fitness"]:
            if fitness_val < global_best:
                global_best = fitness_val
                best_solution = env.gbest_position.copy()
            history.append(global_best)
        if done:
            break
    env.close()
    return best_solution, global_best, np.array(history)

if __name__ == '__main__':
    bounds = [-100, 100]
    dim = 10
    func_id = 1
    # 对于 fade_optimizer 接口，func 参数可以传入 None（实际适应度由环境选择）
    best_sol, best_fit, hist = rlpde_optimizer(bounds, dim, func_id,
                                               pop_size=config.POPULATION_SIZE,
                                               max_eval=config.MAX_ITERATIONS)
    print("Best Solution:", best_sol)
    print("Best Fitness:", best_fit)