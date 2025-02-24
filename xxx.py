import os
import csv
import numpy as np
import scipy.stats as stats
from stable_baselines3 import PPO

import config  # [config.py](config.py)
from env import PSO_Proportional_Env  # [PSO_Proportional_Env](env.py)

def record_rl_log(func_id):
    # 创建保存日志的文件夹，例如 data/rl_logs
    log_dir = os.path.join("data", "rl_logs")
    os.makedirs(log_dir, exist_ok=True)
    csv_file = os.path.join(log_dir, f"F{func_id}_rl_log.csv")
    
    # 初始化环境，指定函数ID
    env = PSO_Proportional_Env(start_function_id=func_id)
    state = env.reset()
    
    # 加载训练好的模型
    model = PPO.load(config.MODEL_PATH)
    
    records = []
    generation = 0
    done = False
    # 主循环，遍历每一代（或直到环境结束）
    while not done:
        generation += 1
        # 使用模型预测动作
        action, _ = model.predict(state, deterministic=True)
        action = [0]
        state, reward, done, info = env.step(action, is_test=True)
        
        # 补充记录信息：当前最佳适应度、p值与种群熵
        best_fitness = env.gbest_fitness
        p_value = action[0]
        # 计算所有个体距离种群中心的距离
        center = np.mean(env.population, axis=0)
        distances = np.linalg.norm(env.population - center, axis=1)
        # 根据种群大小的平方根确定直方图箱数
        pop_size = env.population.shape[0]
        bins = max(1, int(np.sqrt(pop_size)))
        counts, _ = np.histogram(distances, bins=bins)
        probabilities = counts / np.sum(counts)
        # 计算熵（避免对0取对数，加上一个很小的数）
        population_entropy = -np.sum(probabilities * np.log(probabilities + 1e-12))
        
        records.append({
            "generation": generation,
            "best_fitness": best_fitness,
            "p_value": p_value,
            "population_entropy": population_entropy
        })
    
    # 将记录写入 CSV 文件
    fieldnames = ["generation", "best_fitness", "p_value", "population_entropy"]
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            writer.writerow(row)
    print(f"RL日志已保存至: {csv_file}")

if __name__ == "__main__":
    # 针对每个函数进行记录；注意：这里使用 config.FUNC_IDS_TEST 为测试函数ID 列表
    for func_id in config.FUNC_IDS_TEST:
        record_rl_log(func_id)
        break
