# explain_utils.py
import os
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from env import PSO_Proportional_Env
import config  # 导入配置文件

# 单个 dim 和 func_id 组合的处理函数
def process_single(dim, func_id, model):
    cfg = config.DIM_CONFIG[dim]
    max_iter = cfg['max_eval'] // cfg['pop_size']
    
    env = PSO_Proportional_Env(
        dim=dim,
        population_size=cfg['pop_size'],
        max_iter=max_iter,
        num_function=1,
        start_function_id=func_id
    )
    
    data = []
    obs = env.reset()
    # 预先设置一个批量收集列表（batch_size 可调，根据内存情况）
    batch_obs = []
    batch_indices = []
    batch_limit = 4  # 批量预测大小
    
    for i in range(env.max_iter):
        batch_obs.append(obs)
        batch_indices.append(i)
        # 当达到批量大小或最后一次 step 时进行预测
        if len(batch_obs) == batch_limit or i == env.max_iter - 1:
            batch_obs_arr = np.array(batch_obs)
            # 批量调用模型预测
            actions = model.predict(batch_obs_arr, deterministic=True)[0]
            for obs_item, action in zip(batch_obs, actions):
                # 将观测拆分后追加 action 值
                list_obs = [float(x) for x in obs_item]
                list_obs.append(float(action[0]))  # 假设 action 是一个数组
                data.append(list_obs)
            # 重置批量列表
            batch_obs = []
            batch_indices = []
        
        # 单步预测后环境反馈
        obs, _, done, info = env.step(model.predict(obs, deterministic=True)[0], is_test=True)
        if done:
            obs = env.reset()
    return np.array(data), dim

# process_task 调用 process_single
def process_task(task, model_path):
    d, f = task
    local_model = PPO.load(model_path)
    return process_single(d, f, local_model)

def save_data(model_path=config.MODEL_PATH):
    # 定义需要处理的任务列表，任务为 (dim, func_id)
    tasks = []
    for dim in config.DIMS_TEST_EXPLAIN:
        for func_id in config.FUNC_IDS_TEST:
            tasks.append((dim, func_id))
    
    results = {}  # {dim: [data_for_all_func_id...] }
    
    # 同步顺序处理每个任务
    for task in tasks:
        d, f = task
        try:
            data, dim = process_task(task, model_path)
            results.setdefault(dim, []).append((f, data))
            print(f"Processed dim {dim} func {f}")
        except Exception as e:
            print(f"Error processing dim {d} func {f}: {e}")
    
    # 保存每个维度的结果到单独 csv 文件（每个文件含所有 func_id）
    if not os.path.exists(config.EXPLAIN_DATA_DIR):
        os.makedirs(config.EXPLAIN_DATA_DIR)
    
    header = ['gbest_fitness',
              'mean_fitness',
              'std_fitness',
              'gbest_fitness_change',
              'sum_dist_center',
              'sum_dist_gbest',
              'not_updata_count',
              'iter_time',
              'action']
    
    for dim, func_data_list in results.items():
        # 将所有 func_id 数据合并，并可以增加一列标识 func_id
        all_rows = []
        for func_id, data in func_data_list:
            # 为标识加入 func_id 列
            func_col = np.full((data.shape[0], 1), func_id)
            data_with_func = np.hstack([func_col, data])
            all_rows.append(data_with_func)
        all_data = np.vstack(all_rows)
        # 新的表头：第一列为 "func_id" 后续为指标
        out_header = ['func_id'] + header
        csv_path = os.path.join(config.EXPLAIN_DATA_DIR, f'{dim}D_explain_data_{config.PATH_EXPLOIT_DEV}.csv')
        df = pd.DataFrame(all_data, columns=out_header)
        df.to_csv(csv_path, index=False)
        print(f"dim {dim}D, Data saved to {csv_path}")
    return results

# if __name__ == "__main__":
save_data()
print('Data saved successfully')
