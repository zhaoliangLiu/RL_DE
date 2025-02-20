# explain_utils.py
import os
import numpy as np
from stable_baselines3 import PPO
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from env import PSO_Proportional_Env
import config  # 导入配置文件

# 将模型变为一个函数(shap解释)
def get_model(X):
    model = PPO.load(config.MODEL_PATH) # 使用配置文件中的模型路径
    action = model.predict(X, deterministic=True)[0]
    return action

def save_data(model_path=config.MODEL_PATH): # 默认使用配置文件中的模型路径
    data = []
    model = PPO.load(model_path)
    for dim in config.DIMS_TEST_EXPLAIN: # 您可以修改这里需要解释的维度，或者将其参数化
        cfg = config.DIM_CONFIG[dim] # 使用配置文件中的维度配置
        max_iter = cfg['max_eval'] // cfg['pop_size']

        for func_id in config.FUNC_IDS_TEST:
            env = PSO_Proportional_Env(
                dim=dim,
                population_size=cfg['pop_size'],
                max_iter=max_iter,
                num_function=1,
                start_function_id=func_id
            )

            obs = env.reset()

            for i in range(env.max_iter):
                action, _ = model.predict(obs, deterministic=True)
                list_obs = [float(x) for x in obs]
                list_obs.append(float(action[0]))
                data.append(list_obs)
                obs, _, done, info = env.step(action, is_test=True)

        data = np.array(data)

        header = ['gbest_fitness',
                'mean_fitness',
                'std_fitness',
                'gbest_fitness_change',
                'sum_dist_center',
                'sum_dist_gbest',
                'not_updata_count',
                'iter_time',
                'action']

        import pandas as pd
        df = pd.DataFrame(data, columns=header)
        if not os.path.exists(config.EXPLAIN_DATA_DIR): # 使用配置文件中的数据目录
            os.makedirs(config.EXPLAIN_DATA_DIR) # 使用配置文件中的数据目录
        csv_path = f'{config.EXPLAIN_DATA_DIR}/{dim}D_explain_data_{config.PATH_EXPLOIT_DEV}.csv' # 使用配置文件中的数据目录和文件名
        df.to_csv(csv_path, index=False)
        print(f'dim {dim}D,Data saved to {csv_path}')
        data = []

    return data

if __name__ == "__main__":
    save_data()
    print('Data saved successfully')
