import os
import numpy as np
from stable_baselines3 import PPO

# 将父目录加入
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from env import PSO_Proportional_Env

# 将模型变为一个函数(shap解释)
def get_model(X):
    # 用于SHAP分析的模型包装函数 
   
    model = PPO.load("ppo_ea_model")
    action = model.predict(X, deterministic=True)[0]  # 只返回动作值
    return action
   

def save_data(model_path="ppo_ea_model_v3"):
    dim_config = {
        10: {'pop_size': 50, 'max_eval': 10*10000}, 
        30: {'pop_size': 100, 'max_eval': 30*10000},
        50: {'pop_size': 150, 'max_eval': 50*10000},
        100: {'pop_size': 200, 'max_eval': 100*10000}
    }
    data = []
    
    model = PPO.load(model_path) 
    for dim in [10]:
        cfg = dim_config[dim]
        max_iter = cfg['max_eval'] // cfg['pop_size']
        
        for func_id in range(1, 30):
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
                # 记录数据 - 确保所有数据都是浮点数
                list_obs = [float(x) for x in obs]  # 将所有观察值转换为浮点数
                list_obs.append(float(action[0]))   # 添加动作值
                data.append(list_obs)
                obs, _, done, info = env.step(action, is_test=True)
                
            

        # 将数据转换为numpy数组
        data = np.array(data)
        
        # 创建带表头的CSV文件
        header = ['gbest_fitness',
                'mean_fitness', 
                'std_fitness',
                'gbest_fitness_change',
                'sum_dist_center',
                'sum_dist_gbest',
                'not_updata_count',
                'iter_time',
                'action']  # 注意这里添加了action列
        
        # 使用pandas保存带表头的CSV
        import pandas as pd
        df = pd.DataFrame(data, columns=header)
        # 如果没有目录
        if not os.path.exists(f'data/explain'):
            os.makedirs(f'data/explain')
        csv_path = f'data/explain/{dim}D_explain_data_v3.csv'
        df.to_csv(csv_path, index=False)
        print(f'dim {dim}D,Data saved to {csv_path}')
        data = []
    
    return data

if __name__ == "__main__":
    save_data()
    print('Data saved successfully')