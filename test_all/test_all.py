import os
import sys
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from multiprocessing import Pool, cpu_count, set_start_method
import itertools
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 将父目录加入
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from env import PSO_Proportional_Env
from other_ea.other_ea import unified_optimization

# 设置新罗马字体
rcParams['font.family'] = 'Times New Roman'

# 修改配置参数
DIM = 10
REPEAT = 30  # 增加到50次
FUNC_IDS = range(1, 30)
# ALGORITHMS = ["rl", "de", "jade", "lshade", "shade"]
ALGORITHMS = ["rl"]

N_PROCESSES = cpu_count()   
 
try:
    set_start_method("spawn", force=True)
except RuntimeError:
    pass  

# 预加载函数避免重复初始化
def init_worker():
    global GLOBAL_RL_MODEL
    GLOBAL_RL_MODEL = PPO.load("ppo_ea_model_v3")

# 维度配置（与RL环境一致）
DIM_CONFIG = {
    10: {'pop_size': 50, 'max_eval': 10*10000},
    30: {'pop_size': 100, 'max_eval': 30*10000},
    50: {'pop_size': 150, 'max_eval': 50*10000},
    100: {'pop_size': 200, 'max_eval': 100*10000}
}

def run_rl_optimization(func_id, dim, run_id):
    """运行RL-DE优化并返回历史记录"""
    cfg = DIM_CONFIG[dim]
    env = PSO_Proportional_Env(
        dim=dim,
        population_size=cfg['pop_size'],
        max_iter=cfg['max_eval'] // cfg['pop_size'],
        num_function=1,
        start_function_id=func_id
    )
    
    obs = env._get_full_state()
    fitness_history = []
    model = PPO.load("ppo_ea_model_v3")
    
    for _ in range(env.max_iter):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, info = env.step(action, is_test=True)
        fitness_history.extend(info["fitness"])
        if done: break
    
    return env.gbest_fitness, fitness_history

def run_other_optimization(algorithm, func_id, dim):
    """运行其他优化算法并返回历史记录"""
    _, fitness, history = unified_optimization(
        dim=dim,
        algorithm_name=algorithm,
        fitness_function_id=func_id,
        max_evals=DIM_CONFIG[dim]['max_eval']
    )
    return fitness, history

def process_function(func_id):
    """处理单个函数的所有实验并保存为单行格式""" 
    # 如果不存在目录
    result_dir = f"data/compare/{DIM}D"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    csv_path = os.path.join(result_dir, "原始数据v3.csv")
    if os.path.exists(csv_path):
        try:
            existing_df = pd.read_csv(csv_path)
        except pd.errors.EmptyDataError:
            print(f"Warning: {csv_path} is empty. Overwriting.")
    else:
        existing_df = pd.DataFrame()

    # Initialize results dictionary
    results = {algo: {'fitness': [], 'history': []} for algo in ALGORITHMS}

    # Run RL-DE experiments
    print(f"\nProcessing F{func_id} - Running RL-DE experiments...")
    best_rl_fitness = float('inf')
    best_rl_history = None
    rl_fitness_list = []

    for run in range(REPEAT):
        fitness, history = run_rl_optimization(func_id, DIM, run)
        rl_fitness_list.append(fitness)
        if fitness < best_rl_fitness:
            best_rl_fitness = fitness
            best_rl_history = history

    # Save RL-DE convergence history
    convergence_dir = os.path.join(result_dir, "convergence")
    if not os.path.exists(convergence_dir):
        os.makedirs(convergence_dir)

    np.save(os.path.join(convergence_dir, f"F{func_id}_rl.npy"), best_rl_history)
    results['rl'] = {
        'best': np.min(rl_fitness_list),
        'worst': np.max(rl_fitness_list),
        'mean': np.mean(rl_fitness_list),
        'std': np.std(rl_fitness_list),
        'median': np.median(rl_fitness_list)
    }

    # Run other algorithms
    # print(f"Processing F{func_id} - Running other algorithms...")
    # for algo in ["de", "jade", "lshade", "shade"]:
    #     algo_fitness_list = []
    #     for run in range(REPEAT):
    #         fitness, history = run_other_optimization(algo, func_id, DIM)
    #         algo_fitness_list.append(fitness)
    #         if run == REPEAT - 1:  # Save last run's convergence
    #             np.save(os.path.join(convergence_dir, f"F{func_id}_{algo}.npy"), history)

    #     results[algo] = {
    #         'best': np.min(algo_fitness_list),
    #         'worst': np.max(algo_fitness_list),
    #         'mean': np.mean(algo_fitness_list),
    #         'std': np.std(algo_fitness_list),
    #         'median': np.median(algo_fitness_list)
    #     }

    # Create single-row DataFrame
    row_dict = {'Function': f'F{func_id}'}
    for algo in ALGORITHMS:
        prefix = 'rlde' if algo == 'rl' else algo
        for stat in ['best', 'worst', 'mean', 'std', 'median']:
            row_dict[f'{prefix}_{stat}'] = results[algo][stat]

    # Save to CSV (only if function not already processed)
    new_row_df = pd.DataFrame([row_dict])
    if os.path.exists(csv_path) and not existing_df.empty:
        combined_df = pd.concat([existing_df, new_row_df], ignore_index=True)
    else:
        combined_df = new_row_df
    combined_df.to_csv(csv_path, index=False)
    return results

def main():

    # 处理所有函数
    for fid in FUNC_IDS:
        result = process_function(fid)

    print("\nAll functions processed. Starting convergence plots...")

def plot_convergence(func_id):
    """绘制收敛曲线"""
    plt.figure(figsize=(10, 6), dpi=300)

    # 加载并绘制所有算法的收敛历史
    for algo in ['rl', 'de', 'jade', 'lshade', 'shade']:
        history = np.load(f"data/compare/{DIM}D/convergence/F{func_id}_{algo}.npy")
        if algo == 'rl':
            plt.semilogy(history, 'r--', linewidth=2, label='RLPDE')
        else:
            plt.semilogy(history, label=algo.upper(), alpha=0.7)

    plt.title(f'Function {func_id} ({DIM}D) Convergence', fontsize=14, fontname='Times New Roman')
    plt.xlabel('Function Evaluations', fontsize=12, fontname='Times New Roman')
    plt.ylabel('Fitness Value (log scale)', fontsize=12, fontname='Times New Roman')
    plt.legend(prop={'family': 'Times New Roman'})
    plt.grid(True, which='both', linestyle='--', alpha=0.5)

    output_dir = f'graph/compare/{DIM}D/convergence'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, f'F{func_id}.png'), bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main()

    # 绘制所有函数的收敛曲线
    print("\nGenerating convergence plots...")
    for fid in FUNC_IDS:
        plot_convergence(fid)
        print(f"Function {fid} Convergence Plotted")
    from explain.explian_utils import save_data
    from explain.shap_explian import shap_analysis
    save_data()
    shap_analysis()
