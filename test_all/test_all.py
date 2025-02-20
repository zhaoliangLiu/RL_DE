# test_all.py
import os
import sys
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from multiprocessing import Pool, cpu_count, set_start_method
import itertools
import matplotlib.pyplot as plt
from matplotlib import rcParams

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from env import PSO_Proportional_Env
from other_ea.other_ea import unified_optimization
import config # 导入配置文件

rcParams['font.family'] = 'Times New Roman'

DIM = config.DIM_TEST # 使用配置文件中的测试维度
REPEAT = config.REPEAT_TEST # 使用配置文件中的重复次数
FUNC_IDS = config.FUNC_IDS_TEST # 使用配置文件中的函数ID列表
ALGORITHMS = config.ALGORITHM_NAMES # 包含所有要测试的算法

N_PROCESSES = cpu_count()

try:
    set_start_method("spawn", force=True)
except RuntimeError:
    pass

def init_worker():
    global GLOBAL_RL_MODEL
    GLOBAL_RL_MODEL = PPO.load(config.MODEL_PATH) # 使用配置文件中的模型路径

DIM_CONFIG = config.DIM_CONFIG # 使用配置文件中的维度配置

def run_rl_optimization(func_id, dim, run_id):
    cfg = DIM_CONFIG[dim]
    env = PSO_Proportional_Env(
        dim=dim,
        population_size=cfg['pop_size'],
        max_iter=config.MAX_ITERATIONS, # 使用配置文件中的最大迭代次数`,
        num_function=1,
        start_function_id=func_id
    )

    obs = env._get_full_state()
    fitness_history = []
    model = PPO.load(config.MODEL_PATH) # 使用配置文件中的模型路径

    for _ in range(env.max_iter):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, info = env.step(action, is_test=True)
        fitness_history.extend(info["fitness"])
        if done: break

    return env.gbest_fitness, fitness_history

def run_other_optimization(algorithm, func_id, dim):
    _, fitness, history = unified_optimization(
        dim=dim,
        algorithm_name=algorithm,
        fitness_function_id=func_id,
        max_evals=DIM_CONFIG[dim]['max_eval'] # 使用配置文件中的 max_evals
    )
    return fitness, history

def process_function(func_id):
    result_dir = os.path.join(config.COMPARE_DATA_DIR, f"{DIM}D")
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    csv_path = os.path.join(result_dir, f"原始数据{config.PATH_EXPLOIT_DEV}.csv")
    
    # Initialize results dictionary
    results = {algo: {'fitness': [], 'history': []} for algo in ALGORITHMS}
    
    # Handle file reading and initialization
    if func_id == min(FUNC_IDS):
        # For the first function, create empty DataFrame with columns
        columns = ['Function']
        for algo in ALGORITHMS:
            prefix = 'rlde' if algo == 'rl' else algo
            for stat in ['best', 'worst', 'mean', 'std', 'median']:
                columns.append(f'{prefix}_{stat}')
        existing_df = pd.DataFrame(columns=columns)
        existing_df.to_csv(csv_path, index=False)
    else:
        try:
            existing_df = pd.read_csv(csv_path)
        except (pd.errors.EmptyDataError, FileNotFoundError):
            # If file is empty or doesn't exist, create new DataFrame with columns
            columns = ['Function']
            for algo in ALGORITHMS:
                prefix = 'rlde' if algo == 'rl' else algo
                for stat in ['best', 'worst', 'mean', 'std', 'median']:
                    columns.append(f'{prefix}_{stat}')
            existing_df = pd.DataFrame(columns=columns)

    # RL-DE testing
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

    # Testing other algorithms
    for algo in ALGORITHMS:
        if algo != 'rl':
            print(f"\nProcessing F{func_id} - Running {algo.upper()} experiments...")
            algo_fitness_list = []
            best_algo_fitness = float('inf')
            best_algo_history = None

            for run in range(REPEAT):
                fitness, history = run_other_optimization(algo, func_id, DIM)
                algo_fitness_list.append(fitness)
                if fitness < best_algo_fitness:
                    best_algo_fitness = fitness
                    best_algo_history = history

            # Save convergence history
            convergence_dir = os.path.join(result_dir, config.CONVERGENCE_DIR_NAME)
            if not os.path.exists(convergence_dir):
                os.makedirs(convergence_dir)
            np.save(os.path.join(convergence_dir, f"F{func_id}_{algo}.npy"), best_algo_history)

            # Record statistics
            results[algo] = {
                'best': np.min(algo_fitness_list),
                'worst': np.max(algo_fitness_list),
                'mean': np.mean(algo_fitness_list),
                'std': np.std(algo_fitness_list),
                'median': np.median(algo_fitness_list)
            }

    # Save RL results
    convergence_dir = os.path.join(result_dir, config.CONVERGENCE_DIR_NAME)
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

    # Create new DataFrame with results and append to existing data
    row_dict = {'Function': f'F{func_id}'}
    for algo in ALGORITHMS:
        prefix = 'rlde' if algo == 'rl' else algo
        for stat in ['best', 'worst', 'mean', 'std', 'median']:
            row_dict[f'{prefix}_{stat}'] = results[algo][stat]

    new_row_df = pd.DataFrame([row_dict])
    
    # Combine with existing data and save
    combined_df = pd.concat([existing_df, new_row_df], ignore_index=True)
    combined_df.to_csv(csv_path, index=False)
    
    return results

def main():

    for fid in FUNC_IDS:
        result = process_function(fid)

    print("\nAll functions processed. Starting convergence plots...")

def plot_convergence(func_id):
    plt.figure(figsize=(10, 6), dpi=300)

    for algo in ALGORITHMS:
        history = np.load(os.path.join(config.COMPARE_DATA_DIR, f"{DIM}D", config.CONVERGENCE_DIR_NAME, f"F{func_id}_{algo}.npy")) # 使用配置文件中的收敛数据路径
        if algo == 'rl':
            plt.semilogy(history, 'r--', linewidth=2, label='RLPDE')
        else:
            plt.semilogy(history, label=algo.upper(), alpha=0.7)

    plt.title(f'Function {func_id} ({DIM}D) Convergence', fontsize=14, fontname='Times New Roman')
    plt.xlabel('Function Evaluations', fontsize=12, fontname='Times New Roman')
    plt.ylabel('Fitness Value (log scale)', fontsize=12, fontname='Times New Roman')
    plt.legend(prop={'family': 'Times New Roman'})
    plt.grid(True, which='both', linestyle='--', alpha=0.5)

    output_dir = os.path.join(config.COMPARE_GRAPH_DIR, f"{DIM}D", config.CONVERGENCE_DIR_NAME) # 使用配置文件中的图形输出路径
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, f'F{func_id}.png'), bbox_inches='tight') # 使用配置文件中的图形输出路径和文件名
    plt.close()

if __name__ == "__main__":
    from train_ppo import train_ppo
    train_ppo()
    main()
    # 处理排名
    import process.process_data
    print("\nGenerating convergence plots...")
    for fid in FUNC_IDS:
        plot_convergence(fid)
        print(f"Function {fid} Convergence Plotted")
    from explain.explian_utils import save_data
    from explain.shap_explian import shap_analysis
    save_data()
    shap_analysis()

    from test_rl_ea import test_all
    test_all()
