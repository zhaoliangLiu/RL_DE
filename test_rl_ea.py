# test_rl_ea.py
from stable_baselines3 import PPO
from env import PSO_Proportional_Env
import os
import matplotlib.pyplot as plt
import config # 导入配置文件


def test_model(dim, function_id, path, model_path=config.MODEL_PATH): # 默认使用配置文件中的模型路径
    dim_config = config.DIM_CONFIG # 使用配置文件中的维度配置

    model = PPO.load(model_path)

    cfg = dim_config[dim]
    max_iter = cfg['max_eval'] // cfg['pop_size']

    env = PSO_Proportional_Env(
            dim=dim,
            population_size=cfg['pop_size'],
            max_iter=max_iter,
            num_function=1,
            start_function_id=function_id
    )

    obs = env._get_full_state()
    fitness_history = []
    exploit_rates = []

    for i in range(env.max_iter):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, info = env.step(action, is_test=True)

        fitness_history.extend(info["fitness"])
        exploit_rates.extend([info["exploit_rate"][-1]] * len(info["fitness"]))

    # 创建输出目录：在 "勘探开发" 对应的图形和数据目录下，以 {path}/{dim}D/{config.PATH_EXPLOIT_DEV} 作为文件夹路径
    data_dir = os.path.join(config.EXPLOIT_DEV_DATA_DIR, f'{path}/{dim}D/{config.PATH_EXPLOIT_DEV}')
    graph_dir = os.path.join(config.EXPLOIT_DEV_GRAPH_DIR, f'{path}/{dim}D/{config.PATH_EXPLOIT_DEV}')
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(graph_dir, exist_ok=True)

    plt.figure(figsize=(8, 6), dpi=300)
    ax1 = plt.gca()

    ln1 = ax1.semilogy(fitness_history, '#2878B5', linestyle='-',
                       linewidth=2, label='Fitness Value', alpha=0.8)
    ax1.set_xlabel('fps', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Fitness Value', fontsize=12, fontweight='bold', color='#2878B5')
    ax1.tick_params(axis='y', labelcolor='#2878B5')

    ax2 = ax1.twinx()
    ln2 = ax2.plot(exploit_rates, '#C82423', linestyle='--',
                   linewidth=2, label='Exploitation Rate', alpha=0.8)
    ax2.set_ylabel('Exploitation Rate', fontsize=12, fontweight='bold', color='#C82423')
    ax2.set_ylim(-0.05, 1.05)
    ax2.tick_params(axis='y', labelcolor='#C82423')

    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='upper center',
              bbox_to_anchor=(0.5, -0.15),
              ncol=2, fontsize=10)

    ax1.grid(True, linestyle='--', alpha=0.3)
    plt.title(f'Function {function_id} ({dim}D)\nFinal Fitness: {fitness_history[-1]:.2e}',
              pad=20, fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(graph_dir, f'f{function_id}.png'), bbox_inches='tight', dpi=300)

    import pandas as pd
    data_dict = {
        'fps': range(len(fitness_history)),
        'Fitness': fitness_history,
        'Exploitation_Rate': exploit_rates
    }
    df = pd.DataFrame(data_dict)

    csv_path = os.path.join(data_dir, f'F{function_id}.csv')
    df.to_csv(csv_path, index=False)
    return env.gbest_fitness

def test_all(path=config.PATH_EXPLOIT_DEV, dims=config.DIMS_TEST_EXPLOIT_DEV, model=config.MODEL_PATH): # 使用配置文件中的默认参数
    for dim in dims:
        for function_id in config.FUNC_IDS_TEST:
            best_fitness = test_model(dim=dim, function_id=function_id, path = path,model_path=model)
            print(f"Function {function_id} ({dim}D) Best Fitness: {best_fitness}")
if __name__ == "__main__":

    dims = config.DIMS_TEST_EXPLOIT_DEV # 使用配置文件中的测试维度列表
    path = config.PATH_EXPLOIT_DEV # 使用配置文件中的路径
    test_all(path, dims, model = config.MODEL_PATH) # 使用配置文件中的模型路径
