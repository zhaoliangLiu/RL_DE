from stable_baselines3 import PPO
from env import PSO_Proportional_Env
import os
import matplotlib.pyplot as plt


def test_model(dim, function_id, path, model_path="ppo_ea_model_v3"):
    # 配置不同维度的测试参数
    dim_config = {
        10: {'pop_size': 50, 'max_eval': 10*10000},
        30: {'pop_size': 100, 'max_eval': 30*10000},
        50: {'pop_size': 150, 'max_eval': 50*10000},
        100: {'pop_size': 400, 'max_eval': 100*10000}
    }

    # 加载模型
    model = PPO.load(model_path)
    
    cfg = dim_config[dim]
    max_iter = cfg['max_eval'] // cfg['pop_size']
        
    # 创建测试环境
    env = PSO_Proportional_Env(
            dim=dim,
            population_size=cfg['pop_size'],
            max_iter=max_iter,
            num_function=1,
            start_function_id=function_id
    )
            
    # 运行测试
    obs = env._get_full_state()
    fitness_history = []
    exploit_rates = []
    
    # 执行测试并收集数据
    for i in range(env.max_iter):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, info = env.step(action, is_test=True)
        
        # 记录数据 - 每个个体评估后的适应度变化
        fitness_history.extend(info["fitness"])
        # 对应每个适应度评估重复记录当前迭代的开发率
        exploit_rates.extend([info["exploit_rate"][-1]] * len(info["fitness"]))
        
        # if done:
        #     break

    # 确保输出目录存在
    os.makedirs(f'graph/勘探开发{path}/{dim}D', exist_ok=True)
    os.makedirs(f'data/勘探开发{path}/{dim}D', exist_ok=True)

    # 绘制结果图表
    plt.figure(figsize=(8, 6), dpi=300)
    ax1 = plt.gca()
    
    # 适应度曲线（左轴）
    ln1 = ax1.semilogy(fitness_history, '#2878B5', linestyle='-', 
                       linewidth=2, label='Fitness Value', alpha=0.8)
    ax1.set_xlabel('fps', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Fitness Value', fontsize=12, fontweight='bold', color='#2878B5')
    ax1.tick_params(axis='y', labelcolor='#2878B5')
    
    # 策略比例（右轴）
    ax2 = ax1.twinx()
    ln2 = ax2.plot(exploit_rates, '#C82423', linestyle='--', 
                   linewidth=2, label='Exploitation Rate', alpha=0.8)
    ax2.set_ylabel('Exploitation Rate', fontsize=12, fontweight='bold', color='#C82423')
    ax2.set_ylim(-0.05, 1.05)
    ax2.tick_params(axis='y', labelcolor='#C82423')
    
    # 合并图例
    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='upper center', 
              bbox_to_anchor=(0.5, -0.15), 
              ncol=2, fontsize=10)
    
    # 添加网格和标题
    ax1.grid(True, linestyle='--', alpha=0.3)
    plt.title(f'Function {function_id} ({dim}D)\nFinal Fitness: {fitness_history[-1]:.2e}',
              pad=20, fontsize=12, fontweight='bold')
    
    # 保存结果
    plt.tight_layout()
    plt.savefig(f'graph/勘探开发{path}/{dim}D/f{function_id}.png', bbox_inches='tight', dpi=300)
  
    # 保存原始数据
    import pandas as pd

    # 创建DataFrame
    data_dict = {
        'fps': range(len(fitness_history)),
        'Fitness': fitness_history,
        'Exploitation_Rate': exploit_rates
    }
    df = pd.DataFrame(data_dict)

    # 确保目录存在
    if not os.path.exists(f'data/勘探开发{path}/{dim}D'):
        os.makedirs(f'data/勘探开发{path}/{dim}D', exist_ok=True)

    # 保存为CSV文件
    csv_path = f'data/勘探开发{path}/{dim}D/F{function_id}.csv'
    df.to_csv(csv_path, index=False)
    return env.gbest_fitness

def test_all(path, dims=[10], model='ppo_ea_model_v2'):
    for dim in dims:
        for function_id in range(1, 30):
            best_fitness = test_model(dim=dim, function_id=function_id, path = path,model_path=model)
            print(f"Function {function_id} ({dim}D) Best Fitness: {best_fitness}")
if __name__ == "__main__":


    # 测试单个函数
    # dim = 30
    # function_id = 9
    # best_fitness = test_model(dim=dim, function_id=function_id)
    # print(f"Function {function_id} ({dim}D) Best Fitness: {best_fitness}")
    # plt.show()
    # 测试所有函数
    dims = [10]
    path = "v3"
    test_all(path,dims, model = "ppo_ea_model_v3")

    # # explian
    # from explain.explian_utils import save_data
    # save_data()

    # from explain.shap_explian import shap_analysis
    # shap_analysis()

    pass 
