# 加入父路径

import sys
import os
import csv
import statistics
import concurrent.futures
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from other_ea import de, fade, rlpde
import config
from other_ea.utils import run_optimizer

def run_optimizer_wrapper(args):
    """Wrapper function for parallel processing"""
    alg, func_id, bounds, dim, max_eval, pop_size, i = args
    try:
        res = run_optimizer(alg, func_id, bounds, dim, max_eval, pop_size)
        return (func_id, alg, i, res['best_fitness'])
    except Exception as e:
        print(f"Error in {alg} on F{func_id}: {e}")
        return (func_id, alg, i, None)

def run_all_parallel(bounds=config.BOUNDS,
                     dim=config.DIMENSION,
                     max_eval=config.DIM_CONFIG[config.DIMENSION]['max_eval'],
                     pop_size=config.POPULATION_SIZE):
    algorithms = config.ALGORITHMS
    results = {}
    
    # Create task list for parallel processing
    tasks = []
    for func_id in config.FUNC_IDS_TEST:
        func_key = f"F{func_id}"
        results[func_key] = {alg: [] for alg in algorithms}
        
        for alg in algorithms:
            for i in range(config.REPEAT_TEST):
                if alg != "rlpde":  # Skip rlpde as it needs synchronous processing
                    tasks.append((alg, func_id, bounds, dim, max_eval, pop_size, i))

    # Use ProcessPoolExecutor for CPU-bound tasks
    with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count() // 2) as executor:
        # Process non-rlpde algorithms in parallel
        for func_id, alg, i, fitness in executor.map(run_optimizer_wrapper, tasks):
            if fitness is not None:
                func_key = f"F{func_id}"
                results[func_key][alg].append(fitness)
                print(f"Finished {alg} on {func_key}, repeat {i+1}")

    # Process rlpde separately (synchronously)
    for func_id in config.FUNC_IDS_TEST:
        func_key = f"F{func_id}"
        for i in range(config.REPEAT_TEST):
            try:
                res = run_optimizer("rlpde", func_id, bounds, dim, max_eval, pop_size)
                results[func_key]["rlpde"].append(res['best_fitness'])
                print(f"Finished rlpde on {func_key}, repeat {i+1} (synchronous)")
            except Exception as e:
                print(f"Error in rlpde on {func_key}: {e}")

    # Save results to CSV
    output_algs = config.ALGORITHMS
    header = ["function"]
    for alg in output_algs:
        header.extend([f"{alg}_avg", f"{alg}_best", f"{alg}_worst", f"{alg}_median", f"{alg}_var"])
    
    csv_filename = f"{dim}D/{config.PATH_EXPLOIT_DEV}/原始数据_{config.PATH_EXPLOIT_DEV}.csv"
    csv_filepath = os.path.join(config.COMPARE_DATA_DIR, csv_filename)
    os.makedirs(os.path.dirname(csv_filepath), exist_ok=True)
    
    with open(csv_filepath, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        for func, data in results.items():
            row = [func]
            for alg in output_algs:
                values = data[alg]
                avg = sum(values) / len(values) if values else None
                best = min(values) if values else None
                worst = max(values) if values else None
                median = statistics.median(values) if values else None
                var = statistics.variance(values) if len(values) > 1 else 0
                row.extend([avg, best, worst, median, var])
            writer.writerow(row)
    
    print(f"Results saved in {csv_filepath}")

# 绘图比较所有函数
def plot_all():
    # 参数设置
    dim = config.DIM_TEST
    bounds = config.BOUNDS
    max_eval = config.DIM_CONFIG[dim]['max_eval']
    pop_size = config.POPULATION_SIZE

    # 保存收敛数据的路径：data/compare/{dim}D/convergence/
    data_dir = os.path.join(config.COMPARE_DATA_DIR, f"{dim}D", "convergence",f'{config.PATH_EXPLOIT_DEV}')
    os.makedirs(data_dir, exist_ok=True)
    # 保存收敛图像的路径：graph/compare/{dim}D/convergence/
    graph_dir = os.path.join(config.COMPARE_GRAPH_DIR, f"{dim}D", "convergence",f'{config.PATH_EXPLOIT_DEV}')
    os.makedirs(graph_dir, exist_ok=True)

    # 设置论文级别的科研风格
    plt.style.use('seaborn-v0_8-whitegrid')    # 定义各算法的颜色（可根据需求调整）
    colors = {
        "fade": "red",
        "sade": "green",
        "apdsde": "purple",
        "rlpde": "blue"
    }

    # 对每个测试函数执行
    for func_id in config.FUNC_IDS_TEST:
        convergence_data = {}  # 存放各算法的收敛曲线（列表形式）
        max_length = 0
        # 针对 config.ALGORITHMS，每个算法运行一次获取收敛历史
        for alg in config.ALGORITHMS:
            result = run_optimizer(alg, func_id, bounds, dim, max_eval, pop_size)
            # history 可能为 numpy 数组或列表
            history = result["history"]
            if isinstance(history, np.ndarray):
                history = history.tolist()
            convergence_data[alg] = history
            max_length = max(max_length, len(history))
        
        # 如果各算法的长度不一致，则用最后一个值填充至相同长度
        for alg in convergence_data:
            curve = convergence_data[alg]
            if len(curve) < max_length:
                curve.extend([curve[-1]] * (max_length - len(curve)))
                convergence_data[alg] = curve

        # 保存收敛数据CSV，每个函数一个文件，第一列为 1~max_length 的迭代序号
        csv_filename = f"F{func_id}.csv"
        csv_filepath = os.path.join(data_dir, csv_filename)
        header = ["Iteration"] + config.ALGORITHMS
        with open(csv_filepath, "w", newline="") as fcsv:
            writer = csv.writer(fcsv)
            writer.writerow(header)
            for i in range(max_length):
                row = [i + 1]
                for alg in config.ALGORITHMS:
                    row.append(convergence_data[alg][i])
                writer.writerow(row)
        print(f"Convergence CSV saved: {csv_filepath}")

        # 绘制收敛曲线图
        plt.figure(figsize=(8, 6))
        for alg in config.ALGORITHMS:
            # 如果是 rlpde 算法则加粗并使用圆点标记，其它算法保持常规样式
            if (alg == "rlpde"):
                plt.plot(
                    range(1, max_length + 1),
                    convergence_data[alg],
                    label=alg,
                    color=colors.get(alg, None),
                    linewidth=3,
                    marker='o',
                    
                    markersize=5
                )
            else:
                plt.plot(
                    range(1, max_length + 1),
                    convergence_data[alg],
                    label=alg,
                    color=colors.get(alg, None),
                    linewidth=2
                )

        plt.xlabel("Evaluations", fontsize=12)
        plt.ylabel("Best Fitness", fontsize=12)
        plt.yscale("log")  # 对数放缩，使 y 轴以对数刻度显示
        plt.title(f"Convergence Curve for Function F{func_id}", fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, linestyle="--", linewidth=0.5)
        # 保存图像，dpi设为300保证论文级别清晰
        fig_filename = f"F{func_id}.png"
        fig_filepath = os.path.join(graph_dir, fig_filename)
        plt.savefig(fig_filepath, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Convergence plot saved: {fig_filepath}")

    
    pass
if __name__ == "__main__":
    from train_ppo import train_ppo
    # train_ppo()
    run_all_parallel()
    import process.process_data 
    
    plot_all()
    from test_rl_ea import test_all
    # test_all()
    
    
    import explain.explian_utils  
    from explain.shap_explian import shap_analysis
    shap_analysis()
    
    pass


