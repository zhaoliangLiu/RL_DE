# test_ea_virant.py
import numpy as np
import pyade.de
import pyade.jade
import pyade.lshade
import pyade.shade
import matplotlib.pyplot as plt
import config # 导入配置文件

try:
    from opfunu import cec_based
    USE_OPFUNU = True
except ImportError:
    print("Warning: opfunu package not found.  CEC functions will not be available.")
    USE_OPFUNU = False

funcs = config.FUNCS # 使用配置文件中的函数字典
algorithm_map = config.ALGORITHM_MAP # 使用配置文件中的算法映射
algorithm_map["de"] = pyade.de # 确保 algorithm_map 中的值正确设置，如果 config.py 中未初始化
algorithm_map["jade"] = pyade.jade
algorithm_map["lshade"] = pyade.lshade
algorithm_map["shade"] = pyade.shade

def unified_optimization(dim: int, algorithm_name: str, fitness_function_id: int,
                         bounds: np.ndarray = None, max_evals: int = None,
                         **kwargs) -> [np.ndarray, float, list]:

    if algorithm_name not in algorithm_map:
        raise ValueError(
            f"Unknown algorithm: {algorithm_name}.  Choose from: {list(algorithm_map.keys())}")

    if fitness_function_id not in funcs:
        raise ValueError(f"Unknown function id: {fitness_function_id}.  Choose from: {list(funcs.keys())}")

    algorithm = algorithm_map[algorithm_name]
    params = algorithm.get_default_params(dim=dim)

    if bounds is None:
        params['bounds'] = np.array([[config.X_MIN, config.X_MAX]] * dim)  # 使用配置文件中的默认边界
    else:
        params['bounds'] = bounds

    params['func'] = cec_based.__dict__[funcs[fitness_function_id]](ndim=dim).evaluate

    if max_evals is None:
        params['max_evals'] = config.DIM_CONFIG[dim]['max_eval'] # 使用配置文件中的 max_evals
    else:
        params['max_evals'] = max_evals

    params.update(kwargs)
    solution, fitness, fitness_history = algorithm.apply(**params)

    return solution, fitness, fitness_history

if __name__ == '__main__':
    dim = 100
    function_id = 9
    algorithm_names = config.ALGORITHM_NAMES[1:] # 使用配置文件中的算法名称列表 (去除 'rl')

    algorithm_styles = config.ALGORITHM_STYLES # 使用配置文件中的算法样式

    plt.figure(figsize=(10, 6))

    for algorithm_name in algorithm_names:
        solution, fitness, fitness_history = unified_optimization(
            dim=dim,
            algorithm_name=algorithm_name,
            fitness_function_id=function_id
        )

        print(f"Algorithm: {algorithm_name.upper()}")
        print("Solution: ", solution)
        print("Fitness value: ", fitness)

        style = algorithm_styles.get(algorithm_name)
        if style:
            plt.plot(fitness_history, label=algorithm_name.upper(), **style)
        else:
            plt.plot(fitness_history, label=algorithm_name.upper())

    plt.xlabel("Evaluations", fontsize=12)
    plt.ylabel("Fitness", fontsize=12)
    plt.title(f"Convergence Comparison (Function {funcs[function_id]})", fontsize=14)

    plt.yscale("log")
    plt.grid(True, which="both", linestyle='--', linewidth=0.5)

    plt.legend(loc="upper right", fontsize=10)

    plt.tight_layout()

    filename = f"comparison_{funcs[function_id]}_dim{dim}_paper_style.png"
    plt.savefig(filename, dpi=300)
    print(f"Comparison plot saved to {filename}")

    plt.show()
