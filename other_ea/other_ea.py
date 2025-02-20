# other_ea.py
import numpy as np
from typing import Callable, Union, Dict, Any
import pyade.de
import pyade.ilshade
import pyade
import pyade.jade
import pyade.lshade
import pyade.sade
import pyade.shade
# 加入父路径
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import  config # 导入配置文件

try:
    from opfunu import cec_based
    USE_OPFUNU = True
except ImportError:
    print("Warning: opfunu package not found.  CEC functions will not be available.")
    USE_OPFUNU = False

funcs = config.FUNCS # 使用配置文件中的函数字典
algorithm_map = config.ALGORITHM_MAP # 使用配置文件中的算法映射

def unified_optimization(dim: int, algorithm_name: str, fitness_function_id: int,
                          bounds: np.ndarray = None, max_evals: int = None,
                          **kwargs) :

    if algorithm_name not in algorithm_map:
        raise ValueError(f"Unknown algorithm: {algorithm_name}.  Choose from: {list(algorithm_map.keys())}")

    if fitness_function_id not in funcs:
        raise ValueError(f"Unknown function id: {fitness_function_id}.  Choose from: {list(funcs.keys())}")

    algorithm = algorithm_map[algorithm_name]
    params = algorithm.get_default_params(dim=dim)

    if bounds is None:
        params['bounds'] = np.array([[config.X_MIN, config.X_MAX]] * dim)  # 使用配置文件中的默认边界
    else:
        params['bounds'] = bounds

    if USE_OPFUNU:
        params['func'] =  cec_based.__dict__[funcs[fitness_function_id]](ndim=dim).evaluate
    else:
        print("Using a simple sphere function instead of CEC2017 functions.")
        params['func'] = lambda x: np.sum(x**2)

    if max_evals is None:
        params['max_evals'] = config.DIM_CONFIG[dim]['max_eval'] # 使用配置文件中的 max_evals
    else:
        params['max_evals'] = max_evals

    params.update(kwargs)

    try:
        solution, fitness, fitness_history = algorithm.apply(**params)
    except TypeError as e:
        print(f"TypeError calling apply() with **params: {e}")
        print(f"params passed to apply(): {params}")
        raise

    return solution, fitness, fitness_history

if __name__ == '__main__':
    dim = config.DIMENSION # 使用配置文件中的默认维度
    function_id = 9
    algorithm_name = "shade"

    solution, fitness, fitness_history = unified_optimization(
        dim=dim,
        algorithm_name=algorithm_name,
        fitness_function_id = function_id
    )

    print("Solution: ", solution)
    print("Fitness value: ", fitness)

    import matplotlib.pyplot as plt
    plt.plot(fitness_history)
    plt.xlabel("Evaluations")
    plt.ylabel("Fitness")
    plt.title(f"Fitness through evaluations ({algorithm_name.upper()})")
    plt.show()
