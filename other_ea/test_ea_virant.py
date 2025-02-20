import numpy as np
import pyade.de
import pyade.jade
import pyade.lshade
import pyade.shade
import matplotlib.pyplot as plt

# You may need to install opfunu package
try:
    from opfunu import cec_based
    USE_OPFUNU = True
except ImportError:
    print("Warning: opfunu package not found.  CEC functions will not be available.")
    USE_OPFUNU = False

# func 存储函数名字
funcs = {
    1: "F12017", 2: "F22017",
    3: "F32017", 4: "F42017",
    5: "F52017", 6: "F62017",
    7: "F72017", 8: "F82017",
    9: "F92017", 10: "F102017",
    11: "F112017", 12: "F122017",
    13: "F132017", 14: "F142017",
    15: "F152017", 16: "F162017",
    17: "F172017", 18: "F182017",
    19: "F192017", 20: "F202017",
    21: "F212017", 22: "F222017",
    23: "F232017", 24: "F242017",
    25: "F252017", 26: "F262017",
    27: "F272017", 28: "F282017",
    29: "F292017"
}

algorithm_map = {
    "de": pyade.de,
    "jade": pyade.jade,
    "lshade": pyade.lshade,
    "shade": pyade.shade
}


def unified_optimization(dim: int, algorithm_name: str, fitness_function_id: int,
                         bounds: np.ndarray = None, max_evals: int = None,
                         **kwargs) -> [np.ndarray, float, list]:
    """
    Unified optimization function for various differential evolution algorithms from pyade.

    :param dim: Dimensionality of the problem.
    :param algorithm_name: Name of the algorithm to use (de, jade, lshade, shade).
    :param fitness_function_id: id of the fitness_function 1-29.
    :param bounds: (Optional) Numpy array of shape (dim, 2) specifying the lower and upper bounds for each dimension.
    If None, defaults to [-100, 100] for each dimension.
    :param max_evals: (Optional) Maximum number of fitness function evaluations. If None, defaults to 10000 * dim.
    :param kwargs:  Other keyword arguments to be passed to the selected algorithm's `apply` function.
    :return: A tuple containing:
    - Best solution found (NumPy array).
    - Best fitness value (float).
    - List of fitness history values (list of floats).
    """

    if algorithm_name not in algorithm_map:
        raise ValueError(
            f"Unknown algorithm: {algorithm_name}.  Choose from: {list(algorithm_map.keys())}")

    if fitness_function_id not in funcs:
        raise ValueError(f"Unknown function id: {fitness_function_id}.  Choose from: {list(funcs.keys())}")

    algorithm = algorithm_map[algorithm_name]
    params = algorithm.get_default_params(dim=dim)

    # Override default parameters with function arguments
    if bounds is None:
        params['bounds'] = np.array([[-100, 100]] * dim)  # Default bounds
    else:
        params['bounds'] = bounds

    # 如果成功import cec_based包, 适应度函数使用cec_based,否则使用lambda 函数
    
    params['func'] = cec_based.__dict__[funcs[fitness_function_id]](ndim=dim).evaluate
    

    if max_evals is None:
        params['max_evals'] = 10000 * dim
    else:
        params['max_evals'] = max_evals

    # Add any extra kwargs to params
    params.update(kwargs)

    # Run the algorithm.  This try-except handles the different signatures of apply() in pyade.
    
    solution, fitness, fitness_history = algorithm.apply(**params)
  

    return solution, fitness, fitness_history


if __name__ == '__main__':
    dim = 100
    function_id = 9
    algorithm_names = ["de", "jade", "lshade", "shade"]  # List of algorithms to compare

    # Define colors and linestyles for each algorithm
    algorithm_styles = {
        "de": {"color": "blue", "linestyle": "-"},
        "jade": {"color": "red", "linestyle": "--"},
        "lshade": {"color": "green", "linestyle": "-."},
        "shade": {"color": "purple", "linestyle": ":"},
    }

    plt.figure(figsize=(10, 6))  # Adjust figure size for better readability

    # Run each algorithm and plot its fitness history
    for algorithm_name in algorithm_names:
        solution, fitness, fitness_history = unified_optimization(
            dim=dim,
            algorithm_name=algorithm_name,
            fitness_function_id=function_id
        )

        print(f"Algorithm: {algorithm_name.upper()}")
        print("Solution: ", solution)
        print("Fitness value: ", fitness)

        # Apply styling based on the algorithm
        style = algorithm_styles.get(algorithm_name)
        if style:
            plt.plot(fitness_history, label=algorithm_name.upper(), **style)
        else:
            plt.plot(fitness_history, label=algorithm_name.upper())  # Default style

    plt.xlabel("Evaluations", fontsize=12)  # Increase font size
    plt.ylabel("Fitness", fontsize=12)  # Increase font size
    plt.title(f"Convergence Comparison (Function {funcs[function_id]})", fontsize=14)  # Clear title

    plt.yscale("log")  # Use a logarithmic scale for the y-axis
    plt.grid(True, which="both", linestyle='--', linewidth=0.5)  # Add a grid

    plt.legend(loc="upper right", fontsize=10)  # Show legend, adjust location

    plt.tight_layout()  # Adjust layout to prevent labels from overlapping

    # Saving the plot to a file
    filename = f"comparison_{funcs[function_id]}_dim{dim}_paper_style.png"
    plt.savefig(filename, dpi=300)  # Saves the combined plot as a high-resolution PNG
    print(f"Comparison plot saved to {filename}")

    plt.show()  # Display the plot
