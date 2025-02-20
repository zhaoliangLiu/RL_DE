# config.py
import os
import pyade
import pyade.de
import pyade.jade
import pyade.lshade
import pyade.shade

# 模型和路径配置
MODEL_PATH = "ppo_ea_model_v0"
EXPLAIN_DATA_DIR = "data/explain"
EXPLAIN_GRAPH_DIR = "graph/explain"
COMPARE_DATA_DIR = "data/compare"
COMPARE_GRAPH_DIR = "graph/compare"
EXPLOIT_DEV_DATA_DIR = "data/勘探开发"
EXPLOIT_DEV_GRAPH_DIR = "graph/勘探开发"
CONVERGENCE_DIR_NAME = "convergence"

# 维度配置
DIM_CONFIG = {
    10: {'pop_size': 50, 'max_eval': 10 * 1000},
    30: {'pop_size': 100, 'max_eval': 30 * 1000},
    50: {'pop_size': 150, 'max_eval': 50 * 1000},
    100: {'pop_size': 200, 'max_eval': 100 * 1000}
}

# 算法和函数配置
FUNCS = {
    1: "F12017",    2: "F22017",
    3: "F32017",    4: "F42017",
    5: "F52017",    6: "F62017",
    7: "F72017",    8: "F82017",
    9: "F92017",    10: "F102017",
    11: "F112017",  12: "F12017",
    13: "F132017",  14: "F142017",
    15: "F152017",  16: "F162017",
    17: "F172017",  18: "F182017",
    19: "F192017",  20: "F202017",
    21: "F212017", 22: "F222017",
    23: "F232017", 24: "F242017",
    25: "F252017", 26: "F262017",
    27: "F272017", 28: "F282017",
    29: "F292017"
}
ALGORITHM_MAP = {
    "de": pyade.de,
    "jade": pyade.jade,
    "lshade": pyade.lshade,
    "shade": pyade.shade
}
ALGORITHM_NAMES = ["rl", "de", "jade", "lshade", "shade"]
# ALGORITHM_NAMES = ["rl"]

ALGORITHM_STYLES = {
    "de": {"color": "blue", "linestyle": "-"},
    "jade": {"color": "red", "linestyle": "--"},
    "lshade": {"color": "green", "linestyle": "-."},
    "shade": {"color": "purple", "linestyle": ":"},
    "rl": {"color": "red", "linestyle": "--"}
}

# 环境参数 (PSO_Proportional_Env 默认值)
POPULATION_SIZE = 50 # 粒子群大小
DIMENSION = 10 # 维度
MAX_ITERATIONS = 200    # 最大迭代次数
MEMORY_SIZE = 100   # 记忆大小
X_MIN = -100.0 # 粒子位置最小值
X_MAX = 100.0 # 粒子位置最大值
P_MIN = 0.05 # 粒子速度最小值 
NUM_FUNCTION = 1 # 函数数量
START_FUNCTION_ID = 1 # 起始函数ID 

# 训练参数 (passtrain_ppo.py) 
NUM_CPU = 4 # CPU 数量 
TOTAL_TIMESTEPS = 3e1 # 总步数 
LEARNING_RATE = 3e-4 # 学习率 
N_STEPS = 2048 # n_steps 
BATCH_SIZE = 64 # batch_size 
N_EPOCHS = 10 # n_epochs 
GAMMA = 0.99 # gamma 
GAE_LAMBDA = 0.95  # gae_lambda 
CLIP_RANGE = 0.2 # clip_range 
ENT_COEF = 0.0 # ent_coef 
POLICY_NET_ARCH = [dict(pi=[256, 256, 128], vf=[256, 256, 128])]

# 测试参数
DIM_TEST = 10   # 测试维度 
REPEAT_TEST = 2   # 重复次数 
FUNC_IDS_TEST = range(1, 3)        # 测试函数ID
DIMS_TEST_EXPLAIN = [10, 30, 50, 100]  # 解释测试维度    
DIMS_TEST_EXPLOIT_DEV = [10, 30, 50, 100] # 勘探开发测试维度 
PATH_EXPLOIT_DEV = "v0" # 勘探开发测试路径 
EXPLAIN_SAMPLE = 200 # 解释样本数

