# config.py
import os
 
from cec2017.functions import all_functions

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
    10: {'pop_size': 50, 'max_eval': 10 * 10000},
    30: {'pop_size': 100, 'max_eval': 30 * 10000},
    50: {'pop_size': 150, 'max_eval': 50 * 10000},
    100: {'pop_size': 200, 'max_eval': 100 * 10000}
}

# 算法和函数配置
 
# ALGORITHMS = ['de', 'fade', 'sade', 'apdsde', 'rlpde']
ALGORITHMS = ['fade', 'sade', 'apdsde', 'rlpde']

# ALGORITHM_NAMES = ["rlpde"]

ALGORITHM_STYLES = {
    "de": {"color": "blue", "linestyle": "-"},
    "fade": {"color": "red", "linestyle": "--"},
    "sade": {"color": "green", "linestyle": "-."},
    "shapdsadeade": {"color": "purple", "linestyle": ":"},
    "rl": {"color": "red", "linestyle": "--"}
}

# 环境参数 (PSO_Proportional_Env 默认值)
POPULATION_SIZE = 50 # 粒子群大小
DIMENSION = 10 # 维度
MAX_ITERATIONS = 2000    # 最大迭代次数
MEMORY_SIZE = 100   # 记忆大小
X_MIN = -100.0 # 粒子位置最小值
X_MAX = 100.0 # 粒子位置最大值
P_MIN = 0.05 # 粒子速度最小值 
NUM_FUNCTION = 1 # 函数数量
START_FUNCTION_ID = 1 # 起始函数ID 

# 训练参数 (passtrain_ppo.py) 
NUM_CPU = 3 # CPU 数量 
TOTAL_TIMESTEPS = 1e6 # 总步数 
LEARNING_RATE = 1e-4 # 学习率 
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
REPEAT_TEST = 30   # 重复次数 
FUNC_IDS_TEST = range(1, 31)        # 测试函数ID
DIMS_TEST_EXPLAIN = [10]  # 解释测试维度    
DIMS_TEST_EXPLOIT_DEV = [10] # 勘探开发测试维度 
PATH_EXPLOIT_DEV = "v1" # 勘探开发测试路径  (不能改为v1！！！！！！！！！！！！！！！！！！！！！！！！！)
EXPLAIN_SAMPLE = 5000 # 解释样本数
BOUNDS = [-100, 100] # 搜索范围
FUNC_MAP = {i: all_functions[i - 1] for i in FUNC_IDS_TEST}