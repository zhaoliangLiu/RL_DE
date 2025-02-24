# rank_test.py
import pandas as pd
import numpy as np
from scipy.stats import friedmanchisquare
import scikit_posthocs as sp
import os
# 加入父路径
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
# 导入配置文件参数
from config import COMPARE_DATA_DIR, DIM_TEST, PATH_EXPLOIT_DEV # 导入显著性水平和指标列表, 导入维度常数

# 配置参数 (从配置文件读取)
dir_path = os.path.join(COMPARE_DATA_DIR, str(DIM_TEST) + 'D', f'{PATH_EXPLOIT_DEV}') # 使用配置中的数据目录和维度
significance_level = 0.01 
metrics = ['avg', 'best', 'worst', 'median', 'var'] 
def process_metric(metric):
    # 读取排名数据
    ranking_path = os.path.join(dir_path, f'{metric}_ranking.csv')
    df = pd.read_csv(ranking_path)

    # 提取算法名称
    algorithm_cols = [col.replace('_rank', '') for col in df.columns if '_rank' in col]

    # 准备检验数据矩阵
    data_matrix = []
    for _, row in df.iterrows():
        ranks = [row[f"{alg}_rank"] for alg in algorithm_cols]
        data_matrix.append(ranks)
    data_matrix = np.array(data_matrix)

    # Friedman检验
    stat, p_value = friedmanchisquare(*data_matrix.T)

    # 计算平均排名
    avg_ranks = dict(zip(algorithm_cols, data_matrix.mean(axis=0)))

    # 打印结果
    print(f"\n=== {metric.upper()} Metric ===")
    print(f"Friedman统计量: {stat:.4f}")
    print(f"P值: {p_value:.4f}")
    print("\n平均排名:")
    for alg, rank in sorted(avg_ranks.items(), key=lambda x: x[1]):
        print(f"{alg}: {rank:.4f}")

    
# 处理所有指标
for metric in metrics:
    process_metric(metric)
