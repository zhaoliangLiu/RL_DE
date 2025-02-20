import pandas as pd
import numpy as np
from scipy.stats import friedmanchisquare
import scikit_posthocs as sp
import os

# 配置参数
dir_path = 'data/compare/10D'
significance_level = 0.05
metrics = ['best', 'worst', 'mean', 'std', 'median']

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
    
    # 如果Friedman检验显著，执行并打印Nemenyi检验结果
    if p_value < significance_level:
        print("\nNemenyi后续检验 P值矩阵:")
        nemenyi_result = sp.posthoc_nemenyi_friedman(data_matrix)
        print(nemenyi_result.round(4))

# 处理所有指标
for metric in metrics:
    process_metric(metric)