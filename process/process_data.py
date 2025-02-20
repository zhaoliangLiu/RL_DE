# process_data.py
import pandas as pd
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import config 

# 读取CSV文件
csv_file_path = os.path.join(config.COMPARE_DATA_DIR, str(config.DIM_TEST) + 'D', f'原始数据{config.PATH_EXPLOIT_DEV}.csv')
df = pd.read_csv(csv_file_path)

# 定义统计指标
metrics = ['best', 'worst', 'mean', 'std', 'median']

for metric in metrics:
    # 创建新的DataFrame并保留Function列
    result_df = pd.DataFrame()
    result_df['Function'] = df['Function']
    
    # 筛选当前指标对应的所有列
    metric_columns = []
    for algo in config.ALGORITHM_NAMES:
        prefix = 'rlde' if algo == 'rl' else algo
        col_name = f'{prefix}_{metric}'
        if col_name in df.columns:
            metric_columns.append(col_name)
    
    # 计算当前指标的排名（使用密集排名，数值越小排名越高）
    ranks = df[metric_columns].rank(axis=1, method='dense', ascending=True).astype(int)
    
    # 处理每个算法列
    for col in metric_columns:
        # 格式化数值为科学计数法
        formatted_values = df[col].map(lambda x: f"{x:.2e}")
        # 添加原始值和排名列
        result_df[col] = formatted_values
        result_df[f"{col}_rank"] = ranks[col]
    
    # 保存结果到CSV文件
    output_csv_path = os.path.join(config.COMPARE_DATA_DIR, str(config.DIM_TEST) + 'D', f'{metric}_ranking.csv')
    result_df.to_csv(output_csv_path, index=False)

print("处理完成，已生成排名文件：")
for metric in metrics:
    print(f"- {metric}_ranking.csv")
