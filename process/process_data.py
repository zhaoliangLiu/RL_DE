import pandas as pd

# 读取CSV文件（假设文件名为input.csv）
df = pd.read_csv('data/compare/10D/compact_stats.csv')

# 定义需要处理的指标列表
metrics = ['best', 'worst', 'mean', 'std', 'median']

for metric in metrics:
    # 筛选当前指标对应的所有列
    metric_columns = [col for col in df.columns if col.endswith(f'_{metric}')]
    
    # 创建新的DataFrame并保留Function列
    result_df = pd.DataFrame()
    result_df['Function'] = df['Function']
    
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
    result_df.to_csv(f'data/compare/10D/{metric}_ranking.csv', index=False)

print("处理完成，已生成5个排名文件：best_ranking.csv, worst_ranking.csv, mean_ranking.csv, std_ranking.csv, median_ranking.csv")
