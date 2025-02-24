# process_data.py
import pandas as pd
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
import config 

# 读取CSV文件，注意CSV中的第一列为小写 'function'
csv_file_path = os.path.join(
    config.COMPARE_DATA_DIR,
    str(config.DIM_TEST) + "D",
    config.PATH_EXPLOIT_DEV,
    f"原始数据_{config.PATH_EXPLOIT_DEV}.csv"
)
df = pd.read_csv(csv_file_path)

# 新的指标列表（CSV中使用avg和var）
metrics = ['avg', 'best', 'worst', 'median', 'var']

for metric in metrics:
    # 创建新的DataFrame并保留function列
    result_df = pd.DataFrame()
    # 保留原始函数列（不区分大小写）
    if 'Function' in df.columns:
        result_df['Function'] = df['Function']
    elif 'function' in df.columns:
        result_df['Function'] = df['function']
    else:
        raise ValueError("未在CSV中找到 'Function' 或 'function' 列")
    
    # 筛选当前指标对应的所有列，使用 config.ALGORITHMS
    metric_columns = []
    for algo in config.ALGORITHMS:
        # CSV中列名格式为 f"{algo}_{metric}"
        col_name = f"{algo}_{metric}"
        if col_name in df.columns:
            metric_columns.append(col_name)
    
    if not metric_columns:
        print(f"未找到与指标 {metric} 相关的列，跳过...")
        continue
    
    # 计算当前指标的排名（使用密集排名，数值越小排名越高）
    ranks = df[metric_columns].rank(axis=1, method="dense", ascending=True).astype(int)
    
    # 将每个算法的原始值（格式化为科学计数法）及排名加入新的DataFrame
    for col in metric_columns:
        formatted_values = df[col].map(lambda x: f"{x:.2e}")
        result_df[col] = formatted_values
        result_df[f"{col}_rank"] = ranks[col]
    
    # 保存结果到CSV文件，新文件名为 "{metric}_ranking.csv"
    output_dir = os.path.join(
        config.COMPARE_DATA_DIR,
        str(config.DIM_TEST) + "D",
        config.PATH_EXPLOIT_DEV
    )
    os.makedirs(output_dir, exist_ok=True)
    output_csv_path = os.path.join(output_dir, f"{metric}_ranking.csv")
    result_df.to_csv(output_csv_path, index=False)

print("处理完成，已生成排名文件：")
for metric in metrics:
    print(f"- {metric}_ranking.csv")
