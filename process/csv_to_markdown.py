# csv_to_markdown.py
import csv
import os
import sys
# 导入配置文件参数
from config import CSV_TO_MARKDOWN_CSV_FILE

# 将父目录加入
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))


def csv_to_markdown_scientific_from_file(csv_file_path):
    """
    从 CSV 文件读取数据，并将不包含 "rank" 的数字列格式化为科学计数法后转换为 Markdown 表格。

    Args:
        csv_file_path: CSV 文件的路径。

    Returns:
        Markdown 格式的字符串。如果发生错误，则返回错误消息。
    """

    try:
        with open(csv_file_path, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)

            # 格式化表头
            markdown_output = "| " + " | ".join(header) + " |\n"
            markdown_output += "| " + " | ".join(["---"] * len(header)) + " |\n"

            # 格式化数据行
            for row in reader:
                formatted_row = []
                for i, item in enumerate(row):
                    if "rank" not in header[i].lower(): # 忽略大小写，检查列名是否包含 "rank"
                        try:
                            # 尝试将数据转换为浮点数，如果成功则格式化为科学计数法
                            num = float(item)
                            formatted_item = "{:.2e}".format(num)  # 格式化为科学计数法，保留两位小数
                        except ValueError:
                            # 如果不能转换为浮点数，则保持原样
                            formatted_item = item
                    else:
                        formatted_item = item  # "rank" 列保持原样

                    formatted_row.append(formatted_item)
                markdown_output += "| " + " | ".join(formatted_row) + " |\n"

            return markdown_output

    except FileNotFoundError:
        return f"错误：文件 '{csv_file_path}' 未找到。"
    except Exception as e:
        return f"发生错误: {e}"

# 获取 CSV 文件路径（从配置文件读取）
csv_file = CSV_TO_MARKDOWN_CSV_FILE # 使用配置中的 CSV 文件路径

# 转换为 Markdown 并输出
markdown_table = csv_to_markdown_scientific_from_file(csv_file)
print(markdown_table)
