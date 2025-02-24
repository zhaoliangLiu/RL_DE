# shap_explian.py
import shap
import torch
from stable_baselines3 import PPO
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import config # 导入配置文件
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-whitegrid')    # 定义各算法的颜色（可根据需求调整）

# 加载模型并提取策略网络
model = PPO.load(config.MODEL_PATH) # 使用配置文件中的模型路径
policy_net = model.policy

def model_wrapper(X):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        tensor_X = torch.FloatTensor(X).to(device)
        actions, _, _ = policy_net(tensor_X)
    return actions.cpu().numpy().flatten()

def preprocess_data(data_path):
    import pandas as pd
    from sklearn.preprocessing import StandardScaler

    # 使用 index_col=0 避免读入 CSV 中的索引列
    df = pd.read_csv(data_path, index_col=0)
    df = df.sample(n=config.EXPLAIN_SAMPLE, random_state=43)
    # 确保 action 列被删除后，仅保留模型所需的特征
    features = df.drop(columns=['action'])
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    return scaled_features, features.columns, scaler

import matplotlib.pyplot as plt
def shap_analysis(): 
    plt.rcParams['axes.linewidth'] = 1.5
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3

    BLUE = '#2878B5'
    RED = '#C82423'

    dim = config.DIMS_TEST_EXPLAIN # 使用配置文件中的维度列表

    for j, path in enumerate(dim):
        X, feature_names, scaler = preprocess_data(f'{config.EXPLAIN_DATA_DIR}/{path}D_explain_data_{config.PATH_EXPLOIT_DEV}.csv') # 使用配置文件中的数据路径，注意路径是否需要调整

        explainer = shap.KernelExplainer(
            model_wrapper,
            shap.sample(X, 100)
        )

        shap_values = explainer.shap_values(X)

        # 1. 全局特征重要性图
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_values,
            X,
            feature_names=feature_names,
            show=False,
            color=BLUE,
            alpha=0.8
        )
        plt.title(f"Global Feature Importance ({dim[j]}D)",
                 fontsize=14, fontweight='bold', pad=20)
        plt.xlabel("SHAP value", fontsize=12, fontweight='bold')
        plt.ylabel("Feature", fontsize=12, fontweight='bold')
        plt.tight_layout()
        out_dir = os.path.join(config.EXPLAIN_GRAPH_DIR, f'{path}D', config.PATH_EXPLOIT_DEV)
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(os.path.join(out_dir, f'{path}D_global_importance.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

        # 2. 单特征依赖图
        for i, feature in enumerate(feature_names):
            plt.figure(figsize=(8, 6))

            feature_vals = X[:, i]
            feature_shap = shap_values[:, i]

            sort_idx = np.argsort(feature_vals)
            feature_vals = feature_vals[sort_idx]
            feature_shap = feature_shap[sort_idx]

            plt.scatter(feature_vals, feature_shap,
                       c=BLUE,
                       alpha=0.6,
                       s=30)

            z = np.polyfit(feature_vals, feature_shap, 3)
            p = np.poly1d(z)
            x_trend = np.linspace(min(feature_vals), max(feature_vals), 100)
            plt.plot(x_trend, p(x_trend),
                    color=RED,
                    linestyle='--',
                    linewidth=2.5,
                    alpha=0.8)

            plt.xlabel(feature, fontsize=12, fontweight='bold')
            plt.ylabel('SHAP value', fontsize=12, fontweight='bold')
            plt.title(f'Feature Impact: {feature} ({dim[j]}D)',
                     fontsize=14, fontweight='bold', pad=20)

            plt.grid(True, linestyle='--', alpha=0.3)
            plt.tight_layout()
            # 将依赖图保存在维度文件夹下 config.PATH_EXPLOIT_DEV 子文件夹中
            plt.savefig(os.path.join(out_dir, f'{path}D_dependence_{feature}.png'),
                       dpi=300, bbox_inches='tight')
            plt.close()

        # 3. 特征重要性条形图
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_values,
            X,
            feature_names=feature_names,
            plot_type="bar",
            show=False,
            color=BLUE
        )
        plt.title(f"Feature Importance Ranking ({dim[j]}D)",
                 fontsize=14, fontweight='bold', pad=20)
        plt.xlabel("mean(|SHAP value|)", fontsize=12, fontweight='bold')
        plt.ylabel("Feature", fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'{path}D_importance_ranking.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    shap_analysis()
