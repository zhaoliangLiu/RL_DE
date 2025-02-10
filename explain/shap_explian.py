import shap
import torch
from stable_baselines3 import PPO
import numpy as np
import os
# 将父目录加入
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))


# 加载模型并提取策略网络
model = PPO.load("ea_ppo_model")
policy_net = model.policy

def model_wrapper(X):
    """SHAP专用包装函数
    输入: (n_samples, n_features)的numpy数组
    输出: (n_samples,)的动作值
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        tensor_X = torch.FloatTensor(X).to(device)
        actions, _, _ = policy_net(tensor_X)
    return actions.cpu().numpy().flatten()


def preprocess_data(data_path):
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    
    df = pd.read_csv(data_path)
    # 随机1000个样本
    df = df.sample(n=50, random_state=43)
    features = df.drop(columns=['action'])
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    return scaled_features, features.columns, scaler

import matplotlib.pyplot as plt
def shap_analysis():
    # 设置全局绘图样式
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['axes.linewidth'] = 1.5
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    
    # 定义配色方案
    BLUE = '#2878B5'  # 主要蓝色
    RED = '#C82423'   # 主要红色
     
    dim = [10, 30, 50, 100]

    for j, path in enumerate(dim):
        # 加载预处理数据
        X, feature_names, scaler = preprocess_data(f'data/explain/{path}D_explain_data.csv')
        
        # 初始化解释器
        explainer = shap.KernelExplainer(
            model_wrapper, 
            shap.sample(X, 100)  # 减少背景样本数量以提高速度
        )
        
        # 计算SHAP值
        shap_values = explainer.shap_values(X)

        # 1. 全局特征重要性图
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_values, 
            X, 
            feature_names=feature_names, 
            show=False,
            color=BLUE,  # 使用蓝色主题
            alpha=0.8
        )
        plt.title(f"Global Feature Importance ({dim[j]}D)", 
                 fontsize=14, fontweight='bold', pad=20)
        plt.xlabel("SHAP value", fontsize=12, fontweight='bold')
        plt.ylabel("Feature", fontsize=12, fontweight='bold')
        plt.tight_layout()
        # 如果不存在
        if not os.path.exists(f'graph/explain/{path}D'):
            os.makedirs(f'graph/explain/{path}D')
        plt.savefig(f'graph/explain/{path}D/{path}D_global_importance.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()

        # 2. 单特征依赖图
        for i, feature in enumerate(feature_names):
            plt.figure(figsize=(8, 6))
            
            # 创建简化的依赖图
            feature_vals = X[:, i]
            feature_shap = shap_values[:, i]
            
            # 排序以创建平滑的图
            sort_idx = np.argsort(feature_vals)
            feature_vals = feature_vals[sort_idx]
            feature_shap = feature_shap[sort_idx]
            
            # 绘制散点图
            plt.scatter(feature_vals, feature_shap, 
                       c=BLUE,  # 蓝色散点
                       alpha=0.6, 
                       s=30)
            
            # 添加趋势线
            z = np.polyfit(feature_vals, feature_shap, 3)
            p = np.poly1d(z)
            x_trend = np.linspace(min(feature_vals), max(feature_vals), 100)
            plt.plot(x_trend, p(x_trend), 
                    color=RED,  # 红色趋势线
                    linestyle='--', 
                    linewidth=2.5, 
                    alpha=0.8)

            plt.xlabel(feature, fontsize=12, fontweight='bold')
            plt.ylabel('SHAP value', fontsize=12, fontweight='bold')
            plt.title(f'Feature Impact: {feature} ({dim[j]}D)', 
                     fontsize=14, fontweight='bold', pad=20)
            
            plt.grid(True, linestyle='--', alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'graph/explain/{path}D/{path}D_dependence_{feature}.png', 
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
            color=BLUE  # 蓝色条形图
        )
        plt.title(f"Feature Importance Ranking ({dim[j]}D)", 
                 fontsize=14, fontweight='bold', pad=20)
        plt.xlabel("mean(|SHAP value|)", fontsize=12, fontweight='bold')
        plt.ylabel("Feature", fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'graph/explain/{path}D/{path}D_importance_ranking.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    shap_analysis()