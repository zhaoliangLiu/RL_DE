from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from env import PSO_Proportional_Env
import matplotlib.pyplot as plt
import torch
from stable_baselines3.common.vec_env import SubprocVecEnv
import os
import numpy as np
 
def train_ppo(total_timesteps=1e4):
    env = make_vec_env(
        lambda: PSO_Proportional_Env(
            dim=10,
            max_iter=2000,
            num_function=50,
            start_function_id=1
        ),
        n_envs=8,  # 并行环境数量
        vec_env_cls=SubprocVecEnv,  # 使用多进程并行
    )

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=1024,  # 缩短轨迹长度
        batch_size=256,  # 增大批处理大小
        n_epochs=5,  # 减少epoch次数
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=0.2,
        ent_coef=0.005,  # 减少熵系数
        target_kl=None,  # 禁用target_kl限制
        policy_kwargs=dict(
            net_arch=dict(pi=[128, 128], vf=[128, 128]),  # 缩小网络规模
            activation_fn=torch.nn.ReLU,  # 使用更快的ReLU
            ortho_init=True  # 启用正交初始化
        ),
        device='cpu',  # 明确指定使用CPU
        verbose=1,
        seed=42
    )

    # 训练模型
    model.learn(
        total_timesteps=total_timesteps, 
        progress_bar=True
    )

    model.save("ea_ppo_model") 


 
 
if __name__ == "__main__":
    train_ppo(total_timesteps=3e6)  

