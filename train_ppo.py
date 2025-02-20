# passtrain_ppo.py
import gym
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.policies import ActorCriticPolicy
import config # 导入配置文件

from env import PSO_Proportional_Env


def make_env():
    return PSO_Proportional_Env(
        dim=config.DIMENSION, # 使用配置文件中的维度
        population_size=config.POPULATION_SIZE, # 使用配置文件中的种群大小
        max_iter=config.MAX_ITERATIONS, # 使用配置文件中的最大迭代次数
        num_function=config.NUM_FUNCTION, # 使用配置文件中的函数数量
        start_function_id=config.START_FUNCTION_ID # 使用配置文件中的起始函数ID
    )

def train_ppo(total_timesteps=config.TOTAL_TIMESTEPS): # 默认使用配置文件中的训练总步数
    num_cpu = config.NUM_CPU # 使用配置文件中的 CPU 数量
    env = SubprocVecEnv([make_env for _ in range(num_cpu)])
    policy_kwargs = dict(
        net_arch=config.POLICY_NET_ARCH, # 使用配置文件中的网络结构
        activation_fn=torch.nn.ReLU
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=config.LEARNING_RATE, # 使用配置文件中的学习率
        n_steps=config.N_STEPS, # 使用配置文件中的 n_steps
        batch_size=config.BATCH_SIZE, # 使用配置文件中的 batch_size
        n_epochs=config.N_EPOCHS, # 使用配置文件中的 n_epochs
        gamma=config.GAMMA, # 使用配置文件中的 gamma
        gae_lambda=config.GAE_LAMBDA, # 使用配置文件中的 gae_lambda
        clip_range=config.CLIP_RANGE, # 使用配置文件中的 clip_range
        clip_range_vf=None,
        ent_coef=config.ENT_COEF, # 使用配置文件中的 ent_coef
        policy_kwargs=policy_kwargs
    )

    model.learn(total_timesteps=int(total_timesteps), progress_bar=True)
    model.save(config.MODEL_PATH) # 使用配置文件中的模型保存路径
    env.close()

if __name__ == "__main__":
    train_ppo(total_timesteps=config.TOTAL_TIMESTEPS) # 使用配置文件中的训练总步数
