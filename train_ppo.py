import gym
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.policies import ActorCriticPolicy

from env import PSO_Proportional_Env


def make_env():
    return PSO_Proportional_Env(
        dim=10,
        population_size=50,
        max_iter=2000,
        num_function=1,
        start_function_id=1
    )

def train_ppo(total_timesteps=1e6):
    num_cpu = 4  
    env = SubprocVecEnv([make_env for _ in range(num_cpu)])   
    policy_kwargs = dict(
        net_arch=[dict(pi=[256, 256, 128], vf=[256, 256, 128])],  
        activation_fn=torch.nn.ReLU   
    )

    model = PPO(
        "MlpPolicy", 
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        ent_coef=0.0,
        policy_kwargs=policy_kwargs  
    )

    model.learn(total_timesteps=int(total_timesteps), progress_bar=True)
    model.save("ppo_ea_model_v3")  
    env.close()

if __name__ == "__main__":
    train_ppo(total_timesteps=3e6)
