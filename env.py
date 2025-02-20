# env.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces
import gym
import numpy as np
from gym import spaces
import scipy.stats as stats
from opfunu import cec_based
import torch
import config # 导入配置文件


torch.set_num_threads(8)
torch.set_flush_denormal(True)

class PSO_Proportional_Env(gym.Env):
    def __init__(
        self,
        population_size=config.POPULATION_SIZE, # 使用配置文件中的种群大小
        dim=config.DIMENSION, # 使用配置文件中的维度
        max_iter=config.MAX_ITERATIONS, # 使用配置文件中的最大迭代次数
        memory_size=config.MEMORY_SIZE, # 使用配置文件中的记忆大小
        x_min=config.X_MIN, # 使用配置文件中的 x_min
        x_max=config.X_MAX, # 使用配置文件中的 x_max
        p_min=config.P_MIN, # 使用配置文件中的 p_min
        num_function=config.NUM_FUNCTION, # 使用配置文件中的函数数量
        start_function_id=config.START_FUNCTION_ID # 使用配置文件中的起始函数ID
    ):
        super(PSO_Proportional_Env, self).__init__()

        self.population_size = population_size
        self.dim = dim
        self.max_iter = max_iter
        self.x_min = x_min
        self.x_max = x_max
        self.p_min = p_min

        self.memory_size = memory_size
        self.M_CR = np.ones(memory_size) * 0.5
        self.M_F = np.ones(memory_size) * 0.5
        self.k = 0
        self.archive = []

        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_dim = 8
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.observation_dim,),
            dtype=np.float32
        )

        self.fitness_function_id = start_function_id
        self.num_function = num_function
        self.func_mapping = self._create_func_mapping()
        self.reset()

    def _create_func_mapping(self):
        from opfunu import cec_based
        return {
            1: cec_based.F12017,    2: cec_based.F22017,
            3: cec_based.F32017,    4: cec_based.F42017,
            5: cec_based.F52017,    6: cec_based.F62017,
            7: cec_based.F72017,    8: cec_based.F82017,
            9: cec_based.F92017,    10: cec_based.F102017,
            11: cec_based.F112017,  12: cec_based.F12017,
            13: cec_based.F132017,  14: cec_based.F142017,
            15: cec_based.F152017,  16: cec_based.F162017,
            17: cec_based.F172017,  18: cec_based.F182017,
            19: cec_based.F192017,  20: cec_based.F202017,
            21: cec_based.F212017, 22: cec_based.F222017,
            23: cec_based.F232017, 24: cec_based.F242017,
            25: cec_based.F252017, 26: cec_based.F262017,
            27: cec_based.F272017, 28: cec_based.F282017,
            29: cec_based.F292017
        }

    def reset(self):
        func_class = self.func_mapping[self.fitness_function_id]
        self.fitness_function = func_class(ndim=self.dim)

        self.population = np.random.uniform(
            self.x_min, self.x_max,
            (self.population_size, self.dim)
        )

        self.fitness = np.array([self.fitness_function.evaluate(x) for x in self.population])
        self.pbest_positions = self.population.copy()
        self.pbest_fitness = self.fitness.copy()
        self.gbest_idx = np.argmin(self.fitness)
        self.gbest_position = self.population[self.gbest_idx].copy()
        self.gbest_fitness = self.fitness[self.gbest_idx]

        self.M_CR[:] = 0.5
        self.M_F[:] = 0.5
        self.k = 0
        self.archive = []

        self.cur_iter = 0
        self.gbest_fitness_old = self.gbest_fitness
        self.not_update_count = 0
        self.survival = np.ones(self.population_size, dtype=int)

        self.info = {
            "fitness": self.gbest_fitness,
            "reward": 0,
            "exploit_rate": [],
            'fitness_history': []
        }

        self.p_count = 0
        return self._get_full_state()

    def _get_full_state(self):
        center = np.mean(self.population, axis=0)
        dist_center = np.linalg.norm(self.population - center, axis=1)
        dist_gbest = np.linalg.norm(self.population - self.gbest_position, axis=1)
        if self.gbest_fitness < 0:
            print(self.gbest_fitness)
        features = np.array([
            np.log(1 + np.abs(self.gbest_fitness)),
            np.log(1 + np.mean(self.fitness)),
            np.log(1 + np.std(self.fitness)),
            np.log(1 + abs(self.gbest_fitness_old - self.gbest_fitness)),
            np.log(1 + np.sum(dist_center)),
            np.log(1 + np.sum(dist_gbest)),
            self.not_update_count ,
            self.cur_iter / self.max_iter
        ], dtype=np.float32)

        features[:6] = (features[:6] - np.min(features[:6])) / (np.ptp(features[:6]) + 1e-8)
        return features

    def _vectorized_mutation(self, p):
        sorted_idx = np.argsort(self.fitness)
        split_idx = int(self.population_size * p[0])

        r = np.random.randint(0, self.memory_size, self.population_size)
        CR = np.clip(stats.norm(loc=self.M_CR[r], scale=0.1).rvs(), 0, 1)
        F = np.clip(stats.cauchy(loc=self.M_F[r], scale=0.1).rvs(), 0, 1)

        rand_indices = np.array([self._get_rand_indices(i) for i in range(self.population_size)])

        mutants = np.empty_like(self.population)

        if split_idx > 0:
            best_individuals = sorted_idx[:split_idx]
            r1 = rand_indices[best_individuals, 0]
            r2 = rand_indices[best_individuals, 1]
            F_best = F[best_individuals, np.newaxis]
            mutants[best_individuals] = self.population[best_individuals] + F_best * (
                self.population[r1] - self.population[r2]
            )

        if split_idx < self.population_size:
            rand_individuals = sorted_idx[split_idx:]
            r1 = rand_indices[rand_individuals, 0]
            r2 = rand_indices[rand_individuals, 1]
            r3 = rand_indices[rand_individuals, 2]
            F_rand = F[rand_individuals, np.newaxis]
            mutants[rand_individuals] = self.population[r1] + F_rand * (
                self.population[r2] - self.population[r3]
            )

        cross_mask = np.random.rand(*mutants.shape) < CR[:, np.newaxis]
        cross_mask |= np.arange(self.dim) == np.random.randint(self.dim, size=self.population_size)[:, np.newaxis]
        trials = np.where(cross_mask, mutants, self.population)

        return np.clip(trials, self.x_min, self.x_max), F, CR

    def _get_rand_indices(self, idx):
        candidates = np.setdiff1d(np.arange(self.population_size), idx)
        return np.random.choice(candidates, 3, replace=False)


    def update_particles(self, action):
        trials, F, CR = self._vectorized_mutation(action)

        intermediate_fitness = []
        current_gbest = self.gbest_fitness

        for i in range(self.population_size):
            trial_fitness = self.fitness_function.evaluate(trials[i])

            if trial_fitness < self.fitness[i]:
                self.population[i] = trials[i]
                self.fitness[i] = trial_fitness

                if trial_fitness < current_gbest:
                    current_gbest = trial_fitness
                    self.gbest_position = trials[i].copy()
                    self.gbest_fitness = trial_fitness

            intermediate_fitness.append(current_gbest)

        improved = self.fitness < self.fitness_old
        if np.any(improved):
            valid = improved
            self.M_CR[self.k] = np.mean(CR[valid])
            self.M_F[self.k] = np.mean(F[valid])
            self.k = (self.k + 1) % self.memory_size

            self.archive.extend(self.population[~improved].tolist())
            if len(self.archive) > self.population_size:
                del self.archive[:len(self.archive)-self.population_size]

        return np.array(intermediate_fitness)

    def step(self, action, is_test=False):

        if action[0] < 0.1 or action[0] > 0.9:
            self.p_count += 1

        old_gbest = self.gbest_fitness
        self.fitness_old = self.fitness.copy()

        intermediate_fitness = self.update_particles(action)

        progress = self.cur_iter / self.max_iter
        fitness_weight = (1 - np.cos(np.pi * progress)) / 2.0
        diversity_weight = 1 - fitness_weight


        entropy = stats.entropy(np.histogram(self.population, bins=20)[0])

        fitness_reward = (old_gbest - self.gbest_fitness) / (old_gbest + 1e-8)

        reward = fitness_weight * fitness_reward + diversity_weight * entropy

        if self.gbest_fitness < old_gbest:
            self.not_update_count = 0
        else:
            self.not_update_count += 1

        self.cur_iter += 1
        done = self.cur_iter >= self.max_iter
        if not is_test and self.gbest_fitness == 0:
            done = True
            reward += 10
        if done:
            print(f"Function ID: {self.fitness_function_id}, Best Fitness: {self.gbest_fitness}, p_count: {self.p_count}")
            if not is_test:
                self.fitness_function_id = (self.fitness_function_id) % 29 + 1


        self.info["fitness"] = intermediate_fitness
        self.info["fitness_history"].append(self.gbest_fitness)
        self.info["reward"] = reward
        self.info["exploit_rate"].append(action[0])

        return self._get_full_state(), reward, done, self.info

    def render(self, mode='human'):
        pass

    def seed(self, seed=None):
        np.random.seed(seed)
        return [seed]


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import os

    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['axes.linewidth'] = 1.5

    dim = config.DIMENSION # 使用配置文件中的维度
    max_iter = config.MAX_ITERATIONS # 使用配置文件中的最大迭代次数
    func_id = 29

    env = PSO_Proportional_Env(dim=dim, max_iter=max_iter, start_function_id=func_id)

    strategies = {
        '固定比例策略(p=0.5)': [0.5],
        '固定比例策略(p=0)': [0],
        '固定比例策略(p=1)': [1],
    }

    results = {}
    fes_results = {}

    for name, action in strategies.items():
        print(f"Testing {name}...")
        env.reset()
        history = []
        fes_history = []

        for _ in range(max_iter):
            _, _, done, info = env.step(action, is_test=True)
            history.append(env.gbest_fitness)
            fes_history.extend(info['fitness'])

        results[name] = history
        fes_results[name] = fes_history

    plt.figure(1,figsize=(8, 6), dpi=150)
    for name, data in fes_results.items():
        plt.semilogy(data, linewidth=2, alpha=0.8, label=name)

    plt.xlabel('fps', fontsize=12, fontweight='bold')
    plt.ylabel('Fitness Value', fontsize=12, fontweight='bold')
    plt.title('Convergence fps',
              fontsize=12, fontweight='bold', pad=20)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.legend(loc='upper right')
    print(f'fitness_id: {func_id}, dim = {dim}, gbest_fitness: {env.gbest_fitness}')
    plt.show()
