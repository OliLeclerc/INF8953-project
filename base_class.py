import numpy as np
import gym

class Policy():
    def __init__(self, env: gym.Env) -> None:
        self.theta = np.zeros((env.action_space.shape[0], env.observation_space.shape[0]))
    
    def evaluate(self, state, direction: int, std_exp_noise: float, delta: float):
        return (self.theta + direction * std_exp_noise * delta).dot(state)
    
    def sample_deltas(self, n_directions):
        return np.array([np.random.randn(*self.theta.shape) for _ in range(n_directions)])
    
    def update(self, step_size, pos_rewards, neg_rewards, n_best_directions, deltas):
        std_rewards = np.concatenate((pos_rewards, neg_rewards)).std()
        self.theta += (step_size / (n_best_directions * std_rewards)) * np.sum((pos_rewards - neg_rewards).reshape(-1,1,1) * deltas, axis=0)


class BaseNormalizer():

    def observe(self, x):
        pass

    def normalize(self, inputs):
        return inputs

class Normalizer():

    def __init__(self, nb_inputs):
        self.n = np.zeros(nb_inputs)
        self.mean = np.zeros(nb_inputs)
        self.mean_diff = np.zeros(nb_inputs)
        self.var = np.zeros(nb_inputs)

    def observe(self, x):
        self.n += 1.
        last_mean = self.mean.copy()
        self.mean += (x - self.mean) / self.n
        self.mean_diff += (x - last_mean) * (x - self.mean)
        self.var = (self.mean_diff / self.n).clip(min=1e-2)

    def normalize(self, inputs):
        self.observe(inputs)
        obs_mean = self.mean
        obs_std = np.sqrt(self.var)
        return (inputs - obs_mean) / obs_std