import gym
#from gym.wrappers import Monitor
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from pickle import load, dump
import mujoco_py



from enum import Enum
class Environment(Enum): # See https://github.com/benelot/pybullet-gym
    ANT = "Ant-v2"
    CHEETAH = "HalfCheetah-v2"
    HOPPER = "Hopper-v2"
    SWIMMER = "Swimmer-v2"
    WALKER = "Walker2D-v2"
    HUMAN = "Humanoid-v2"

class Direction(Enum):
    POSITIVE = 1
    NEGATIVE = -1
    BEST = 0


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

class ARS():
    """Augmented Random Search"""
    def __init__(self, env: gym.Env, step_size: float, n_directions: int, std_exp_noise: float, n_best_directions: int, is_V2=False, output_path="./outputs") -> None:
        assert n_best_directions <= n_directions
        self.step_size = step_size
        self.n_directions = n_directions
        self.n_best_directions = n_best_directions
        self.std_exp_noise = std_exp_noise
        self.env = env
        #self.eval_env = Monitor(env, output_path, force=True)
        if is_V2:
            self.normalizer = Normalizer(env.observation_space.shape[0])
        else:
            self.normalizer = BaseNormalizer()

        
    
    
    def run_episode(self, policy: Policy, direction: Direction = Direction.BEST.value, delta=0, render=False):
        temp_env = self.env
        #if render:
        #    temp_env = self.eval_env
        state = temp_env.reset()
        done = False
        total_reward = 0
        steps = 0
        while not done and steps < 2000:
            state = self.normalizer.normalize(state)
            action = policy.evaluate(state, direction, self.std_exp_noise, delta)
            state, reward, done, _  = temp_env.step(action)
            if direction != Direction.BEST.value:
                #reward = max(min(reward, 1), -1)
                reward = reward - 1
            total_reward += reward
            steps +=1
        
        return total_reward

    def execute(self, n_steps, policy = None, render=False):
        if policy is None:
            policy = Policy(self.env)
        
        rewards = []
        for i in tqdm(range(n_steps)):
            deltas = policy.sample_deltas(self.n_directions)
            positive_rewards = np.array([self.run_episode(policy, Direction.POSITIVE.value, delta) for delta in deltas])
            negative_rewards = np.array([self.run_episode(policy, Direction.NEGATIVE.value, delta) for delta in deltas])

            scores = {k: max(r_pos, r_neg) for k, (r_pos, r_neg) in enumerate(zip(positive_rewards, negative_rewards))}
            order = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:self.n_best_directions]
            positive_rewards = np.array([positive_rewards[k] for k in order])
            negative_rewards = np.array([negative_rewards[k] for k in order])
            deltas = np.array([deltas[k] for k in order])

            policy.update(self.step_size, positive_rewards, negative_rewards, self.n_best_directions, deltas)
            reward = self.run_episode(policy, render=render)
            if render and i % 10 == 0:
                print(f"Step {i * 2 * self.n_best_directions}: {reward}")
            rewards.append(reward)
        
        return policy, rewards


from dataclasses import dataclass
@dataclass
class HyperParameters:
    nb_steps: int = 1000
    max_episode_length = 100
    learning_rate: float = 0.01
    std_exp_noise: float = 0.025
    nb_directions: int = 8
    nb_best_directions:int = 4
    seed: int = 1

@dataclass
class ARSOutput():
    policy: Policy
    rewards: list
    hyper_parameters: HyperParameters
    env: str

def save_output(path:str, policy: Policy, rewards: list, hp: HyperParameters, env: str):
    output = ARSOutput(policy, rewards, hp, env)
    with open(path, "wb") as f:
        dump(output, f)

def load_output(path:str)-> ARSOutput:
    with open(path, "rb") as f:
        return load(f)

hp = HyperParameters()
np.random.seed(hp.seed)
env = gym.make(Environment.HOPPER.value)
ars = ARS(env, hp.learning_rate, hp.nb_directions, hp.std_exp_noise, hp.nb_best_directions, is_V2=True)
policy, rewards = ars.execute(hp.nb_steps, render=True)

#plt.plot(rewards)
save_output('save/exp.pickle', policy, rewards, hp, Environment.HOPPER.value)
