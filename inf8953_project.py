import gym
from gym.wrappers import Monitor
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
from base_class import *
from utils import *

class Direction(Enum):
    POSITIVE = 1
    NEGATIVE = -1
    BEST = 0

class ARS():
    """Augmented Random Search"""
    def __init__(self, env: gym.Env, hp: HyperParameters, is_V2=False, output_path="./outputs") -> None:
        print(hp)
        assert hp.nb_best_directions <= hp.nb_directions
        self.step_size = hp.learning_rate
        self.n_directions = hp.nb_directions
        self.n_best_directions =  hp.nb_best_directions
        self.std_exp_noise = hp.std_exp_noise
        self.env = env
        self.eval_env = Monitor(env, output_path, force=True)
        if is_V2:
            self.normalizer = Normalizer(env.observation_space.shape[0])
        else:
            self.normalizer = BaseNormalizer()
        
        self.kp_penalty = hp.keep_alive_penalty
        
        self.seed = hp.seed
        self.max_nb_ep = hp.max_ep
        self.max_ep_length = hp.max_episode_length
    
    def run_episode(self, policy: Policy, direction: Direction = Direction.BEST.value, delta=0, render=False):
        temp_env = self.env
        if render:
            temp_env = self.eval_env
        state = temp_env.reset()
        done = False
        total_reward = 0
        steps = 0
        while not done and steps < self.max_ep_length:
            state = self.normalizer.normalize(state)
            action = policy.evaluate(state, direction, self.std_exp_noise, delta)
            state, reward, done, _  = temp_env.step(action)
            if direction != Direction.BEST.value:
                reward = reward - self.kp_penalty
            total_reward += reward
            steps +=1
        if render:
            temp_env.close()
        
        return total_reward

    def execute(self, policy = None, render=False):
        if policy is None:
            policy = Policy(self.env)
        
        np.random.seed(self.seed)
        self.env.seed(self.seed)
        self.eval_env.seed(self.seed)
        
        rewards = []
        for i in tqdm(range(0, self.max_nb_ep, 2 * self.n_directions)):
            deltas = policy.sample_deltas(self.n_directions)
            positive_rewards = np.array([self.run_episode(policy, Direction.POSITIVE.value, delta, render = render) for delta in deltas])
            negative_rewards = np.array([self.run_episode(policy, Direction.NEGATIVE.value, delta, render = render) for delta in deltas])

            scores = {k: max(r_pos, r_neg) for k, (r_pos, r_neg) in enumerate(zip(positive_rewards, negative_rewards))}
            order = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:self.n_best_directions]
            positive_rewards = np.array([positive_rewards[k] for k in order])
            negative_rewards = np.array([negative_rewards[k] for k in order])
            deltas = np.array([deltas[k] for k in order])

            policy.update(self.step_size, positive_rewards, negative_rewards, self.n_best_directions, deltas)
            reward = self.run_episode(policy, render=render)
            if i % (10 * 2 * self.n_directions) == 0:
                print(f"Step {i}: {reward}")
            rewards.append(reward)
        
        return policy, rewards


if __name__ == '__main__':
    args = sys.argv
    print(args)
    env_name = 'Swimmer-v2'
    render = False
    if len(args) > 1:
        env_name = args[1]
    if len(args) > 2:
        render = args[2] == 'True'
    print()
    print()
    print()
    print()
    print()
    print('==============')
    print(f'mujoco_env: {env_name}')
    print(f'render: {render}')
    print(f'is_V2: {True}')
    print('==============')
    DEFAULT_SEED = [1, 23455, 920847]
    env, hp = get_env_and_hyper_params(env_name, is_v2=True, truncated=True, seed=1)
    ars = ARS(env, hp, is_V2=True)
    policy, rewards = ars.execute(render=render)

    #plt.plot(rewards)
    save_output('save/exp.pickle', policy, rewards, hp, env_name)
