import gym
from gym.wrappers import Monitor
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from base_class import *
from utils import *
import argparse
import mujoco_py

class Direction(Enum):
    POSITIVE = 1
    NEGATIVE = -1
    BEST = 0

class ARS():
    """Augmented Random Search"""
    def __init__(self, env: gym.Env, hp: HyperParameters, is_V2=False, output_path="./outputs") -> None:

        assert hp.nb_best_directions <= hp.nb_directions
        self.step_size = hp.learning_rate
        self.n_directions = hp.nb_directions
        self.n_best_directions =  hp.nb_best_directions
        self.std_exp_noise = hp.std_exp_noise
        self.env = env
        if is_V2:
            self.normalizer = Normalizer(env.observation_space.shape[0])
        else:
            self.normalizer = BaseNormalizer()
        
        self.kp_penalty = hp.keep_alive_penalty
        
        self.seed = hp.seed
        self.max_nb_ep = hp.max_ep
        self.max_ep_length = hp.max_episode_length

        self.monitor_env = Monitor(env, output_path, force=True, video_callable = lambda episode_id: True)
    
    def run_episode(self, policy: Policy, direction: Direction = Direction.BEST.value, delta=0, render=False):
        temp_env = self.env
        if render:
            temp_env = self.monitor_env
        temp_env.seed(self.seed)
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
        
        return total_reward

    def execute(self, policy = None, render=False):
        if policy is None:
            policy = Policy(self.env)
        
        np.random.seed(self.seed)
        rewards = []
        for i in tqdm(range(0, self.max_nb_ep, 2 * self.n_directions)):
            deltas = policy.sample_deltas(self.n_directions)
            positive_rewards = np.array([self.run_episode(policy, Direction.POSITIVE.value, delta, render = False) for delta in deltas])
            negative_rewards = np.array([self.run_episode(policy, Direction.NEGATIVE.value, delta, render = False) for delta in deltas])

            scores = {k: max(r_pos, r_neg) for k, (r_pos, r_neg) in enumerate(zip(positive_rewards, negative_rewards))}
            order = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:self.n_best_directions]
            positive_rewards = np.array([positive_rewards[k] for k in order])
            negative_rewards = np.array([negative_rewards[k] for k in order])
            deltas = np.array([deltas[k] for k in order])

            policy.update(self.step_size, positive_rewards, negative_rewards, self.n_best_directions, deltas)
            reward = self.run_episode(policy, render=False)
            if i % (10 * 2 * self.n_directions) == 0:
                print(f"Step {i}: {reward}")
            rewards.append(reward)
        if render:
            final_reward = self.run_episode(policy, render=True)
            print(f'Final reward: {final_reward}')
        
        return policy, rewards



def get_args():
    parser = argparse.ArgumentParser(description='Run ARS')
    parser.add_argument('env_name', type=str,
                        help='Name of the MuJoCo Env')

    parser.add_argument('seed', type=int, default = 1,
                        help='Name of the MuJoCo Env')

    parser.add_argument('--r', action='store_true', default = False,
                        help='Whether or not sample video will be saved')

    parser.add_argument('--is_v2', action='store_true',
                        help='Whether or not sample video will be saved')
            
    parser.add_argument('--t', action='store_true',
                        help='If ars algo should be version -t')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()

    print()
    print('==============')
    print(f'mujoco_env: {args.env_name} with seed : {args.seed}')
    print(f'render: {args.r}')
    print(f'is_V2: {args.is_v2}, is truncated: {args.t}')
    print('==============')

    version = 'v2' if args.is_v2 else 'v1'
    version += '-t' if args.t else ''
    full_output_path = f'./outputs/{args.env_name}/{version}/{args.seed}/'
    env, hp = get_env_and_hyper_params(args.env_name, is_v2=args.is_v2, truncated=args.t, seed=args.seed)
    ars = ARS(env, hp, is_V2=args.is_v2, output_path = full_output_path)
    policy, rewards = ars.execute(render=args.r)

    #plt.plot(rewards)
    save_output(f'{full_output_path}save/', policy, rewards, hp, args.env_name)
