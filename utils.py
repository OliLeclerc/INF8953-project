from hyper_params_config import *
from dataclasses import dataclass
import gym
from base_class import *
from pickle import load, dump
import os
DEFAULT_SEED = [1, 23455, 920847]

@dataclass
class HyperParameters:
    max_episode_length:int
    max_ep: int 
    learning_rate: float
    std_exp_noise: float
    nb_directions: int
    nb_best_directions:int
    seed: int
    keep_alive_penalty: int

valid_algo = ['V1', 'V1-t', 'V2', 'V2-t']

def get_env_and_hyper_params(env_name: str, is_v2: bool = True, truncated: bool = True, seed: int = 0):
    
    env = Environment(env_name)

    #because no hyper params for V1 right now.
    version = 'V'
    version += '2' if is_v2 else '2'
    version += '-t' if truncated else ''

    hp = HyperParameters(
        max_episode_length = max_episode_length,
        max_ep = hp_config[env]['max_ep'],
        learning_rate = hp_config[env][version]['lr'],
        std_exp_noise = hp_config[env][version]['std_noise'],
        nb_directions = hp_config[env][version]['n'], 
        nb_best_directions = hp_config[env][version]['b'],
        seed = seed,
        keep_alive_penalty = hp_config[env]['ka_penalty']
    )

    return gym.make(env.value), hp


@dataclass
class ARSOutput():
    policy: Policy
    rewards: list
    hyper_parameters: HyperParameters
    env: str

def save_output(path:str, policy: Policy, rewards: list, hp: HyperParameters, env: str):
    name = 'exp.pickle'
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    output = ARSOutput(policy, rewards, hp, env)
    with open(path + name, "wb") as f:
        dump(output, f)

def load_output(path:str)-> ARSOutput:
    with open(path, "rb") as f:
        return load(f)