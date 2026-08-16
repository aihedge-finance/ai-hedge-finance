import os
import ast
import time
import sys
from copy import deepcopy

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any
from ahf.utils.utils import readable_error


def get_gym_env_args(env, if_print) -> dict:  # [ElegantRL.2021.12.12]
    """
    Get a dict ``env_args`` about a standard OpenAI gym env information.
    :param env: a standard OpenAI gym env
    :param if_print: [bool] print the dict about env information.
    :return: env_args [dict]
    env_args = {
        'env_num': 1,               # [int] the environment number, 'env_num>1' in vectorized env
        'env_name': env_name,       # [str] the environment name, such as XxxXxx-v0
        'max_step': max_step,       # [int] the steps in an episode. (from env.reset to done).
        'state_dim': state_dim,     # [int] the dimension of state
        'action_dim': action_dim,   # [int] the dimension of action or the number of discrete action
        'if_discrete': if_discrete, # [bool] action space is discrete or continuous
    }
    """
    import gym

    env_num = getattr(env, "env_num") if hasattr(env, "env_num") else 1

    if {'unwrapped', 'observation_space', 'action_space', 'spec'}.issubset(dir(env)):  # isinstance(env, gym.Env):
        env_name = getattr(env, 'env_name', None)
        env_name = env.unwrapped.spec.id if env_name is None else env_name

        state_shape = env.observation_space.shape
        state_dim = state_shape[0] if len(state_shape) == 1 else state_shape  # sometimes state_dim is a list

        max_step = getattr(env, 'max_step', None)
        max_step_default = getattr(env, '_max_episode_steps', None)
        if max_step is None:
            max_step = max_step_default
        if max_step is None:
            max_step = 2 ** 10

        if_discrete = isinstance(env.action_space, gym.spaces.Discrete)
        if if_discrete:  # make sure it is discrete action space
            action_dim = env.action_space.n
        elif isinstance(env.action_space, gym.spaces.Box):  # make sure it is continuous action space
            action_dim = env.action_space.shape[0]
            if not any(env.action_space.high - 1):
                print('WARNING: env.action_space.high', env.action_space.high)
            if not any(env.action_space.low - 1):
                print('WARNING: env.action_space.low', env.action_space.low)
        else:
            raise RuntimeError('\n| Error in get_gym_env_info()'
                               '\n  Please set these value manually: if_discrete=bool, action_dim=int.'
                               '\n  And keep action_space in (-1, 1).')
    else:
        env_name = env.env_name
        max_step = env.max_step
        state_dim = env.state_dim
        action_dim = env.action_dim
        if_discrete = env.if_discrete

    env_args = {'env_num': env_num,
                'env_name': env_name,
                'max_step': max_step,
                'state_dim': state_dim,
                'action_dim': action_dim,
                'if_discrete': if_discrete, }
    if if_print:
        env_args_repr = repr(env_args)
        env_args_repr = env_args_repr.replace(',', f",\n   ")
        env_args_repr = env_args_repr.replace('{', "{\n    ")
        env_args_repr = env_args_repr.replace('}', ",\n}")
        print(f"env_args = {env_args_repr}")
    return env_args


def kwargs_filter(func, kwargs: Dict[str, Any]):
    """
    Filter the variable in env func.
    :param func: the function for creating an env.
    :param kwargs: args for the env.
    :return: filtered args.
    """
    import inspect

    sign = inspect.signature(func).parameters.values()
    sign = {val.name for val in sign}

    common_args = sign.intersection(kwargs.keys())
    return {key: kwargs[key] for key in common_args}  # filtered kwargs


def build_env(env_func=None, env_args=None):
    env = env_func(**kwargs_filter(env_func.__init__, env_args.copy()))

    for attr_str in ('state_dim', 'action_dim', 'max_step', 'if_discrete', 'target_return'):
        if (not hasattr(env, attr_str)) and (attr_str in env_args):
            setattr(env, attr_str, env_args[attr_str])

    return env


class ReplayBufferList(list):  # for on-policy
    def __init__(self):
        list.__init__(self)

    def update_buffer(self, traj_list):
        cur_items = list(map(list, zip(*traj_list)))
        self[:] = [torch.cat(item, dim=0) for item in cur_items]

        steps = self[1].shape[0]
        r_exp = self[1].mean().item()
        return steps, r_exp


class Arguments:
    def __init__(self, agent, env_func=None, env_args=None):
        self.env_func = env_func  # env = env_func(*env_args)
        self.env_args = env_args  # env = env_func(*env_args)

        self.env_num = self.env_args["env_num"]  # env_num = 1. In vector env, env_num > 1.
        self.max_step = self.env_args['max_step']  # the max step of an episode
        self.env_name = self.env_args['env_name']  # the env name. Be used to set 'cwd'.
        self.state_dim = self.env_args['state_dim']  # vector dimension (feature number) of state
        self.action_dim = self.env_args['action_dim']  # vector dimension (feature number) of action
        self.if_discrete = self.env_args['if_discrete']  # discrete or continuous action space

        self.agent = agent  # DRL algorithm
        self.net_dim = 2 ** 7  # the middle layer dimension of Fully Connected Network
        self.batch_size = 2 ** 7  # num of transitions sampled from replay buffer.
        self.mid_layer_num = 1  # the middle layer number of Fully Connected Network
        self.if_off_policy = self.get_if_off_policy()  # agent is on-policy or off-policy
        self.if_use_old_traj = False  # save old data to splice and get a complete trajectory (for vector env)
        if self.if_off_policy:  # off-policy
            self.max_memo = 2 ** 21  # capacity of replay buffer
            self.target_step = 2 ** 10  # repeatedly update network to keep critic's loss small
            self.repeat_times = 2 ** 0  # collect target_step, then update network
        else:  # on-policy
            self.max_memo = 2 ** 12  # capacity of replay buffer
            self.target_step = self.max_memo  # repeatedly update network to keep critic's loss small
            self.repeat_times = 2 ** 4  # collect target_step, then update network

        '''Arguments for training'''
        self.gamma = 0.99  # discount factor of future rewards
        self.reward_scale = 2 ** 0  # an approximate target reward usually be closed to 256
        self.learning_rate = 2 ** -12  # 2 ** -15 ~= 3e-5
        self.soft_update_tau = 2 ** -8  # 2 ** -8 ~= 5e-3

        '''Arguments for device'''
        self.worker_num = 2  # rollout workers number pre GPU (adjust it to get high GPU usage)
        self.thread_num = 8  # cpu_num for pytorch, `torch.set_num_threads(self.num_threads)`
        self.random_seed = 0  # initialize random seed in self.init_before_training()
        self.learner_gpus = 0  # `int` means the ID of single GPU, -1 means CPU

        '''Arguments for evaluate'''
        self.cwd = None  # current working directory to save model. None means set automatically
        self.if_remove = True  # remove the cwd folder? (True, False, None:ask me)
        self.break_step = +np.inf  # break training if 'total_step > break_step'

        '''Arguments for evaluate'''
        self.eval_gap = 2 ** 7  # evaluate the agent per eval_gap seconds
        self.eval_times = 2 ** 4  # number of times that get episode return

    def init_before_training(self):
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        torch.set_num_threads(self.thread_num)
        torch.set_default_dtype(torch.float32)

        '''auto set cwd (current working directory)'''
        if self.cwd is None:
            self.cwd = f'./{self.env_name}_{self.agent.__name__[5:]}_{self.learner_gpus}'

        '''remove history'''
        if self.if_remove is None:
            self.if_remove = bool(input(f"| Arguments PRESS 'y' to REMOVE: {self.cwd}? ") == 'y')
        elif self.if_remove:
            import shutil
            shutil.rmtree(self.cwd, ignore_errors=True)
            print(f"| Arguments Remove cwd: {self.cwd}")
        else:
            print(f"| Arguments Keep cwd: {self.cwd}")
        os.makedirs(self.cwd, exist_ok=True)

    def get_if_off_policy(self):
        name = self.agent.__name__
        return all((name.find('PPO') == -1, name.find('A2C') == -1))  # if_off_policy


def get_trade_args(file):
    try:
        # file = 'opts.params'
        with open(file, 'r') as f:
            data_dict = ast.literal_eval(f.read())

            return data_dict
    except Exception as e:
        err = readable_error(e, __file__)
        print(err)
        sys.exit()
