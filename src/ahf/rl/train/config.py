import os
import sys
import ast
import torch
import time
import json
import yaml

import numpy as np
from loguru import logger
from copy import deepcopy
from deprecated import deprecated
from typing import Dict, Any

from sys_config import TRAINED_MODEL_DIR
from ahf.core.enums import PriceEnv
from ahf.utils.utils import readable_error, get_project_root, is_dir_exist

"""
class ReplayBufferList(list):  # for on-policy
    def __init__(self):
        list.__init__(self)

    def update_buffer(self, traj_list):
        cur_items = list(map(list, zip(*traj_list)))
        self[:] = [torch.cat(item, dim=0) for item in cur_items]

        steps = self[1].shape[0]
        r_exp = self[1].mean().item()
        return steps, r_exp
"""

class Arguments:
    def __init__(self, agent, env=None, env_func=None, env_args=None):
        """
        package everything here to be init together later
        Parameters
        ----------
        agent: RL training agent class, e.g. AgentPPO, AgentDDPG etc.
        env: env instance initiated already, why
        env_func: it is env_cls, e.g. StockTradingEnv, BrunhildEnv
        env_args: parameters of the EnvClass e.g. env = env_func(*env_args)
        """
        self.env = env
        self.env_func = env_func  # env = env_func(*env_args)
        self.env_args = env_args  # env = env_func(*env_args)

        self.env_num = self.env_args["env_num"]  # env_num = 1. In vector env, env_num > 1.
        self.max_step = self.env_args["max_step"]  # the max step of an episode
        self.env_name = self.env_args["env_name"]  # the env name. Be used to set "cwd".
        self.state_dim = self.env_args["state_dim"]  # vector dimension (feature number) of state
        self.action_dim = self.env_args["action_dim"]  # vector dimension (feature number) of action
        self.if_discrete = self.env_args["if_discrete"]  # discrete or continuous action space
        self.init_steps = self.env_args.get("init_steps", 0)  # starting steps if we restart
        self.target_return = self.update_attr("target_return")  # target average episode return

        self.agent = agent  # DRL algorithm
        # self.batch_size = 2 ** 7  # num of transitions sampled from replay buffer.
        self.mid_layer_num = 2  # the middle layer number of Fully Connected Network
        self.if_off_policy = self.get_if_off_policy()  # agent.if_off_policy # agent is on-policy or off-policy
        self.if_use_old_traj = True  # save old data to splice and get a complete trajectory (for vector env)
        self.if_act_target = False
        self.if_cri_target = True

        # hyperparameter tuning, if does not exist here, it will be setup with default in PPO or PPH35
        if "tau" in self.env_args:
            self.tau = self.env_args["tau"]
        if "ratio_clip" in self.env_args:
            self.ratio_clip = self.env_args["ratio_clip"]
        if "lambda_gae_adv" in self.env_args:
            self.lambda_gae_adv = self.env_args["lambda_gae_adv"]
        if "lambda_entropy" in self.env_args:
            self.lambda_entropy = self.env_args["lambda_entropy"]
        if "if_use_gae" in self.env_args:
            self.if_use_gae = self.env_args["if_use_gae"]
        if "lr_gamma" in self.env_args:
            self.lr_gamma = self.env_args["lr_gamma"]
        if "min_learning_rate" in self.env_args:
            self.min_learning_rate = self.env_args["min_learning_rate"]

        """max risk: 還沒用到"""
        self.max_risk = None

        """on/off-policy"""
        if self.if_off_policy:  # off-policy
            self.net_dim = env_args.get("net_dim", 2 ** 8)  # the network width: the middle layer dimension of Fully Connected Network
            self.max_memo = env_args.get("max_memo", 2 ** 21)  # origin 2 **21 capacity of replay buffer
            self.batch_size = self.net_dim  # num of transitions sampled from replay buffer.
            self.target_step = 2 ** 10  # repeatedly update network to keep critic"s loss small
            self.repeat_times = 2 ** 0  # collect target_step, then update network
            self.if_per_or_gae = False  # use PER (Prioritized Experience Replay) for sparse reward

        else:  # on-policy
            self.net_dim = env_args.get("net_dim", 2 ** 9)  # the network width: the middle layer dimension of Fully Connected Network
            self.max_memo = env_args.get("max_memo", 2 ** 16)  # origin 2 ** 12 capacity of replay buffer
            self.batch_size = self.net_dim * 2  # num of transitions sampled from replay buffer.
            self.target_step = self.max_memo  # repeatedly update network to keep critic"s loss small
            self.repeat_times = 2 ** 4  # collect target_step, then update network
            self.if_per_or_gae = True  # use PER: GAE (Generalized Advantage Estimation) for sparse reward

        """Arguments for reward shaping"""
        self.gamma = env_args.get("gau", 0.99)  # discount factor of future rewards
        self.reward_scale = 2 ** 0  # an approximate target reward usually be closed to 256

        """Arguments for training"""
        self.learning_rate = env_args.get("learning_rate", 2 ** -12)  # 2 ** -15 ~= 3e-5
        self.soft_update_tau = env_args.get("tau", 2 ** -8)  # 2 ** -8 ~= 5e-3

        """Arguments for device"""
        self.worker_num = 1  # rollout workers number pre GPU (adjust it to get high GPU usage)
        self.thread_num = 8  # cpu_num for pytorch, `torch.set_num_threads(self.num_threads)`
        self.random_seed = 0  # initialize random seed in self.init_before_training()
        self.learner_gpus = env_args.get("learner_gpus", -1)  # env_args.get("gpu_id")  # `int` means the ID of single GPU, -1 means CPU
        self.learner_gpus = self.learner_gpus.split(",") if "," in self.learner_gpus else self.learner_gpus
        self.learner_gpus = [int(gpu_id) for gpu_id in self.learner_gpus]
        self.workers_gpus = self.learner_gpus  # for GPU_VectorEnv (such as isaac gym)

        """Arguments for evaluate"""
        self.cwd = None  # current working directory to save model. None means set automatically
        self.if_remove = None  # remove the cwd folder? (True, False, None:ask me)
        self.break_step = env_args.get("break_step", 2000 * 1e6)  # +np.inf  # break training if "total_step > break_step"
        self.if_over_write = False  # overwrite the best policy network (actor.pth)
        self.if_allow_break = False  # allow break training when reach goal (early termination)

        self.eval_env = None  # the environment for evaluating. None means set automatically.
        self.eval_env_func = None  # env = env_func(*env_args)
        self.eval_env_args = None  # env = env_func(*env_args)

        """Arguments for evaluate"""
        self.eval_gap = 2 ** 7  # evaluate the agent per eval_gap seconds
        self.eval_times = 2 ** 4  # number of times that get episode return
        self.eval_times1 = 2 ** 2  # number of times that get episode return in first
        self.eval_times2 = 2 ** 4  # number of times that get episode return in second
        self.eval_gpu_id = None  # -1 means use cpu, >=0 means use GPU, None means set as learner_gpus[0]

        """ensemble DRL"""
        self.save_gap = 2 ** 9  # save the agent per save_gap seconds (for ensemble DRL)
        self.save_dir = f"./{TRAINED_MODEL_DIR}"  # a directory to save the `pod_save_{episode_returns}` for ensemble DRL

    def init_before_training(self, agent_id=0, load_if_exists_prompt=True):
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        torch.set_num_threads(self.thread_num)
        torch.set_default_dtype(torch.float32)

        """env"""
        assert isinstance(self.env_num, int)
        assert isinstance(self.env_name, str)
        assert isinstance(self.max_step, int)
        assert isinstance(self.state_dim, int) or isinstance(self.state_dim, tuple)
        assert isinstance(self.action_dim, int) or isinstance(self.action_dim, tuple)
        assert isinstance(self.if_discrete, int) or isinstance(self.if_discrete, bool)
        assert isinstance(self.target_return, int) or isinstance(self.target_return, float)

        """agent"""
        # assert hasattr(self.agent, "init")
        assert hasattr(self.agent, "update_net")
        # assert hasattr(self.agent, "explore_env")  # PPO changed to "explore_one_env" and "explore_vec_env"
        # assert hasattr(self.agent, "select_actions")

        """auto set"""
        if isinstance(self.learner_gpus, int):
            self.learner_gpus = (self.learner_gpus,)

        train_mode = self.env.trade_args.get("train_mode", "DEV")

        """max risk: 還沒用到"""
        self.max_risk = self.env.trade_args.get("max_risk", 1200000)

        agent_name = self.agent.__name__[5:]
        has_gpu = True if self.learner_gpus[agent_id] != -1 else False
        gpus_str = "CPU" if self.learner_gpus[agent_id] == -1 or not torch.cuda.is_available() else "GPU"
        print(f"| torch.cuda available:{torch.cuda.is_available()} and learner GPU: {has_gpu} ==> gpus_str: {gpus_str}")

        if self.cwd is None:  # if it is None, it is not tuning
            train_config_dir = f"{train_mode}/{self.env.trade_args.get('symbol')}_{self.env_name}" \
                               f"_{self.env.trade_args.get('tech_id')}_{self.env.trade_args.get('long_short')}" \
                               f"_{self.env.trade_args.get('trade_interval')}_{agent_id:02}" \
                               f"_{agent_name}_{gpus_str}_{self.env.trade_args.get('job_id')}"

            from collections.abc import Iterable

            if isinstance(self.learner_gpus, Iterable):  # ensemble DRL
                self.cwd = f"{self.save_dir}/pod_{agent_id:04}/{train_config_dir}"
            else:
                self.cwd = self.update_value(self.cwd, f"{self.save_dir}/{train_config_dir}")

        # print(f"[config] cwd: {self.cwd}")
        self.eval_env_func = self.update_value(self.eval_env_func, self.env_func)
        self.eval_env_args = self.update_value(self.eval_env_args, self.env_args)
        self.eval_gpu_id = self.update_value(self.eval_gpu_id, self.learner_gpus[agent_id])
        self.eval_env = self.update_value(self.eval_env, self.env)

        """remove history"""
        dir_exist = is_dir_exist(self.cwd)
        if load_if_exists_prompt:
            self.if_remove = False
        else:
            if self.if_remove is None and dir_exist:
                self.if_remove = bool(input(f"| Arguments PRESS 'y' to REMOVE: {self.cwd}? ") == "y")

        if self.if_remove:
            import shutil
            shutil.rmtree(self.cwd, ignore_errors=False)
            print(f"| Arguments Remove cwd: {self.cwd}")
        elif dir_exist:
            print(f"| Arguments Keep cwd: {self.cwd}")

        # check again, it may be deleted just now
        dir_exist = is_dir_exist(self.cwd)
        if not dir_exist:
            os.makedirs(self.cwd, exist_ok=True)
            print(f"| folder {self.cwd} is created")

    def get_if_off_policy(self):
        name = self.agent.__name__
        return all((name.find("PPO") == -1, name.find("A2C") == -1))  # if_off_policy

    @staticmethod
    def update_value(src, dst):
        if src is None:
            src = dst
        return src

    def update_attr(self, attr: str):
        if self.env_args is None:
            value = getattr(self.env, attr)
        else:
            value = self.env_args[attr]
        return value

def get_gym_env_args(env, hyper_args, if_print) -> dict:  # [ElegantRL.2021.12.12]
    """
    I use this to check all parameter needed by environment are loaded here
    Assign values to None to check if anything missing or not loaded as expected.

    ** SPECIAL note **
    "max_risk": max_risk
        - description: for tolerance, but I have not figured out what it can help me
        - required: False
    "init_steps": init_steps
        - description: for training that is done halve way and want to continue
        - required: False


    return a dict ``env_args`` about a standard my custom env information.

    :param hyper_args:
    :param env: a standard OpenAI gym env
    :param if_print: [bool] print the dict about env information.
    :return: env_args [dict]
    env_args = {
        "env_num": 1,               # [int] the environment number, "env_num>1" in vectorized env
        "env_name": env_name,       # [str] the environment name, such as XxxXxx-v0
        "max_step": max_step,       # [int] the steps in an episode. (from env.reset to done).
        "state_dim": state_dim,     # [int] the dimension of state
        "action_dim": action_dim,   # [int] the dimension of action or the number of discrete action
        "if_discrete": if_discrete, # [bool] action space is discrete or continuous
        "max_memo": max_memo,       # [int]
        "target_return": target_return, # [float]
        "job_id": job_id,
        "learner_gpus": learner_gpus,
        "net_dim": net_dim,
        "hyper_args": hyper_args,
        "break_step": break_step,
        "max_risk": max_risk,  # not required
        "init_steps": init_steps  # not required
    }

    """

    # args are required, do NOT use default
    """
    I DISABLE THIS TO make it None so that to check if we accidentally change any data
    env_num = hyper_args.get("env_num", 1)
    max_risk = getattr(env, "max_risk", 1200)
    learner_gpus = hyper_args.get("learner_gpus", -1)
    net_dim = hyper_args.get("net_dim", 2 ** 9)
    break_step = hyper_args.get("break_step", 2000 * 1e6)
    init_steps = hyper_args.get("init_steps", 0)
    max_memo = hyper_args.get("max_memo")
    job_id = env.exch_env.ds.job_id
    
    
    if {"unwrapped", "observation_space", "action_space", "spec"}.issubset(dir(env)):  # isinstance(env, gym.Env):
        print("We are deprecating gym, existing program....")
        sys.exit(-1)

        import gym

        env_name = getattr(env, "env_name", None)
        env_name = env.unwrapped.spec.id if env_name is None else env_name

        state_shape = env.observation_space.shape
        state_dim = state_shape[0] if len(state_shape) == 1 else state_shape  # sometimes state_dim is a list

        max_step = getattr(env, "max_step") if hasattr(env, "max_step") else env.max_step
        max_step_default = getattr(env, "_max_episode_steps", None)
        if max_step is None:
            max_step = max_step_default
        if max_step is None:
            max_step = 2 ** 10

        target_return = getattr(env.exch_env.ds, "target_return") if hasattr(env.exch_env.ds, "target_return") else env.target_return

        if_discrete = isinstance(env.action_space, gym.spaces.Discrete)
        if if_discrete:  # make sure it is discrete action space
            action_dim = env.action_space.n
        elif isinstance(env.action_space, gym.spaces.Box):  # make sure it is continuous action space
            action_dim = env.action_space.shape[0]
            if not any(env.action_space.high - 1):
                print("WARNING: env.action_space.high", env.action_space.high)
            if not any(env.action_space.low - 1):
                print("WARNING: env.action_space.low", env.action_space.low)
        else:
            raise RuntimeError("\n| Error in get_gym_env_info()"
                               "\n  Please set these value manually: if_discrete=bool, action_dim=int."
                               "\n  And keep action_space in (-1, 1).")
    else:
        # these already included in env, so we need not worry
        env_name = env.env_name
        max_step = env.max_step
        state_dim = env.exch_env.ds.state_dim
        action_dim = env.action_dim
        if_discrete = env.if_discrete
        target_return = env.exch_env.ds.target_return
    """

    # hyper_args
    env_num = hyper_args.get("env_num", 1) or 1
    learner_gpus = hyper_args.get("learner_gpus", None)
    net_dim = hyper_args.get("net_dim", None)
    break_step = hyper_args.get("break_step", None)
    # max_memo = hyper_args.get("max_memo", None)  # use Agent_base default
    init_steps = hyper_args.get("init_steps", 0)  # not required
    learning_rate = hyper_args.get("learning_rate", None)
    num_layer = hyper_args.get("num_layer", None)
    ent_coef = hyper_args.get("ent_coef", None)
    lr_gamma = hyper_args.get("lr_gamma", None)
    min_learning_rate = hyper_args.get("min_learning_rate", None)

    # for PPO
    tau = hyper_args.get("tau", None)
    ratio_clip = hyper_args.get("ratio_clip", None)
    lambda_gae_adv = hyper_args.get("lambda_gae_adv", None)
    lambda_entropy = hyper_args.get("lambda_entropy", None)
    if_use_gae = hyper_args.get("if_use_gae", None)

    # trade_args
    job_id = env.trade_args["job_id"]  # 不要用 get, 確認欄位都在
    max_risk = env.trade_args.get("max_risk", None)  # not required, not in use
    target_return = env.trade_args.get("target_return", +np.inf)

    # BrunhildEnv
    env_name = env.env_name
    max_step = env.max_step
    state_dim = env.state_dim
    action_dim = env.action_dim
    if_discrete = env.if_discrete

    assert break_step is not None, "break_step cannot be None, please check hyper_args"
    if env.trade_args.get("price_env") == PriceEnv.TRAIN:
        assert max_step > 0, "max_step cannot be zero, you have to brunhild.reset() first"

    gym_env_args = {
        "env_num": env_num,
        "env_name": env_name,
        "max_step": max_step,
        "state_dim": state_dim,
        # "max_memo": max_memo,  # use Agent_base default
        "action_dim": action_dim,
        "if_discrete": if_discrete,
        "target_return": target_return,
        "job_id": job_id,
        "learner_gpus": learner_gpus,
        "learning_rate": learning_rate,
        "net_dim": net_dim,
        "hyper_args": hyper_args,
        "break_step": break_step,
        "max_risk": max_risk,  # not strictly required
        "init_steps": init_steps  # not strictly required
    }

    # hyperparameter tuning
    if num_layer is not None:
        gym_env_args["num_layer"] = num_layer
    if ent_coef is not None:
        gym_env_args["ent_coef"] = ent_coef
    if tau is not None:
        gym_env_args["tau"] = tau
    if ratio_clip is not None:
        gym_env_args["ratio_clip"] = ratio_clip
    if lambda_gae_adv is not None:
        gym_env_args["lambda_gae_adv"] = lambda_gae_adv
    if lambda_entropy is not None:
        gym_env_args["lambda_entropy"] = lambda_entropy
    if if_use_gae is not None:
        gym_env_args["if_use_gae"] = if_use_gae
    if lr_gamma is not None:
        gym_env_args["lr_gamma"] = lr_gamma
    if min_learning_rate is not None:
        gym_env_args["min_learning_rate"] = min_learning_rate
    # if if_print:
    #     print(json.dumps(env_args, indent=4, default=str))
        # env_args_repr = repr(env_args)
        # env_args_repr = env_args_repr.replace(",", f",\n   ")
        # env_args_repr = env_args_repr.replace("{", "{\n    ")
        # env_args_repr = env_args_repr.replace("}", ",\n}")
        # print(f"env_args = {env_args_repr}")

    # check if None
    has_none_value = any(value is None for value in gym_env_args.values())
    if has_none_value:
        print("We are experimenting gym_env_args function, existing program....")
        print(f"env_args: {json.dumps(gym_env_args, indent=4, default=str)}")
        sys.exit(-1)

    return gym_env_args


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


"""
def build_env(env_func=None, env_args=None):
    env = env_func(**kwargs_filter(env_func.__init__, env_args.copy()))

    for attr_str in ("state_dim", "action_dim", "max_step", "if_discrete", "target_return"):
        if (not hasattr(env, attr_str)) and (attr_str in env_args):
            setattr(env, attr_str, env_args[attr_str])

    return env
"""


def build_env(env=None, env_func=None, env_args=None, gpu_id=-1):  # [ElegantRL.2021.12.12]
    if env is not None:
        env = deepcopy(env)
    else:
        try:
            env_args0 = deepcopy(env_args)
            env_args0["device_id"] = gpu_id  # -1 means CPU, int >=1 means GPU id
            env_args1 = kwargs_filter(env_func.__init__, env_args0)

            env = env_func(**env_args1)
        except TypeError as error:
            if repr(error) == "TypeError(\"make() missing 1 required positional argument: 'id'\")":
                import gym
                gym.logger.set_level(40)
                env = env_func(id=env_args["id"])
            else:
                raise TypeError(f"Meet ERROR: {error}\n"
                                f"Check env_args: {env_args}")

    env.max_step = env.max_step if hasattr(env, "max_step") else env_args["max_step"]
    env.if_discrete = env.if_discrete if hasattr(env, "if_discrete") else env_args["if_discrete"]
    return env


def get_trade_args(file, file_type: str = "json"):
    """
    read trade_Args.json at project root, unless you assign another file to overt-write the default
    Parameters
    ----------
    file_type
    file: str, the file that you specify by force to over-write the original setting

    Returns
    -------
    return content of the file

    """
    try:
        assert file is not None, "get_trade_args(file) cannot be None"

        if file[0] == "/":
            # absolute file path
            dir_file = file
        else:
            # relative file path
            project_root = get_project_root()
            dir_file = f"{project_root}/{file}"

        logger.info(f"[train.config] Loading trade_args from '{dir_file}'")

        if file_type == "object_eval":
            with open(dir_file, "r") as f:
                data_dict = ast.literal_eval(f.read())
                return data_dict

        if file_type == "json":
            with open(dir_file, newline="") as json_file:
                data_dict = json.load(json_file)

                return data_dict
        raise Exception(f"get_trade_args unknown file_type {file_type}")
    except Exception as e:
        print(f"get_trade_args {readable_error(e, __file__)}")
        time.sleep(3)
        sys.exit()



def get_tech_args(file_name: str, file_type: str = "json"):
    """
    開啟使用的 technical indicators
    """

    try:
        assert file_name is not None, "get_tech_args(file) cannot be None"

        if file_name[0] == "/":
            # absolute file path
            dir_file = file_name
        else:
            # relative file path
            project_root = get_project_root()
            dir_file = f"{project_root}/{file_name}"

        logger.info(f"[train.config] Loading tech_args from {dir_file}")

        if file_type == "object_eval":
            with open(dir_file, "r") as f:
                data_dict = ast.literal_eval(f.read())
                return data_dict

        if file_type == "json":
            with open(dir_file, newline="") as json_file:
                data_dict = json.load(json_file)
                return data_dict

        raise Exception(f"get_tech_args unknown file_type {file_type}")
    except Exception as e:
        print(f"get_tech_args {readable_error(e, __file__)}")
        time.sleep(3)
        sys.exit()


def get_hyper_args(file, alg, env_name):
    """
    取出 hyper_args 的資料
    預設在根目錄，但是如果有加入 pod_dir，那就會存取其他目錄
    """
    try:
        assert file is not None, "get_hyper_args(file) cannot be None"

        if file[0] == "/":
            # absolute file path
            dir_file = file
        else:
            # relative file path
            project_root = get_project_root()
            dir_file = f"{project_root}/{file}"

        logger.info(f"[train.config] opening hyper_args.json {dir_file}")

        file_type = file.split(".")[-1]
        if file_type == "yml":
            with open(dir_file, "r") as f:
                _hyper_args = yaml.safe_load(f)[alg][env_name]
                return _hyper_args

        if file_type == "json":
            with open(dir_file, newline="") as json_file:
                data_dict = json.load(json_file)
                return data_dict

        raise Exception(f"get_hyper_args unknown file_type {file_type}")
    except Exception as e:
        print(f"get_hyper_args {readable_error(e, __file__)}")
        time.sleep(3)
        sys.exit()


def get_env_args(file):
    """
    取出 hyper_args 的資料
    預設在根目錄，但是如果有加入 pod_dir，那就會存取其他目錄
    """
    try:
        assert file is not None, "get_env_args(file) cannot be None"

        if file[0] == "/":
            # absolute file path
            dir_file = file
        else:
            # relative file path
            project_root = get_project_root()
            dir_file = f"{project_root}/{file}"

        logger.info(f"[train.config] opening env_args.json {dir_file}")

        file_type = file.split(".")[-1]

        if file_type == "json":
            with open(dir_file, newline="") as json_file:
                data_dict = json.load(json_file)
                return data_dict

        raise Exception(f"get_hyper_args unknown file_type {file_type}")
    except Exception as e:
        print(f"get_env_args {readable_error(e, __file__)}")
        time.sleep(3)
        sys.exit()


def get_all_args(_cmd_args):
    """舊的，但還在使用，最主要是 `preprocessor/plotter` 和 `Gen_Indicator`"""

    _trade_args = get_trade_args(_cmd_args["trade_args_path"])

    if _cmd_args.get("tech_id") is not None:
        _trade_args.update({"tech_id": _cmd_args.get("tech_id")})

    if _cmd_args.get("trade_interval") is not None:
        _trade_args.update({"trade_interval": _cmd_args.get("trade_interval")})

    # different setting for DEV and PROD
    print(f"preprocessor is running at {_trade_args['tech_id']}")

    tech_id = _trade_args["tech_id"]

    # technical indicators
    tech_file_name = f"{tech_id}.json"
    tech_args = get_tech_args(tech_file_name)
    _tech_list = tech_args.get("tech_list")

    return {
        "exch_mode": _trade_args["exch_mode"],
        "tech_id": tech_id,
        "trade_args": _trade_args,
        "tech_list": _tech_list,
    }


@deprecated(version='?', reason="Please use `/Trade/Binance/BinanceTrade/load_args`")
def load_args(env_name=None, cmd_args=None):
    """

    Parameters
    ----------
    env_name
    cmd_args
        - config_dir:
            - type: str
            - description: config dir that is relative to project root

    Returns
    -------

    """
    # get trade_args
    # IMPORTANT: load pod or directory default
    has_pod_config = cmd_args is not None and "pod_dir" in cmd_args
    config_dir = ""
    if has_pod_config:
        config_dir = cmd_args["pod_dir"]

        # get trade_args
        trade_args = get_trade_args(f"{config_dir}/trade_args.json")
    else:
        trade_args = get_trade_args(cmd_args["trade_args_path"])

    # 如有取代的，就拉出取代，但 pod 最大
    """
    if cmd_args is not None and not has_pod_config:
        if "trade_args" in cmd_args and cmd_args["trade_args_path"] is not None:
            trade_args_specific = get_trade_args(cmd_args["trade_args_path"])
            trade_args.update(trade_args_specific)
    """

    # cmd_args overwrite 所有，除了 pod_config 中的 trade_args and market_args
    trade_args.update(cmd_args)

    env_name = env_name if env_name is not None else trade_args["env_name"]
    hyper_args = get_hyper_args(f"{config_dir}/hyper_args.yml", "PPO", env_name)

    # if trade_args["tech_id"] not in trade_args:
    #     raise Exception(f"Trade_args for {trade_args["tech_id"]} not found")
    # else:
    exch_mode = trade_args["exch_mode"]
    tech_id = trade_args["tech_id"]
    tech_list = trade_args[tech_id]["tech_list"]

    # make app_env from str to list
    trade_args["app_env"] = parse_app_env(trade_args["app_env"])

    order_trade_args_checker(trade_args)

    return exch_mode, hyper_args, trade_args, tech_list
