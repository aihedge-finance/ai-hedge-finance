"""
DRL models is a container to bind everything together
"""
from __future__ import annotations

import torch

from deprecated import deprecated

from ahf.rl.agents.AgentPPO import AgentPPO
# from agents.AgentDDPG import AgentDDPG
# from agents.AgentSAC import AgentSAC
# from agents.AgentTD3 import AgentTD3
from ahf.rl.train.config import Arguments
from ahf.rl.train.run import init_agent
from ahf.rl.train.run import train_and_evaluate, train_and_evaluate_mp
from ahf.rl.train.evaluator import Evaluator
from ahf.rl.train.config import get_gym_env_args
from ahf.core.enums import PriceEnv
from ahf.utils.utils import readable_error
from api.PriceFetcher import PriceFetcherTrain

# from elegantrl.agents import AgentA2C

# MODELS = {"ddpg": AgentDDPG, "td3": AgentTD3, "sac": AgentSAC, "ppo": AgentPPO}
MODELS = {"ppo": AgentPPO}
OFF_POLICY_MODELS = ["ddpg", "td3", "sac"]
ON_POLICY_MODELS = ["ppo"]


# MODEL_KWARGS = {x: config.__dict__[f"{x.upper()}_PARAMS"] for x in MODELS.keys()}
#
# NOISE = {
#     "normal": NormalActionNoise,
#     "ornstein_uhlenbeck": OrnsteinUhlenbeckActionNoise,
# }


class DRLAgent:
    """Implementations of DRL algorithms
    Attributes
    ----------
        env: environment class
             user-defined class
    Methods
    -------
        get_model()
            setup DRL algorithms
        train_model()
            train DRL algorithms in a train dataset
            and output the trained model
        DRL_prediction()
            make a prediction in a test dataset and get results
    """

    def __init__(self, env_cls, check_env, hyper_args, env_args, trade_args,
                 tech_args, strategy_cls, exch_api, logger,  Tuning_mode=False):
        self.price_fetcher = None
        self.env = None
        self.env_cls = env_cls
        self.check_env = check_env
        self.hyper_args = hyper_args
        self.env_args = env_args
        self.trade_args = trade_args
        self.tech_args = tech_args
        self.strategy_cls = strategy_cls
        self.exch_api = exch_api
        self.tuning_mode = Tuning_mode

        # self.turbulence_array = turbulence_array
        self.logger = logger

        if check_env is not None and trade_args["train_mode"] == "DEV":
            check_env(hyper_args,
                      env_args,
                      trade_args,
                      tech_args,
                      strategy_cls,
                      logger)

    def get_model(self, model_name, model_kwargs=None):
        """
        _summary_
            This is to construct model by using init env_cls -> get_gym_env_args -> Arguments => model
            But we use model_kwargs to overwrite some hyperparameter setting in order to achieve tuning purpose.
            We can also do ensemble training when we specify different model_name name such as PPO, DDPG etc.
        Args:
            model_name (str): model that we are using, such as PPO or DDPG
            model_kwargs (_type_): args to override hyperparameter and trading_args in order to achieve tuning or ensemble training

        Raises:
            NotImplementedError: Model has to be implemented, such as PPO or DDPG
            ValueError: This is used to load overwriting arg to do hyper-tuning

        Returns:
            _type_: _description_

        """
        agent = MODELS[model_name]
        if model_name not in MODELS:
            raise NotImplementedError("NotImplementedError")

        # Hyperparameter tuning purpose.
        if model_kwargs is not None:
            try:
                # Trade_args
                self.trade_args["trade_interval"] = model_kwargs.get("trade_interval", self.trade_args["trade_interval"])
                self.trade_args["stacking_lookback"] = model_kwargs.get("stacking_lookback", self.trade_args["stacking_lookback"])
                self.trade_args["take_profit_margin"] = model_kwargs.get("take_profit_margin", self.trade_args["take_profit_margin"])
                self.trade_args["stop_loss_margin"] = model_kwargs.get("stop_loss_margin", self.trade_args["stop_loss_margin"])

                # IMPORTANT tech_args
                self.tech_args["tech_list"] = model_kwargs.get("tech_list", self.tech_args["tech_list"])
                self.tech_args.update(model_kwargs)

                # hyper_args
                self.hyper_args["net_dim"] = model_kwargs.get("net_dim",  self.hyper_args["net_dim"])
                self.hyper_args["learning_rate"] = model_kwargs.get("learning_rate", self.hyper_args["learning_rate"])
                self.hyper_args["batch_size"] = model_kwargs.get("batch_size", self.hyper_args["batch_size"])
                self.hyper_args["num_layer"] = model_kwargs.get("num_layer", self.hyper_args["num_layer"])
                # self.hyper_args["max_memo"] = model_kwargs.get("max_memo", self.hyper_args["max_memo"])  # use default
                self.hyper_args["rl_mode"] = model_kwargs.get("rl_mode", self.hyper_args["rl_mode"])

                if "tau" in [self.hyper_args, model_kwargs]:
                    self.hyper_args["tau"] = model_kwargs.get("tau", self.hyper_args["tau"])
                if "ratio_clip" in [self.hyper_args, model_kwargs]:
                    self.hyper_args["ratio_clip"] = model_kwargs.get("ratio_clip", self.hyper_args["ratio_clip"])
                if "lambda_gae_adv" in [self.hyper_args, model_kwargs]:
                    self.hyper_args["lambda_gae_adv"] = model_kwargs.get("lambda_gae_adv", self.hyper_args["lambda_gae_adv"])
                if "lambda_entropy" in [self.hyper_args, model_kwargs]:
                    self.hyper_args["lambda_entropy"] = model_kwargs.get("lambda_entropy", self.hyper_args["lambda_entropy"])
                if "if_use_gae" in [self.hyper_args, model_kwargs]:
                    self.hyper_args["if_use_gae"] = model_kwargs.get("if_use_gae", self.hyper_args["if_use_gae"])

                # env_args
                self.env_args["DEFAULT_SELL_DELTA_CHANGE"] = model_kwargs.get("default_sell_delta_change", self.env_args["DEFAULT_SELL_DELTA_CHANGE"])
                self.env_args["SELL_DELTA_LIMIT_UPPER"] = model_kwargs.get("sell_delta_limit_upper", self.env_args["SELL_DELTA_LIMIT_UPPER"])

                # model.ent_coef = model_kwargs.get("ent_coef", model.ent_coef)
                # model.gamma = model_kwargs.get("gamma", model.gamma)
                # see only work in sb -> gym.Env.seed
                # model.seed = model_kwargs["seed"] if "seed" in model_kwargs else model.seed
                # Brunhild Specific operation

            except BaseException:
                raise ValueError("Fail to read arguments, please check 'model_kwargs' input.")

        try:
            self.price_fetcher = PriceFetcherTrain(self.trade_args,
                                              PriceEnv(self.trade_args["price_env"]),
                                              self.logger,
                                              catchup_price=False)

            # IMPORTANT
            # init environment + tech data is initiated at `def create_tech_if_not_exist` this line
            self.env = self.env_cls(self.hyper_args, self.env_args, self.trade_args, self.tech_args,
                                    self.strategy_cls, self.logger,
                                    price_fetcher=self.price_fetcher,
                                    exch_api=self.exch_api,
                                    rl_mode=self.hyper_args.get("rl_mode"))
            self.env.reset()

            self.env.env_num = self.hyper_args.get("env_num", 1) or 1
            gym_env_args = get_gym_env_args(self.env, self.hyper_args, if_print=True)

            # This step join everything together as final model with parameters
            model = Arguments(agent, env=self.env, env_func=self.env_cls, env_args=gym_env_args)
            # no need anymore, it has get_if_off_policy()
            # model.if_off_policy = model_name in OFF_POLICY_MODELS

            return model
        except Exception as e:
            err_str = readable_error(e, __file__)
            raise Exception(err_str)


    @staticmethod
    def train_model(model,
                    agent_id=0,
                    cwd=None,
                    total_time_steps=None,
                    load_if_exists_prompt=False,
                    logger=None,
                    trial=None) -> Evaluator:
        """
        REQUIRE CODE UPGRADE

        取得參數後開始執行訓練和 evaluation
        get_model() --> train_model()
        """
        # we do this because we may need to overwrite cwd from other entry point
        model.cwd = cwd if cwd is not None else model.cwd
        model.break_step = total_time_steps
        if model.env_num == 1:
            evaluator = train_and_evaluate(agent_id, model, load_if_exists_prompt, logger, trial)
        else:
            evaluator = train_and_evaluate_mp(agent_id, model, load_if_exists_prompt, logger, trial)

        return evaluator

    def DRL_prediction(self, model: Arguments, pod_cwd):

        self.price_fetcher.reset()
        model.env.env_num = 1
        model.cwd = pod_cwd if pod_cwd is not None else model.cwd

        """init"""
        env = model.env

        # load agent
        try:
            agent = init_agent(model, gpu_id=0)
            act = agent.act
            device = agent.device
        except BaseException:
            raise ValueError("Fail to load agent!")

        # test on the testing env
        _torch = torch
        state = env.reset()

        assert hasattr(env, "strategy"), "env has to contain strategy"
        assert hasattr(env, "exch_env") and hasattr(env.exch_env, "ds"), "env has to contain exch_env and exch_env.ds"

        episode_returns = []  # the cumulative_return / initial_account
        with _torch.no_grad():
            for i in range(env.max_step):
                s_tensor = _torch.as_tensor((state,), device=device)
                a_tensor = act(s_tensor)  # action_tanh = act.forward()
                action = (
                    a_tensor.detach().cpu().numpy()[0]
                )  # not need detach(), because with torch.no_grad() outside
                state, reward, done, _ = env.step(action)
                env.render(mode="console")

                episode_return = env.exch_env.cumulative_returns
                episode_returns.append(episode_return)
                if done:
                    break
        print("Test Finished!")
        # return episode total_assets on testing data
        print("episode_return", episode_return)

        return episode_returns, env.exch_env.ds, env.strategy


