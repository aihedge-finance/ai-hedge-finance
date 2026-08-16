import os
import time
import torch
import optuna
from typing import Tuple
import multiprocessing as mp
from collections import deque
from ahf.utils.utils import convert_to_min
from ahf.utils.helper import save_args_prior_train

from optuna.exceptions import TrialPruned
from ahf.utils.utils import readable_error, get_project_root
from ahf.rl.train.utils import (
    init_agent,
    init_buffer,
    init_evaluator
)
from ahf.rl.train.evaluator import Evaluator
from ahf.rl.train.config import Arguments

from ahf.rl.agents.AgentBase import AgentBase
from ahf.rl.train.config import build_env


def train_and_evaluate(agent_id, args, load_if_exists_prompt, logger,
                       trial=None, patience=50
                       ) -> Evaluator:
    """
    The training and evaluating loop.

    Parameters
    ----------
    patience: for early stopping, if lose patience, then break
    args: an object of ``Arguments`` class, which contains all hyperparameters.
    load_if_exists_prompt: if folder already exist, it will not ask if deletion id required.
    logger: logger
    agent_id:
    trial:
    """
    try:
        torch.set_grad_enabled(False)
        args.init_before_training(agent_id, load_if_exists_prompt=load_if_exists_prompt)
        gpu_id = int(args.learner_gpus[agent_id])

        """init"""
        env = args.env
        steps = int(float(args.init_steps) * 10000)  # in W

        agent = init_agent(args, gpu_id, env)
        buffer = init_buffer(args, gpu_id)
        evaluator: Evaluator = init_evaluator(args, gpu_id)

        # agent.state = env.reset()  # init_agent 已經執行過 reset()
        if args.if_off_policy:
            trajectory, step = agent.explore_env(env, args.num_seed_steps * args.num_steps_per_episode, True)
            buffer.update_buffer(trajectory)
            steps += step

        # save StockTradeEnv and env_args for record
        save_args_prior_train(args.cwd, env, logger)

        """start training"""
        cwd = args.cwd
        break_step = args.break_step
        horizon_len = args.target_step  # args.env.max_step
        if_allow_break = args.if_allow_break
        if_off_policy = args.if_off_policy

        ####### START OF EARLY STOPPING LOGIC #######
        best_reward = float("-inf")
        episodes_without_improvement = 0
        reward_history = deque(maxlen=100)
        degradation_threshold = 0.05
        ####### END OF EARLY STOPPING LOGIC #######

        episode = 0
        if_train = True
        while if_train:
            episode += 1
            _state = env.reset()
            assert _state is not None, "state cannot be None after reset"

            # start from 1, zero position is loaded at reset
            # env.exch_env.ds.step_idx(AppEnv.TRAIN)

            # trajectory, step = agent.explore_env(env, horizon_len, False)
            trajectory = agent.explore_env(env, horizon_len, False)

            if env.exch_env.ds.done_kelly_active and env.exch_env.ds.done_kelly_mode in ("auto", "true"):
                period = int(convert_to_min(env.trade_args["must_trade_max"]) / convert_to_min(env.trade_args["trade_interval"]))
                perfs = env.exch_env.ds.get_cumulative_realized_pnl_range(agent.idx - period, agent.idx)
                # if all are positive then stop kelly_cap stopping
                kelly_enabled = False if len(perfs) >= period // 2 and all(perf > 0.1 for perf in perfs) else True
                logger.info(f"====== kelly enabled: {kelly_enabled} =======")

                env.exch_env.ds.done_kelly_active = kelly_enabled
            else:
                logger.info(f"====== kelly enabled: {env.exch_env.ds.done_kelly_active} =======")

            steps += agent.idx  # this is PPO's own idx, do not change this
            logger.info("\n===== Process Report =========================================================\n"
                        f"| agent_id: {agent_id}\n"
                        f"| total episode trained: {episode/1_000:.3f}k steps\n"
                        f"| evaluator.total_step: {evaluator.total_step / 1_000_000:.3f}M / "
                        f"break_step: {break_step / 1_000_000:.3f}M "
                        f"[{evaluator.total_step / break_step * 100:.3f}%]\n"
                        f"| done_kelly_active: {env.exch_env.ds.done_kelly_active}\n"
                        f"===============================================================================\n")

            current_lr = agent.act_optimizer.param_groups[0]['lr']
            if current_lr <= agent.min_learning_rate:
                print(f"Reached minimum learning rate after episode {episode}")

            if if_off_policy:
                buffer.update_buffer(trajectory)
                torch.set_grad_enabled(True)
                logging_tuple = agent.update_net(buffer)
                torch.set_grad_enabled(False)
            else:
                torch.set_grad_enabled(True)
                buffer[:] = trajectory
                logging_tuple = agent.update_net(buffer)
                torch.set_grad_enabled(False)

            # mean episode_reward
            r_exp = agent.reward_tracker.mean()
            # step_exp = agent.step_tracker.mean()

            _state = env.reset()
            assert _state is not None, "state cannot be None after reset"

            (if_reach_goal, if_save) = evaluator.evaluate_save_and_plot(agent.act, steps, r_exp,
                                                                        logging_tuple)  # step_exp,
            dont_break = not if_allow_break
            not_reached_goal = not if_reach_goal
            stop_dir_absent = not os.path.exists(os.path.expanduser(f"{cwd}/stop"))
            if_train = (
                    (dont_break or not_reached_goal)
                    and evaluator.total_step <= break_step
                    and stop_dir_absent
            )

            ####### START OF EARLY STOPPING LOGIC #######
            reward_history.append(r_exp)
            if len(reward_history) >= 100:
                avg_reward = sum(reward_history) / len(reward_history)

                if avg_reward > best_reward:
                    best_reward = avg_reward
                    episodes_without_improvement = 0
                    # beast actor is saved below at `evaluator.evaluate_save_and_plot`
                    # torch.save(model.state_dict(), 'best_model.pth')
                elif avg_reward < best_reward - degradation_threshold:
                    episodes_without_improvement += 1

                if episodes_without_improvement >= patience:
                    logger.info(f"Early stopping at episode {episode}")
                    logger.info(f"Reason: Degradation")
                    if_train = False

            if not if_train:
                logger.info(f"don't_break:{dont_break}, not_reached_goal:{not_reached_goal},"
                            f"evaluator.total_step:{evaluator.total_step / 1000000:.3f}M, "
                            f"break_step:{break_step / 1000000:.3f}M, "
                            f"episode run: {episode/1000}k, "
                            f"episodes without improvement {episodes_without_improvement} >= patience {patience} "
                            f"[{evaluator.total_step / break_step * 100:.3f}%]\n"
                            f"stop_dir_absent:{stop_dir_absent}")

            if if_save:
                agent.save_or_load_agent(cwd, if_save=True)

            if env.trade_args["image"]:
                # TODO
                # env.draw_trades()
                pass

            # Optuna reporting and pruning mechanism
            # Report intermediate objective value.
            if trial is not None:
                intermediate_value = evaluator.max_reward()
                trial.report(intermediate_value, episode-1)

                # Handle pruning based on the intermediate value.
                if trial.should_prune():
                    raise optuna.TrialPruned()

        logger.info(f"| UsedTime: {time.time() - evaluator.start_time:.0f} | SavedDir: {cwd}")

        # Draw the graph
        env.draw_cumulative_return(args, torch)

        return evaluator

    except TrialPruned as e:
        raise
    except Exception as e:
        err = readable_error(e, __file__)
        logger.error(err)
        raise


"""train multiple process"""


def train_and_evaluate_mp(args: Arguments, logger):
    args.init_before_training()

    # save StockTradeEnv and env_args for record
    save_args_prior_train(args.cwd, args.env, logger)

    process = list()
    mp.set_start_method(method="spawn", force=True)  # force all the multiprocessing to "spawn" methods

    evaluator_pipe = PipeEvaluator()
    process.append(mp.Process(target=evaluator_pipe.run, args=(args,)))

    worker_pipe = PipeWorker(args.worker_num)
    process.extend([mp.Process(target=worker_pipe.run, args=(args, worker_id))
                    for worker_id in range(args.worker_num)])

    learner_pipe = PipeLearner()
    process.append(mp.Process(target=learner_pipe.run, args=(args, evaluator_pipe, worker_pipe)))

    for p in process:
        p.start()

    process[-1].join()  # waiting for learner
    process_safely_terminate(process)


class PipeWorker:
    def __init__(self, worker_num: int):
        self.worker_num = worker_num
        self.pipes = [mp.Pipe() for _ in range(worker_num)]
        self.pipe1s = [pipe[1] for pipe in self.pipes]

    def explore(self, agent: AgentBase):
        act_dict = agent.act.state_dict()

        for worker_id in range(self.worker_num):
            self.pipe1s[worker_id].send(act_dict)

        traj_lists = [pipe1.recv() for pipe1 in self.pipe1s]
        return traj_lists

    def run(self, args: Arguments, worker_id: int):
        torch.set_grad_enabled(False)
        gpu_id = args.learner_gpus

        """init"""
        env = build_env(args.env, args.env_func, args.env_args)
        agent = init_agent(args, gpu_id, env)

        """loop"""
        target_step = args.target_step
        if args.if_off_policy:
            trajectory = agent.explore_env(env, args.target_step)
            self.pipes[worker_id][0].send(trajectory)
        del args

        while True:
            act_dict = self.pipes[worker_id][0].recv()
            agent.act.load_state_dict(act_dict)
            trajectory = agent.explore_env(env, target_step)
            self.pipes[worker_id][0].send(trajectory)


# import wandb
class PipeLearner:
    def __init__(self):
        # wandb.init(project="DDPG_H")
        pass

    @staticmethod
    def run(args: Arguments, comm_eva: mp.Pipe, comm_exp: mp.Pipe):
        torch.set_grad_enabled(False)
        gpu_id = args.learner_gpus
        cwd = args.cwd
        # wandb.init(project="DDPG_H")

        """init"""
        agent = init_agent(args, gpu_id)
        buffer = init_buffer(args, gpu_id)

        """loop"""
        if_train = True
        while if_train:
            traj_list = comm_exp.explore(agent)
            steps, r_exp = buffer.update_buffer(traj_list)

            torch.set_grad_enabled(True)
            logging_tuple = agent.update_net(buffer)
            torch.set_grad_enabled(False)
            # wandb.log({"obj_cri": logging_tuple[0], "obj_act": logging_tuple[1]})
            if_train, if_save = comm_eva.evaluate_and_save_mp(agent.act, steps, r_exp, logging_tuple)
        agent.save_or_load_agent(cwd, if_save=True)
        print(f"| Learner: Save in {cwd}")

        env = build_env(env_func=args.env_func, env_args=args.env_args)
        buffer.get_state_norm(
            cwd=cwd,
            state_avg=getattr(env, "state_avg", 0.0),
            state_std=getattr(env, "state_std", 1.0),
        )
        if hasattr(buffer, "save_or_load_history"):
            print(f"| LearnerPipe.run: ReplayBuffer saving in {cwd}")
            buffer.save_or_load_history(cwd, if_save=True)


class PipeEvaluator:
    def __init__(self):
        self.pipe0, self.pipe1 = mp.Pipe()

    def evaluate_and_save_mp(self, act, steps: int, r_exp: float, logging_tuple: tuple) -> Tuple[bool, bool]:
        if self.pipe1.poll():  # if_evaluator_idle
            if_train, if_save_agent = self.pipe1.recv()
            act_state_dict = act.state_dict().copy()  # deepcopy(act.state_dict())
        else:
            if_train = True
            if_save_agent = False
            act_state_dict = None

        self.pipe1.send((act_state_dict, steps, r_exp, logging_tuple))
        return if_train, if_save_agent

    def run(self, args: Arguments):
        torch.set_grad_enabled(False)
        gpu_id = args.learner_gpus

        """init"""
        agent = init_agent(args, gpu_id)
        evaluator = init_evaluator(args, gpu_id)

        """loop"""
        cwd = args.cwd
        act = agent.act
        break_step = args.break_step
        if_allow_break = args.if_allow_break
        save_gap = args.save_gap
        del args

        if_save = False
        if_train = True
        if_reach_goal = False
        save_counter = 0
        while if_train:
            act_dict, steps, r_exp, logging_tuple = self.pipe0.recv()

            if act_dict:
                act.load_state_dict(act_dict)
                if_reach_goal, if_save = evaluator.evaluate_save_and_plot(act, steps, r_exp, logging_tuple)

                save_counter += 1
                if save_counter == save_gap:
                    save_counter = 0
                    torch.save(act.state_dict(), f"{cwd}/actor_{evaluator.total_step:012}.pth")
            else:
                evaluator.total_step += steps

            if_train = not ((if_allow_break and if_reach_goal)
                            or evaluator.total_step > break_step
                            or os.path.exists(os.path.expanduser(f"{cwd}/stop")))
            self.pipe0.send((if_train, if_save))

        print(f"| UsedTime: {time.time() - evaluator.start_time:>7.0f} | SavedDir: {cwd}")

        while True:  # wait for the forced stop from main process
            self.pipe0.recv()
            self.pipe0.send((False, False))


def process_safely_terminate(process: list):
    for p in process:
        try:
            p.kill()
        except OSError as e:
            print(e)
