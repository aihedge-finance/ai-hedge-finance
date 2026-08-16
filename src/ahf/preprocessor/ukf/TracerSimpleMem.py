import gc
import os
import sys
import math
import numpy as np
import pandas as pd
from os import path
import datetime as dt
from typing import Dict
from loguru import logger
from ahf.utils.utils import readable_error, datesffmt, create_dir_if_non_exist, remove_char

from ahf.preprocessor.ukf.TracerSimple import TracerSimple

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def update_slope(pos, pos_last):
    if not math.isnan(pos):
        re = (pos - pos_last) / pos_last * 100 if pos_last > 0 else 0
        return re
    else:
        return 0


class TracerSimpleMem(TracerSimple):
    def __init__(self,
                 trade_args: Dict,
                 strategy: str,
                 job_id: str,
                 proc_cov: float,
                 logger,
                 initial_state_mean: float=None,
                 is_unittest=False, verbose=True):

        """

        Parameters
        ----------
        trade_args
        strategy:
        job_id:
        proc_cov: 最重要的控制項 process_noise_covariance
        logger:
        initial_state_mean: initial position, only
        is_unittest:
        verbose:
        """

        self.name = "TracerSimpleMem"
        self.trade_args = trade_args
        self.tracer_hist = None
        self.slope = None
        self.verbose = True if is_unittest else verbose

        self.proc_cov: float = proc_cov

        self.initial_state_mean: float = initial_state_mean  # pos only, no vel, acc
        self.initial_state_cov = None  # 不知道就不要給

        self.theta_id = f"{self.trade_args['exchange']}_{self.trade_args['symbol']}_" \
                        f"{self.trade_args['trade_interval']}_{job_id}"

        self.dt: float = remove_char(trade_args["trade_interval"])

        if is_unittest:
            unittest_str = "_UNITTEST"
        else:
            unittest_str = ""
        self.file_dir = os.path.expanduser(f"{self.trade_args['ukf_data_path']}/{self.name}_{self.theta_id}{unittest_str}.csv")
        create_dir_if_non_exist(self.file_dir)

        TracerSimple.__init__(self, strategy, job_id, self.dt, proc_cov, logger, initial_state_mean)

        self.tracer = np.zeros(10000)
        self.state_cov = np.zeros(10000)

    def catchup(self, priceA, tracer_hist=None):
        # IMPORTANT+ CAREFUL: This is used with side-line (Jobber cron job),
        # don't mix parameters with online (running program)
        try:
            if self.verbose:
                self.logger.info(f"[TracerSimpleMem] catching up for TracerSimpleMem proc_cov:{self.proc_cov}")
            if "date" not in priceA and priceA.index.name != "date":
                raise Exception("date index column does not exist")

            if priceA.index.name != "date":
                price_pd = priceA.copy().set_index("date", drop=False, inplace=False)
            else:
                price_pd = priceA.copy()

            if tracer_hist is None:
                tracer_loaded, tracer_hist = self.load_tracer()
                self.tracer_hist = tracer_hist
            else:
                tracer_loaded = True

            if not tracer_loaded or tracer_hist is None:
                self.logger.info("[TracerSimpleMem] tracer_history does not exist, building up")
                last_price_dt_index = last_save_dt_index = price_pd.index[0].to_pydatetime()  # .strftime(datefmt)
                last_save_idx = 0

                self.initial_state_mean = initial_state_mean = price_pd["open"].iloc[0]
                self.initial_state_cov = initial_state_cov = 0.
            else:
                last_price_dt_index = price_pd.index[-1].to_pydatetime()
                last_save_dt_index = tracer_hist["dt_idx"].iloc[-1]
                last_save_idx = tracer_hist.index[-1]
                # last_save_idx = int(tracer_hist["idx"][-1])  # record it to truncate data for saving

                # load previous stored ukf params
                self.initial_state_mean = initial_state_mean = tracer_hist["tracer"].iloc[-1]
                self.initial_state_cov = initial_state_cov = tracer_hist["state_cov00"].iloc[-1]

            # capture only since from last save part for training
            # a = last_price_dt_index.astimezone(dt.timezone.utc)
            # b = last_save_dt_index.tz_convert("UTC")
            t_proc_cov = last_price_dt_index - last_save_dt_index
            if t_proc_cov.total_seconds() < 0:
                s = f"{self.trade_args['symbol']} price did not catch up with history data, " \
                    f"price:  {last_price_dt_index} size {len(price_pd.index)}, \n" \
                    f"history:{last_save_dt_index}  size {len(tracer_hist.index)}"
                raise Exception(s)

            priceA_train = price_pd[last_save_dt_index:]
            # priceA_train = priceA_train[1:]

            n = len(priceA_train.index)

            """
            # right now, we don't need it, we may need that later

            # do nothing when it is below trade_interval
            if n <= min_elapsed:  # 1 hour
                self.logger.info('[UKF] too few new prices, no need to train new mu and sd.')
                # too few data, not need to do anything
                return tracer_hist["mu"][-1], tracer_hist["sd"][-1]
            """

            # Setup variables
            dt_idx = np.zeros(n, dtype=dt.datetime)  # dt.datetime datetime_index

            if 0 <= n <= 1:
                # self.logger.info(f"Tracer_hist is up-to-date till {last_save_dt_index}")
                if self.verbose:
                    self.logger.info("[TracerSimpleMem] Tracer_hist is up to date")
                tracer_hist.set_index("dt_idx", inplace=True)
                del tracer_hist["idx"]
                tracer_hist = tracer_hist[~tracer_hist.index.duplicated(keep="first")]
                return True, tracer_hist

            # disable gc to speed up calculation in case there are too many data
            gc.disable()
            self.logger.info("[TracerSimpleMem] Start training new ukf market.")

            price_np = priceA_train["open"].to_numpy()
            self.slope = np.zeros(len(price_np))
            last_tracer = 0.

            l = len(price_np)
            for i, v in enumerate(price_np):
                idx_now = self.idx_now

                if idx_now >= len(self.tracer):
                    self.tracer = np.append(self.tracer, np.zeros(10000))
                    self.state_cov = np.append(self.state_cov, np.zeros(10000))

                tracer_new, state_cov_new, _, _ = self.update(self.dt, v)

                index = priceA_train.index[i]
                dt_idx[idx_now] = index.to_pydatetime()

                if i == 0:
                    self.slope[i] = update_slope(self.initial_state_mean, tracer_new)
                else:
                    self.slope[i] = update_slope(last_tracer, tracer_new)

                self.tracer[idx_now] = tracer_new
                self.state_cov[idx_now] = state_cov_new

                last_tracer = tracer_new

                if idx_now % 1000 == 0:
                    msg = "\r{0}, INFO {1} [UKF] " \
                          "Training at {2:.2f}%\r".format(dt.datetime.now().strftime(datesffmt),
                                                          f"{self.trade_args['exchange']}_"
                                                          f"{self.trade_args['symbol']}",
                                                          idx_now / l * 100)
                    sys.stdout.write(msg)
                    sys.stdout.flush()

            # add idx from last saved
            idx_run = self.idx_now + last_save_idx + 1

            # save slope and intercept
            need_header = True if last_save_idx == 0 else False

            idx_np = np.arange(start=0, stop=self.idx_now, step=1)

            tracer_np = self.tracer[:self.idx_now]
            proc_cov_np = np.empty(self.idx_now)
            proc_cov_np.fill(self.proc_cov)
            slope_np = self.slope[:self.idx_now]
            state_cov_np = self.state_cov[:self.idx_now]

            # only save when it is larger than 1, otherwise there are lots of zeros
            if self.idx_now > 1:
                self._save_tracer(self.file_dir, idx_np, dt_idx, price_np, tracer_np, slope_np,
                                  proc_cov_np, state_cov_np, need_header)

            # save
            gc.enable()

            new = {
                "idx": idx_np, "dt_idx": dt_idx, "price": price_np, "tracer": tracer_np,
                "slope": slope_np, "proc_cov": proc_cov_np, "state_cov00": state_cov_np
            }

            new_pd = pd.DataFrame(new)
            if len(new_pd.index) > 0:
                concat_pd = pd.concat([tracer_hist, new_pd], axis=0) if tracer_hist is not None else new_pd
            else:
                concat_pd = tracer_hist

            concat_pd.set_index("dt_idx", drop=True, inplace=True)
            del concat_pd["idx"]
            concat_pd = concat_pd[~concat_pd.index.duplicated(keep="first")]
            # concat_pd.drop_duplicates(subset=["dt_idx"], keep="first", inplace=True)

            return True, concat_pd

        except Exception as e:
            err = readable_error(e, __file__)
            self.logger.error(f"[TracerSimpleMem] {err}")
            return False, {"err": err}

    def load_tracer(self):
        """
        result from parent update()=tracer_new
        :return:
        """
        try:
            if not path.exists(self.file_dir):
                return False, None
                # r = pd.DataFrame([], columns=["idx", "slope", "intercept", "tracer", "mu", "sd",
                #                              "state_cov00", "state_cov01", "state_cov10", "state_cov11"])
                # r.to_csv(file_dir, mode="w", header=True, index=True, na_rep="NA", index_label="dt_idx")

            tracer_hist = pd.read_csv(self.file_dir,
                                      usecols=[
                                          "idx", "dt_idx", "price", "tracer", "slope",
                                          "proc_cov", "state_cov00"
                                      ],
                                      # index_col=0, parse_dates=True,
                                      dtype={
                                          "idx": int, "dt_idx": str, "price": float,
                                          "tracer": float, "slope": float, "proc_cov": float,
                                          "state_cov00": float
                                      })

            tracer_hist["dt_idx"] = pd.to_datetime(tracer_hist["dt_idx"])
            return True, tracer_hist
        except Exception as e:
            err = readable_error(e, __file__)
            self.logger.error(f"[TracerSimpleMem] {err}")
            sys.exit(-1)
            # return False, None


def generate_complex_signal(t):
    # Base signal
    signal = 2 * np.sin(2 * np.pi * 0.2 * t) + np.cos(2 * np.pi * 0.5 * t)

    # Add sudden jumps
    jumps = np.zeros_like(t)
    jump_points = [25, 50, 75]
    for jp in jump_points:
        jumps[jp:] += 1.5

    # Add exponential decay
    decay = 0.5 * np.exp(-0.2 * t)

    # Add polynomial trend
    trend = 0.01 * t ** 2 - 0.1 * t

    # Combine all components
    return signal + jumps + decay + trend

def main_complex():
    dt = 0.1
    proc_cov = 0.01
    ukf_poly = TracerSimple(strategy="TracerSimple",
                            job="UnitTest",
                            dt=dt,
                            proc_cov=proc_cov,
                            logger=logger)

    num_points = 200
    time = np.linspace(0, 20, num_points)

    # Generate complex true signal
    true_position = generate_complex_signal(time)

    # Calculate true velocity and acceleration using finite differences
    true_velocity = np.gradient(true_position, time)
    true_acceleration = np.gradient(true_velocity, time)

    # Add non-uniform noise
    base_noise = 0.2
    varying_noise = base_noise * (1 + 0.5 * np.sin(2 * np.pi * 0.1 * time))
    measurements = true_position + np.random.normal(0, varying_noise, size=num_points)

    estimated_position = []
    position_cov= []
    estimated_velocity = []
    estimated_acceleration = []

    for z in measurements:
        pos, pos_cov, vel, acc = ukf_poly.update(dt, z)
        estimated_position.append(pos)
        position_cov.append(pos_cov)
        estimated_velocity.append(vel)
        estimated_acceleration.append(acc)

    # Calculate RMSEs
    position_rmse = np.sqrt(np.mean((true_position - estimated_position) ** 2))
    velocity_rmse = np.sqrt(np.mean((true_velocity - estimated_velocity) ** 2))
    acceleration_rmse = np.sqrt(np.mean((true_acceleration - estimated_acceleration) ** 2))

    print(f"Position RMSE: {position_rmse:.4f}")
    print(f"Velocity RMSE: {velocity_rmse:.4f}")
    print(f"Acceleration RMSE: {acceleration_rmse:.4f}")

    # Enhanced visualization
    fig = plt.figure(figsize=(15, 12))
    gs = GridSpec(3, 1, height_ratios=[1, 1, 1], hspace=0.3)

    # Position plot
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(time, true_position, 'b-', label='True Position', alpha=0.7)
    ax1.plot(time, estimated_position, 'r-', label='UKF Estimate', linewidth=2)
    ax1.plot(time, measurements, 'mo', label='Measurements', alpha=0.8, markersize=12)  # Magenta circles for measurements

    ax1.set_title('Position Tracking')
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Position')
    ax1.grid(True)
    ax1.legend()

    # Velocity plot
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(time, true_velocity, 'b-', label='True Velocity', alpha=0.7)
    ax2.plot(time, estimated_velocity, 'r-', label='UKF Estimate', linewidth=2)
    ax2.set_title('Velocity Tracking')
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Velocity')
    ax2.grid(True)
    ax2.legend()

    # Acceleration plot
    ax3 = fig.add_subplot(gs[2])
    ax3.plot(time, true_acceleration, 'b-', label='True Acceleration', alpha=0.7)
    ax3.plot(time, estimated_acceleration, 'r-', label='UKF Estimate', linewidth=2)
    ax3.set_title('Acceleration Tracking')
    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('Acceleration')
    ax3.grid(True)
    ax3.legend()

    plt.suptitle('UKF State Estimation of Complex Motion', fontsize=16)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # main_simple()
    main_complex()