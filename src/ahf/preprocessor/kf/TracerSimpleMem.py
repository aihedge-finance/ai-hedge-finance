import gc
import os
import sys
import math
import numpy as np
import pandas as pd
from os import path
import datetime as dt
from ahf.utils.utils import readable_error, datesffmt, create_dir_if_non_exist, get_project_root


from ahf.preprocessor.kf.TracerSimple import TracerSimple


def update_slope(market, market_last):
    if not math.isnan(market):
        re = (market - market_last) / market_last * 100 if market_last > 0 else 0
        return re
    else:
        return 0


class TracerSimpleMem(TracerSimple):
    def __init__(self, trade_args, strategy, job_id, delta, obs_cov, logger,
                 initial_state_mean=None, initial_state_cov=None, is_unittest=False, verbose=True):

        self.name = "TracerSimpleMem"
        self.trade_args = trade_args
        self.tracer_hist = None
        self.slope = None
        self.verbose = True if is_unittest else verbose

        self.delta = delta

        self.theta_id = f"{self.trade_args['exchange']}_{self.trade_args['symbol']}_" \
                        f"{self.trade_args['trade_interval']}_{job_id}"

        if is_unittest:
            unittest_str = "_UNITTEST"
        else:
            unittest_str = ""
        self.file_dir = os.path.expanduser(f"{self.trade_args['kf_data_path']}/{self.name}_{self.theta_id}{unittest_str}.csv")
        create_dir_if_non_exist(self.file_dir)

        TracerSimple.__init__(self, strategy, job_id, delta, obs_cov, logger, initial_state_mean, initial_state_cov)

    def catchup(self, priceA, tracer_hist=None):
        # IMPORTANT+ CAREFUL: This is used with side-line (Jobber cron job),
        # don't mix parameters with online (running program)
        try:
            if self.verbose:
                self.logger.info(f"[TracerSimpleMem] catching up for TracerSimpleMem delta:{self.delta}")
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

                # load previous stored kf params
                self.initial_state_mean = initial_state_mean = tracer_hist["tracer"].iloc[-1]
                self.initial_state_cov = initial_state_cov = tracer_hist["state_cov00"].iloc[-1]

            # capture only since from last save part for training
            # a = last_price_dt_index.astimezone(dt.timezone.utc)
            # b = last_save_dt_index.tz_convert("UTC")
            t_delta = last_price_dt_index - last_save_dt_index
            if t_delta.total_seconds() < 0:
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
                self.logger.info('[KF] too few new prices, no need to train new mu and sd.')
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
            self.logger.info("[TracerSimpleMem] Start training new kf market.")

            price_np = priceA_train["open"].to_numpy()
            self.slope = np.zeros(len(price_np))
            last_tracer = 0.

            l = len(price_np)
            for i, v in enumerate(price_np):
                idx_now = self.idx_now

                tracer_new = self.update(v, is_save=False)

                index = priceA_train.index[i]
                dt_idx[idx_now] = index.to_pydatetime()

                if i == 0:
                    self.slope[i] = update_slope(self.initial_state_mean, tracer_new)
                else:
                    self.slope[i] = update_slope(last_tracer, tracer_new)

                last_tracer = tracer_new

                if idx_now % 1000 == 0:
                    msg = "\r{0}, INFO {1} [KF] " \
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
            delta_np = np.empty(self.idx_now)
            delta_np.fill(self.delta)
            slope_np = self.slope[:self.idx_now]
            state_cov_np = self.state_cov[:self.idx_now]

            # only save when it is larger than 1, otherwise there are lots of zeros
            if self.idx_now > 1:
                self._save_tracer(self.file_dir, idx_np, dt_idx, price_np, tracer_np, slope_np,
                                  delta_np, state_cov_np, need_header)

            # save
            gc.enable()

            new = {
                "idx": idx_np, "dt_idx": dt_idx, "price": price_np, "tracer": tracer_np,
                "slope": slope_np, "delta": delta_np, "state_cov00": state_cov_np
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
                                          "delta", "state_cov00"
                                      ],
                                      # index_col=0, parse_dates=True,
                                      dtype={
                                          "idx": int, "dt_idx": str, "price": float,
                                          "tracer": float, "slope": float, "delta": float,
                                          "state_cov00": float
                                      })

            tracer_hist["dt_idx"] = pd.to_datetime(tracer_hist["dt_idx"])
            return True, tracer_hist
        except Exception as e:
            err = readable_error(e, __file__)
            self.logger.error(f"[TracerSimpleMem] {err}")
            sys.exit(-1)
            # return False, None
