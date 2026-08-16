import sys
import os
import simplejson as json
import numpy as np
import time
import datetime as dt
from typing import Deque
from collections import deque
from typing import Optional, Tuple, Dict, Any
from decimal import Decimal

from api.Binance.BacktestOrderData import BacktestOrderData
from ahf.utils.utils import readable_error
from ahf.rl.envs.TradeEnum import TradeAction
from ahf.utils.utils import is_dt_offset_aware, d, d_round, d_round_fee

from ahf.core.enums import AppEnv, BotEnv


class Decision:
    """
    This class is to contain logics regarding `long` / `dual` situation
    We have implemented long, but for `dual`, we have not fully implemented it.
    """

    def __init__(self, long_short):
        self.long_short = long_short

    @staticmethod
    def step(min_kelly_cap, max_kelly_cap, kelly_cap, active_long_short):
        kelly_cap = np.clip(kelly_cap, min_kelly_cap, max_kelly_cap)
        # print(f"min_kelly_cap: {min_kelly_cap} max_kelly_cap: {max_kelly_cap}")
        return kelly_cap, active_long_short


class BrunhildDatastore_v11(BacktestOrderData):

    def __init__(self,
                 hyper_args,
                 trade_args,
                 tech_args,
                 exch_api,
                 price_fetcher,
                 read_write,
                 strategy,
                 logger):
        symbol = trade_args["symbol"]
        target_asset = trade_args["target_asset"]
        home_asset = symbol.replace(target_asset, "")
        # read trading settings
        self.trade_args = trade_args

        silo_size = int(tech_args.get("silo_size", trade_args.get("silo_size")))
        if silo_size is None:
            raise Exception("Silo size must be in tech_args or trade_args, and cannot be None")

        # 下面是 reset 會需要的，所以首先 init
        # ==== price_fetcher ====
        # if it is None or training mode, then it means the same thing.
        self.price_fetcher = price_fetcher

        init_trade_cash = d(trade_args.get("init_trade_cash", 0))
        init_target_cash = d(trade_args.get("init_target_cash", 0))


        kelly_cap_args = {
            "min_kelly_cap": self.trade_args.get("min_kelly_cap", 1),
            "max_kelly_cap": self.trade_args.get("max_kelly_cap", 1),
            "must_trade_max": self.trade_args.get("must_trade_max", "6w"),
            "trade_interval": self.trade_args.get("trade_interval")
        }

        super().__init__(symbol,
                         home_asset,
                         target_asset,
                         exch_api,
                         silo_size,
                         init_trade_cash,
                         init_target_cash,
                         trade_args["exch_mode"],
                         kelly_cap_args,
                         logger)

        self.logger.info(f"[Datastore] {__name__} Class loaded")

        self.read_write = read_write
        self.exch_api = exch_api

        # ========== init data START =============
        self._cumulative_returns: Decimal = d(0.0)
        self.total_reward: float = 0.0

        self.num_trades = 0
        self.profit_trades = deque(maxlen=10)

        # 這階層的 running 欄位
        self._rewards: Optional[Deque[Optional[float]]] = None
        self.episode_returns: Optional[Deque[Optional[float]]] = None

        # self.strategy = strategy # move to upper BrunhildEnv
        # ========== init data END =============


        self.max_risk = trade_args.get("max_risk")

        # Require Init
        self.long_short = trade_args.get("long_short")

        self.render_mode = trade_args.get("render_mode", "console")
        self.target_return = trade_args.get("target_return", +np.inf)
        self.done_kelly_mode = trade_args.get("done_kelly_mode")

        if self.done_kelly_mode is None:
            raise Exception("self.done_kelly_active cannot be None")

        self.done_kelly_active = True if self.done_kelly_mode in ("auto", "true") else False

        self.job_id = trade_args.get("job_id")

        if self.price_fetcher is None:
            raise Exception("env requires self.price_fetcher in this version")

        self.app_env = self.trade_args.get("app_env")
        self.price_env = self.trade_args.get("price_env")

        if self.app_env != AppEnv.TRAIN:
            self.trade_args["form_end"] = None

        # Append Horizontally
        # ============= Assign value START ===============
        # environment information # TODO hard-code first
        self.shares_num = 1  # self.price_ary.shape[1]
        self.logger.info(f"[Datastore] app currently at {self.app_env.name} mode")


        # ============= Assign value END ===============

        self.stacking_lookback = trade_args.get("stacking_lookback", 0)
        self.buy_cost_rate = d(trade_args.get("buy_fee_pct", 0.001))
        self.sell_cost_rate = d(trade_args.get("sell_fee_pct", 0.001))
        self.short_cost_rate = d(trade_args.get("short_fee_pct", 0.001))
        self.cover_cost_rate = d(trade_args.get("cover_fee_pct", 0.001))


        if self.init_trade_cash is None:
            self.init_trade_cash = trade_args["init_trade_cash"] = d(0)
            self.logger.warning("init_trade_cash has to be assigned")

        if self.init_target_cash is None:
            self.init_target_cash = trade_args["init_target_cash"] = d(0)
            self.logger.warning("init_target_cash has to be assigned")

        self.logger.info(f">> init_trade_cash: {self.init_trade_cash} {self.home_asset}")
        self.logger.info(f">> init_target_cash: {self.init_target_cash} {self.target_asset}")

        # capture loss
        self._last_recorded_date = None
        self._starting_asset_cash = d(self.init_trade_cash)

    def round_price(self, price):
        return self.exch_api.round_price(self.symbol, price)

    def round_qty(self, qty):
        return self.exch_api.round_qty(self.symbol, qty)

    def step_idx(self, app_env):
        idx = super().step_idx(app_env)

        self._rewards.append(0)
        self._buysell_lvl.append(None)
        self.episode_returns.append(0)

        # 檢查長度一樣
        deque_len = len(self._rewards)
        assert (len(self._buysell) == deque_len and
                len(self._buysell_lvl) == deque_len and
                len(self.episode_returns) == deque_len), \
            f"長度有問題 _buysell:{len(self._buysell)}, " \
            f"_rewards: {len(self._rewards)}, " \
            f"buysell_lvl: {len(self._buysell_lvl)}, " \
            f"episode_returns: {len(self.episode_returns)}, "

        return idx


    def reset(self):
        super().reset()

        if self.price_fetcher is None:
            raise Exception("env requires self.price_fetcher to be available at this version")

        # Basic property Init
        self.long_short = self.trade_args.get("long_short")

        # DEPRECATED
        # self.decision = Decision(self.long_short)

        # reward  # REQUIRED IN STEP
        self._rewards = deque([0.0], maxlen=self.max_deque)
        self._cumulative_returns = d(0.0)
        self.total_reward = 0.0

        self._buysell_lvl = deque([None], maxlen=self.max_deque)
        self.episode_returns = deque([0], maxlen=self.max_deque)

        # num_trades  # REQUIRED IN STEP
        self.num_trades = 0
        self.profit_trades = deque(maxlen=10)

        # Kelly_cap
        self._realized_pnl_pct_hist = np.zeros(1)

    def load_txn_hist_simple(self, txn_order_filename: str, row_count_from_bottom:int=100) -> Tuple[bool, Optional[str]]:

        with open(txn_order_filename, "r") as f:
            total_rows = sum(1 for line in f)
            # Calculate how many rows to skip
            skip_rows = max(total_rows - row_count_from_bottom, 0)

        skip_rows=range(1, skip_rows + 1)

        # check if there is active order or old record
        found_txn_order, txn_order = self.read_write.load_txn_order(txn_order_filename,
                                                                    self.trade_cols, self.trade_cols_type,
                                                                    self.logger,
                                                                    skip_rows=skip_rows, row_count=row_count_from_bottom)
        if not bool(found_txn_order):
            return False, None

        col_names = ("dt_idx", "buy_sell", "executed_qty", "realized_pnl", "orderId")

        rows = []
        for row in txn_order:
            if row["buy_sell"] != "HOLD":
                # Filter the row to include only the specified columns
                filtered_dict = {key: row[key] for key in col_names if key in row}
                rows.insert(0, filtered_dict)

        # Include metadata (header)
        data_to_send = {
            "header": col_names,
            "data": rows
        }

        # Serialize the structured data to JSON
        json_data = json.dumps(data_to_send)

        return True, json_data



    def load_txn_hist(
            self, trade_args: Dict[str, Any], txn_order_filename: str, row_count_from_bottom:int=3
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        這是給 BinanceTrade 交易時用的，會把就交易紀錄拉出後塞到第一筆資料
        如果是單純訓練使用請用
        """
        try:
            if self.app_env in (AppEnv.TRADE, AppEnv.BOT):
                pass
            elif self.trade_args.get("bot_env") in (BotEnv.SIMULATION, ):
                pass
            else:
                raise Exception(f"只能在交易使用 ( AppEnv.TRADE | BOT ), got {self.app_env}")

            # this is awkward, if it is within opts.interval, then use the old data

            # get latest price
            price_actual_new, depth = self.price_fetcher.get_price()

            dt_idx = price_actual_new.index[-1].to_pydatetime().replace(tzinfo=dt.timezone.utc)
            price = d(price_actual_new[-1])
            # price = self.round_price(price)
            price = d_round(price, self.decimal_price)

            # create directory if not exist
            txn_order_dir = os.path.dirname(os.path.abspath(txn_order_filename))
            os.makedirs(txn_order_dir, exist_ok=True)

            skip_rows = 0
            if os.path.exists(txn_order_filename):
                with open(txn_order_filename, "r") as f:
                    total_rows = sum(1 for line in f)
                    # Calculate how many rows to skip
                    skip_rows = max(total_rows - row_count_from_bottom, 0)

                skip_rows=range(1, skip_rows + 1)

            # check if there is active order or old record
            found_txn_order, txn_order = self.read_write.load_txn_order(txn_order_filename,
                                                                        self.trade_cols, self.trade_cols_type,
                                                                        self.logger,
                                                                        skip_rows=skip_rows,
                                                                        row_count=row_count_from_bottom)

            # ["dt_idx", "price", "idx_trade", "signal", "position", "share", "buysell",
            #         "executedQty", "executedAmt", "fee1", "fee2", "paper_pnl",
            #         "cash", "target_cash", "asset", "orderId"]
            if bool(found_txn_order):
                need_header = False
                # retrieve data for past order
                # price = txn_order["price"][-1]
                # dt_idx = txn_order["dt_idx"][-1]

                position = d(txn_order["position"][-1])
                # share = txn_order["share"][-1]

                # if reset_trade_cash is larger than zero, then use it
                cash = d(txn_order["cash"][-1])
                trade_cash = d(txn_order["trade_cash"][-1])
                target_cash = d(txn_order["target_cash"][-1])
                borrowed_cash = d(txn_order["borrowed_cash"][-1])
                if trade_args["reset_trade_cash"] > 0:
                    cash = d(trade_args["reset_trade_cash"])

                asset = position * price
                buysell, orderId = TradeAction.HOLD, trade_args["orderId"]
                executedQty, executedAmt, fee = d(0.), d(0.), d(0.)
                paper_pnl = d(txn_order["paper_pnl"][-1])

                kelly_cap = float(txn_order["kelly_cap"][-1])

                silo_pos = txn_order["silo_pos"][-1] if txn_order["silo_pos"][-1] != "None" else None
                silo_amt = txn_order["silo_amt"][-1] if txn_order["silo_amt"][-1] != "None" else None

                total_position = d(txn_order["total_position"][-1])
                total_amt_bought = d(txn_order["total_amt_bought"][-1])

                cumulative_realized_pnl = d(txn_order["cumulative_realized_pnl"][-1])
                drawdown = d(txn_order["drawdown"][-1])
                drawdown_pct = d(txn_order["drawdown_pct"][-1])
                buysell_lvl = float(txn_order["buysell_lvl"][-1] or 0.05)

            else:
                # IMPORTANT: not found or AppEnv.TRAIN mode
                need_header = True

                cash = trade_cash = d_round(trade_args.get("init_trade_cash"), self.quote_asset_precision)
                target_cash = d_round(trade_args.get("init_target_cash", 0), self.base_asset_precision)
                borrowed_cash = d_round(0.0, self.base_asset_precision)
                asset, executedQty, executedAmt, fee, paper_pnl, asset = d(0.), d(0.), d(0.), d(0.), d(0.), d(0.)
                signal, position = d(0), d(0)

                # kelly
                kelly_cap = round(self.KellyCls.kelly_cap, 3)
                buysell, orderId = TradeAction.HOLD, None
                silo_pos, silo_amt = None, None
                total_position, total_amt_bought = d(0.), d(0.)

                cumulative_realized_pnl, drawdown, drawdown_pct = d(0.), d(0.), d(0.)
                # 起始買賣總資金百分比
                buysell_lvl = float(trade_args.get("buysell_lvl", 0.05))

            fee1, fee2 = d(0.), d(0.)
            realized_pnl = d(0.)

            acct1 = self.exch_api.get_account_balance(self.home_asset, self.trade_args.get("spot_margin"), self.symbol,
                                                      balance=cash, margin_level=d(999))
            home_cash = d(acct1[self.home_asset]["free"])
            acct2 = self.exch_api.get_account_balance(self.target_asset, self.trade_args.get("spot_margin"), self.symbol,
                                                      balance=target_cash, margin_level=d(999))  # "isolated"
            target_cash = d(acct2[self.target_asset]["free"])

            self.logger.info(f"[BrunhildDatastore] ===== active balance for home {home_cash:>15} {self.home_asset} ======")
            self.logger.info(f"[BrunhildDatastore] ===== active balance for target {target_cash:>15} {self.target_asset} ======")

            # ["dt_idx", "idx_trade", "price", "idx_trade", "signal", "position", "share", "buysell",
            #         "executedQty", "executedAmt", "fee1", "fee2", "starting_asset",
            #         "cash", "target_cash", "asset", "orderId"]
            # dt_idx_str = dt_idx.strftime(datesffmt)[:-3]  # microseconds

            executedAmt_precision = max(self.base_asset_precision, self.quote_asset_precision)

            txn_order_dict = {
                "dt_idx": dt_idx.replace(tzinfo=None),  # dt_idx_str
                "idx_trade": 0,
                "price": price,
                "position": d_round(position, self.quote_asset_precision),
                "buysell": buysell,
                "cash": d_round(cash, self.quote_asset_precision),
                "borrowed_cash": d_round(borrowed_cash, self.base_asset_precision),
                "asset": d_round(asset, self.quote_asset_precision),
                "cash_asset": d_round(cash + asset, self.quote_asset_precision),
                "realized_pnl": d_round(realized_pnl, self.quote_asset_precision),
                "kelly_cap": kelly_cap,
                "silo_pos": silo_pos,
                "silo_amt": silo_amt,
                "cumulative_realized_pnl": d_round(cumulative_realized_pnl, self.quote_asset_precision),
                "drawdown": drawdown,
                "drawdown_pct": drawdown_pct,
                "target_cash": target_cash,
                "executed_qty": d_round(executedQty, self.decimal_qty),
                "executed_amt": d_round(executedAmt, executedAmt_precision),
                "fee1": fee1, "fee2": fee2,
                "paper_pnl": d_round(paper_pnl, self.quote_asset_precision),
                "total_position": d_round(total_position, self.quote_asset_precision),
                "total_amt_bought": d_round(total_amt_bought, executedAmt_precision),
                "buysell_lvl": buysell_lvl,
                "trade_cash": d_round(trade_cash, self.quote_asset_precision),
                "orderId": orderId
            }

            trade_data_new_col = list(txn_order_dict.keys())

            # append result
            col_diff = list(set(trade_data_new_col) - set(self.trade_cols))
            assert len(col_diff) == 0, f"columns of self.trade_data differs from trade_data_new: {col_diff}"

            if txn_order is not None and txn_order["realized_pnl"].shape[0] >= 1:
                a = txn_order["realized_pnl"]
                self.populate_realized_pnl_pct_hist(txn_order["realized_pnl"].to_numpy())

            self.logger.debug(f"[BrunhildDatastore] load_txn_hist.txn_order_dict: {txn_order_dict}")
            self._set_init_row(**txn_order_dict)

            # reality check
            has_position = (position == 1 or position == -1)
            if has_position and trade_args["orderId"] is not None:
                self.set_order_id(trade_args["orderId"], d(0), d(0))
            elif has_position and trade_args["orderId"] is None:
                raise Exception("[Trade] You have to specify orderId for existing active order, exiting bot")
            else:
                pass

            # We save it on purpose, just to check if everything is alright
            self.read_write.write_txn_order(
                data_dict=txn_order_dict,
                trade_cols=self.trade_cols,
                txn_order_filename=txn_order_filename,
                logger=self.logger
            )

            return found_txn_order, txn_order_dict

        except Exception as e:
            err_str = readable_error(e, __file__)
            self.logger.error(f"[Trade] {err_str}")
            time.sleep(3)
            sys.exit()

    # def as_dict(self, child_dict=None):
    #     __d = self.__dict__.copy()
    #     _d = super().as_dict(__d)
    #     if child_dict is not None:
    #         _d.update(child_dict)


        # del _d["read_write"]
        # del _d["silo"]
        # del _d["KellyCls"]
        #
        # del _d["_realized_pnl_pct_hist"]
        # del _d["_cumulative_returns"]
        # del _d["episode_returns"]
        # del _d["_kelly_p"]
        # del _d["_kelly_b_win"]
        # del _d["_kelly_b_loss"]
        # del _d["_kelly_cap"]
        # del _d["_buysell_lvl"]
        # del _d["_rewards"]
        #
        # for key in list(_d.keys()):
        #     if isinstance(_d[key], np.ndarray) or isinstance(_d[key], Deque):
        #         del _d[key]

        return _d

    # ======== SETTER =============
    def set_price(self,
                  dt_idx: dt.datetime,
                  price: Decimal,
                  app_env: AppEnv = AppEnv.TRAIN,
                  unit_test: bool = False):

        # 取得 APP 運行模式
        is_real_trade = (app_env == AppEnv.TRADE and self.exch_mode == "SpotAPI")
        # is_paper_trade = (app_env == AppEnv.PAPER_TRADE)
        is_unit_test = (self.exch_mode == "SpotTest" or unit_test)

        # 如果是 TRAIN/SIM/BACKTEST 就不用，因為資料 load 出來是 np.array，不好處理
        # TODO 這裡和 BacktestOrderData->BinanceOrderData 要整合一下
        if is_unit_test:
            self.price_ary[-1] = [price]
            self._dt_idx[-1] = dt_idx

            return
        elif is_real_trade:  #  or is_paper_trade:
            assert is_dt_offset_aware(dt_idx)
        elif app_env == AppEnv.TRAIN:
            self.price_ary[-1] = [price]
            self._dt_idx[-1] = dt_idx
            return
        else:
            # 如果不是 trade, 其他模式已經有資料，回檔
            # TODO 觀察一下，這裡寫很差
            raise Exception("觀察一下，這裡寫很差")

        now = dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc)
        a = (now - dt_idx).total_seconds()
        diff = (now - dt_idx).total_seconds()
        if diff > 10 and is_real_trade:
            m = int(diff // 60)
            s = diff - (m * 60)
            self.logger.warning(f"[Datastore] DataStore set_price is off by {m}m{round(s, 3)}s now_utc {now} ")

        self.price_ary[-1] = [price]
        # self.price_ary_T[idx] = price
        self._dt_idx[-1] = dt_idx.replace(tzinfo=None)

    def set_buysell(self, _buysell: TradeAction):
        self._buysell[-1] = _buysell

    def set_starting_asset_cash(self, starting_asset_cash: Decimal):
        self._starting_asset_cash = starting_asset_cash

    def set_rewards(self, reward: float):
        if reward is not None:
            self._rewards[-1] = reward

    def set_last_rewards(self, reward: float):
        if reward is not None:
            self._rewards[-2] = reward

    def set_buysell_lvl(self, lvl: float):
        if lvl is not None:
            self._buysell_lvl[-1] = lvl

    def inc_num_trades(self):
        """
        增加一次交易次數
        """
        self.num_trades += 1

    def get_num_trades(self) -> int:
        """
        讀取交易次數
         # 原本的 param:, period_start, period_end
        """
        # assert period_end > period_start

        # return (self._realized_pnl[period_start:period_end] != 0).sum() / 2
        return self.num_trades

    def append_profit_trade(self, is_profit: bool):
        """
        用來紀錄是否賺錢
        """
        self.profit_trades.append(is_profit)

    def get_num_profit_trades(self) -> int:
        """
        取得幾次是賺錢的
        """
        return self.profit_trades.count(True)

    def populate_realized_pnl_pct_hist(self, data: np.array):
        if "numpy.ndarray" not in str(type(data)):
            raise Exception(f"populate_realized_pnl_pct_hist param has to be numpy.ndarray got {type(data)}")
        for v in data:
            if not isinstance(v, Decimal):
                raise Exception(f"populate_realized_pnl_pct_hist data has to be Decimal type got {type(v)}")

        self._realized_pnl_pct_hist = data

    def _set_init_row(self, dt_idx, idx_trade, price, position, buysell, executed_qty, executed_amt,
                      fee1, fee2, paper_pnl, cash, borrowed_cash, target_cash, asset, cash_asset,
                      realized_pnl, kelly_cap, drawdown, drawdown_pct, cumulative_realized_pnl,
                      total_position, total_amt_bought,
                      silo_pos, silo_amt, trade_cash,
                      orderId, buysell_lvl):
        # trade_data
        self._dt_idx[0] = dt_idx

        self.price_ary[0] = [d(price)]
        self._position[0] = position
        self._buysell[0] = buysell
        self._asset[0] = asset
        self._cash[0] = cash
        self._borrowed_cash[0] = borrowed_cash
        self._cash_asset[0] = cash_asset
        self._realized_pnl[0] = realized_pnl

        self._kelly_cap[0] = kelly_cap
        self._buysell_lvl[0] = buysell_lvl

        self._silo_pos[0] = silo_pos
        self._silo_amt[0] = silo_amt

        # _silo_pos = ast.literal_eval(silo_pos.replace("|", ",")) if silo_pos is not None else None
        _silo_pos = silo_pos.split("|") if silo_pos is not None else None
        silo_pos_list = [d(item) for item in _silo_pos] if _silo_pos is not None else []
        silo_pos_deque = deque(silo_pos_list, maxlen=10)

        # _silo_amt = ast.literal_eval(silo_amt.replace("|", ",")) if silo_amt is not None else None
        _silo_amt = silo_amt.split("|") if silo_amt is not None else None
        silo_amt_list = [d(item) for item in _silo_amt] if _silo_amt is not None else []
        silo_amt_deque = deque(silo_amt_list, maxlen=10)

        if _silo_pos is not None and _silo_amt is not None:
            self.logger.debug(f"[BrunhildDatastore] _set_init_row silo_pos_deque: {silo_pos_deque}, silo_amt_deque: {silo_amt_deque}")
            self.silo.populate_pos_amt(silo_pos_deque, silo_amt_deque)

        self._cumulative_realized_pnl[0] = cumulative_realized_pnl
        self._drawdown[0] = drawdown
        self._drawdown_pct[0] = drawdown_pct

        self._target_cash[0] = target_cash

        self._executed_qty[0] = executed_qty
        self._executed_amt[0] = executed_amt
        self._fee1[0] = fee1
        self._fee2[0] = fee2

        self._paper_pnl[0] = paper_pnl
        self._total_position[0] = total_position
        self._total_amt_bought[0] = total_amt_bought

        self._trade_cash[0] = trade_cash
        self._orderId[0] = orderId

        # ==== Data Source ====
        # _dt_idx = dt_idx.replace(tzinfo=dt.timezone.utc)
        # self.set_price(0, _dt_idx, price)
        self.set_trade(buysell, position, cash, borrowed_cash, asset)
        self.set_pnl(d(0.), d(0.), cumulative_realized_pnl)

        """
        # no need update
        self.KellyCls[0] = kelly_cap
        self.KellyCls_short[0] = kelly_cap_short
        self.silo[0] = silo
        self.silo_short[0] = silo_short

        self.set_fee1(0, 0)
        self.set_fee2(0, 0)
        """

        # self._paper_pnl[0] = paper_pnl
        # self._total_position[0] = total_position
        # self._total_amt_bought[0] = total_amt_bought


    # ======== GETTER =============
    def get_current_dt_idx(self) -> dt.datetime:
        return self._dt_idx[-1]

    def get_buysell_lvl(self) -> float:
        return self._buysell_lvl[-1]

    def get_last_buysell_lvl(self) -> float:
        return self._buysell_lvl[-2] or self._buysell_lvl[-2] or 0.05

    def get_cumulative_realized_pnl_range(self, idx_start: int, idx_end: int) -> Decimal:
        if idx_start <= 0:
            idx_start = 0
        return self._cumulative_realized_pnl[idx_start:idx_end]

    def get_cumulative_returns(self) -> Decimal:
        return self._cumulative_returns

    def get_starting_asset_cash(self) -> Decimal:
        return self._starting_asset_cash

    def get_rewards(self) -> float:
        return self._rewards[-1]

    def get_last_rewards(self) -> float:
        return self._rewards[-2]

    def get_mean_reward(self) -> float:
        return np.mean(list(self._rewards))

    def get_idx(self):
        return self._idx

    def get_current_row(self):
        try:
            idx = -2 if self.price_ary[-1] is None or self._position[-1] is None else -1
            idx_minus_one = idx-1 if len(self._position) > abs(idx) else idx

            all_zeros = not np.any(self.silo.position)
            silo_pos_str = self.silo.get_position_str() if not all_zeros else None
            silo_amt_str = self.silo.get_amt_str() if not all_zeros else None

            self.logger.debug(f"[BrunhildDatastore] get_current_row self._idx={self._idx}, idx={idx}:\n"
                             f"  dt_idx: {self._dt_idx[idx].replace(tzinfo=None)},\n"
                             f"  idx: {self._idx+idx+1},\n"
                             f"  price: {self.price_ary[idx][0]},\n"
                             f"  pos: [{self._position[idx_minus_one]}, {self._position[idx]}],\n"
                             f"  buysell: [{self._buysell[idx_minus_one]}, {self._buysell[idx].name}],\n"
                             f"  cash: [{self._cash[idx_minus_one]}, {self._cash[idx]}],\n"
                             f"  borrowed_cash: [{self._borrowed_cash[idx_minus_one]}, {self._borrowed_cash[idx]}],\n"
                             f"  asset: [{self._asset[idx_minus_one]}, {self._asset[idx]}],\n"
                             f"  cash_asset: [{self._cash_asset[idx_minus_one]}, {self._cash_asset[idx]}],\n"
                             f"  realized_pnl: [{self._realized_pnl[idx_minus_one]}, {self._realized_pnl[idx]}],\n"
                             f"  silo_pos: {silo_pos_str},\n"
                             f"  silo_amt: {silo_amt_str},\n"
                             f"  cumulative_realized_pnl: [{self._cumulative_realized_pnl[idx_minus_one]}, {self._cumulative_realized_pnl[idx]}],\n"
                             f"  drawdown: [{self._drawdown[idx_minus_one]}, {self._drawdown[idx]}],\n"
                             f"  drawdown_pct: [{self._drawdown_pct[idx_minus_one]}, {self._drawdown_pct[idx]}],\n"
                             f"  buysell_lvl: [{self._buysell_lvl[idx_minus_one]}, {self._buysell_lvl[idx]}]\n"
                             )

            # prepare data
            # ================================================================
            # Note: we keep Datastore simple for training only, all trading
            # moves to TradeDate class for logic separation
            # ================================================================
            # last_executed = self.trade_data.get_just_executed()
            data_dict = {
                "dt_idx": self._dt_idx[idx].replace(tzinfo=None),
                "idx_trade": self._idx+idx+1,
                "price": d_round(self.price_ary[idx][0], self.decimal_price),
                "position": d_round(self._position[idx] or 0, self.quote_asset_precision),
                "buysell": self._buysell[idx],
                "cash": d_round(self._cash[idx], self.quote_asset_precision),
                "borrowed_cash": d_round(self._borrowed_cash[idx], self.base_asset_precision),
                "asset": d_round(self._asset[idx], self.base_asset_precision),
                "cash_asset": d_round(self._cash_asset[idx], self.quote_asset_precision),
                "realized_pnl": d_round(self._realized_pnl[idx], self.quote_asset_precision),
                "kelly_cap": round(self._kelly_cap[idx], 3),
                "silo_pos": silo_pos_str,
                "silo_amt": silo_amt_str,

                # extra non-essential fields
                "cumulative_realized_pnl": d_round(self._cumulative_realized_pnl[idx], 8),
                "drawdown": d_round(self._drawdown[idx], self.quote_asset_precision),
                "drawdown_pct": d_round(self._drawdown_pct[idx], 3),
                "buysell_lvl": d_round(self._buysell_lvl[idx] or 0, 3),

                # the following are moved to TradeData
                # "target_cash": self.trade_data.get_target_cash(),
                # "total_position": round(last_executed["total_position"], 5),
                # "total_amt_bought":  round(last_executed["total_amt_bought"], 5),
                # "executedQty": last_executed["executed_qty"],
                # "executedAmt": last_executed["executed_amt"],
                # "fee1": round(last_executed["fee1"], 4),
                # "fee2": round(last_executed["fee2"], 8),
                # "starting_asset": last_executed["starting_asset"],
                # "orderId": last_executed["orderId"],
            }

            return data_dict
        except Exception as e:
            err_str = readable_error(e, __file__)
            self.logger.error(f"[Trade] {err_str}")
            raise


    def get_just_executed(self):
        try:
            idx = -2 if self._buysell[-1] is None else -1
            idx_minus_one = idx-1 if len(self._position) > abs(idx) else idx

            self.logger.debug(f"[BrunhildDatastore] get_just_executed idx={idx}:\n"
                              f"  price: {self.price_ary[idx][0]},\n"
                              f"  buysell: [{self._buysell[idx_minus_one]}, {self._buysell[idx].name}],\n"
                              f"  pos: [{self._position[idx_minus_one]}, {self._position[idx]}],\n"
                              f"  cash: [{self._cash[idx_minus_one]}, {self._cash[idx]}],\n"
                              f"  borrowed_cash: [{self._borrowed_cash[idx_minus_one]}, {self._borrowed_cash[idx]}],\n"
                              f"  asset: [{self._asset[idx_minus_one]}, {self._asset[idx]}],\n"
                              f"  cash_asset: [{self._cash_asset[idx_minus_one]}, {self._cash_asset[idx]}],\n"
                              f"  target_cash: [{self._target_cash[idx_minus_one]}, {self._target_cash[idx]}],\n"
                              f"  executed_qty: [{self._executed_qty[idx_minus_one]}, {self._executed_qty[idx]}],\n"
                              f"  executed_amt: [{self._executed_amt[idx_minus_one]}, {self._executed_amt[idx]}],\n"
                              f"  fee1: [{self._fee1[idx_minus_one]}, {self._fee1[idx]}],\n"
                              f"  fee2: [{self._fee2[idx_minus_one]}, {self._fee2[idx]}],\n"
                              f"  paper_pnl: [{self._paper_pnl[idx_minus_one]}, {self._paper_pnl[idx]}],\n"
                              f"  total_position: [{self._total_position[idx_minus_one]}, {self._total_position[idx]}],\n"
                              f"  total_amt_bought: [{self._total_amt_bought[idx_minus_one]}, {self._total_amt_bought[idx]}],\n"
                              f"  cumulative_realized_pnl: [{self._cumulative_realized_pnl[idx_minus_one]}, {self._cumulative_realized_pnl[idx]}],\n"
                              f"  buysell_lvl: [{self._buysell_lvl[idx_minus_one]}, {self._buysell_lvl[idx]}]\n"
                              f"  trade_cash: [{self._trade_cash[idx_minus_one]}, {self._trade_cash[idx]}]\n"
                              f"  home_asset: {self.home_asset}\n"
                              f"  target_asset: {self.target_asset}\n"
                              f"  orderId: [{self._orderId[idx_minus_one]}, {self._orderId[idx]}]\n"
                             )

            executedAmt_precision = self.base_asset_precision if self._buysell[idx] == TradeAction.BUY else self.quote_asset_precision
            re = {
                "price": d_round(self.price_ary[idx][0], self.decimal_price),
                "buysell": self._buysell[idx],
                "position": d_round(self._position[idx], self.quote_asset_precision),
                "cash": d_round(self._cash[idx], self.quote_asset_precision),
                "borrowed_cash": d_round(self._borrowed_cash[idx], self.base_asset_precision),
                "asset": d_round(self._asset[idx], self.quote_asset_precision),
                "cash_asset": d_round(self._cash_asset[idx], self.quote_asset_precision),
                "target_cash": d_round(self._target_cash[idx], self.base_asset_precision),
                "executed_qty": d_round(self._executed_qty[idx], self.decimal_qty),
                "executed_amt": d_round(self._executed_amt[idx], executedAmt_precision),
                "fee1": d_round_fee(self._fee1[idx], self.quote_comm_precision),
                "fee2": d_round_fee(self._fee2[idx], self.base_comm_precision),
                "paper_pnl": d_round(self._paper_pnl[idx], self.quote_precision),
                "total_position": d_round(self._total_position[idx], self.base_asset_precision),
                "total_amt_bought": d_round(self._total_amt_bought[idx], self.quote_asset_precision),
                "cumulative_realized_pnl": d_round(self._cumulative_realized_pnl[idx], 8),
                "buysell_lvl": d_round(self._buysell_lvl[idx], 3),
                "trade_cash": d_round(self._trade_cash[idx], self.quote_asset_precision),
                # "home_asset": self.home_asset,
                # "target_asset": self.target_asset,
                "orderId": self._orderId[idx],
            }
            return re
        except Exception as e:
            err_str = readable_error(e, __file__)
            self.logger.error(f"[Trade] {err_str}")
            raise
