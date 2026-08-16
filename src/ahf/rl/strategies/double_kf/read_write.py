import os
import sys
import time

import optuna
import asyncio
import numpy as np
import pandas as pd
from os import path
import datetime as dt

from typing import Optional, Tuple, Dict, List
import dask.dataframe as dd
from ahf.rl.train.config import get_tech_args
from dask.diagnostics import ProgressBar

from ahf.utils.utils import is_dir_exist, d
from ahf.utils.utils import readable_error, save_data, get_project_root
# TODO(v2): port TradeBot DB CRUD — see ahf domain layer
# TODO(v2): port TradeBot schema — see ahf domain layer


def load_data(trade_args, tech_args, logger):
    try:
        # load tech
        _tech_concat_df = _load_tech(trade_args, tech_args, logger)

        if trade_args["form_end"] is None:
            tech_df = _tech_concat_df[trade_args["form_start"]:]
        else:
            tech_df = _tech_concat_df[trade_args['form_start']:trade_args['form_end']]

        # price is the first column, do transpose because it is 1 x n array
        price_open = tech_df['open'].apply(lambda x: d(x))
        price_open = price_open.to_numpy()
        dt_idx = np.array(tech_df.index.to_pydatetime(), dtype=np.datetime64)

        if dt_idx[0] > dt.datetime.strptime(trade_args['form_start'], '%Y-%m-%d'):
            logger.warning(f"tech data has earlier form_start {dt_idx[0]}"
                                f" expect {trade_args['form_start']} from trade_args")

        tech_ary = tech_df.loc[:, ~tech_df.columns.str.contains('market')]
        drop_cols = ['tic', 'open', 'close', 'high', 'low', 'exchange', 'symbol', 'interval']
        tech_ary = tech_ary.drop(columns=drop_cols, errors='ignore')

        # DEPRECATING
        # market = None
        # market_ary = []
        # if 'market' in tech_df.columns:
        #    market = tech_df['market']
        #    market_ary = tech_df.filter(regex='market_')

        if len(tech_ary.index) == 0:
            raise Exception(f'Empty data set by _load_data() input form_start {trade_args["form_start"]} but '
                            f'saved tech data starts at {_tech_concat_df.index[0]}')

        return dt_idx, price_open, tech_ary
    except Exception as e:
        err = readable_error(e, __file__)
        logger.error(err)
        sys.exit()


def _load_tech(trade_args, tech_args, logger=None):
    try:
        logger.debug("_load_tech entered")
        start_time = time.time()
        trade_interval = trade_args['trade_interval']

        # backward compatibility, and binance is default to 1T
        trade_args["trade_interval"] = '1T' if trade_interval is None else trade_interval

        source_dir = get_tech_dir(trade_args, tech_args)

        # with warnings.catch_warnings():
        #    warnings.simplefilter("ignore")
        logger.info(f'Loading tech indicator from {source_dir}') if logger else print(
                    f'Loading tech indicator from {source_dir}')
        if not is_dir_exist(source_dir):
            raise Exception(f'folder does not exist {source_dir}')

        with ProgressBar():
            tech_dd = dd.read_parquet(source_dir, engine='pyarrow', aggregate_files=True)  #.persist()  # , index=index

        # print(f'partition: {tech_dd.npartitions},  division:{tech_dd.divisions[0:5]}, known_divisions:{tech_dd.known_divisions}')

        # REFERENCE
        # https://stackoverflow.com/a/46798024/1596886
        # https://docs.dask.org/en/latest/dataframe-design.html#partitions
        # divisions = pd.date_range('2021-02-02T00:15:00', '2021-03-01T00:00:00', freq=interval)
        # tech_dd = tech_dd.set_index('date', sorted=True, divisions=divisions)
        # tech_dd = tech_dd.resample(rule=interval).first().ffill()

        tech_pd = tech_dd.compute()
        tech_pd = tech_pd[~tech_pd.index.duplicated(keep='last')]
        end_time = time.time() - start_time
        if end_time > 3:
            if logger is not None:
                logger.info(f'[load_tech] takes {end_time:.2f}s to load and resample tech indicators from \n{source_dir}')
            else:
                print(f'[load_tech] takes {end_time:.2f}s to load and resample tech indicators from \n{source_dir}')

        return tech_pd

    except Exception as e:
        err = readable_error(e, __file__)
        logger.error(err) if logger else print(err)
        sys.exit(-1)


def load_txn_order(
        txn_order_filename: str,
        trade_cols: List,
        trade_cols_type: Dict,
        logger,
        skip_rows: int = 0,
        row_count: Optional[int] = None,
        bot_id: Optional[str] = None,
        user_id: Optional[str] = None
) -> Tuple[bool, Optional[pd.DataFrame]]:
    """
    Load txn_orders from the database if bot_id and user_id are provided, otherwise call load_txn_order_csv.

    :param txn_order_filename: Path to the CSV file
    :param trade_cols: List of columns to load from the file or database
    :param trade_cols_type: Dictionary specifying column data types for CSV
    :param logger: Logger instance for error/debugging
    :param skip_rows: Number of rows to skip while reading the CSV
    :param row_count: Number of rows to read (for the CSV reader)
    :param bot_id: Optional bot_id to query txn_orders from the database
    :param user_id: Optional user_id to query txn_orders from the database
    :return: Tuple (found: bool, dataframe: pd.DataFrame)
    """
    try:
        logger.debug("load_txn_order entered")
        # If bot_id and user_id are provided, query the DB
        if bot_id is not None and user_id is not None:
            logger.info(f"Loading txn_order for bot_id: {bot_id} and user_id: {user_id} from DB.")

            # Fetch data from the database using the async service function
            txn_orders = asyncio.run(get_txn_orders_by_bot_and_user(bot_id, user_id))  # Run async service in sync mode

            if not txn_orders or len(txn_orders) == 0:
                logger.warning(f"No records found for bot_id: {bot_id} and user_id: {user_id}")
                return False, None

            # Convert the data into a Pandas DataFrame
            txn_order_hist = pd.DataFrame([txn.dict() for txn in txn_orders])  # Convert to dict and load into DataFrame

            # Rename the column
            txn_order_hist = txn_order_hist.rename(columns={'exch_order_id': 'orderId'})

            # Ensure DataFrame contains only requested trade_cols
            txn_order_hist = txn_order_hist[trade_cols]
            logger.info(f"Loaded {len(txn_order_hist)} records from the database.")

            return True, txn_order_hist
        else:
            # Fallback: Read from CSV file
            logger.info(f"Reading txn_order from CSV file: {txn_order_filename}")
            return _load_txn_order_csv(
                txn_order_filename=txn_order_filename,
                trade_cols=trade_cols,
                trade_cols_type=trade_cols_type,
                logger=logger,
                skip_rows=skip_rows,
                row_count=row_count
            )
    except Exception as e:
        err_str = readable_error(e, __file__)
        logger.error(f"[load_txn_order] {err_str}")
        sys.exit()


def _load_txn_order_csv(
    txn_order_filename: str,
    trade_cols: List,
    trade_cols_type: Dict,
    logger,
    skip_rows: int = 0,
    row_count: Optional[int] = None
) -> Tuple[bool, Optional[pd.DataFrame]]:
    try:
        logger.debug("_load_txn_order_csv entered")
        found = False

        if not path.exists(txn_order_filename):
            return False, None

        txn_order_hist = pd.read_csv(
            txn_order_filename,
            usecols=trade_cols,
            index_col=0,
            parse_dates=True,
            dtype=trade_cols_type,
            skiprows=skip_rows,
            nrows=row_count,
            converters={
                "price": lambda v: d(v),
                "position": lambda v: d(v),
                "executed_qty": lambda v: d(v),
                "executed_amt": lambda v: d(v),
                "fee1": lambda v: d(v),
                "fee2": lambda v: d(v),
                "paper_pnl": lambda v: d(v),
                "cash": lambda v: d(v),
                "borrowed_cash": lambda v: d(v),
                "target_cash": lambda v: d(v),
                "asset": lambda v: d(v),
                "cash_asset": lambda v: d(v),
                "realized_pnl": lambda v: d(v),
                "total_position": lambda v: d(v),
                "total_amt_bought": lambda v: d(v),
                "silo_pos": lambda v: None if v == "None" else v,
                "silo_amt": lambda v: None if v == "None" else v,
                "orderId": lambda v: None if v == "None" else v,
            }
        )

        # Get all columns except the ones that can have None
        columns_that_shouldnt_have_nan = [col for col in txn_order_hist.columns
                                          if col not in ['silo_pos', 'silo_amt', 'orderId']]

        # Check only those columns
        if txn_order_hist[columns_that_shouldnt_have_nan].isnull().any().any():
            raise Exception(f"[read_write] txn_order_hist contains NaN (missing data), "
                            f"please check your {txn_order_filename}")

        if txn_order_hist is not None and txn_order_hist.shape[0] >= 1:
            found = True
            return found, txn_order_hist
        else:
            return found, None
    except Exception as e:
        err_str = readable_error(e, __file__)
        logger.error(f"[Trade] {err_str}")
        sys.exit()



def write_txn_order(
        data_dict: Dict,
        trade_cols: List,
        txn_order_filename: str,
        bot_id: Optional[str] = None,
        user_id: Optional[str] = None,
        logger=None
) -> bool:
    """
    Write a single row of txn_order to either the database (if bot_id and user_id are provided) or a CSV file.

    :param data_dict: Dictionary containing the txn_order data
    :param trade_cols: List of columns to include in the output
    :param txn_order_filename: Path to the CSV file for fallback writing
    :param bot_id: Optional bot_id for database writing
    :param user_id: Optional user_id for database writing
    :param logger: Logger instance for logging
    :return: True if the write operation was successful, False otherwise
    """
    try:
        # Write to CSV file
        logger.info(f"Writing txn_order to CSV file: {txn_order_filename}")

        # Create a DataFrame from the data_dict
        trade_data_new = pd.DataFrame.from_records([data_dict], columns=trade_cols)
        trade_data_new.set_index("dt_idx", inplace=True)

        file_path = os.path.expanduser(txn_order_filename)
        include_header = True if not os.path.exists(file_path) else False
        write_mode = "w" if include_header else "a"

        # Append the result to the CSV file
        trade_data_new.to_csv(
            txn_order_filename,
            mode=write_mode,
            header=include_header,
            index=True,
            na_rep="None",
            float_format="%.8f"
        )

        logger.info(f"Successfully wrote txn_order to CSV file: {txn_order_filename}")
    except Exception as e:
        err_str = readable_error(e, __file__)
        logger.error(f"[write_txn_order] csv failed err:{err_str}")

    try:
        logger.debug(f"write_txn_order entered bot_id:{bot_id}, user_id:{user_id}")
        # If bot_id and user_id are provided, write to the database
        if bot_id is not None and user_id is not None:
            logger.info(f"Writing txn_order to the database for bot_id: {bot_id}, user_id: {user_id}")

            # Add bot_id and user_id to the data_dict
            data_dict["bot_id"] = bot_id
            data_dict["user_id"] = user_id
            data_dict["exch_order_id"] = str(data_dict.get("orderId"))
            del data_dict["orderId"]

            # Convert the data_dict to a CreateTxnOrderSchema object
            txn_order_data = CreateTxnOrderSchema(**data_dict)

            # Write to the database using the async service function
            result = asyncio.run(create_txn_order(txn_order_data))

            if result:
                logger.info(f"Successfully wrote txn_order to the database for bot_id: {bot_id}, user_id: {user_id}")
                return True
            else:
                logger.error(f"Failed to write txn_order to the database for bot_id: {bot_id}, user_id: {user_id}")
                return False


    except Exception as e:
        err_str = readable_error(e, __file__)
        logger.error(f"[write_txn_order] db failed err:{err_str}")

        return False


@DeprecationWarning
def _get_tech_dir_old(trade_args, alfa1, alfa2=None):
    source_dir = f"{trade_args['tech_data_path']}/tech={trade_args['tech_id']}/" \
                 f"exchange={trade_args['exchange']}/" \
                 f"symbol={trade_args['symbol']}/interval={trade_args['trade_interval']}/" \
                 f"alfa1={alfa1}"

    if alfa2 is not None:
        source_dir += f"|alfa2={alfa2}"

    # source_dir = os.path.expanduser(source_dir)
    if source_dir[:2] == "./":
        project_root = get_project_root()
        source_dir = f"{project_root}/{source_dir}"

    return source_dir

# def extract(_d):
#     # price_alfa_001 -> 001 -> 0.01
#     _delta = float(f"0.{_d.split('_')[2][1:]}")
#     return _delta

def get_tech_dir(trade_args: Dict, tech_args: Dict):
    """
    source_dir = f"{trade_args['tech_data_path']}/tech={trade_args['tech_id']}/" \
                 f"exchange={trade_args['exchange']}/" \
                 f"symbol={trade_args['symbol']}/interval={trade_args['trade_interval']}"
    依照設定檔來組成 tech 檔案位置
    """
    required_params = ["tech_data_path", "tech_id", "exchange", "symbol", "trade_interval"]
    for param in required_params:
        if trade_args.get(param) is None:
            raise Exception(f"double_kf.read_write.get_tech_dir.trade_args required {param}")

    alfa1 = str(tech_args.get("buy_delta")).replace(".", "")
    alfa2 = str(tech_args.get("sell_delta")).replace(".", "")

    # tech_list = tech_args.get("tech_list")
    # if not isinstance(tech_list, list) or len(tech_list) == 0:
    #     raise Exception(f"double_kf.read_write.get_tech_dir.trade_args required tech_list got {tech_list}")
    #
    # alfa1, alfa2 = None, None
    # # 這部分是 Training 時他會是空的，所以要補上
    # for i, v in enumerate(tech_list):
    #     if "price_alfa" in v:
    #         if alfa1 is None:
    #             alfa1 = extract(v)
    #         else:
    #             alfa2 = extract(v)
    #
    # # not specified, then it is training
    # if alfa1 == "xxx" or not isinstance(alfa1, float):
    #     raise Exception(f"alfa1 expect type float but got {alfa1} of type {type(alfa1)}")
    #
    # if alfa2 is not None and (alfa2 == "yyy" or not isinstance(alfa2, float)):
    #     raise Exception(f"alfa2 expect type float but got {alfa2} of type {type(alfa2)}")

    source_dir = f"{trade_args['tech_data_path']}/tech={trade_args['tech_id']}/" \
                 f"exchange={trade_args['exchange']}/" \
                 f"symbol={trade_args['symbol']}/interval={trade_args['trade_interval']}/" \
                 f"alfa1={alfa1}|alfa2={alfa2}"

    # source_dir = os.path.expanduser(source_dir)
    if source_dir[:2] == "./":
        project_root = get_project_root()
        source_dir = f"{project_root}/{source_dir}"

    return source_dir


def save_tech(symbol, trade_args, tech_args, df, append, dest_dir=None, logger=None):
    """
    to_save = bool(input(f"| PRESS 'y' to SAVE Result: {trade_args['tech_data_path']}? ") == 'y')

    if not to_save:
        return
    """

    # folder location
    # dest_dir = f"{trade_args['tech_data_path']}/" \
    #            f"tech={trade_args['tech_id']}/" \
    #            f"exchange={trade_args['exchange']}/" \
    #            f"symbol={symbol}/interval={trade_args['trade_interval']}" if dest_dir is None else dest_dir

    dest_dir = get_tech_dir(trade_args, tech_args) if dest_dir is None else dest_dir

    # extract column names
    cols = list(df.columns.values)
    train_data_schema = {}

    # check data validity
    if 'date' not in cols:
        if df.index.name != 'date':
            raise Exception(f"expect 'date' field in column or in index, but found index named {df.index.name} and \n"
                            f"{cols}")
        elif df.index.name == 'date':
            df.reset_index(inplace=True)
            cols = cols + ['date']
        else:
            raise Exception(f'expect index name to date but got {df.index.name}')

    # setup parquet schema
    for col in cols:
        if col == 'date':
            train_data_schema[col] = 'datetime64[ns]'
        elif col == 'tic':
            train_data_schema[col] = 'str'
        else:
            train_data_schema[col] = 'float64'

    # save
    s = f'[{symbol}] saving {df.shape[0]} rows to {dest_dir} [showing last row]:'
    if logger is not None:
        logger.info(f"[helper] {s}\n{df.iloc[-1].to_dict()}")
    else:
        print(s)
        print(repr(df.iloc[-1].to_dict()))

    if len(df.index) > 0:
        save_data(df, dest_dir, train_data_schema, index='date', append=append, logger=logger)


def create_tech_if_not_exist(trade_args, tech_args, logger, dest_dir=None):
    """
    工具箱，如果在 RL 時檢查是否已有資料，沒有的話就執行
    Parameters
    ----------
    trade_args:
    tech_args
    logger:
    dest_dir: saving destination
    """
    # from ahf.preprocessor.kf.Price_Alfa_Processor import gen_data_v2
    from ahf.rl.strategies.double_kf.Strategy import run_ind


    _form_start = trade_args.get("form_start")
    symbol = trade_args.get("symbol")
    _form_end = None
    _append = False
    _save_result = True
    tech_list = tech_args.get("tech_list")

    assert tech_list is not None, "tech_list cannot be None"

    # Check Alfa_xxx indicator exists, otherwise, generate it
    dest_dir = get_tech_dir(trade_args, tech_args)

    if not is_dir_exist(dest_dir):
        logger.info(f"tech_args does not exist {dest_dir}")
        tech_df = run_ind(_form_start, trade_args, tech_args, logger)
        # Check Alfa_xxx indicator exists, otherwise, generate it
        save_tech(symbol, trade_args, tech_args, tech_df,
                  append=False, dest_dir=dest_dir)



