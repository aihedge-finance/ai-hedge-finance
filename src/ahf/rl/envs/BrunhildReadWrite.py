import sys
import time

import numpy as np
import pandas as pd
from os import path
import datetime as dt
from dask.diagnostics import ProgressBar

from typing import Optional, Tuple
import dask.dataframe as dd
from ahf.rl.train.config import get_tech_args
from ahf.utils.utils import is_dir_exist, d
from ahf.utils.utils import readable_error, save_data, get_project_root


def load_data(trade_args, hyper_args, logger):
    try:
        # load tech
        _tech_concat_df = _load_tech(trade_args, hyper_args, logger)

        if trade_args['form_end'] is None:
            tech_df = _tech_concat_df[trade_args['form_start']:]
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
        tech_ary = tech_ary[tech_ary.columns.drop(drop_cols)]

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
        time.sleep(3)
        sys.exit()


def _load_tech(trade_args, tech_list, logger=None):
    try:
        start_time = time.time()
        trade_interval = trade_args['trade_interval']

        # backward compatibility, and binance is default to 1T
        trade_interval = '1T' if trade_interval is None else trade_interval

        source_dir = get_tech_dir(trade_args, tech_list)

        # with warnings.catch_warnings():
        #    warnings.simplefilter("ignore")
        logger.info(f'Loading tech indicator from {source_dir}') if logger else print(
            f'Loading tech indicator from {source_dir}')
        if not is_dir_exist(source_dir):
            raise Exception(f'folder does not exist {source_dir}')

        with ProgressBar():
            tech_dd = dd.read_parquet(source_dir, engine='pyarrow', aggregate_files=True) #.persist()  # , index=index

        # print(f'partition: {tech_dd.npartitions},  division:{tech_dd.divisions[0:5]}, known_divisions:{tech_dd.known_divisions}')

        # REFERENCE
        # https://stackoverflow.com/a/46798024/1596886
        # https://docs.dask.org/en/latest/dataframe-design.html#partitions
        # divisions = pd.date_range('2021-02-02T00:15:00', '2021-03-01T00:00:00', freq=interval)
        # tech_dd = tech_dd.set_index('date', sorted=True, divisions=divisions)
        # tech_dd = tech_dd.resample(rule=interval).first().ffill()

        tech_pd = tech_dd.compute()
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


def load_txn_order(txn_order_filename, trade_cols, trade_cols_type, logger, skip_rows=0, row_count=None) -> Tuple[bool, Optional[pd.DataFrame]]:
    try:
        found = False

        if not path.exists(txn_order_filename):
            return False, None

        txn_order_hist = pd.read_csv(txn_order_filename, usecols=trade_cols, index_col=0, parse_dates=True,
                                     dtype=trade_cols_type, skiprows=skip_rows, nrows=row_count)

        if txn_order_hist is not None and txn_order_hist.shape[0] >= 1:
            found = True
            return found, txn_order_hist
        else:
            return found, None
    except Exception as e:
        err_str = readable_error(e, __file__)
        logger.error(f"[Trade] {err_str}")
        time.sleep(3)
        sys.exit()




def get_tech_dir(trade_args, tech_list):
    """
    source_dir = f"{trade_args['tech_data_path']}/tech={trade_args['tech_id']}/" \
                 f"exchange={trade_args['exchange']}/" \
                 f"symbol={trade_args['symbol']}/interval={trade_args['trade_interval']}"
    依照設定檔來組成 tech 檔案位置
    """
    pass

def save_tech(symbol, trade_args, df, append, dest_dir=None, logger=None):
    """
    to_save = bool(input(f"| PRESS 'y' to SAVE Result: {trade_args['tech_data_path']}? ") == 'y')

    if not to_save:
        return
    """

    # folder location
    dest_dir = f"{trade_args['tech_data_path']}/" \
               f"tech={trade_args['tech_id']}/" \
               f"exchange={trade_args['exchange']}/" \
               f"symbol={symbol}/interval={trade_args['trade_interval']}" if dest_dir is None else dest_dir

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
