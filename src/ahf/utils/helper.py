import os
import sys
import time
import talib
import pprint

import warnings
import numpy as np
import pandas as pd
import datetime as dt
from typing import Tuple, Optional
import simplejson as json
from dask.diagnostics import ProgressBar

from ahf.utils import utils
from pathlib import Path
import dask.dataframe as dd
from dask.delayed import Delayed

from ahf.utils.dask_division_helper import setup_divisions
from ahf.utils.utils import readable_error, save_data, get_project_root


def load_tech_last_dt(trade_args, logger=None):
    """
    load technical indicators and get its last updated datetime

    Parameters
    ----------
    trade_args
        - description: parameter needed to construct the path the necessary data
    logger
        - description: logger
    Returns
    -------
    - last updated datetime
    """
    try:
        start_time = time.time()
        trade_interval = trade_args['trade_interval']

        # backward compatibility, and binance is default to 1T
        trade_interval = '1T' if trade_interval is None else trade_interval

        source_dir = f"{trade_args['tech_data_path']}/tech={trade_args['tech_id']}/" \
                     f"exchange={trade_args['exchange']}/" \
                     f"symbol={trade_args['symbol']}/interval={trade_interval}"

        parquet_folder = Path(source_dir).expanduser()
        parquets_ary = set(parquet_folder.glob('part.*.parquet'))
        if len(parquets_ary) > 0:
            parquets_int = (int(str(v.stem).split('.')[-1]) for v in parquets_ary)
            parquets_last_int = sorted(parquets_int)[-1]
            last_file = os.path.expanduser(f'{source_dir}/part.{parquets_last_int}.parquet')
            last_file = [last_file]

            with ProgressBar():
                data_origin_dd: Delayed = dd.from_map(pd.read_parquet, last_file, columns=['close'], engine='pyarrow')

            a = data_origin_dd.index

            last_datetime = data_origin_dd.index.max().compute().to_pydatetime()
            end_time = time.time() - start_time

            s = f'[load_tech_last_dt] last tech dt_idx {last_datetime}, takes {end_time:.2f}s ' \
                f'to load from \n{os.path.expanduser(source_dir)}'
            logger.info(s) if logger else print(s, flush=True)

            return last_datetime

        return None

    except Exception as e:
        err = readable_error(e, __file__)
        logger.error(err) if logger else print(err, flush=True)
        time.sleep(5)
        sys.exit()


def get_tech_dir(trade_args):
    source_dir = f"{trade_args['tech_data_path']}/tech={trade_args['tech_id']}/" \
                     f"exchange={trade_args['exchange']}/" \
                     f"symbol={trade_args['symbol']}/interval={trade_args['trade_interval']}"

    # source_dir = os.path.expanduser(source_dir)
    if source_dir[:2] == "./":
        project_root = get_project_root()
        source_dir = f"{project_root}/{source_dir}"

    return source_dir


def load_tech(trade_args, logger=None):
    try:
        start_time = time.time()
        trade_interval = trade_args['trade_interval']

        # backward compatibility, and binance is default to 1T
        trade_interval = '1T' if trade_interval is None else trade_interval

        source_dir = get_tech_dir(trade_args)

        # with warnings.catch_warnings():
        #    warnings.simplefilter("ignore")
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
        end_time = time.time() - start_time
        if end_time > 3:
            if logger is not None:
                logger.info(f'[load_tech] takes {end_time:.2f}s to load and resample tech indicators from \n{source_dir}')
            else:
                print(f'[load_tech] takes {end_time:.2f}s to load and resample tech indicators from \n{source_dir}', flush=True)

        return tech_pd

    except Exception as e:
        err = readable_error(e, __file__)
        logger.error(err) if logger else print(err, flush=True)
        time.sleep(5)
        sys.exit()


def load_raw_price(exchange, symbol, price_data_path, interval_base, logger=None):
    try:
        # backward compatibility, and binance is default to 1T
        # interval_base = '1T' if interval_base is None else interval_base

        # data_path = './appData/trainData_crypto/prices_v3.parquet'
        start_time = time.time()

        klines = utils.read_data(exchange, symbol, price_data_path, interval=interval_base).compute()
        price_pd = pd.DataFrame(klines, index=klines.index)
        price_raw = price_pd.resample(rule=interval_base).first().ffill()

        end_time = time.time() - start_time
        duration_str = str(dt.timedelta(seconds=end_time))
        s = f'[{exchange}] {symbol} load_raw_price takes {duration_str[:-5]} to complete'
        logger.info(s) if logger else print(s, flush=True)

        return price_raw

    except Exception as e:
        err = readable_error(e, __file__)
        logger.error(err) if logger else print(err, flush=True)
        time.sleep(5)
        sys.exit()


def load_base_price(exchange, symbol, price_data_path, interval_base, logger=None,
                    cols=None, index=None) -> Optional[dd.DataFrame]:
    # exchange, symbol, price_data_path, interval, interval_base, logger,
    #                                      truncated, n_look_back, verbose, source
    try:
        # backward compatibility, and binance is default to 1T
        # interval_base = '1T' if interval_base is None else interval_base

        # data_path = './appData/trainData_crypto/prices_v3.parquet'
        start_time = time.time()

        klines = utils.read_data(exchange, symbol, price_data_path,
                                 interval=interval_base, cols=cols, index=index)

        # price_pd = pd.DataFrame(klines, columns=cols, index=klines.index)
        # price_raw = price_pd.resample(rule=interval_base).first().ffill()

        end_time = time.time() - start_time
        duration_str = str(dt.timedelta(seconds=end_time))
        s = f'[{exchange}] {symbol} load_base_price takes {duration_str} to complete'
        logger.info(s) if logger else print(s, flush=True)

        return klines

    except Exception as e:
        err = readable_error(e, __file__)
        logger.error(err) if logger else print(err, flush=True)
        return None



def load_price(exchange, symbol, price_data_path, trade_interval, interval_base,
               logger=None, verbose=False) -> Tuple[dd.DataFrame, None]:
    try:
        start_time = time.time()

        # Validate interval
        pd.Timedelta(trade_interval)  # Will raise if invalid

        cols_old = ['symbol', 'open', 'close', 'high', 'low']

        dtype_dict = {
            'exchange': 'string',
            'symbol': 'string',
            'yymm': 'string',
            'open': 'float64',
            'high': 'float64',
            'low': 'float64',
            'close': 'float64',
            'volume': 'float64'
        }
        index = ["date"]
        cols = list(dtype_dict.keys())

        # price_raw = load_base_price(exchange, symbol, price_data_path, interval_base, logger,
        #                             cols=cols, index=index)
        #
        # price_pd_actual = price_raw.copy().resample(rule=trade_interval).first().ffill()

        price_dd = load_base_price(exchange, symbol, price_data_path, interval_base, logger,
                                   cols=cols, index=index)

        if price_dd is None:
            raise Exception("load_base_price failed... ")

        # Ensure we have datetime index
        if not pd.api.types.is_datetime64_any_dtype(price_dd.index.dtype):
            raise ValueError("Index must be datetime type")

        # Check index type
        # print(price_dd.index.dtype)
        #
        # a = price_dd.npartitions
        # b = price_dd.divisions

        # First set known divisions using repartition
        #price_dd = price_dd.map_partitions(lambda x: x.set_index(x.index))
        #price_dd = price_dd.map_partitions(lambda x: x.sort_index())

        # If the index is already set
        # min_date = price_dd.index.min().compute()
        # max_date = price_dd.index.max().compute()
        # divisions = pd.date_range(start=min_date, end=max_date, freq='1T')
        #
        # # Repartition with known divisions
        # price_dd = price_dd.repartition(divisions=divisions)

        # def get_partition_bounds(df):
        #     return [df.index.min(), df.index.max()]
        #
        # bounds = price_dd.map_partitions(_get_partition_bounds).compute()
        # bounds = bounds.tolist()
        #
        # divisions = [bounds[0][0]]  # Start with first partition's start
        # for bound in bounds[:-1]:
        #     divisions.append(bound[1])  # Use the end time of each partition except the last
        # divisions.append(bounds[-1][1])  # Add the end time of the last partition
        #
        # # Sort the data and set divisions
        # price_dd = price_dd.reset_index()  # Reset index to make it a column
        # price_dd = price_dd.set_index('date', sorted=True, divisions=divisions)  # Set index back with divisions

        price_dd = setup_divisions(price_dd)

        # Verify
        # print("Number of partitions:", price_dd.npartitions)
        # print("New divisions:", price_dd.divisions)

        # c = price_dd.npartitions
        # d = price_dd.divisions

        price_dd = price_dd.resample(rule=trade_interval).first().ffill()

        # Still DaskDataFrame
        # price_dd = price_dd.persist()

        end_time = time.time() - start_time
        print(f'[priceFetcher] takes {end_time:.2f}s to load and resample prices', flush=True)

        return price_dd, None

    except Exception as e:
        err = readable_error(e, __file__)
        logger.error(err) if logger else print(err, flush=True)
        time.sleep(3)
        sys.exit()


def load_kf_est(exchange, symbol, kf_est_data_path, interval_eval, n_look_back=None, logger=None):
    try:
        cols = ['delta', 'rmse', 'obv_cov']
        if n_look_back is None:
            kf_est = utils.read_data(exchange, symbol, kf_est_data_path, cols).compute()
        else:
            kf_est = utils.read_data(exchange, symbol, kf_est_data_path, cols).compute()[-n_look_back:]

        kf_est_pd = kf_est.resample(rule=interval_eval).first().ffill()

        return kf_est_pd

    except Exception as e:
        err = readable_error(e, __file__)
        logger.error(err) if logger else print(err, flush=True)
        time.sleep(3)
        sys.exit()


def load_pdfo_est(exchange, symbol, pdfo_est_data_path, interval_eval, n_look_back=None, logger=None):
    try:
        dt_col = ['form_start', 'form_end']
        x_col = ['buy_delta', 'sell_delta', 'long_level', 'short_level', 'long_exit_level', 'short_exit_level']
        y_col = ['gain']
        other_col = ['buy_sd_delta', 'buy_sd_delta', 'num_trade', 'accuracy']
        all_col = dt_col + y_col + x_col + other_col

        if n_look_back is None:
            pdfo_est = utils.read_data(exchange, symbol, pdfo_est_data_path, all_col).compute()
        else:
            pdfo_est = utils.read_data(exchange, symbol, pdfo_est_data_path, all_col).compute()[-n_look_back:]

        pdfo_est_pd = pdfo_est.resample(rule=interval_eval).first().ffill()

        return pdfo_est_pd

    except Exception as e:
        err = readable_error(e, __file__)
        print(err, flush=True) if logger else logger.error(err)
        time.sleep(3)
        sys.exit()


def save_result(data, brunhild_data_path, exchange, symbol, interval, append=False, logger=None):
    try:
        # folder location
        dest_dir = '{0}/exchange={1}/symbol={2}/interval={3}'.format(brunhild_data_path, exchange, symbol, interval)

        train_data_schema = {
            'price': 'float64',
            'buy_tracer': 'float64',
            'buy_alfa': 'float64',
            'buy_sd_mv': 'float64',
            'sell_tracer': 'float64',
            'sell_alfa': 'float64',
            'sell_sd_mv': 'float64'
        }
        utils.save_data(data, dest_dir, train_data_schema, append)
    except Exception as e:
        err = readable_error(e, __file__)
        if logger:
            logger.error(err)
        raise Exception(e)


def save_pdfo_result(data, parquet_file, append=False, logger=None):
    try:
        psfo_data_schema = {
            'form_start': 'datetime64[ns]',
            'form_end': 'datetime64[ns]',
            'buy_delta': 'float64',
            'sell_delta': 'float64',
            'long_level': 'float64',
            'short_level': 'float64',
            'long_exit_level': 'float64',
            'short_exit_level': 'float64',
            'buy_sd_delta_band': 'float64',
            'sell_sd_delta_band': 'float64'
        }
        utils.save_data(data, parquet_file, psfo_data_schema, append)

    except Exception as e:
        err = readable_error(e, __file__)
        if logger:
            logger.error(err)
        raise Exception(e)


def save_kf_est_result(data, kf_est_data_path, exchange, symbol, append, logger=None):
    try:
        # folder location
        dest_dir = '{0}/exchange={1}/symbol={2}'.format(kf_est_data_path, exchange, symbol)

        kf_est_data_schema = {
            # 'dt_index': 'datetime64[ns]',
            'delta': 'float64',
            'obv_cov': 'float64',
            'rmse': 'float64'
        }
        utils.save_data(data, dest_dir, kf_est_data_schema, append)
    except Exception as e:
        err = readable_error(e, __file__)
        if logger:
            logger.error(err)
        raise Exception(e)


def to_timestamp(from_dt):
    if isinstance(from_dt, pd.Timestamp):
        to_ts = from_dt.timestamp()
    elif isinstance(from_dt, dt.datetime):
        to_ts = from_dt.replace(tzinfo=dt.timezone.utc).timestamp()
    else:
        raise Exception('unknown type')

    return to_ts


def print_layer(net, net_name, logger):
    logger.info('[Brunhild] {0} with following layers'.format(net_name))

    s = ''
    for layer in net:
        s += '{0} {1}\n {1}\n'.format(layer.name, layer, layer.params)

    logger.info(s)


def save_param(net, net_name, brunhild_data_path, exchange, interval, logger):
    # ==================================================
    # REFERENCE
    # Saving and Loading Gluon Models
    # https://mxnet.apache.org/versions/1.9.0/api/python/docs/tutorials/packages/gluon/blocks/save_load_params.html
    # ==================================================

    try:
        # folder location
        dest_dir = '{0}/exchange={1}/params/{2}_{3}.params' \
                   ''.format(brunhild_data_path, exchange, net_name, interval)
        net.save_parameters(dest_dir)
    except Exception as e:
        err = readable_error(e, __file__)
        logger.error(err)
        raise Exception(e)


def load_param(net, net_name, ctx, brunhild_data_path, exchange, interval, logger):
    try:
        # folder location
        dest_dir = '{0}/exchange={1}/params/{2}_{3}.params' \
                   ''.format(brunhild_data_path, exchange, net_name, interval)
        net.load_parameters(dest_dir, ctx=ctx[0])
    except Exception as e:
        err = readable_error(e, __file__)
        logger.error(err)
        raise Exception(e)


def transform(data, label):
    return data.astype('float32') / 255, label.astype('float32')


def reverse_one_hot(predictions):
    reversed_x = []
    for x in predictions:
        reversed_x.append(np.argmax(np.array(x)))
    return


def HMA(close, period):
    """
    赫爾移動平均線(HMA)
    Hull Moving Average.
    Formula:
    HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    """
    hma = talib.WMA(2 * talib.WMA(close, int(period / 2)) - talib.WMA(close, period), int(np.sqrt(period)))
    return hma


def save_args_prior_train(cwd: str, env, logger=None):
    # save merged to StockTradeEnv.txt

    file_name = f"{cwd}/Env.json"
    data_dict = env.exch_env.spec
    logger.info("> save_args_prior_train\n%s", pprint.pformat(data_dict))

    assert "hyper_args" in data_dict, "env.exch_env.spec must contain hyper_args"
    assert "env_args" in data_dict, "env.exch_env.spec must contain env_args"
    assert "trade_args" in data_dict, "env.exch_env.spec must contain trade_args"
    assert "tech_args" in data_dict, "env.exch_env.spec must contain tech_args"

    assert data_dict["hyper_args"] == env.hyper_args
    assert data_dict["env_args"] == env.env_args
    assert data_dict["trade_args"] == env.trade_args
    assert data_dict["tech_args"] == env.tech_args

    # to prevent json dumpy encounter numpy array and raises an error
    for key, value in list(data_dict.items()):
        if isinstance(value, np.ndarray):
            data_dict[key] = value.tolist()

        if value.__class__.__module__ not in ("builtins", "decimal", "app.enums"):
            del data_dict[key]

        if isinstance(value, dict):
            for sub_key, sub_value in list(value.items()):
                if isinstance(sub_value, np.ndarray):
                    value[sub_key] = sub_value.tolist()

    with open(file_name, "w") as fp:
        json.dump(data_dict, fp, indent=4)

    # save hyper_args
    dest_file_name = f"{cwd}/hyper_args.json"
    assert hasattr(env, "hyper_args"), "hyper_args not found env"
    with open(dest_file_name, "w") as fp:
        json.dump(env.hyper_args, fp, indent=4)

    # save env_args
    dest_file_name = f"{cwd}/env_args.json"
    assert hasattr(env, "env_args"), "env_args not found env"
    with open(dest_file_name, "w") as fp:
        json.dump(env.env_args, fp, indent=4)

    # trade_args
    dest_file_name = f"{cwd}/trade_args.json"
    assert hasattr(env, "trade_args"), "trade_args not found at env.strategy or env.ds"
    trade_args = env.trade_args
    # if "trade_args" in trade_args:
    #     del trade_args["trade_args"]
    with open(dest_file_name, "w") as fp:
        json.dump(trade_args, fp, indent=4)

    # save tech_args.json for record
    dest_file_name = f"{cwd}/tech_args.json"
    assert hasattr(env, "tech_args"), "tech_args not found at env.strategy or env.ds"
    with open(dest_file_name, "w") as fp:
        json.dump(env.tech_args, fp, indent=4)

    print('saving trade_args, tech_args and hyper_args.yml to json file', flush=True)


def unit_test():
    exchange, symbols = 'Binance', ['SOLUSDT', 'ETHUSDT']
    price_data_path = './appData/trainData_crypto/prices_v3.parquet'
    interval, interval_base, interval_eval = '1h', '1T', '1h'
    price_pd = load_price(exchange, symbols, price_data_path, interval, interval_base)

    a = 1


# DEBUG
if __name__ == '__main__':
    unit_test()
