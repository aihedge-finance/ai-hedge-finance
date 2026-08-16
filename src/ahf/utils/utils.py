import re
import json
import sys
import pytz
import time
import random
import string
import subprocess
import logging
import decimal
import pprint
import numpy as np
import pandas as pd

from pathlib import Path
from collections import deque
from typing import Optional
import http.client as httplib
from dateutil.parser import parse as dateparse
from decimal import Decimal, ROUND_HALF_UP, ROUND_DOWN

import dask.dataframe as dd
import pyarrow.parquet as pq
import pyarrow.dataset as ds
from dask.diagnostics import ProgressBar

# from dask.cache import Cache
#
# cache = Cache(2e9)  # 2GB cache
# cache.register()

from ahf.core.settings import tw_tz, current_time_zone
import matplotlib.pyplot as plt
from datetime import datetime as dt
from pandas.api.types import is_string_dtype
from typing import Union, List

import warnings

warnings.filterwarnings("ignore")

datefmt4log = "%Y-%b-%d %H:%M:%S"
datefmt = "%Y-%m-%d %H:%M:%S"  # e.g. 2019-11-16 23:16:15
datesffmt = "%Y-%m-%d %H:%M:%S.%f"  # nano-seconds e.g. 2022-02-22 23:23:25,91499
# Sample for microseconds
# dt_idx.strftime(utils.datesffmt)[:-3]  # microseconds

binance_columns = ["date", "open",
                   "high", "low", "close", "volume", "close_time", "quote_av",
                   "trades", "tb_base_av", "tb_quote_av", "ignore"]

binance_columns_type = {"date": str,
                        "open": float,
                        "high": float,
                        "low": float,
                        "close": float,
                        "volume": float,
                        "close_time": str,
                        "quote_av": str,
                        "trades": str, "tb_base_av": str, "tb_quote_av": str, "ignore": str}

price_columns = ["date", "open", "high", "low", "close", "volume"]

position_columns = ["date", "position"]
position_columns_type = {"date": str, "position": int}

signal_columns = ["date", "signal"]
signal_columns_type = {"date": str, "signal": int}

prc_level_columns = ["date", "prc_level"]
prc_level_columns_type = {"date": str, "prc_level": int}

cash_pos_columns = ["date", "cash_pos"]
cash_pos_columns_type = {"date": str, "cash_pos": float}

num_share_columns = ["date", "share"]
num_share_columns_type = {"date": str, "share": float}

CoSpreadTrade_columns = ["date", "spread"]
CoSpreadTrade_columns_type = {"date": str, "spread": float}


def return_not_matches(a, b):
    return [[x for x in a if x not in b], [x for x in b if x not in a]]


def get_project_root() -> Path:
    p = Path(__file__).expanduser().parent.parent
    return p


def InvertP2B_interval(p):
    """
    Pandas's string alias
    N -> Nano ;us -> Micro; ms -> Milli; S -> seconds; T -> minutes; H -> hours; D -> days; W -> weeks;  M -> months

    m -> minutes; h -> hours; d -> days; w -> weeks; M -> months
    1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M
    :param p:
    :return:
    """
    p = p.replace("W", "w")
    p = p.replace("D", "d")
    p = p.replace("H", "h")
    p = p.replace("T", "m")
    p = p.replace("min", "m")

    return p

def remove_char(p: str):
    p = p.lower()
    p = p.replace("w", "")
    p = p.replace("d", "")
    p = p.replace("h", "")
    p = p.replace("t", "")
    p = p.replace("min", "")
    p = p.replace("m", "")

    return p

def calculate_days_to_download(s):
    # ignore hrs and days for now
    # 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M
    try:
        temp = re.compile("([0-9]+)([a-zA-Z]+)")
        res = temp.match(s).groups()
        days = int(1000 / (24 * 60 / int(res[0])) / 2)

        if days == 0:
            days = 1
        return days

    except Exception as e:
        print(readable_error(e, __file__), flush=True)
        return 1


def utcToLocal(utc, time_zone=pytz.timezone(current_time_zone), display_offset=False):
    # Reference
    # https://learningactors.com/working-with-datetime-objects-and-timezones-in-python/
    # https://howchoo.com/g/ywi5m2vkodk/working-with-datetime-objects-and-timezones-in-python
    if isinstance(utc, str):
        utc = dt.strptime(utc, datefmt)

    if isinstance(time_zone, str):
        time_zone = pytz.timezone(time_zone)

    t = pytz.utc.localize(utc, is_dst=None).astimezone(time_zone)
    if not display_offset:
        t = dt.strftime(t, datefmt)
    return t


def utc2LocalDf(df, time_zone=current_time_zone, display_offset=False):
    # Reference:
    # https://stackoverflow.com/questions/46295355/pandas-cant-compare-offset-naive-and-offset-aware-datetimes

    if is_string_dtype(df):
        df = pd.to_datetime(df)

    t = df.dt.tz_localize("utc").dt.tz_convert(time_zone)
    if not display_offset:
        t = t.dt.tz_localize(None)

    return t


def date2tw(_d):
    _d = date2datetime(_d)
    _d = _d.replace(tzinfo=tw_tz)
    return _d


def date2datetime(_d):
    _d = dt(
        year=_d.year,
        month=_d.month,
        day=_d.day,
    )
    return _d


def normalize_fraction(num_string) -> decimal.Decimal:
    # REFERENCE
    # Remove trailing zeros
    # https://stackoverflow.com/a/11227743/1596886
    d = decimal.Decimal(num_string)
    normalized = d.normalize()
    sign, digits, exponent = normalized.as_tuple()
    if exponent > 0:
        return decimal.Decimal((sign, digits + (0,) * exponent, 0))
    else:
        return normalized


def least_significant_digit_power(num_string) -> int:
    # REFERENCE
    # https://stackoverflow.com/a/25571529/1596886

    if "." in num_string:
        # There"s a decimal point. Figure out how many digits are to the right
        # of the decimal point and negate that.
        return len(num_string.partition(".")[2])
    else:
        # No decimal point. Count trailing zeros.
        return -(len(num_string) - len(num_string.rstrip("0")))


def have_internet():
    conn = httplib.HTTPSConnection("8.8.8.8", timeout=5)
    try:
        conn.request("HEAD", "/")
        return True
    except Exception:
        return False
    finally:
        conn.close()


def convert_interval_to_wait_time(interval):
    # Pandas's string alias
    # N -> Nano ;us -> Micro; ms -> Milli; S -> seconds; T -> minutes; H -> hours; D -> days; W -> weeks;  M -> months
    try:
        temp = re.compile("([0-9]+)([a-zA-Z]+)")
        res = temp.match(interval).groups()
        unit = res[1]

        def f(x):
            return {
                "N": 10 ** -9,
                "us": 10 ** -6,
                "ms": 10 ** -3,
                "S": 1,
                "T": 60,
                "m": 60,
                "H": 60 * 60,
                "h": 60 * 60,
                "D": 60 * 60 * 24,
                "d": 60 * 60 * 24,
                "W": 60 * 60 * 24 * 7,
                "M": 60 * 60 * 24 * 30
            }.get(x, -1)

        r = f(unit)
        if r == -1:
            print("[utils.py] convert_interval_to_wait_time(interval) unit cannot be found: {}".format(interval), flush=True)
            sys.exit(-3)

        r = r * res[0]
        return r

    except Exception as e:
        print("converting interval to wait time failed.... interval:{0}, {1}".format(interval, e), flush=True)
        sys.exit(-3)

def is_iso_format_str(date_text):
    try:
        if not isinstance(date_text, str):
            return False

        dateparse(date_text)

        return True
    except ValueError:
        return False

def convert_iso_to_datetime(date_text):
    try:
        return dateparse(date_text)
    except ValueError:
        raise Exception(f"convert_iso_to_datetime date_text {date_text} is not iso=8601 format")

def read_data(exchange, symbol, price_root_dir, interval=None, cols:List = None, index: [List, str] = None):
    try:
        dest_no_interval_dir = f"{price_root_dir}/exchange={exchange}/symbol={symbol}"
        dest_interval_dir = f"{price_root_dir}/exchange={exchange}/symbol={symbol}/interval={interval}"

        if dest_interval_dir[:2] == "./" or dest_interval_dir[:1] != "/":
            dest_no_interval_dir = f"{get_project_root()}/{dest_no_interval_dir}"
            dest_interval_dir = f"{get_project_root()}/{dest_interval_dir}"

        if is_dir_exist(dest_interval_dir):
            dest_dir = dest_interval_dir
        else:
            dest_dir = dest_no_interval_dir

        # Load the dataset using PyArrow. PyArrow will recognize partition columns automatically (hive-style).
        dataset = ds.dataset(price_root_dir, format="parquet", partitioning="hive")
        print("== = PyArrow Schema == = ", flush=True)
        print(dataset.schema, flush=True)
        print("== Partition expression == ", flush=True)
        print(dataset.partitioning.schema, flush=True)

        # if True:
            # Retrieve and inspect the schema
            # schema = dataset.schema
            # print("Schema in the entire dataset:")
            # print(schema)

            # Read the schema from the _metadata file
            # metadata_path = f"{price_root_dir}/_metadata"
            # metadata_schema = pq.read_metadata(metadata_path).schema
            # print("Schema from _metadata:")
            # print(metadata_schema)
            #
            # # reading single file
            # table = pq.read_table(dest_no_interval_dir)
            # print("Columns in file:")
            # print(table.column_names)
            #
            # df_dask = dd.read_parquet(price_root_dir, engine="pyarrow")
            #
            # print("=== Dask Schema ===")
            # print(df_dask.dtypes)
            # print("Partitioning info: ")
            # print(df_dask.divisions)

        # table = pq.read_table(f"{price_root_dir}/exchange=Binance/symbol=BTCUSDT/part.0.parquet")
        # print("Partition schema:", table.schema)

        # Define filters for hive-partitioned data
        filters = [
            ("exchange", "=", exchange),
            ("symbol", "=", symbol)
        ]
        with ProgressBar():
            data_dd = dd.read_parquet(price_root_dir,
                                      engine="pyarrow",
                                      columns=cols,  # list(dtype_dict.keys()),
                                      # dtype=dtype_dict,
                                      index = index,
                                      filters=filters,
                                      infer_divisions = True,
                                      use_nullable_dtypes=True,  # New in recent versions
                                      dtype_backend="pyarrow",  # New in recent versions
                                      aggregate_files=True).persist()

        return data_dd
    except Exception as e:
        print(f"Error read_data: {readable_error(e, __file__)}", flush=True)
        raise


def save_data(data, dest_dir: str, schema, index: [List, str] = None, append: bool = True, partition_on: List = None, logger=None):
    # make directory
    os.makedirs(dest_dir, exist_ok=True)
    # make sure type is correct
    data = data.astype(dtype=schema)
    # create index
    # data.set_index("date", inplace=True)

    try:
        data_dd = dd.from_pandas(data, npartitions=1)

        if index is not None:
            data_dd = data_dd.set_index(index, sorted=True)

        s = f"[save_data] parquet division: {data_dd.divisions[0]} ~ {data_dd.divisions[-1]}"
        logger.info(s) if logger is not None else print(s, flush=True)

        # data_dd = data_dd.repartition(freq=interval)

        data_dd.to_parquet(dest_dir,
                           engine="pyarrow",
                           write_metadata_file=True,
                           append=append,
                           partition_on=partition_on,
                           overwrite=False, write_index=True, compute=True,
                           )  # ignore_divisions=True, append=True, overwrite=False, ignore_divisions=True
    except Exception as e:
        print("Error saving data: {0}".format(e), flush=True) if not logger else logger.error("Error saving data: {0}".format(e))
        time.sleep(3)
        sys.exit()


def d(data):
    if not isinstance(data, Decimal):
        # if isinstance(data, float) or isinstance(data, int) or isinstance(data, str):
        data = Decimal(str(data))
    return data


def d_round(data, decimal_place, rounding=ROUND_DOWN):
    """
    基本上我們要 ROUND_DOWN，因為錢不可能變多，你只能少得到或少給，多給做不到，少給無法成交
    手續費似乎是四捨五入 ROUND_HALF_UP

    詳細解說
    https://blog.csdn.net/czx840624424/article/details/108556904
    """
    if not isinstance(data, Decimal):
        data = d(data)

    if not isinstance(decimal_place, int):
        raise Exception(f"Decimal place must be an integer: {decimal_place}")

    if decimal_place < 0:
        raise Exception(f"Decimal place must be non-negative: {decimal_place}")

    quantized_data = data.quantize(d(f"1e-{decimal_place}"), rounding=rounding)
    return quantized_data


def d_round_fee(data, decimal_place, rounding=ROUND_HALF_UP):
    """
    基本上我們要 ROUND_DOWN，因為錢不可能變多，你只能少得到或少給，多給做不到，少給無法成交
    手續費似乎是四捨五入 ROUND_HALF_UP

    詳細解說
    https://blog.csdn.net/czx840624424/article/details/108556904
    """
    if not isinstance(data, Decimal):
        data = d(data)

    if not isinstance(decimal_place, int):
        raise Exception(f"Decimal place must be an integer: {decimal_place}")

    if decimal_place < 0:
        raise Exception(f"Decimal place must be non-negative: {decimal_place}")

    quantized_data = data.quantize(d(f"1e-{decimal_place}"), rounding=rounding)
    return quantized_data


def d_is_close(a: Decimal, b: Decimal, decimal_place: int):
    sameSign = (a * b) > 0
    if sameSign:
        return d_abs(d_abs(a) - d_abs(b)) <= d(f"1e-{decimal_place}")
    else:
        # 嚴格一點，abs 後加起來不能超過 significance
        return d_abs(d_abs(a) + d_abs(b)) <= d(f"1e-{decimal_place}")


def d_negate(data):
    return Decimal.copy_negate(d(data))


def d_abs(data):
    return d(data).copy_abs()

class DecimalEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle Decimal objects"""
    def default(self, obj):
        if isinstance(obj, Decimal):
            return str(obj)  # Convert Decimal to string to preserve precision
        return super(DecimalEncoder, self).default(obj)

# def readable_error(e, file_name):
#     s = f"{type(e).__name__} at line {e.__traceback__.tb_lineno} of {file_name}: {e}"
#     return s


def tz_now(hours_from_utc):
    """
    取得指定時區的現在時間
    """
    return dt.now(tz.utc) + timedelta(hours=hours_from_utc)


def tz_today(hours_from_utc):
    """
    取得指定時區的今天日期
    """
    return dt.fromordinal((dt.now(tz.utc) + timedelta(hours=hours_from_utc)).date().toordinal())


def tw_now():
    """
    取得台灣時區的現在時間
    """
    return tz_now(8)


def tw_today():
    """
    取得台灣時區的今天日期
    """
    return tz_today(8)


def readable_error(e, file_name):
    exc_type, exc_value, exc_traceback = sys.exc_info()
    tb_lineno = exc_traceback.tb_lineno
    s = f"{exc_type.__name__} at line {tb_lineno} of {file_name}: {e}"
    return s


def _normalize_price(priceA, interval="1T", logger=None):
    try:
        # check out test for debugging visually

        priceA = priceA.resample(interval, closed="right", label="left").mean()

        # priceA = priceA.where(pd.notnull(priceA), None)
        # priceB = priceB.where(pd.notnull(priceB), None)

        priceA = priceA.interpolate()

        # fig, axes = plt.subplots(nrows=2,
        #                         ncols=1,
        #                         sharex=True)
        # priceA.plot(ax=axes[0], linewidth=2, color="b", linestyle="solid")
        # priceB.plot(ax=axes[1], linewidth=2, color="b", linestyle="solid")

        # df = pd.concat([priceA, priceB], axis=1)
        # df.columns = ["priceA", "priceB"]

        # axB = priceB.plot()
        # (df["priceB"]-10).plot(ax=axB)

        first_valid_index = priceA.first_valid_index()
        last_valid_index = priceA.last_valid_index()

        return priceA, first_valid_index, last_valid_index

    except Exception as e:
        if logger:
            logger(readable_error(e, __file__))
        else:
            print(readable_error(e, __file__), flush=True)


def setup_logger(file_name, symbol, log_level=logging.INFO):
    """
    完全概念，忘了就看
    https://jacychu.medium.com/python-logging模組介紹-f678707808ed
    """
    project_root = get_project_root()

    logger = logging.getLogger(symbol)

    # Clear any existing handlers
    if logger.handlers:
        logger.handlers.clear()

    log_file = f"{project_root}/logs/{file_name}"
    log_dir = os.path.dirname(os.path.abspath(log_file))
    dir_exist = is_dir_exist(log_dir)
    if not dir_exist:
        os.makedirs(log_dir, exist_ok=True)
        print(f"| folder {log_dir} is created", flush=True)

    # file handler to save all levels
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)

    # stream handler (console) to show only INFO level and above
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(log_level)  # Only INFO level and above

    # create formatter
    # OLD formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s: %(message)s")
    formatter = logging.Formatter("%(asctime)s|%(levelname)s|%(name)s|%(filename)s:%(lineno)d|[tid:%(thread)d]|%(message)s", datefmt="%Y-%m-%d %H:%M:%S.%f")

    # add formatter to handlers
    stream_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add handlers to logger
    # logger.addHandler(stream_handler)  # it will output to stream even it is file_handler
    logger.addHandler(file_handler)

    logger.setLevel(log_level)
    logger.handlers[0].flush()

    return logger


def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return "".join(random.choice(chars) for _ in range(size))


def safe_div(a, b):
    re = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    return re


def convert_to_day(s):
    day_per_unit = {"T": 1 / (60 * 24), "h": 1 / 24, "d": 1, "w": 7}
    return int(s[:-1]) * day_per_unit[s[-1]]


def convert_to_hour(s):
    hour_per_unit = {"T": 1 / 60, "h": 1, "d": 24, "w": 1440}
    return int(s[:-1]) * hour_per_unit[s[-1]]


def convert_to_min(s):
    min_per_unit = {"T": 1, "h": 60, "d": 60 * 24, "w": 10080}
    return int(s[:-1]) * min_per_unit[s[-1]]


def is_dir_exist(dir_path) -> bool:
    if dir_path[:2] == "./":
        dir_path = f"{get_project_root()}/{dir_path}"
    dir_path_obj = Path(dir_path).expanduser()
    is_exist = dir_path_obj.exists()
    is_dir = dir_path_obj.is_dir()
    return is_exist and is_dir


def pct_change(a: Union[np.ndarray, List[float]], include_first: bool = False) -> np.ndarray:
    """
    Calculate percentage change for a numeric array or list of floats.

    Args:
        a (Union[np.ndarray, List[float]]): Input array (prices as float values).
        include_first (bool): If True, include the first value as 0% change.

    Returns:
        np.ndarray: Array of percentage changes (dtype=float).
    """
    if not isinstance(a, (np.ndarray, list)):
        raise TypeError("Invalid prices type, it must be a list or a numpy array")

    # Ensure input is a NumPy array backed by float type
    a = np.asarray(a, dtype=float)

    # Calculate percentage changes
    re_pair = (np.diff(a) / a[:-1]) * 100
    if include_first:
        re_pair = np.insert(re_pair, 0, 0.0, axis=0)  # Add 0% change for the first value
        re_pair[0] = 0.0  # Set explicitly to 0.0 for consistency
    return re_pair


def d_pct_change(a: Union[np.ndarray, List[Decimal]], include_first: bool = False) -> np.ndarray:
    """
    Calculate percentage change for a numeric array or list of decimals.

    Args:
        a (Union[np.ndarray, List[Decimal]]): Input array (prices as `decimal.Decimal` values).
        include_first (bool): If True, include the first value as 0% change.

    Returns:
        np.ndarray: Array of percentage changes (dtype=Decimal).
    """
    if not isinstance(a, (np.ndarray, list)):
        raise TypeError("Invalid prices type, it must be a list or a numpy array")

    # If using NumPy array, ensure it contains decimals
    if isinstance(a, np.ndarray) and not np.issubdtype(a.dtype, np.object_):
        raise TypeError("If input is a NumPy array, it must have dtype=object to support Decimal")

    # Ensure input values are of type Decimal
    if isinstance(a, list):
        if not all(isinstance(i, Decimal) for i in a):
            raise TypeError("All elements in the input list must be of type `decimal.Decimal`")

    # Convert list to a numpy array of dtype=object (to safely hold Decimals)
    a = np.asarray(a, dtype=object)

    # Calculate percentage changes with Decimal precision
    re_pair = np.array([(a[i + 1] - a[i]) / a[i] * Decimal("100") for i in range(len(a) - 1)], dtype=object)

    if include_first:
        # Add 0% change for the first value
        re_pair = np.concatenate((np.array([Decimal("0")], dtype=object), re_pair))

    return re_pair


def pretty_dict(data):
    pprint.PrettyPrinter(indent=4)
    data_repr = pprint.pformat(data)
    # usage
    # print(f"{name} = {data_repr}") if logger is None else logger.info(f"{name} = {data_repr}")

    return data_repr


def plot_sim(opts, obv, final_pnl, final_gain, hash_tag, logger):
    if opts.image or opts.plot:
        # plotting
        plt.figure(1, figsize=(15, 10))

        l = len(obv.buy_tracer)
        ax1 = plt.subplot(711)
        plt.plot(obv.dt_index[:l], obv.price_mu[:l], label="price mu")
        plt.plot(obv.dt_index[:l], obv.buy_simple, label="buy simple", ls="-.", lw=1)
        ax1.set_ylabel("price")
        plt.grid()
        plt.legend(loc="upper left")
        plt.title("{0} TradingStrategy {1}, pnl: ${2:.2f}, "
                  "gain {3:.2f}%".format(opts.strategy, opts.symbol, final_pnl, final_gain))

        ax2 = plt.subplot(712, sharex=ax1)
        plt.step(obv.dt_index[:l], obv.signal[:l], label="Signal")
        plt.axhline(0, color="black", ls="-.", lw=1)
        plt.axhline(1, color="green", ls="-.", lw=1)
        plt.axhline(-1, color="green", ls="-.", lw=1)
        ax2.set_ylabel("signal")
        plt.grid()
        plt.legend(loc="upper right")

        ax3 = plt.subplot(713, sharex=ax1)
        plt.step(obv.dt_index[:l], obv.position[:l], label="Position")
        plt.axhline(0, color="black", ls="-.", lw=1)
        plt.axhline(1, color="green", ls="-.", lw=1)
        plt.axhline(-1, color="green", ls="-.", lw=1)
        ax3.set_ylabel("position")
        plt.grid()
        plt.legend(loc="upper right")

        ax4 = plt.subplot(714, sharex=ax1)
        # plt.step(obv.dt_index[:l], obv.share[:l], label="share", lw=1)
        # ax4.set_ylabel("share")
        y = obv.realized_pnl[:l] / obv.cash_asset[:l] * 100
        plt.bar(obv.dt_index[:l], y, width=0.1,
                color=np.where(obv.pnl < 0, "crimson", "blue"),
                label="realized_pnl %")
        # ax4.set_yscale("log")
        plt.axhline(0, color="black", ls="-.", lw=0.5)
        ax4.set_ylabel("realized pnl")

        plt.grid()
        plt.legend(loc="upper left")

        ax5 = plt.subplot(715, sharex=ax1)
        plt.plot(obv.dt_index[:l], obv.cash_asset[:l], label="cash + asset", lw=1)
        ax5.set_ylabel("cash+asset")
        ax5.set_yscale("log")
        plt.grid()
        plt.legend(loc="upper left")

        ax6 = plt.subplot(716, sharex=ax1)
        plt.plot(obv.dt_index[:l], obv.buy_alfa, label="buy_alfa", lw=1)
        plt.plot(obv.dt_index[:l], obv.sell_alfa, label="sell_alfa", lw=1)
        plt.axhline(0, color="black", ls="-.", lw=1)
        # plt.axhline(opts.long_level, color="green", ls="-.", lw=1)
        # plt.axhline(opts.short_level, color="green", ls="-.", lw=1)
        plt.plot(obv.dt_index[:l], obv.long_level[:l], color="green", label="long_level", ls="--", lw=0.5)
        plt.plot(obv.dt_index[:l], obv.long_exit_level[:l], color="blue", label="long_exit_level", ls="-.", lw=0.5)
        plt.plot(obv.dt_index[:l], obv.short_level[:l], color="green", label="short_level", ls="--", lw=0.5)
        plt.plot(obv.dt_index[:l], obv.short_exit_level[:l], color="blue", label="short_exit_level", ls="-.", lw=0.5)
        plt.plot(obv.dt_index[:l], obv.buy_sd[:l], color="blue", label="buy_sd", ls="dotted", lw=0.5)

        ax6.set_ylabel("buy_alfa")
        plt.grid()
        plt.legend(loc="upper right")

        # ax7 = plt.subplot(717, sharex=ax1)
        # plt.plot(re_train["buy_sd_mv"], label="buy_sd_mv", lw=1)
        # plt.axhline(0, color="black", ls="-.", lw=1)
        # plt.axhline(tracer_entry_level, color="green", ls="-.", lw=1)
        # plt.axhline(-tracer_entry_level, color="green", ls="-.", lw=1)
        # ax7.set_ylabel("buy_sd_mv")

        # num_profit_trade_np = np.array(obv.num_profit_trade)/10
        ax7 = plt.subplot(717, sharex=ax1)
        plt.plot(obv.dt_index[1:l], obv.buy_delta[1:], label="buy delta", lw=1)
        plt.plot(obv.dt_index[1:l], obv.sell_delta[1:], label="sell delta", lw=1)
        # plt.plot(obv.dt_index[1:l], num_profit_trade_np[1:], label="last 10 successful trade", lw=1)

        ax7.set_ylabel("leader delta")
        plt.grid()
        plt.legend(loc="upper right")

        if opts.image:
            logger.info("Generating PNG image ....")
            # logger.info("=================================================")
            project_root = get_project_root()
            plt.savefig("{0}/img/RunSim_{1}_gain{2:.0f}_{3}.png"
                        "".format(project_root, opts.symbol, final_gain, hash_tag),
                        bbox_inches="tight", dpi=300)

        if opts.plot:
            plt.show()

        return plt


def extract_num(sentence):
    import re

    s = [int(s) for s in re.findall(r"-?\d+\.?\d*", sentence)]

    return s


def save_file(file_path, file_data, write_mode="w"):
    if isinstance(file_data, str):
        with open(file_path, write_mode) as saved_file:
            saved_file.write(file_data)
    elif isinstance(file_data, bytes):
        with open(file_path, f"{write_mode}b") as saved_file:
            saved_file.write(file_data.encode("utf-8"))


def is_dt_offset_aware(d):
    is_offset = d.tzinfo is not None and d.tzinfo.utcoffset(d) is not None

    return is_offset


def create_dir_if_non_exist(file_dir: str):
    dir_path = os.path.dirname(file_dir)
    os.makedirs(dir_path, exist_ok=True)


def set_reset_trade_cash(amt: float, user_bot_id :str, logger):
    try:
        project_root = get_project_root()
        with open(f"{project_root}/appData/user_tradebot/{user_bot_id}/tradeBot/RESET_TRADE_CASH.txt", "w") as file:
            file.write(str(amt))

        return True

    except Exception as e:
        err = readable_error(e, __file__)
        logger.exception(f"[set_reset_trade_cash] {err}")
        return False


def get_reset_trade_cash_txt(user_bot_id: str, logger) -> Optional[Decimal]:
    try:
        project_root = get_project_root()
        with open(f"{project_root}/appData/user_tradebot/{user_bot_id}/tradeBot/RESET_TRADE_CASH.txt", "r") as file:
            value = d(file.read().strip())

        return value
    except Exception as e:
        err = readable_error(e, __file__)
        logger.exception(f"[set_reset_trade_cash] {err}")
        raise


def get_txt_file(file_name: str, logger) -> str:
    try:
        project_root = get_project_root()
        with open(f"{project_root}/{file_name}", "r") as file:
            value = file.read().strip()

        return value
    except Exception as e:
        err = readable_error(e, __file__)
        logger.exception(f"[get{file_name}] {err}")
        raise Exception(f"unable to open file {file_name}")

def set_txt_file(file_name: str, data: str, logger):
    try:
        project_root = get_project_root()
        with open(f"{project_root}/{file_name}", "w") as file:
            file.write(str(data))
        return True

    except Exception as e:
        err = readable_error(e, __file__)
        logger.exception(f"set_txt_file failed [{file_name}]; {err}")
        return False


def get_decimal_place(f):
    n = len(str(f).split(".")[1])
    return n


def convert_trade_interval(trade_interval):
    # read trading time interval
    if trade_interval == "1s":
        return 1
    elif trade_interval == "5s":
        return 5
    elif trade_interval == "1T":
        return 60
    elif trade_interval == "3T":
        return 60 * 3
    elif trade_interval == "5T":
        return 60 * 5
    elif trade_interval == "10T":
        return 60 * 10
    elif trade_interval == "15T":
        return 60 * 15
    elif trade_interval == "1d":
        return 24 * 60 * 60
    else:
        raise ValueError("Time interval input is NOT supported yet.")


def normalize_decreasing_values(x, lower=0.0001, upper=0.9999, min_value=1e-7):
    """
    Please CHECK OUT README.md at /envs/README.md
    Normalize decreasing values less than 1 to a range between lower and upper bounds.

    :param x: Input value or array to normalize (should be less than 1)
    :param lower: Lower bound of the output range (default: 0.0001)
    :param upper: Upper bound of the output range (default: 0.9999)
    :param min_value: Minimum value to prevent log(0) (default: 1e-10)
    :return: Normalized value(s) between lower and upper bounds
    """
    # Ensure x is at least min_value to prevent log(0)
    x = np.maximum(x, min_value)

    # Apply log transformation
    log_x = np.log(x)

    # Calculate the range for log values
    log_min = np.log(min_value)
    log_max = np.log(1)  # log(1) = 0

    # Normalize log values to 0-1 range
    normalized = (log_x - log_min) / (log_max - log_min)

    # Invert the normalization because smaller input values should map to larger output values
    normalized = 1 - normalized

    # Scale to desired output range
    scaled = lower + (upper - lower) * normalized

    return scaled


def normalize_decreasing_values_centered(x, lower=0.0001, upper=0.9999):
    """
    Normalize decreasing values less than 1 to a range between lower and upper bounds,
    attempting to center the distribution.

    :param x: Input value or array to normalize (should be less than 1)
    :param lower: Lower bound of the output range (default: 0.0001)
    :param upper: Upper bound of the output range (default: 0.9999)
    :return: Normalized value(s) between lower and upper bounds
    """
    # Automatically set min_value and center_value based on lower and upper bounds
    min_value = lower / 10  # Set min_value to one-tenth of the lower bound
    center_value = np.sqrt(lower * upper)  # Geometric mean of lower and upper bounds

    # Ensure x is at least min_value to prevent log(0)
    x = np.maximum(x, min_value)

    # Apply log transformation
    log_x = np.log(x)
    log_center = np.log(center_value)

    # Calculate the range for log values
    log_min = np.log(min_value)
    log_max = np.log(1)  # log(1) = 0

    # Normalize log values to -1 to 1 range, centered around log_center
    normalized = (log_x - log_center) / max(log_center - log_min, log_max - log_center)

    # Scale and shift to desired output range
    scaled = lower + (upper - lower) * (normalized + 1) / 2

    return scaled


def to_dict(obj, depth=0):
    def serialize_value(value):
        if isinstance(value, (int, float, str, bool, Decimal)):
            return value
        elif isinstance(value, type):
            return value.__name__
        elif isinstance(value, (np.ndarray, deque)):
            return list(value)[-5:]
        elif hasattr(value, "__dict__"):
            return value.__class__.__name__
        return None  # Default case

    result = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            result[key] = serialize_value(value)
    else:
        for attr_name, attr_value in vars(obj).items():
            if isinstance(attr_value, dict) and depth == 0:
                result[attr_name] = to_dict(attr_value, depth + 1)
            else:
                result[attr_name] = serialize_value(attr_value)

    return result


import os
from datetime import datetime as dt, timedelta, timezone as tz
from pathlib import Path


def rename_file_with_datetime(filepath):
    """
    Renames a file by adding datetime stamp before the extension.
    Supports ~ expansion for home directory.

    Args:
        filepath (str): Path to file (can include ~/ for home directory)

    Returns:
        str: New filepath with datetime stamp (with same path format as input)
    """
    try:
        # Expand ~ to full home directory path
        expanded_path = Path(filepath).expanduser()

        # Ensure the file exists
        if not expanded_path.exists():
            raise FileNotFoundError(f"File not found: {expanded_path}")

        # Split the filepath into parts
        directory = expanded_path.parent
        filename = expanded_path.name
        name, ext = os.path.splitext(filename)

        # Generate datetime stamp
        datetime_stamp = dt.now().strftime("%Y%m%d_%H%M%S")

        # Create new filename with datetime stamp
        new_filename = f"{name}_{datetime_stamp}{ext}"

        # Create new full filepath
        new_filepath = directory / new_filename

        # Rename the file
        expanded_path.rename(new_filepath)

        # If the input used ~/, return the path with ~/ notation
        if str(filepath).startswith("~/"):
            return str(new_filepath).replace(str(Path.home()), "~")
        return str(new_filepath)

    except Exception as e:
        print(f"Error renaming file: {readable_error(e, __file__)}", flush=True)
        return None


def get_git_commit_hash():
    try:
        # Run the git command to get the current commit hash
        commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("utf-8")
        return commit_hash
    except subprocess.CalledProcessError as e:
        print("Error retrieving commit hash:", e, flush=True)
        return None