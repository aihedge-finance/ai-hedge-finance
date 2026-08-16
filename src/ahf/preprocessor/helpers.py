import sys
import time
import datetime
import numpy as np
import pandas as pd

from ahf.utils.utils import readable_error, pretty_dict
from ahf.utils.helper import load_base_price


def preprocessor_load_data(data_args, form_start, form_end, logger):
    df_dict = {}
    if 'symbols' in data_args:
        symbols = data_args['symbols']
    elif 'symbol' in data_args:
        symbols = [data_args['symbol']]
    else:
        raise Exception(f'data_args do not contain symbol or symbols {data_args}')

    start_time = time.time()
    for s in symbols:
        price_pd = load_base_price(data_args['exchange'], s,
                                   data_args.get('price_data_path'),
                                   data_args.get('interval_base'),
                                   logger,
                                   cols=['symbol', 'open', 'close', 'high', 'low'])

        if price_pd is None:
            print(f"price for {s} is empty")
            sys.exit(-1)

        if hasattr(price_pd, 'compute'):
            price_pd = price_pd.compute()

        # clip range if there is any
        if form_end is None:
            # this is used when running bot and we need to generate data on the fly, so
            # we only need later part of the data, no need the entire stretch
            price_pd = price_pd[form_start:]
        else:
            price_pd = price_pd[form_start: form_end]

        # resample according to interval
        price_pd = price_pd.resample(rule=data_args.get('interval_base')).first().ffill()
        # no need to do it again if they are the same
        if data_args['interval_base'] != data_args.get('trade_interval'):
            price_pd = price_pd.resample(rule=data_args.get('trade_interval')).first().ffill()

        # rename column name to fit FinRL
        price_pd = price_pd.rename(columns={"symbol": "tic"})
        price_pd.index.names = ['date']

        # Convert index into a column
        # price_pd = price_pd.reset_index(level=0)

        # df_concat = pd.concat([df_concat, price_pd], axis=0)
        df_dict[s] = price_pd

    end_time = time.time() - start_time
    duration_str = str(datetime.timedelta(seconds=end_time))

    logger.info(f'load_data took {duration_str[:-5]} to complete')

    # key_list = list(df_dict.keys())
    # if len(key_list) > 1:
    #     return df_dict
    # else:
    #     return df_dict[s]

    return df_dict


def price_ta_job(price_ta_processor, price_pd, tech_custom_list, logger):
    try:
        price_ta_arr = []
        history_arr = price_ta_processor.catchup(price_pd)

        # pick first one as role model
        rows_price = len(history_arr[0].index)

        for i, row in enumerate(history_arr):
            assert rows_price == len(row.index), 'length of history price_ta_job record must match'
            tmp = row['rsi']
            price_ta_arr.append(tmp)

        price_ta_np = np.array(price_ta_arr)
        price_ta_arr_T = np.transpose(price_ta_np)
        price_ta_pd = pd.DataFrame(price_ta_arr_T)

        price_ta_pd.columns = tech_custom_list
        price_ta_pd = price_ta_pd.set_index(history_arr[0].index)

        # no need, we use history_arr's index
        # _price_kf_pd = price_kf_pd[~price_kf_pd.index.duplicated(keep='first')]

        return price_ta_pd
    except Exception as e:
        err_str = readable_error(e, __file__)
        logger.error(err_str)

