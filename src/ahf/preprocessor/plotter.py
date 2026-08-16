import argparse
import matplotlib.pyplot as plt
from ahf.utils.helper import load_tech
from ahf.rl.train.config import get_all_args


def plot_gen_data(_form_start, _form_end, _trade_args, _tech_list, _market_list, _market_kf_list):
    exchange = _trade_args['exchange']
    symbol = _trade_args['symbol']
    data_path = _trade_args['tech_data_path']
    trade_interval = _trade_args['trade_interval']

    tech_pd = load_tech(_trade_args, logger=None)

    if _form_end is None:
        tech_pd = tech_pd[_form_start:]
    else:
        tech_pd = tech_pd[_form_start:_form_end]

    rows = []
    chart_id = 0
    for i, v in enumerate(_tech_list):
        if i == 0:
            rows.append(1)
        # move to next group
        elif chart_id == 0 and v.startswith('price_alfa'):
            chart_id += 1
            rows.append(1)
        else:
            rows[chart_id] += 1

    # move to next group: _market_kf_list
    rows.append(1)
    chart_id += 1
    rows[chart_id] = len(_market_kf_list)

    ax1_list = []
    row_col_list = []

    tech_list0 = [v for v in _tech_list if not v.startswith('price_alfa')]
    tech_list1 = [v for v in _tech_list if v.startswith('price_alfa')]

    chart_id = 0
    for i, col in enumerate(tech_list0):
        if i == 0:
            m = 11 + 100 * (rows[chart_id] + 1)
            row_col_list.append(m)

            plt.figure(chart_id + 1, figsize=(15, 8))
            ax1 = plt.subplot(m)
            ax1_list.append(ax1)

            plt.plot(tech_pd['open'], label="open")

            plt.grid()
            plt.legend(loc="best")
            plt.title("price indicators")

        x = row_col_list[chart_id] + i + 1
        plt.subplot(x, sharex=ax1_list[chart_id])
        plt.plot(tech_pd[col], label=col, lw=1)

        plt.grid()
        plt.legend(loc="lower right")

    chart_id = 1
    for i, col in enumerate(tech_list1):
        if i == 0:
            m = 11 + 100 * (rows[chart_id] + 1)
            row_col_list.append(m)

            plt.figure(chart_id + 1, figsize=(15, 8))
            ax1 = plt.subplot(m)
            ax1_list.append(ax1)

            plt.plot(tech_pd['open'], label="open")

            plt.grid()
            plt.legend(loc="best")
            plt.title("price alfa")

        x = row_col_list[chart_id] + i + 1
        plt.subplot(x, sharex=ax1_list[chart_id])
        plt.plot(tech_pd[col][800:], label=col, lw=1)

        plt.grid()
        plt.legend(loc="lower right")

    chart_id = 2
    for i, col in enumerate(_market_kf_list):
        # is_new = (last_chart_id != chart_id)
        if i == 0:
            m = 11 + 100 * (rows[chart_id] + 1)
            row_col_list.append(m)

            plt.figure(chart_id + 1, figsize=(15, 8))
            ax1 = plt.subplot(m)
            ax1_list.append(ax1)

            plt.plot(tech_pd['open'], label="open")

            plt.grid()
            plt.legend(loc="best")
            plt.title("_market_kf_list")

        plt.subplot(row_col_list[chart_id] + i + 1, sharex=ax1_list[chart_id])
        plt.plot(tech_pd[col][800:], label=col, lw=1)

        plt.grid()
        plt.legend(loc="lower right")

        # last_chart_id = chart_id
    plt.show()

    return plt


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--tech_id', type=str, required=False,
                        help='which tech_list indicators to use , e.g. LITE, PROD')

    """
    parser.add_argument('--job_id', type=str, required=True,
                        help='job id, a unique job name to identify training')

    parser.add_argument('--env_name', type=str, required=True,
                        help='env name. e.g. StockTradingEnv-v2')
    """

    parser.add_argument('--trade_args_path', type=str, required=False,
                        help='trade_args path')

    opts = parser.parse_args()

    return opts


def main():

    _cmd_args = vars(parse_arguments())
    all_args = get_all_args(_cmd_args)
    _tech_id, _trade_args, _market_args, _tech_list, _market_list, _market_kf_list = all_args.values()

    _form_start = _trade_args['form_start']
    _form_end = None
    plot_gen_data(_form_start, _form_end, _trade_args, _tech_list, _market_list, _market_kf_list)


if __name__ == '__main__':
    main()
