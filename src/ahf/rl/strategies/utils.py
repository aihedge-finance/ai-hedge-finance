import numpy as np
import matplotlib.pyplot as plt
from ahf.utils.utils import convert_to_min, pretty_dict, is_dir_exist, readable_error


def order_trade_args_checker(trade_args):
    if trade_args.get('exch_mode') not in ['SpotTest', 'SpotAPI']:
        raise Exception(f'exch_mode can only be SpotTest or SpotAPI but got {trade_args.get("exch_mode")}, bot existing')

    if trade_args.get('train_mode') not in ['DEV', 'PROD']:
        raise Exception(f'train_mode can only be DEV but got {trade_args.get("train_mode")}, bot existing')

    # if trade_args['tech_id'] not in trade_args:
    #     print(f"Trade_args for {trade_args['tech_id']} not found")
    #     sys.exit()

    if 'symbol' not in trade_args:
        raise Exception(f"Trade_args for 'symbol' not found")


def plot_sim(opts, re, logger):
    try:
        plt.figure(1, figsize=(15, 8))

        # RMSE
        price_sell_tracer_diff = re['prices_mu'] - re['sell_indi'].tracer
        sell_buy_tracer_diff = re['sell_indi'].tracer - re['buy_indi'].tracer

        dt_index = re['obv'].dt_index

        ax1 = plt.subplot(711)
        plt.plot(re['prices_actual'], label="price actual", lw=1)

        ax1.set_ylabel('price')
        plt.grid()
        plt.legend(loc="best")

        plt.title('{0} TradingStrategy {1}, Gain:{2:.2f}%'
                  ''.format(opts.strategy, opts.symbol, re['gain']))

        ax2 = plt.subplot(712, sharex=ax1)
        # plt.step(signal_pd.index, signal_pd, label="Signal", lw=1)
        plt.step(dt_index, re['obv'].signal, label="Signal ref", lw=1)

        plt.axhline(0, color='black', ls='-.', lw=1)
        plt.axhline(1, color='green', ls='-.', lw=1)
        plt.axhline(-1, color='green', ls='-.', lw=1)
        ax2.set_ylabel('signal')
        plt.grid()
        plt.legend(loc="upper right")

        ax3 = plt.subplot(713, sharex=ax1)
        plt.step(dt_index, re['obv'].position, label="Position", lw=1)
        plt.axhline(0, color='black', ls='-.', lw=1)
        plt.axhline(1, color='green', ls='-.', lw=1)
        plt.axhline(-1, color='green', ls='-.', lw=1)
        ax3.set_ylabel('position')
        plt.grid()
        plt.legend(loc="upper right")

        ax4 = plt.subplot(714, sharex=ax1)
        plt.step(dt_index, re['obv'].share, label="share", lw=1)
        plt.axhline(0, color='black', ls='-.', lw=1)
        ax4.set_ylabel('share')
        plt.grid()
        plt.legend(loc="best")

        ax5 = plt.subplot(715, sharex=ax1)
        plt.plot(dt_index, re['obv'].cash_asset, label="cash + asset", lw=1)
        ax5.set_ylabel('cash+asset')
        plt.grid()
        plt.legend(loc="upper left")

        ax6 = plt.subplot(716, sharex=ax1)
        plt.plot(re['obv'].dt_index, re['buy_indi'].alfa, label="buy_alfa", lw=1)
        # plt.plot(obv.dt_index, buy_indi['price_pct'], label="price_pct", lw=1)
        is_active = np.where(re['buy_indi'].sd < 0.95, None, re['buy_indi'].sd)
        plt.step(re['obv'].dt_index, re['buy_indi'].sd, label="buy_sd", lw=1)
        plt.step(re['obv'].dt_index, is_active, label="buy_sd active", lw=1)

        plt.axhline(0, color='black', ls='-.', lw=1)
        plt.axhline(opts.long_level, color='green', ls='-.', lw=1)
        plt.axhline(opts.short_level, color='green', ls='-.', lw=1)

        ax6.set_ylabel('buy_tracer')
        plt.grid()
        plt.legend(loc="upper right")

        ax7 = plt.subplot(717, sharex=ax1)
        plt.plot(dt_index, re['sell_indi'].alfa, label="sell_alfa", lw=1)
        plt.axhline(0, color='black', ls='-.', lw=1)
        plt.axhline(opts.long_exit_level, color='green', ls='-.', lw=1)
        plt.axhline(opts.short_exit_level, color='green', ls='-.', lw=1)
        ax7.set_ylabel('sell_tracer')
        plt.grid()
        plt.legend(loc="upper right")

        # ===== Figure 2 ========
        plt.figure(2, figsize=(15, 8))

        ax21 = plt.subplot(611)
        plt.plot(re['prices_actual'], label="price actual")
        plt.plot(dt_index, re['buy_simple'].tracer, label="BuySimple, delta:{0}".format(re['buy_simple'].delta),
                 ls='-.', lw=1)
        plt.plot(dt_index, re['sell_simple'].tracer, label="SellSimple, delta:{0}".format(re['sell_simple'].delta),
                 ls='--', lw=1)

        ax21.set_ylabel('price')
        plt.grid()
        plt.legend(loc="best")
        plt.title('{0} TradingStrategy {1} Train'.format(opts.strategy, opts.symbol))

        dt_index = re['obv'].dt_index
        ax22 = plt.subplot(612, sharex=ax21)

        plt.plot(dt_index, re['sell_indi'].alfa, label="sell_alfa (price_pct - its' tracer)", lw=1)
        plt.plot(dt_index, re['buy_indi'].alfa, ls='-.', label="buy_alfa (price_pct - its' tracer)", lw=1)

        plt.axhline(0, color='black', ls='-.', lw=1)
        plt.axhline(opts.long_level, color='green', ls='-.', lw=1)
        plt.axhline(opts.short_level, color='green', ls='-.', lw=1)
        ax22.set_ylabel('buy sell alfa')
        plt.grid()
        plt.legend(loc="upper right")

        ax23 = plt.subplot(613, sharex=ax21)
        plt.plot(dt_index, re['buy_indi'].tracer - re['sell_indi'].tracer, label="buy tracer - sell tracer", lw=1)

        plt.axhline(0, color='black', ls='-.', lw=1)
        plt.axhline(opts.long_level, color='green', ls='-.', lw=1)
        plt.axhline(opts.short_level, color='green', ls='-.', lw=1)
        plt.grid()
        plt.legend(loc="upper right")

        ax24 = plt.subplot(614, sharex=ax21)
        ax24.set_ylabel('buy_delta')
        plt.plot(dt_index, re['buy_indi'].alfa - re['sell_indi'].alfa, label="buy alfa - sell alfa", lw=1)

        plt.axhline(0, color='black', ls='-.', lw=1)
        plt.grid()
        plt.legend(loc="upper right")

        ax25 = plt.subplot(615, sharex=ax21)
        ax25.set_ylabel('buy_delta')
        plt.plot(dt_index, re['obv'].buy_delta, label="buy_delta", lw=1)

        plt.axhline(0, color='black', ls='-.', lw=1)
        plt.grid()
        plt.legend(loc="upper right")

        ax26 = plt.subplot(616, sharex=ax21)
        ax26.set_ylabel('sell_delta')
        plt.plot(dt_index, re['obv'].sell_delta, label="sell_delta", lw=1)

        plt.axhline(0, color='black', ls='-.', lw=1)

        plt.grid()
        plt.legend(loc="upper right")

        plt.show()

    except Exception as e:
        err = readable_error(e, __file__)
        logger.error(err)
