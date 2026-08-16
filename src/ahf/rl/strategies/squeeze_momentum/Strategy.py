"""
@version=4
@author echepata
@credits LazyBear for the Squeeze Momentum indicator.
@credits see LazyBear's other indicators here https://www.tradingview.com/v/4IneGo8h/
"""

import talib
import numpy as np

# strategy("Crypto momentum strategy", overlay=false, pyramiding=0, default_qty_value=100, default_qty_type= strategy.percent_of_equity, precision=7, currency=currency.AUD, commission_value=0.1, commission_type=strategy.commission.percent, initial_capital=initialBalance)


class OPTS:
    def __init__(self):
        self.length = 20  # BB Length
        self.mult = 2.0  # BB MultFactor
        self.len_KC = 20  # KC Length
        self.mult_KC = 1.5  # KC MultFactor
        self.ema_periods = 50  # EMA periods

        self.use_true_range = True  # Use TrueRange (KC)
        self.consider_market = True  # Consider crypto market behavior


def tr(high, low, last_close):
    a = max(high - low, abs(high - last_close), abs(low - last_close))
    return a


def highest(data, n):
    return np.max(data[-n:])


def lowest(data, n):
    return np.min(data[-n:])


def avg(x, y):
    return (x + y) / 2


def cross_over(series, value):
    return True if series[0] <= value <= series[1] else False


def cross_under(series, value):
    return True if series[0] >= value >= series[1] else False


class SqueezeMomentum_v4:
    def __init__(self, opts):

        self.length = opts.length
        self.mult = opts.mult
        self.len_KC = opts.len_KC
        self.mult_KC = opts.mult_KC
        self.ema_periods = opts.ema_periods

        self.use_true_range = opts.use_true_range
        self.consider_market = opts.consider_market

        self.time_frame = opts.time_frame

        self.market_ema = np.zeros(10000)
        self.market_ema_slope = np.zeros(10000)
        self.val = np.zeros(10000)
        self.ema = np.zeros(10000)

        self.idx = 0

    def step(self, market, price):
        """

        :param market: close price of entire market [market = security(marketTicker, timeframe.period, close, true)]
        :param price: price for individual asset
        :return:
        """
        idx = self.idx

        self.market_ema[idx] = talib.EMA(market, self.ema_periods)
        self.market_ema_slope[idx] = self.market_ema[idx] - self.market_ema[idx - 1]

        # Calculate BB

        source = close = price['close']
        high = price['high']
        low = price['low']
        basis = talib.SMA(source, self.length)
        self.ema[idx] = talib.EMA(source, self.ema_periods)
        dev = self.mult_KC * np.std(source, self.length)
        upperBB = basis + dev
        lowerBB = basis - dev

        # Calculate KC
        ma = talib.SMA(source, self.len_KC)
        range_p = tr if self.use_true_range else high - low
        range_ma = talib.SMA(range_p, self.len_KC)
        upperKC = ma + range_ma * self.mult_KC
        lowerKC = ma - range_ma * self.mult_KC

        sqzOn = lowerBB > lowerKC and upperBB < upperKC
        sqzOff = lowerBB < lowerKC and upperBB > upperKC
        noSqz = not sqzOn and not sqzOff  # sqzOn == False and sqzOff == False

        self.val[idx] = talib.LINEARREG(source - avg(avg(highest(high, self.len_KC), lowest(low, self.len_KC)),
                                                     talib.SMA(close, self.len_KC)), self.len_KC, 0)

        slope = (self.val[idx] - self.val[idx - 2])
        ema_slope = (self.ema[idx] - self.ema[idx - 1])

        # bcolor = iff(slope > 0, color.lime, color.red)
        # scolor = noSqz ? color.green : sqzOn ? color.black : color.green
        squeeze = 0 if noSqz else 1 if sqzOn else 0

        co = cross_over(slope / abs(slope), 0)
        cu = cross_under(slope / abs(slope), 0)

        if co and source > self.ema[idx] and ema_slope > 0 and \
                (not self.consider_market or (market > self.market_ema[idx] and self.market_ema_slope[idx] > 0)):
            # strategy.entry("long", strategy.long, comment="long")
            return 1
        if cu:
            # strategy.close("long")
            return -1

    def plot(self):
        # plot(val, color=color.gray, style=plot.style_line, linewidth=1, title="momentum")
        # plot(slope, color=bcolor, style=plot.style_circles, linewidth=2, title="slope")
        # plot(0, color=scolor, style=plot.style_line, linewidth=2, title="squeeze-zero")
        pass
