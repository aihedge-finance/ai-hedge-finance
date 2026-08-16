from matplotlib import pyplot as plt
import mplfinance as mpf
import gif
import numpy as np

# plt settings
plt.style.use('seaborn')
gif.options.matplotlib['dpi'] = 300

"""
REFERENCE:
1.  How to use financial analysis visualization module mplfinance
    https://www.grenade.tw/blog/how-to-use-the-python-financial-analysis-visualization-module-mplfinance/
2.  Animation 1
    https://github.com/matplotlib/mplfinance/blob/master/examples/mpf_animation_demo1.py
3.  Animation all and MACS
    https://github.com/matplotlib/mplfinance/tree/master/examples
    
    https://github.com/matplotlib/mplfinance/blob/master/examples/mpf_animation_macd.py
    
"""


class Tech_Plot:
    def __init__(self, fname):

        self.fig = None
        self.axes = None
        self.fname = fname

        self.dt_idx = None
        self.tech_col = None
        self.tech_ary = None
        self.price_ary = None
        self.market_ary = None
        self.market_cols = None

        # n = price_ary.shape[0]
        # self.kelly_cap = np.empty(n)
        # self.kelly_cap[:] = np.nan

        self.apds = []

    def reset(self):
        self.apds = []
        self.fig, self.axes = None, None
        self.dt_idx = None
        self.tech_col = None
        self.tech_ary = None
        self.price_ary = None
        self.market_ary = None
        self.market_cols = None

    def plot(self, price_ary, tech_ary, tech_col, market_ary, market_cols, show=False, save=False):
        for i, v, in enumerate(tech_col):
            sub_plt = mpf.make_addplot(tech_ary[:, i], panel=i + 1)
            self.apds.append(sub_plt)

        j = len(tech_col)
        for i, v, in enumerate(market_cols):
            sub_plt = mpf.make_addplot(market_ary[:, i], panel=i + j + 1)
            self.apds.append(sub_plt)

        s = mpf.make_mpf_style(base_mpf_style='yahoo', rc={'figure.facecolor': 'lightgray'})

        self.fig, self.axes = mpf.plot(price_ary, type='line', addplot=self.apds, figscale=1.5,
                                       figratio=(7, 5), title='Tech Indicators',
                                       style=s, panel_ratios=(6, 3, 2), returnfig=True)

        if save:
            dpi = 300
            saving_params = dict(fname=self.fname, dpi=dpi, figsize=(324 / dpi, 252 / dpi),
                                 pad_inches=0.1)
            self.fig.savefig(self.fname, **saving_params)

        if show:
            self.fig.show()


class Trade_Plot:
    def __init__(self):
        pass

    def update(self):
        pass

    def plot(self):
        pass

    def save(self):
        pass

    def to_gif(self):
        pass