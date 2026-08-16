from collections import namedtuple


PlotSimParam = namedtuple("PlotSimParam",
                         ["ds",
                          "strategy",
                          "tech_args",
                          "trade_args",
                          "plot_args",
                          "gain_pct"])