# Strategy Analysis & Improvement TODOs
*Session Date: 2026-07-25 | Author: Analysis Session*

---

## 1. Backtest Results Summary

### BTCUSDT — `double_kf` / PPO / Dual / 8h interval
- **Out-of-sample period**: 2025-01-01 onwards
- **Peak P&L**: +14.5% (reached May 2, 2025, step 363)
- **Termination**: August 9, 2025 (step 661) at **cum_pnl: -10.517%**
- **Cause**: `done_total_loss = -0.1` safety limit triggered

### ETHUSDT — `double_kf` / PPO / Dual / 8h interval
- **Out-of-sample period**: 2025-01-01 onwards
- **Termination**: February 5, 2025 (step 105) at **cum_pnl: -10.527%**
- **Cause**: Same safety limit triggered early
- **Note**: ETHUSDT deteriorated much faster, suggesting worse generalization

### RSI_MACD
- **Status**: No trained model weights available in `Tuner/LeaderBoard/PROD/`
- **Code**: Strategy code fully implemented in `TradingStrategy/RSI_MACD/Strategy.py`
- **Conclusion**: Cannot backtest without first training a model

---

## 2. Root Cause Analysis — Why `double_kf` Fails Out-of-Sample

### 2.1 Chasing Spikes (Entry Problem)
- **Problem**: Strategy enters Long when `buy_alfa` (Kalman return residual) crosses **above** `long_level` — i.e., it buys immediately after a sharp positive return spike.
- **Effect**: In range-bound/choppy markets (2025 ETH/BTC), a spike is frequently followed by immediate mean-reversion. The agent constantly buys the local top.

### 2.2 Asymmetric Exit Logic (Exit Problem)
- **Problem**: Under `sell_tracer` rule, exit only triggers on a further positive spike (`sell_alfa >= long_exit_level`). If the trade goes negative, no dynamic exit signal fires — the agent holds until the hard stop-loss (`-7%` paper PnL threshold).
- **Effect**: Small, frequent profits, but catastrophic hold-to-stop-loss losses → negative risk/reward expectancy.

### 2.3 Regime Shift (Training Data Problem)
- **Trained on**: 2018–2023 (sustained trends: 2021 bull run, 2022 bear market).
- **Tested on**: 2025–2026 (sideways consolidation, high-frequency fakeouts, whipsaws).
- **Effect**: Momentum-optimized parameters are systematically chopped up in range-bound markets.

---

## 3. TODO — Improvement Recommendations

### HIGH PRIORITY — Fix `double_kf` Exit Logic
- [ ] **Add Kalman State Cross Exit**: Exit a Long when price closes **below the Kalman Filter tracer mean** (not waiting for the next positive spike).
  - *File*: `TradingStrategy/double_kf/Strategy.py` → `cal_signal_run()`
  - *Signal logic*: Add `in_position == 1 and price_list[1] < buy_tracer_list[1]` → `signal = 2`
- [ ] **Add Time-Based Timeout**: Auto-exit if a position has not reached profit target within N bars (e.g., 5 × 8h = 40 hours).
  - *Rationale*: Crypto momentum half-life is short; stale positions bleed fees.
- [ ] **Add Trailing Stop-Loss**: Move per-trade stop-loss up dynamically as paper P&L increases (e.g., protect 50% of peak gain).

### HIGH PRIORITY — Fix `double_kf` Entry Logic
- [ ] **Multi-Timeframe Trend Filter**: Only allow BUY signal if the 1-day (24h) Kalman trend is also positive. Prevents entering longs during macro downtrends.
  - *Implementation*: Add a second, slower KF tracer with 24h lookback; gate BUY signal on its sign.
- [ ] **Volume Confirmation**: Filter out low-volume fakeout spikes. Only enter if 8h volume is above its 20-period moving average.

### MEDIUM PRIORITY — Regime-Aware Retraining
- [ ] **Retrain `double_kf` models on 2024–2026 data** to adapt to range-bound market conditions.
  - *Caveat*: Temporary patch only — models will likely overfit to sideways regime and underperform when strong trends return.
  - *Recommendation*: Retrain AND implement exit/entry fixes together.
- [ ] **Retrain with risk-adjusted reward function**: Optimize PPO reward for Sortino ratio or risk-adjusted returns (subtract drawdown penalty), not just raw cumulative P&L.

### MEDIUM PRIORITY — Train RSI_MACD Strategy
- [ ] **Train `RSI_MACD` for BTCUSDT and ETHUSDT** using the existing Hyperband tuner (`Hyperband/RSI_MACD/`).
  - *Config available*: `tech_args/rsi_macd_deploy/BTCUSDT.json`, `tech_args/rsi_macd_deploy/ETHUSDT.json`
  - *Why RSI_MACD is better*: Combines trend-following (MACD) + mean-reversion oscillator (RSI) + trend baseline (EMA). Naturally adapts across trending vs. ranging regimes.
- [ ] **Ensure dual (Long + Short) training** for RSI_MACD so the model learns both BUY and SHORT entry conditions (see Section 4 for signal spec).

### LOW PRIORITY — New Strategy: ETH/BTC Pair Trading
- [ ] **Investigate applying `double_kf` to the ETH/BTC spread** (not raw price).
  - *Rationale*: Spreads between co-integrated assets are structurally mean-reverting. KF tracking a spread is far more predictable than tracking raw price returns.
  - *Config reference*: `trade_args/ETHBTC_LONG.json` exists — extend to dual long/short.
  - *Benefit*: Largely insulated from broader crypto trend regime changes.

---

## 4. RSI_MACD Dual Strategy — Signal Specification

### Long Entry (BUY)
| Condition | Requirement |
| :--- | :--- |
| Macro trend | `price > EMA` (bullish baseline) |
| Momentum | MACD bar crosses **above** `macd_bar_long_level` |
| Oscillator | RSI crosses **above** `rsi_long_level` |

### Long Exit (SELL)
| Rule | Condition |
| :--- | :--- |
| Take Profit | RSI falls from recent peak: `max_rsi * 0.9 >= rsi[1]` |
| Stop Loss | RSI crosses below `rsi_long_loss_level` |

### Short Entry (SHORT)
| Condition | Requirement |
| :--- | :--- |
| Macro trend | `price < EMA` (bearish baseline) |
| Momentum | MACD bar crosses **below** `macd_bar_short_level` |
| Oscillator | RSI crosses **below** `rsi_short_level` (e.g., <= 40) |

### Short Exit (COVER)
| Rule | Condition |
| :--- | :--- |
| Take Profit | RSI rebounds from recent trough: `max_rsi * 1.1 <= rsi[1]` |
| Stop Loss | RSI crosses above `rsi_short_loss_level` (e.g., >= 70) |

*All conditions are coded in `TradingStrategy/RSI_MACD/Strategy.py` under `entry_rule="ema_macd_rsi"` and `exit_rule="ema_macd_rsi"` / `"macd_cross"` / `"ind_cross"` / `"trailing_rsi"`.*

---

## 5. Bug Fixes Applied in This Session

| File | Fix |
| :--- | :--- |
| `preprocessor/kf/Price_Alfa_Processor.py` | Changed `@deprecated(..., action="error")` to `action="ignore"` |
| `preprocessor/kf/TracerMem_v2.py` | Removed invalid `is_save=False` argument from `self.update(v)` call |
| `Gen_Indicator.py` | Changed subprocess `python` to `sys.executable` to respect virtual environment |
| `TradingStrategy/double_kf/read_write.py` | Added `tech_pd = tech_pd[~tech_pd.index.duplicated(keep='last')]` |
| `TradingStrategy/double_ukf/read_write.py` | Same duplicate-dropping fix |
| `TradingStrategy/RSI_MACD/read_write.py` | Same duplicate-dropping fix |
| `preprocessor/helpers.py` | Removed unsupported `truncated=False, n_look_back=0` args; added `price_pd.compute()` guard |
| `preprocessor/ta/MACD.py` | `pyximport.install()` → `pyximport.install(setup_args={'include_dirs': np.get_include()})` |
| `preprocessor/ta/RSI.py` | Same Cython NumPy include fix |

---

## 6. Backtest Command Reference

### BTCUSDT
```bash
./venv/bin/python ./RL_Backtest.py \
  --job_id 20240206_TRAIN_LONG_Power_2022Q12 \
  --pod_dir "Tuner/LeaderBoard/PROD/symbl-BTCUSDT|env-BrunhildEnv-v11|techid-double_kf|ls-dual|agt-PPO|gpu-GPU|jobid-20240206_TRAIN_LONG_Power_2022Q12/pod_000001" \
  --form_start 2025-01-01 \
  --env_name BrunhildEnv-v11 \
  --train_mode PROD
```

### ETHUSDT
```bash
./venv/bin/python ./RL_Backtest.py \
  --job_id 20240206_TRAIN_LONG_Power_2022Q12 \
  --pod_dir "Tuner/LeaderBoard/PROD/symbl-ETHUSDT|env-BrunhildEnv-v11|techid-double_kf|ls-dual|agt-PPO|gpu-GPU|jobid-20240206_TRAIN_LONG_Power_2022Q12/pod_000001" \
  --form_start 2025-01-01 \
  --env_name BrunhildEnv-v11 \
  --tech_data_path ./appData/trainData_crypto/brunhild_eth_v1.parquet \
  --train_mode PROD
```

> NOTE: Always pass `--train_mode PROD` to skip the `check_env()` sanity check that fails without a live `price_fetcher`.
