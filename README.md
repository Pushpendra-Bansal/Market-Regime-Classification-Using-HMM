# 📈 HMM Market Regime Detection

> **IITI SOC · Finalytics Advanced PS**  
> A production-grade Hidden Markov Model pipeline for detecting market regimes (Bull / Chop / Bear) in SPY, with walk-forward backtesting, strategy comparison, and forward-looking next-day regime prediction.

**🌐 Live Demo:** [website-mewfdzpk2julnnsvqdzmnj.streamlit.app](https://website-mewfdzpk2julnnsvqdzmnj.streamlit.app/)
(if it doesnt work due to inactivity then run it locally from "Interactive Dashboard Web App" present in our repository)
---

## 🧠 What This Project Does

This project uses a **3-state Gaussian Hidden Markov Model (HMM)** trained on SPY (S&P 500 ETF) data from 2005–2024 to:

- Classify each trading day as **Bull**, **Chop**, or **Bear** regime
- Run a **walk-forward out-of-sample backtest** (no lookahead bias)
- Apply **regime-based position sizing** to outperform buy-and-hold
- Compare multiple strategies across return, Sharpe, drawdown, and Calmar
- Provide a **forward-looking next-day regime prediction** using transition matrix math

👉 Try the interactive version here: **[Streamlit App](https://website-mewfdzpk2julnnsvqdzmnj.streamlit.app/)**

---

## 🏆 Key Results (OOS Walk-Forward Backtest)

| Strategy | Total Return | Sharpe | Max Drawdown |
|---|---|---|---|
| **Aggressive** (2.3x / 1.5x / 0.5x) | ~2149% | ~1.05 | ~-29% |
| **Balanced** (1.0x / 0.7x / 0.3x) | ~401% | ~0.90 | ~-18% |
| **Conservative** (1.0x / 0.3x / 0.0x) | ~180% | ~0.85 | ~-11% |
| Buy & Hold SPY | ~638% | ~0.73 | ~-34% |

> All results are purely out-of-sample. No in-sample data is used in performance evaluation.

---

## 🗂️ Project Structure

```
├── feature_engineering_and_scaling.py      # Downloads SPY data, computes technical features, saves CSVs
├── hmm_production.py           # HMM walk-forward training, position sizing, backtest
├── regime_prediction_engine.py           # Forward-looking prediction engine (π_{t+1} = π_t @ A)
├── trading_strategies.py       # Strategy comparison, metrics table, matplotlib charts
│
├── features_improved.csv       # Recommended feature set (6 features) — output of step 1
├── features_original.csv       # Baseline feature set (4 features) — output of step 1
├── all_features.csv            # Full feature set — output of step 1
├── price_data.csv              # SPY daily OHLCV data
├── scaler.pkl                  # Saved RobustScaler instance
│
├── hmm_oos_results.csv         # OOS regime predictions + backtest returns — output of step 2
├── forward_regime_results.csv  # Forward-looking signals + 3-method comparison — output of step 3
├── best_strategies_results.csv             # Strategy metrics comparison table — output of step 4
├── hmm_oos_results_with_anomalies.csv      # OOS results with anomaly flags — output of anomaly_detection.py
├── fold_diagnostics_<TICKER>.csv           # Per-fold emission means & self-transition probs
├── performance_summary_<TICKER>.json       # Full performance report in JSON format
├── feature_clean_<TICKER>.csv             # Raw unscaled features per asset
├── price_data_<TICKER>.csv               # OHLCV data per asset
├── anomaly_chart.png                      # Two-panel anomaly visualization
│
└── requirements.txt
```

---

## ⚙️ Pipeline Overview

```
Raw OHLCV Data (yfinance)
        │
        ▼
1. Feature Engineering          ← feature_engineering_and_scaling.py
   log_return, parkinson_vol,
   rsi, ma_cross, momentum,
   volume_zscore
        │
        ▼
2. Walk-Forward HMM Training    ← hmm_production.py
   Train window : 1260 days
   Step size    : 63 days
   States       : 3 (Bull / Chop / Bear)
   Position sizing + backtest
        │
        ▼
3. Forward Prediction Engine    ← regime_prediction_engine.py
   π_{t+1} = π_t @ A
   Next-day regime probabilities,
   bear warning signal, stability score
        │
        ▼
4. Strategy Comparison          ← trading_strategies.py
   5 strategies, matplotlib charts,
   metrics: return, Sharpe, Calmar, drawdown
```

---

## 🚀 Quick Start

### 0. Try it online first (no setup required)

Explore the model interactively without installing anything:

**🌐 [https://website-mewfdzpk2julnnsvqdzmnj.streamlit.app/](https://website-mewfdzpk2julnnsvqdzmnj.streamlit.app/)**

### 1. Clone the repo
```bash
git clone https://github.com/<your-username>/hmm-regime-detection.git
cd hmm-regime-detection
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add required data files

Before running, place the following file in the project root directory:

- **`price_data.csv`** — SPY daily OHLCV data (Date, Open, High, Low, Close, Volume).  
  Download from [Yahoo Finance](https://finance.yahoo.com/quote/SPY/history/) or export via yfinance:
  ```python
  import yfinance as yf
  df = yf.download("SPY", start="2005-01-01", end="2024-12-31")
  df.to_csv("price_data.csv")
  ```

> `price_data.csv` is required by `hmm_production.py`, `regime_prediction_engine.py`, and `trading_strategies.py`.  
> `feature_engineering_and_scaling.py` downloads data automatically via yfinance and does not need this file.

### 4. Run the full pipeline
```bash
# Step 1: Download SPY data and compute features
python feature_engineering_and_scaling.py

# Step 2: Train HMM walk-forward and run backtest
python hmm_production.py

# Step 3: Run forward-looking prediction engine
python regime_prediction_engine.py

# Step 4: Compare all strategies and generate charts
python trading_strategies.py

# Step 5: Run anomaly detection on OOS results
# Requires hmm_oos_results_SPY.csv in the working directory
python anomaly_detection.py

# Step 6: Run cross-asset pipeline (SPY, GLD, TLT, BTC-USD)
python Code_for_Different_Regimes.py SPY
python Code_for_Different_Regimes.py GLD
python Code_for_Different_Regimes.py TLT
python Code_for_Different_Regimes.py BTC-USD 2014-09-17
```

### 5. Run the Streamlit app locally (optional)

If the repo includes a Streamlit entry point (e.g. `app.py` / `streamlit_app.py`):
```bash
streamlit run app.py
```

---

## 📦 Requirements

```
numpy
pandas
scikit-learn
hmmlearn
yfinance
joblib
pandas_ta
matplotlib
scipy
statsmodels
streamlit
```

Install all at once:
```bash
pip install numpy pandas scikit-learn hmmlearn yfinance joblib pandas_ta matplotlib scipy statsmodels streamlit
```

---

## 🔬 Model Details

### HMM Configuration
| Parameter | Value |
|---|---|
| States | 3 (Bull, Chop, Bear) |
| Emission type | Gaussian (diagonal covariance) |
| Training iterations | 200 |
| Random seed | 42 |

### Features Used
| Feature | Description |
|---|---|
| `log_return` | Daily log return of closing price |
| `volume_zscore` | Volume deviation from 20-day rolling mean |
| `momentum` | 5-day rolling mean of log returns |
| `parkinson_vol` | High-Low volatility estimator (20-day) |
| `rsi` | Relative Strength Index (14-day) |
| `ma_cross` | % difference between 20-day and 50-day MA |

### Walk-Forward Methodology
- **Expanding window** — training set grows by one quarter each step
- **No lookahead** — OOS predictions use only past data at each step
- **Winsorization** — 1st/99th percentile clipping before scaling
- **StandardScaler** — fitted on train window only, applied to OOS

### Regime Labelling
States are sorted by mean log return:
- **Lowest mean return** → Bear
- **Middle mean return** → Chop
- **Highest mean return** → Bull

### Position Sizing Strategies
| Strategy | Bull | Chop | Bear |
|---|---|---|---|
| Aggressive | 2.3x | 1.5x | 0.5x |
| Balanced | 1.0x | 0.7x | 0.3x |
| Conservative | 1.0x | 0.3x | 0.0x |
| Advanced Max | 2.5x | 2.0x | 1.0x |
| Advanced Safest | 1.0x | 0.5x | 0.2x |

Transaction cost: **2 bps per trade**

---

## 🔮 Forward-Looking Engine

Extends the standard HMM with **transition probability forecasting**:

```
π_{t+1} = π_t @ A
```

Where:
- `π_t` = today's posterior state distribution from `predict_proba()`
- `A` = HMM learned transition matrix
- `π_{t+1}` = tomorrow's predicted regime probability distribution

This produces:
- **Forward-weighted position** — blend of caps weighted by predicted regime probs
- **Bear warning flag** — raised when P(Bear tomorrow) > 30%
- **Regime stability score** — probability of staying in current regime
- **5-day rolling forecast** — dominant regime for each of next 5 days

### Method Comparison (from `regime_prediction_engine.py`)
| Method | Position Basis |
|---|---|
| A. Hard-Cap | Fixed cap per today's regime label |
| B. Soft Today | Prob-weighted using today's posterior |
| C. Forward (new) | Prob-weighted using tomorrow's forecast |

---

## 🚨 Anomaly Detection (`anomaly_detection.py`)

Analyzes OOS HMM posterior probabilities to flag high-uncertainty days where the model is not confident in its regime classification.

**Three posterior metrics evaluated:**

| Metric | Threshold | Meaning |
|---|---|---|
| `posterior_top_prob` | < 0.70 | Top state probability too low |
| `posterior_margin` | < 0.50 | Gap between top two states too small |
| `posterior_entropy` | > 0.50 | Too much uncertainty across all states |

**Anomaly score** (0–1, higher = more anomalous):
```
score = (1 - top_prob) × 0.4 + (1 - margin) × 0.3 + (entropy / max_entropy) × 0.3
```

**Outputs:**
- `hmm_oos_results_with_anomalies.csv` — original OOS data with `is_anomaly` and `anomaly_score` columns
- `anomaly_chart.png` — two-panel chart: anomaly flags overlaid on cumulative returns + anomaly score over time

**Required input:** `hmm_oos_results_SPY.csv` must contain columns `posterior_top_prob`, `posterior_margin`, `posterior_entropy`, `regime`. These are produced by `Code_for_Different_Regimes.py` (not the original `hmm_production.py`).

> ⚠️ **IMPORTANT:** Do NOT use the `hmm_oos_results.csv` generated by `hmm_production.py` — it does not contain the posterior probability columns required by this script. You must use the **pre-computed CSV provided in the `Anomaly Detection` folder of this repo**. Place that file in the same directory as `anomaly_detection.py` before running.

---

## 🌐 Cross-Asset Pipeline (`Code_for_Different_Regimes.py`)

A fixed, production-grade version of the HMM pipeline that runs on any asset ticker. Key improvements over `hmm_production.py`:

**Bug fixes & improvements:**

| Issue | Fix |
|---|---|
| `model.predict()` has lookahead leakage (Viterbi decodes full block jointly) | Replaced with causal forward filter `π_{t+1} = π_t @ A` — day t can only see data up to t |
| Single EM restart — unstable | `N_INIT=5` restarts per fold, keeps best log-likelihood |
| Convergence silently swallowed | Real convergence check — logs warnings per fold |
| Hardcoded K=3 | Generalized to any N_STATES with `generate_regime_labels()` |
| No model selection | `run_model_order_selection()` sweeps K∈{2,3,4} with BIC/AIC + holdout log-likelihood |
| No per-fold diagnostics | Saves emission means, variances, self-transition probs per fold to CSV |
| BTC annualization wrong | Auto-detects BTC and uses 365 trading days instead of 252 |

**Usage:**
```bash
python Code_for_Different_Regimes.py SPY
python Code_for_Different_Regimes.py GLD
python Code_for_Different_Regimes.py TLT
python Code_for_Different_Regimes.py BTC-USD 2014-09-17
```

**Outputs per asset (suffixed by ticker):**
- `hmm_oos_results_<TICKER>.csv` — OOS regime predictions with posterior metrics
- `fold_diagnostics_<TICKER>.csv` — per-fold emission means, variances, self-transition probabilities
- `performance_summary_<TICKER>.json` — full performance metrics in JSON
- `feature_clean_<TICKER>.csv` — raw unscaled features
- `price_data_<TICKER>.csv` — downloaded OHLCV data

---

## 📄 License

MIT License — free to use, modify, and distribute with attribution.

---

## 👥 Authors

**IIT Indore — Financial Machine Learning Research**  
Walk-Forward HMM Market Regime Classification · 2024–2025
