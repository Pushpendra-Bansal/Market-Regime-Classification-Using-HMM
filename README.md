# 📈 HMM Market Regime Detection

> **IITI SOC · Finalytics Advanced PS**  
> A production-grade Hidden Markov Model pipeline for detecting market regimes (Bull / Chop / Bear) in SPY, with walk-forward backtesting, strategy comparison, and forward-looking next-day regime prediction.

---

## 🧠 What This Project Does

This project uses a **3-state Gaussian Hidden Markov Model (HMM)** trained on SPY (S&P 500 ETF) data from 2005–2024 to:

- Classify each trading day as **Bull**, **Chop**, or **Bear** regime
- Run a **walk-forward out-of-sample backtest** (no lookahead bias)
- Apply **regime-based position sizing** to outperform buy-and-hold
- Compare multiple strategies across return, Sharpe, drawdown, and Calmar
- Provide a **forward-looking next-day regime prediction** using transition matrix math

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
├── feature_engineering.py      # Downloads SPY data, computes technical features, saves CSVs
├── hmm_production.py           # HMM walk-forward training, position sizing, backtest
├── forward_engine.py           # Forward-looking prediction engine (π_{t+1} = π_t @ A)
├── compare_strategies.py       # Strategy comparison, metrics table, matplotlib charts
│
├── features_improved.csv       # Recommended feature set (6 features) — output of step 1
├── features_original.csv       # Baseline feature set (4 features) — output of step 1
├── all_features.csv            # Full feature set — output of step 1
├── price_data.csv              # SPY daily OHLCV data
├── scaler.pkl                  # Saved RobustScaler instance
│
├── hmm_oos_results.csv         # OOS regime predictions + backtest returns — output of step 2
├── forward_regime_results.csv  # Forward-looking signals + 3-method comparison — output of step 3
├── best_strategies_results.csv # Strategy metrics comparison table — output of step 4
│
└── requirements.txt
```

---

## ⚙️ Pipeline Overview

```
Raw OHLCV Data (yfinance)
        │
        ▼
1. Feature Engineering          ← feature_engineering.py
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
3. Forward Prediction Engine    ← forward_engine.py
   π_{t+1} = π_t @ A
   Next-day regime probabilities,
   bear warning signal, stability score
        │
        ▼
4. Strategy Comparison          ← compare_strategies.py
   5 strategies, matplotlib charts,
   metrics: return, Sharpe, Calmar, drawdown
```

---

## 🚀 Quick Start

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
```

Install all at once:
```bash
pip install numpy pandas scikit-learn hmmlearn yfinance joblib pandas_ta matplotlib
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

## 📄 License

MIT License — free to use, modify, and distribute with attribution.

---

## 👥 Authors

**IIT Indore — Financial Machine Learning Research**  
Walk-Forward HMM Market Regime Classification · 2024–2025
