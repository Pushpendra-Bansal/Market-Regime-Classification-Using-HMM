# Market-Regime-Classification-Using-HMM
# Market Regime Classification Using HMM

## 📊 Project Overview

This project implements a **Hidden Markov Model (HMM)** based trading strategy that automatically detects market regimes (Bull/Chop/Bear) and dynamically adjusts trading positions for optimal risk-adjusted returns.

The strategy uses a **3-state HMM** with walk-forward validation to ensure zero look-ahead bias, making it robust and realistic for live trading applications.

---

## 🎯 Key Results

| Metric | HMM Strategy | Buy & Hold | Improvement |
|--------|--------------|------------|-------------|
| **Annual Return** | **14.21%** | 11.86% | **+2.35%** |
| **Sharpe Ratio** | **0.96** | 0.69 | **+39%** |
| **Max Drawdown** | **-24.52%** | -35.75% | **31% less loss** |
| **Win Rate** | **55.02%** | 53.00% | **+2%** |

✅ **Strategy beats Buy & Hold by 2.35% annually with 31% less drawdown!**

---

## 🏆 Features

- **3-State HMM** — Detects Bull / Chop / Bear regimes automatically
- **Walk-Forward Validation** — No look-ahead bias, realistic backtesting
- **Volatility Targeting** — Dynamic position sizing based on market volatility
- **Transaction Costs** — Realistic backtesting with 2bps per trade
- **Aggressive Strategy** — Bull: 2.0x leverage, Chop: 1.5x, Bear: 0.6x

---

## 📁 Project Structure
Market-Regime-Classification-Using-HMM/
├── README.md # Project documentation
├── requirements.txt # Python dependencies
├── .gitignore # Excluded files
├── hmm_production.py # Main HMM strategy (Python script)
├── feature_engineering.py # Feature creation script
└── Trading_Startegy.py
==================================================================
  OOS PERFORMANCE REPORT (Walk-Forward, No In-Sample Data)
==================================================================
  Metric                         HMM Strategy      Buy & Hold
------------------------------------------------------------------
  Annualised Return                   14.21%          11.86%
  Annualised Volatility               14.87%          17.21%
  Sharpe Ratio                         0.96            0.69
  Maximum Drawdown                   -24.52%         -35.75%
==================================================================

  REGIME OCCUPANCY
------------------------------------------------------------------
  Bull        :  1866 days (51.1%)
  Chop        :   922 days (25.2%)
  Bear        :   866 days (23.7%)
==================================================================

  CLUSTERING METRICS
------------------------------------------------------------------
  Silhouette Score: +0.1654
  Directional Hit Rate: 54.43%
  Trade Win Rate: 55.02%
==================================================================
