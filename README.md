# Market-Regime-Classification-Using-HMM
# Market Regime Classification Using HMM

## 📊 Project Overview

This project implements a **Hidden Markov Model (HMM)** based trading strategy that automatically detects market regimes (Bull/Chop/Bear) and dynamically adjusts trading positions for optimal risk-adjusted returns.

The strategy uses a **3-state HMM** with walk-forward validation to ensure zero look-ahead bias, making it robust and realistic for live trading applications.


##  Features

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


##  Key Results

### Best Strategy: Aggressive (Bull=2.3x, Chop=1.5x, Bear=0.5x)

| Metric | HMM Strategy | Buy & Hold | Improvement |
|--------|--------------|------------|-------------|
| **Total Return** | **2,149.02%** | 637.88% | **+1,511.14%** |
| **Annual Return** | **23.96%** | 14.52% | **+9.44%** |
| **Sharpe Ratio** | **102.12** | 85.00 | **+20%** |
| **Max Drawdown** | **-27.28%** | -33.72% | **6% less loss** |
| **Calmar Ratio** | **78.76** | 18.92 | **+316%** |

✅ **Strategy beats Buy & Hold by 1,511% total return with 16% less drawdown!**

---

## 📊 Strategy Comparison Table

| Strategy | Bull | Chop | Bear | Return | Annual Return | Sharpe | Max DD | Calmar |
|----------|------|------|------|--------|---------------|--------|--------|--------|
| **Aggressive** 🏆 | 2.3 | 1.5 | 0.5 | **2,149.02%** | 23.96% | 102.12 | -27.28% | **78.76** |
| **Advanced Max** | 2.5 | 2.0 | 1.0 | 1,369.93% | 20.37% | 94.73 | -27.51% | 49.79 |
| **Balanced** | 1.0 | 0.7 | 0.3 | 352.98% | 10.98% | 101.70 | -15.08% | 23.41 |
| **Advanced Safest** | 1.0 | 0.5 | 0.2 | 221.71% | 8.39% | 102.75 | -11.03% | 20.10 |
| **Conservative** | 1.0 | 0.3 | 0.0 | 199.74% | 7.87% | 99.69 | -10.95% | 18.24 |
| **Buy & Hold** | - | - | - | 637.88% | 14.52% | 85.00 | -33.72% | 18.92 |

---

## 📈 Performance Visualization

![Strategy Performance](results/strategy_performance.png)


### Strategy Parameters

| Regime | Aggressive | Balanced | Conservative | Advanced Max | Advanced Safest |
|--------|------------|----------|--------------|--------------|-----------------|
| **Bull** | 2.3x | 1.0x | 1.0x | 2.5x | 1.0x |
| **Chop** | 1.5x | 0.7x | 0.3x | 2.0x | 0.5x |
| **Bear** | 0.5x | 0.3x | 0.0x | 1.0x | 0.2x |


