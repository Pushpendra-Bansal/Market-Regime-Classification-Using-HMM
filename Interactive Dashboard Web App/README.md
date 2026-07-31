# 📈 HMM Regime Trading Strategy — Interactive Dashboard

An interactive Streamlit dashboard for exploring, tuning, and comparing regime-based trading strategies built on top of Hidden Markov Model (HMM) regime detection (Bull / Chop / Bear).

The app lets you:
- Visualize price history with detected regimes overlaid
- Backtest 5 preset strategies (Aggressive, Balanced, Conservative, Advanced Max, Advanced Safest) or build your own custom exposure profile
- Inspect cumulative returns, drawdowns, position sizing, and regime transition probabilities
- Compare all strategies side-by-side on key performance metrics

---

## ✨ Features

| Tab | Description |
|---|---|
| 📈 Price & Regimes | Price chart shaded by detected regime, plus a regime distribution pie chart |
| 💰 Performance | Cumulative strategy vs. buy & hold returns, and position sizing over time |
| 📉 Drawdown | Strategy vs. buy & hold drawdown comparison |
| 🔁 Transition Matrix | Heatmap of regime-to-regime transition probabilities |
| 📋 Compare All | Side-by-side metrics table, cumulative return chart, and bar chart across all strategies |

Strategy parameters (bull/chop/bear exposure, drawdown penalty, volatility scaling) are fully adjustable from the sidebar.

---

## 🗂 Requirements

- Python 3.9+
- Two data files in the project root:
  - **`hmm_oos_results.csv`** — your out-of-sample HMM results. Must be indexed by date and include a `regime` column with values `Bull`, `Chop`, or `Bear`.
  - **`price_data.csv`** — price history in a yfinance-style export format (indexed by date, with a `close` or `price` column). The loader skips the first two rows after the header to accommodate yfinance's multi-index CSV export format.

Python packages:
```
streamlit
pandas
numpy
plotly
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate      # on Windows: venv\Scripts\activate
```

### 3. Install dependencies

Create a `requirements.txt` (if you don't already have one):

```
streamlit
pandas
numpy
plotly
```

Then install:

```bash
pip install -r requirements.txt
```

### 4. Add your data files

Place `hmm_oos_results.csv` and `price_data.csv` in the same directory as `app.py`.

### 5. Run the app

```bash
streamlit run app.py
```

Streamlit will start a local server and open the dashboard in your browser, typically at:

```
http://localhost:8501
```

---

## 🎛 Using the Dashboard

1. **Choose a strategy** from the sidebar — pick one of the 5 presets or select "Custom — Build Your Own" to set your own bull/chop/bear exposures and toggle drawdown penalty / volatility scaling.
2. **Fine-tune** a preset strategy by checking "Adjust parameters" to override its default exposures.
3. **Compare strategies** by checking "Show comparison across ALL strategies" in the sidebar, or by clicking the button in the "Compare All" tab.
4. Explore the tabs to review regime detection quality, performance, drawdowns, and regime transition behavior.

### Strategy Presets

| Strategy | Bull | Chop | Bear | Drawdown Penalty | Volatility Scaling |
|---|---|---|---|---|---|
| 🚀 Aggressive — Max Return | 2.3 | 1.5 | 0.5 | ❌ | ❌ |
| ⚖️ Balanced | 1.0 | 0.7 | 0.3 | ❌ | ❌ |
| 🛡️ Conservative — Safest | 1.0 | 0.3 | 0.0 | ❌ | ❌ |
| 🔥 Advanced Max — Risk-Managed Max Return | 2.5 | 2.0 | 1.0 | ✅ | ✅ |
| 🏆 Advanced Safest — Best Sharpe | 1.0 | 0.5 | 0.2 | ✅ | ✅ |

Exposure values represent leverage/position size applied while the model is in that regime (values can be negative to represent short positions).

---

## 📊 Metrics Reported

- Total Return / Annualized Return
- Sharpe Ratio
- Max Drawdown
- Calmar Ratio
- Win Rate
- Average / Max Position Size
- Average Turnover
- Excess Return vs. Buy & Hold

Backtests apply a flat transaction cost of `0.02%` per unit of turnover.

---

## 📁 Project Structure

```
.
├── app.py                  # Streamlit dashboard (this app)
├── hmm_oos_results.csv     # Your HMM regime output (required, not included)
├── price_data.csv          # Your price data (required, not included)
├── requirements.txt        # Python dependencies
└── README.md
```

---

## ⚠️ Disclaimer

This dashboard is for research and educational purposes only. Nothing here constitutes financial advice. Past backtest performance does not guarantee future results.

---

## 📝 License

Add your preferred license here (e.g., MIT).
