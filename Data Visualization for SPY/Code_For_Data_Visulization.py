"""
HMM Regime Analysis — standalone script (converted from notebook cells).

NOTE: This script assumes `hmm_oos_results.csv` already exists, so please download it from github repo and then run it.



"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import yfinance as yf

# ============================================================
# LOAD DATA
# ============================================================
oos_df = pd.read_csv("hmm_oos_results.csv", index_col=0, parse_dates=True)

if "oos_df" not in dir():
    raise NameError(
        "oos_df is not defined. Load or construct it before running this "
        "script (see the note at the top of this file)."
    )


# ============================================================
# REGIME SMOOTHING
# ============================================================
def smooth_regimes(regimes, window=60):
    smoothed = []
    for i in range(len(regimes)):
        start = max(0, i - window // 2)
        end = min(len(regimes), i + window // 2)
        window_regimes = regimes[start:end]
        values, counts = np.unique(window_regimes, return_counts=True)
        smoothed.append(values[np.argmax(counts)])
    return smoothed


def enforce_min_duration(regimes, min_days=40):
    regimes = list(regimes)
    result = regimes.copy()
    i = 0
    while i < len(regimes):
        current = regimes[i]
        j = i
        while j < len(regimes) and regimes[j] == current:
            j += 1
        duration = j - i
        if duration < min_days:
            prev_regime = result[i - 1] if i > 0 else regimes[j] if j < len(regimes) else current
            for k in range(i, j):
                result[k] = prev_regime
        i = j
    return result


# apply both with aggressive parameters
temp = smooth_regimes(oos_df['regime'].values, window=60)
temp = enforce_min_duration(temp, min_days=40)
temp = smooth_regimes(temp, window=30)  # second pass
oos_df['regime_smooth'] = enforce_min_duration(temp, min_days=40)  # second enforcement

print(oos_df['regime_smooth'].value_counts())


# ============================================================
# PLOT 1: SPY price with regime shading
# ============================================================
spy = yf.download("SPY", start="2005-01-01", end="2024-12-31")
spy_close = spy['Close'].squeeze()
fig, ax = plt.subplots(figsize=(16, 7))

ax.plot(spy_close.index, spy_close.values,
        color='#1a1a2e', linewidth=1.2, label='SPY Close Price', zorder=3)

regime_colors = {
    'Bull': '#90EE90',   # soft green
    'Chop': '#FFD700',   # soft yellow
    'Bear': '#FF6B6B'    # soft red
}

prev_date = oos_df.index[0]
prev_regime = oos_df['regime'].iloc[0]

for i in range(1, len(oos_df)):
    current_regime = oos_df['regime'].iloc[i]
    current_date = oos_df.index[i]

    if current_regime != prev_regime or i == len(oos_df) - 1:
        ax.axvspan(prev_date, current_date,
                   alpha=0.3,
                   color=regime_colors[prev_regime],
                   zorder=1)
        prev_date = current_date
        prev_regime = current_regime

bull_patch = mpatches.Patch(color='#90EE90', alpha=0.5, label='Bull Regime')
chop_patch = mpatches.Patch(color='#FFD700', alpha=0.5, label='Chop Regime')
bear_patch = mpatches.Patch(color='#FF6B6B', alpha=0.5, label='Bear Regime')
price_line = plt.Line2D([0], [0], color='#1a1a2e', linewidth=1.5, label='SPY Close')

ax.legend(handles=[price_line, bull_patch, chop_patch, bear_patch],
          loc='upper left', fontsize=11, framealpha=0.9)

ax.set_title('SPY Price History with HMM Detected Regimes (2005–2024)',
             fontsize=16, fontweight='bold', pad=15)
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('SPY Price (USD)', fontsize=12)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_facecolor('#f8f9fa')
fig.patch.set_facecolor('white')

events = {
    '2008-09-15': '2008\nCrisis',
    '2020-03-23': 'COVID\nCrash',
    '2022-01-03': '2022\nBear'
}

for date_str, label in events.items():
    date = pd.Timestamp(date_str)
    if date in spy_close.index:
        price = spy_close[date]
        ax.annotate(label,
                    xy=(date, price),
                    xytext=(date, price * 0.85),
                    fontsize=9,
                    color='#cc0000',
                    fontweight='bold',
                    ha='center',
                    arrowprops=dict(arrowstyle='->', color='#cc0000', lw=1.5))

plt.tight_layout()
plt.savefig('plot1_spy_regimes.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print("Saved as plot1_spy_regimes.png")


# ============================================================
# PLOT 2: Feature distributions by regime
# ============================================================
feature_cols = ['log_return', 'volume_zscore', 'momentum', 'parkinson_vol', 'rsi', 'ma_cross']
feature_labels = ['Log Return', 'Volume Z-Score', 'Momentum', 'Parkinson Volatility', 'RSI', 'MA Cross']

regime_colors = {
    'Bull': '#2ecc71',
    'Chop': '#f39c12',
    'Bear': '#e74c3c'
}

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for idx, (feature, label) in enumerate(zip(feature_cols, feature_labels)):
    ax = axes[idx]

    for regime in ['Bull', 'Chop', 'Bear']:
        data = oos_df[oos_df['regime_smooth'] == regime][feature].dropna()

        lower = data.quantile(0.02)
        upper = data.quantile(0.98)
        data = data.clip(lower, upper)

        ax.hist(data,
                bins=50,
                alpha=0.5,
                color=regime_colors[regime],
                label=regime,
                density=True,
                edgecolor='none')

        ax.axvline(data.mean(),
                   color=regime_colors[regime],
                   linestyle='--',
                   linewidth=2,
                   alpha=0.9)

    ax.set_title(label, fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel('Value', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.legend(fontsize=9, framealpha=0.8)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_facecolor('#f8f9fa')

fig.suptitle('Feature Distributions Across Market Regimes\n(Dashed lines show regime means)',
             fontsize=16, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig('plot2_feature_distributions.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print("Saved as plot2_feature_distributions.png")


# ============================================================
# PLOT 6: Features over time with regime shading
# ============================================================
regime_colors = {
    'Bull': '#90EE90',
    'Chop': '#FFD700',
    'Bear': '#FF6B6B'
}

fig, axes = plt.subplots(6, 1, figsize=(18, 22))

for idx, (feature, label) in enumerate(zip(feature_cols, feature_labels)):
    ax = axes[idx]

    ax.plot(oos_df.index, oos_df[feature],
            color='#1a1a2e', linewidth=0.8, alpha=0.9, zorder=3)

    prev_date = oos_df.index[0]
    prev_regime = oos_df['regime_smooth'].iloc[0]

    for i in range(1, len(oos_df)):
        current_regime = oos_df['regime_smooth'].iloc[i]
        current_date = oos_df.index[i]

        if current_regime != prev_regime or i == len(oos_df) - 1:
            ax.axvspan(prev_date, current_date,
                      alpha=0.25,
                      color=regime_colors[prev_regime],
                      zorder=1)
            prev_date = current_date
            prev_regime = current_regime

    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)

    ax.set_ylabel(label, fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.set_facecolor('#f8f9fa')

    if idx < len(feature_cols) - 1:
        ax.set_xticklabels([])

bull_patch = mpatches.Patch(color='#90EE90', alpha=0.6, label='Bull Regime')
chop_patch = mpatches.Patch(color='#FFD700', alpha=0.6, label='Chop Regime')
bear_patch = mpatches.Patch(color='#FF6B6B', alpha=0.6, label='Bear Regime')

axes[0].legend(handles=[bull_patch, chop_patch, bear_patch],
               loc='upper right', fontsize=10, framealpha=0.9)

axes[-1].set_xlabel('Date', fontsize=12)

fig.suptitle('Feature Behaviour Over Time with HMM Detected Regimes (2005–2024)',
             fontsize=15, fontweight='bold', y=1.005)

plt.tight_layout()
plt.savefig('plot6_features_over_time.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print("Saved as plot6_features_over_time.png")


# ============================================================
# PLOT 4: Regime transition probability matrix
# ============================================================
regimes = oos_df['regime_smooth'].values
regime_labels = ['Bull', 'Chop', 'Bear']

transition_matrix = np.zeros((3, 3))

for i in range(len(regimes) - 1):
    from_regime = regimes[i]
    to_regime = regimes[i + 1]
    from_idx = regime_labels.index(from_regime)
    to_idx = regime_labels.index(to_regime)
    transition_matrix[from_idx][to_idx] += 1

row_sums = transition_matrix.sum(axis=1, keepdims=True)
transition_matrix = transition_matrix / row_sums

fig, ax = plt.subplots(figsize=(8, 6))

sns.heatmap(
    transition_matrix,
    annot=True,
    fmt='.3f',
    cmap='RdYlGn',
    vmin=0,
    vmax=1,
    xticklabels=regime_labels,
    yticklabels=regime_labels,
    linewidths=2,
    linecolor='white',
    annot_kws={'size': 14, 'weight': 'bold'},
    ax=ax,
    square=True
)

ax.set_title('HMM Regime Transition Probability Matrix',
             fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('To Regime', fontsize=12, fontweight='bold')
ax.set_ylabel('From Regime', fontsize=12, fontweight='bold')

ax.set_xticklabels(regime_labels, fontsize=12, fontweight='bold')
ax.set_yticklabels(regime_labels, fontsize=12, fontweight='bold', rotation=0)

fig.text(0.5, -0.05,
         'Diagonal values show regime persistence probability\nHigh diagonal = stable regimes, Low diagonal = frequent switching',
         ha='center', fontsize=10, style='italic', color='gray')

plt.tight_layout()
plt.savefig('plot4_transition_matrix.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print("Saved as plot4_transition_matrix.png")
