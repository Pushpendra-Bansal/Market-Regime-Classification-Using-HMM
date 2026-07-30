import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==============================================================================
# 0. FILE VERIFICATION & INPUT DEFINITIONS
# ==============================================================================

# File requirements mapped to assets
ASSET_CONFIG = {
    'SPY': {'summary': 'performance_summary_SPY.json',     'oos': 'hmm_oos_results_SPY.csv'},
    'GLD': {'summary': 'performance_summary_GLD.json',     'oos': 'hmm_oos_results_GLD.csv'},
    'TLT': {'summary': 'performance_summary_TLT.json',     'oos': 'hmm_oos_results_TLT.csv'},
    'BTC': {'summary': 'performance_summary_BTC-USD.json', 'oos': 'hmm_oos_results_BTC-USD.csv'}
}

# Collect missing files across all expected inputs
missing_files = []
for asset, files in ASSET_CONFIG.items():
    if not os.path.exists(files['summary']):
        missing_files.append(files['summary'])
    if not os.path.exists(files['oos']):
        missing_files.append(files['oos'])

# Exit gracefully if missing required files
if missing_files:
    print("❌ ERROR: Required data files are missing to run this analysis:")
    for file in missing_files:
        print(f"   - {file}")
    print("\nPlease generate or place these files in the current working directory before proceeding.")
    sys.exit(1)
else:
    print("✅ All required data files found. Proceeding with execution...\n")


# ==============================================================================
# 1. LOAD DATA & HELPER FUNCTIONS FOR REGIME TIMELINE
# ==============================================================================

# Helper functions for regime post-processing
def smooth_regimes(regimes, window=30):
    smoothed = []
    for i in range(len(regimes)):
        start = max(0, i - window // 2)
        end = min(len(regimes), i + window // 2)
        window_regimes = regimes[start:end]
        values, counts = np.unique(window_regimes, return_counts=True)
        smoothed.append(values[np.argmax(counts)])
    return smoothed

def enforce_min_duration(regimes, min_days=30):
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
            prev_regime = result[i-1] if i > 0 else (regimes[j] if j < len(regimes) else current)
            for k in range(i, j):
                result[k] = prev_regime
        i = j
    return result

# Load OOS dataset into dataframe dictionary
dfs_oos = {}
for asset, files in ASSET_CONFIG.items():
    df = pd.read_csv(files['oos'], index_col=0, parse_dates=True)
    dfs_oos[asset] = df

# Apply smoothing logic to loaded dataframes
dfs_timeline = {}
for asset, df in dfs_oos.items():
    df_temp = df.copy()
    temp = smooth_regimes(df_temp['regime'].values, window=30)
    df_temp['regime_smooth'] = enforce_min_duration(temp, min_days=30)
    dfs_timeline[asset] = df_temp


# ==============================================================================
# 2. PLOT 1: CROSS-ASSET REGIME TIMELINE
# ==============================================================================

regime_colors = {
    'Bull': '#2ecc71',
    'Chop': '#f39c12',
    'Bear': '#e74c3c'
}

fig, axes = plt.subplots(4, 1, figsize=(18, 10), gridspec_kw={'hspace': 0.08})
asset_names_timeline = list(dfs_timeline.keys())

for idx, (name, df) in enumerate(dfs_timeline.items()):
    ax = axes[idx]

    # Fill regime background as solid colored blocks
    prev_date = df.index[0]
    prev_regime = df['regime_smooth'].iloc[0]

    for i in range(1, len(df)):
        current_regime = df['regime_smooth'].iloc[i]
        current_date = df.index[i]

        if current_regime != prev_regime or i == len(df) - 1:
            ax.axvspan(prev_date, current_date,
                       alpha=0.85,
                       color=regime_colors.get(prev_regime, '#95a5a6'),
                       zorder=1)
            prev_date = current_date
            prev_regime = current_regime

    # Asset label on Y axis
    ax.set_ylabel(name, fontsize=11, fontweight='bold', rotation=0,
                  labelpad=120, va='center')
    ax.set_yticks([])
    ax.set_xlim(pd.Timestamp('2010-01-01'), pd.Timestamp('2024-12-31'))

    # Add BTC start marker if present
    if 'BTC' in name:
        ax.axvline(x=pd.Timestamp('2018-04-08'),
                   color='white', linestyle='--', linewidth=1.5, alpha=0.8)
        ax.text(pd.Timestamp('2018-06-01'), 0.5, 'Data\nStart',
                transform=ax.get_xaxis_transform(),
                fontsize=8, color='white', va='center', fontweight='bold')
        # Fill pre-2018 with gray for BTC
        ax.axvspan(pd.Timestamp('2010-01-01'), pd.Timestamp('2018-04-08'),
                   alpha=0.3, color='gray', zorder=2)
        ax.text(pd.Timestamp('2013-01-01'), 0.5, 'No Data',
                transform=ax.get_xaxis_transform(),
                fontsize=9, color='white', va='center',
                ha='center', fontweight='bold')

    # Grid lines
    ax.grid(True, axis='x', alpha=0.2, linestyle='--', color='white', zorder=3)

    # Remove X labels except for bottom plot
    if idx < len(asset_names_timeline) - 1:
        ax.set_xticklabels([])
    else:
        ax.set_xlabel('Date', fontsize=12)

# Global key event annotations
key_events = {
    '2020-03-23': 'COVID\nCrash',
    '2022-01-03': '2022\nBear',
    '2015-08-24': '2015\nCorrection',
}

for date_str, label in key_events.items():
    date = pd.Timestamp(date_str)
    for ax in axes:
        ax.axvline(x=date, color='white',
                   linestyle=':', linewidth=1.5, alpha=0.8, zorder=4)
    axes[0].text(date, 1.02, label,
                transform=axes[0].get_xaxis_transform(),
                fontsize=8, color='gray', ha='center',
                fontweight='bold')

# Legend setup
bull_patch = mpatches.Patch(color='#2ecc71', alpha=0.85, label='Bull Regime')
chop_patch = mpatches.Patch(color='#f39c12', alpha=0.85, label='Chop Regime')
bear_patch = mpatches.Patch(color='#e74c3c', alpha=0.85, label='Bear Regime')
gray_patch = mpatches.Patch(color='gray',    alpha=0.30, label='No Data')

fig.legend(handles=[bull_patch, chop_patch, bear_patch, gray_patch],
           loc='upper right', fontsize=11, framealpha=0.9,
           bbox_to_anchor=(0.98, 0.98))

fig.suptitle('Cross-Asset Regime Timeline: SPY | GLD | TLT | BTC (2010–2024)',
             fontsize=15, fontweight='bold', y=1.01)

plt.tight_layout()
plt.savefig('cross_asset_regime_timeline.png', dpi=300,
            bbox_inches='tight', facecolor='#1a1a2e')
plt.show()
print("Saved as cross_asset_regime_timeline.png")


# ==============================================================================
# 3. PLOT 2: PERFORMANCE DASHBOARD (STRATEGY VS BUY & HOLD)
# ==============================================================================

assets_perf_map = {
    'SPY\n(Equity)':  ASSET_CONFIG['SPY']['summary'],
    'GLD\n(Gold)':    ASSET_CONFIG['GLD']['summary'],
    'TLT\n(Bonds)':   ASSET_CONFIG['TLT']['summary'],
    'BTC\n(Crypto)':  ASSET_CONFIG['BTC']['summary'],
}

strategy_data = {}
bah_data = {}

for name, filepath in assets_perf_map.items():
    with open(filepath, 'r') as f:
        summary = json.load(f)
    strategy_data[name] = summary['strategy_metrics']
    bah_data[name] = summary['bah_metrics']

asset_names_perf = list(assets_perf_map.keys())

def get_metric(data, key, multiply=100):
    return [data[name][key] * multiply if data[name][key] is not None else 0
            for name in asset_names_perf]

strategy_returns  = get_metric(strategy_data, 'annualised_return')
bah_returns       = get_metric(bah_data,      'annualised_return')
strategy_sharpe   = get_metric(strategy_data, 'sharpe', multiply=1)
bah_sharpe        = get_metric(bah_data,      'sharpe', multiply=1)
strategy_drawdown = get_metric(strategy_data, 'max_drawdown')
bah_drawdown      = get_metric(bah_data,      'max_drawdown')
strategy_vol      = get_metric(strategy_data, 'annualised_vol')
bah_vol           = get_metric(bah_data,      'annualised_vol')

x = np.arange(len(asset_names_perf))
width = 0.35
colors_strategy = '#2ecc71'
colors_bah      = '#e74c3c'

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.patch.set_facecolor('white')

# --- Subplot 1: Annualised Return ---
ax = axes[0, 0]
bars1 = ax.bar(x - width/2, strategy_returns, width, label='HMM Strategy', color=colors_strategy, alpha=0.85, edgecolor='white')
bars2 = ax.bar(x + width/2, bah_returns, width, label='Buy & Hold', color=colors_bah, alpha=0.85, edgecolor='white')

for bar in bars1:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.3, f'{h:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold', color='#27ae60')
for bar in bars2:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.3, f'{h:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold', color='#c0392b')

ax.set_title('Annualised Return (%)', fontsize=13, fontweight='bold', pad=10)
ax.set_xticks(x)
ax.set_xticklabels(asset_names_perf, fontsize=10)
ax.set_ylabel('Return (%)', fontsize=11)
ax.legend(fontsize=10, framealpha=0.9)
ax.grid(True, alpha=0.3, linestyle='--', axis='y')
ax.set_facecolor('#f8f9fa')
ax.axhline(y=0, color='gray', linewidth=0.8)

# --- Subplot 2: Sharpe Ratio ---
ax = axes[0, 1]
bars1 = ax.bar(x - width/2, strategy_sharpe, width, label='HMM Strategy', color=colors_strategy, alpha=0.85, edgecolor='white')
bars2 = ax.bar(x + width/2, bah_sharpe, width, label='Buy & Hold', color=colors_bah, alpha=0.85, edgecolor='white')

for bar in bars1:
    h = bar.get_height()
    if h != 0:
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f'{h:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold', color='#27ae60')
for bar in bars2:
    h = bar.get_height()
    if h != 0:
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f'{h:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold', color='#c0392b')

ax.set_title('Sharpe Ratio', fontsize=13, fontweight='bold', pad=10)
ax.set_xticks(x)
ax.set_xticklabels(asset_names_perf, fontsize=10)
ax.set_ylabel('Sharpe Ratio', fontsize=11)
ax.legend(fontsize=10, framealpha=0.9)
ax.grid(True, alpha=0.3, linestyle='--', axis='y')
ax.set_facecolor('#f8f9fa')
ax.axhline(y=0, color='gray', linewidth=0.8)

# --- Subplot 3: Maximum Drawdown ---
ax = axes[1, 0]
bars1 = ax.bar(x - width/2, strategy_drawdown, width, label='HMM Strategy', color=colors_strategy, alpha=0.85, edgecolor='white')
bars2 = ax.bar(x + width/2, bah_drawdown, width, label='Buy & Hold', color=colors_bah, alpha=0.85, edgecolor='white')

for bar in bars1:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h - 1, f'{h:.1f}%', ha='center', va='top', fontsize=9, fontweight='bold', color='#27ae60')
for bar in bars2:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h - 1, f'{h:.1f}%', ha='center', va='top', fontsize=9, fontweight='bold', color='#c0392b')

ax.set_title('Maximum Drawdown (%)', fontsize=13, fontweight='bold', pad=10)
ax.set_xticks(x)
ax.set_xticklabels(asset_names_perf, fontsize=10)
ax.set_ylabel('Drawdown (%)', fontsize=11)
ax.legend(fontsize=10, framealpha=0.9)
ax.grid(True, alpha=0.3, linestyle='--', axis='y')
ax.set_facecolor('#f8f9fa')
ax.axhline(y=0, color='gray', linewidth=0.8)

# --- Subplot 4: Annualised Volatility ---
ax = axes[1, 1]
bars1 = ax.bar(x - width/2, strategy_vol, width, label='HMM Strategy', color=colors_strategy, alpha=0.85, edgecolor='white')
bars2 = ax.bar(x + width/2, bah_vol, width, label='Buy & Hold', color=colors_bah, alpha=0.85, edgecolor='white')

for bar in bars1:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.3, f'{h:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold', color='#27ae60')
for bar in bars2:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.3, f'{h:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold', color='#c0392b')

ax.set_title('Annualised Volatility (%)', fontsize=13, fontweight='bold', pad=10)
ax.set_xticks(x)
ax.set_xticklabels(asset_names_perf, fontsize=10)
ax.set_ylabel('Volatility (%)', fontsize=11)
ax.legend(fontsize=10, framealpha=0.9)
ax.grid(True, alpha=0.3, linestyle='--', axis='y')
ax.set_facecolor('#f8f9fa')

fig.suptitle('Cross-Asset Performance Dashboard: HMM Strategy vs Buy & Hold', fontsize=15, fontweight='bold', y=1.01)

plt.tight_layout()
plt.savefig('cross_asset_performance_dashboard.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print("Saved as cross_asset_performance_dashboard.png")


# ==============================================================================
# 4. PLOT 3: DRAWDOWN COMPARISON
# ==============================================================================

def compute_drawdown(cum_returns):
    rolling_max = cum_returns.cummax()
    return (cum_returns - rolling_max) / rolling_max * 100

asset_drawdown_map = {
    'SPY (Equity)': dfs_oos['SPY'],
    'GLD (Gold)':   dfs_oos['GLD'],
    'TLT (Bonds)':  dfs_oos['TLT'],
    'BTC (Crypto)': dfs_oos['BTC'],
}

asset_colors = {
    'SPY (Equity)': '#3498db',
    'GLD (Gold)':   '#f39c12',
    'TLT (Bonds)':  '#2ecc71',
    'BTC (Crypto)': '#9b59b6'
}

fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.patch.set_facecolor('white')
axes = axes.flatten()

for idx, (name, df) in enumerate(asset_drawdown_map.items()):
    ax = axes[idx]
    color = asset_colors[name]

    strategy_dd = compute_drawdown(df['strategy_cum'])
    bah_dd = compute_drawdown(df['bah_cum'])

    ax.fill_between(df.index, strategy_dd, 0, alpha=0.6, color=color, label='HMM Strategy', zorder=3)
    ax.fill_between(df.index, bah_dd, 0, alpha=0.25, color='#e74c3c', label='Buy & Hold', zorder=2)

    ax.plot(df.index, strategy_dd, color=color, linewidth=1.2, zorder=4)
    ax.plot(df.index, bah_dd, color='#c0392b', linewidth=1.2, zorder=4, linestyle='--')

    strategy_max = strategy_dd.min()
    bah_max = bah_dd.min()
    strategy_max_date = strategy_dd.idxmin()
    bah_max_date = bah_dd.idxmin()

    ax.annotate(f'HMM: {strategy_max:.1f}%',
                xy=(strategy_max_date, strategy_max),
                xytext=(30, -15), textcoords='offset points',
                fontsize=9, fontweight='bold', color=color,
                arrowprops=dict(arrowstyle='->', color=color, lw=1.2))

    ax.annotate(f'B&H: {bah_max:.1f}%',
                xy=(bah_max_date, bah_max),
                xytext=(30, 15), textcoords='offset points',
                fontsize=9, fontweight='bold', color='#c0392b',
                arrowprops=dict(arrowstyle='->', color='#c0392b', lw=1.2))

    events = {'2020-03-23': 'COVID', '2022-01-03': '2022'}
    for date_str, label in events.items():
        date = pd.Timestamp(date_str)
        if date >= df.index[0]:
            ax.axvline(x=date, color='gray', linestyle='--', linewidth=1, alpha=0.6)
            ax.text(date, ax.get_ylim()[0] * 0.95 if ax.get_ylim()[0] != 0 else -1,
                    label, fontsize=8, color='gray', ha='center', style='italic')

    dd_reduction = abs(bah_max) - abs(strategy_max)
    textstr = f'DD Reduction: {dd_reduction:.1f}%'
    props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor=color)
    ax.text(0.02, 0.05, textstr, transform=ax.transAxes,
            fontsize=9, verticalalignment='bottom', bbox=props, color=color)

    ax.set_title(f'{name} — Drawdown Comparison', fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel('Date', fontsize=10)
    ax.set_ylabel('Drawdown (%)', fontsize=10)
    ax.legend(fontsize=10, loc='lower right', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_facecolor('#f8f9fa')
    ax.axhline(y=0, color='gray', linewidth=0.8)

fig.suptitle('Cross-Asset Drawdown Comparison: HMM Strategy vs Buy & Hold', fontsize=15, fontweight='bold', y=1.01)

plt.tight_layout()
plt.savefig('cross_asset_drawdown.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print("Saved as cross_asset_drawdown.png")


# ==============================================================================
# 5. PLOT 4 & 5: PLOTLY REGIME ALIGNMENT & CONDITIONAL HEATMAPS
# ==============================================================================

dfs_regimes = {asset: dfs_oos[asset][['regime']] for asset in ASSET_CONFIG.keys()}

combined = pd.concat(dfs_regimes, axis=1)
combined.columns = list(ASSET_CONFIG.keys())
combined = combined.dropna()

print(f"Common trading days across all assets: {len(combined)}")

assets_list = list(ASSET_CONFIG.keys())
alignment_matrix = pd.DataFrame(index=assets_list, columns=assets_list, dtype=float)

for a1 in assets_list:
    for a2 in assets_list:
        same = (combined[a1] == combined[a2]).sum()
        alignment_matrix.loc[a1, a2] = round(same / len(combined) * 100, 1)

print("\nSame Regime Alignment Matrix (%):")
print(alignment_matrix)

# Heatmap 1 — Standard Alignment Heatmap
fig1 = go.Figure(data=go.Heatmap(
    z=alignment_matrix.values.astype(float),
    x=assets_list,
    y=assets_list,
    colorscale=[
        [0.0, '#0a0e1a'],
        [0.3, '#1e3a5f'],
        [0.6, '#1d4ed8'],
        [1.0, '#4ade80']
    ],
    text=[[f'{v:.1f}%' for v in row] for row in alignment_matrix.values.astype(float)],
    texttemplate='%{text}',
    textfont=dict(size=16, family='JetBrains Mono', color='#f1f5f9'),
    hovertemplate='%{y} vs %{x}<br>Same Regime: %{text}<extra></extra>',
    showscale=True,
    zmin=0,
    zmax=100,
    colorbar=dict(
        title=dict(text='% Same Regime', font=dict(color='#94a3b8')),
        tickfont=dict(color='#64748b', family='JetBrains Mono'),
        ticksuffix='%',
    )
))

fig1.update_layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(10,14,26,0.8)',
    font=dict(family='Inter', color='#94a3b8', size=12),
    height=420,
    title=dict(
        text='Regime Alignment — % of Days Assets Share Same Regime',
        font=dict(color='#f1f5f9', size=15),
        x=0.5, xanchor='center'
    ),
    xaxis=dict(tickfont=dict(color='#94a3b8', size=13), side='bottom'),
    yaxis=dict(tickfont=dict(color='#94a3b8', size=13)),
    margin=dict(l=10, r=10, t=60, b=10),
)

fig1.show()

# Heatmap 2 — Conditional Regime Distribution
anchor_asset = 'SPY'
other_assets = [a for a in assets_list if a != anchor_asset]
regimes = ['Bull', 'Chop', 'Bear']

rows = []
for regime in regimes:
    mask = combined[anchor_asset] == regime
    subset = combined[mask]
    n_days = len(subset)

    for other in other_assets:
        for other_regime in regimes:
            pct = (subset[other] == other_regime).sum() / n_days * 100 if n_days > 0 else 0
            rows.append({
                'anchor_regime': regime,
                'other_asset': other,
                'other_regime': other_regime,
                'pct': round(pct, 1)
            })

cond_df = pd.DataFrame(rows)

fig2 = make_subplots(
    rows=1, cols=3,
    subplot_titles=[f'When SPY is {r}' for r in regimes],
    horizontal_spacing=0.08
)

for col_idx, spy_regime in enumerate(regimes, start=1):
    subset = cond_df[cond_df['anchor_regime'] == spy_regime]

    z_vals = []
    y_labels = []
    x_labels = regimes

    for other in other_assets:
        row_vals = []
        for r in regimes:
            val = subset[
                (subset['other_asset'] == other) &
                (subset['other_regime'] == r)
            ]['pct'].values
            row_vals.append(val[0] if len(val) > 0 else 0)
        z_vals.append(row_vals)
        y_labels.append(other)

    color = {'Bull': '#4ade80', 'Chop': '#facc15', 'Bear': '#f87171'}[spy_regime]

    fig2.add_trace(go.Heatmap(
        z=z_vals,
        x=x_labels,
        y=y_labels,
        colorscale=[
            [0.0, '#0a0e1a'],
            [0.5, '#1e3a5f'],
            [1.0, color]
        ],
        text=[[f'{v:.0f}%' for v in row] for row in z_vals],
        texttemplate='%{text}',
        textfont=dict(size=13, family='JetBrains Mono', color='#f1f5f9'),
        showscale=False,
        zmin=0,
        zmax=100,
        hovertemplate='%{y} is %{x}: %{text}<extra></extra>',
    ), row=1, col=col_idx)

fig2.update_layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(10,14,26,0.8)',
    font=dict(family='Inter', color='#94a3b8', size=11),
    height=380,
    title=dict(
        text='Conditional Regime Distribution — Given SPY Regime, What Are Other Assets In?',
        font=dict(color='#f1f5f9', size=14),
        x=0.5, xanchor='center'
    ),
    margin=dict(l=10, r=10, t=80, b=10),
)

for i in range(1, 4):
    fig2.update_xaxes(tickfont=dict(color='#94a3b8', size=11), gridcolor='rgba(30,45,74,0.4)', row=1, col=i)
    fig2.update_yaxes(tickfont=dict(color='#94a3b8', size=11), gridcolor='rgba(30,45,74,0.4)', row=1, col=i)

for ann in fig2.layout.annotations:
    ann.font.color = '#94a3b8'
    ann.font.size = 12

fig2.show()
