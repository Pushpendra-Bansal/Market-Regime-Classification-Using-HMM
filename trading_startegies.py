import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("  HMM TRADING STRATEGY — BEST PERFORMERS")
print("="*70)

# Data Loading

print("\n📂 Loading data...")

# Load HMM results
oos_df = pd.read_csv('hmm_oos_results.csv', index_col=0, parse_dates=True)

# Load price data
price_df = pd.read_csv('price_data.csv', skiprows=[1, 2], index_col=0, parse_dates=True)
price_df.columns = price_df.columns.str.strip().str.lower()

# Find close column
close_col = [c for c in price_df.columns if 'close' in c or 'price' in c][0]
price_df = price_df[[close_col]].rename(columns={close_col: 'close'})

# Merge
oos_df = oos_df.join(price_df, how='left')
oos_df['close'] = oos_df['close'].ffill()
oos_df['asset_return'] = oos_df['close'].pct_change()

print(f"   ✅ Loaded {len(oos_df)} rows")
print(f"   📅 {oos_df.index[0]} to {oos_df.index[-1]}")

# Strategy Definition

BEST_STRATEGIES = {

    # HIGHEST RETURN 
    'aggressive': {
        'bull': 2.3,
        'chop': 1.5,
        'bear': 0.5,
        'description': ' MAX RETURN (2157.19%)',
        'use_drawdown_penalty': False,
        'use_volatility_scaling': False,
    },

    # BEST BALANCE
    'balanced': {
        'bull': 1.0,
        'chop': 0.7,
        'bear': 0.3,
        'description': 'BEST BALANCE (401.32%)',
        'use_drawdown_penalty': False,
        'use_volatility_scaling': False,
    },

    #  SAFEST 
    'conservative': {
        'bull': 1.0,
        'chop': 0.3,
        'bear': 0.0,
        'description': ' SAFEST (-8.98% DD)',
        'use_drawdown_penalty': False,
        'use_volatility_scaling': False,
    },

    #  ADVANCED MAX RETURN 
    'advanced_max': {
        'bull': 2.5,
        'chop': 2.0,
        'bear': 1.0,
        'description': '📈 ADVANCED MAX (1560.28%)',
        'use_drawdown_penalty': True,
        'use_volatility_scaling': True,
    },

    #  BEST SHARPE 
    'advanced_safest': {
        'bull': 1.0,
        'chop': 0.5,
        'bear': 0.2,
        'description': '📊 BEST SHARPE (104.55)',
        'use_drawdown_penalty': True,
        'use_volatility_scaling': True,
    },
}

# Positon Sizing Engine

def calculate_position(regime, current_drawdown, volatility, params):
    """Calculate position size with optional risk controls"""

    # Base position by regime
    if regime == 'Bull':
        base = params['bull']
    elif regime == 'Chop':
        base = params['chop']
    else:  # Bear
        base = params['bear']

    # Apply drawdown penalty if enabled
    if params.get('use_drawdown_penalty', False):
        if current_drawdown < -0.10:
            penalty = 1.0 + current_drawdown * 2
            penalty = max(penalty, 0.3)
        elif current_drawdown < -0.05:
            penalty = 1.0 - (abs(current_drawdown) - 0.05) * 5
            penalty = max(penalty, 0.5)
        else:
            penalty = 1.0
        base = base * penalty

    # Apply volatility scaling if enabled
    if params.get('use_volatility_scaling', False):
        target_vol = 0.15
        vol_scalar = min(1.0, target_vol / max(volatility, 0.01))
        base = base * vol_scalar

    return np.clip(base, -2.0, 3.0)

# Backtest Engine

def run_backtest(oos_df, strategy_name, params):
    """Run backtest for a single strategy"""

    df = oos_df.copy()
    df['position'] = 0.0
    TRANSACTION_COST = 0.0002

    # Calculate volatility
    df['volatility'] = df['asset_return'].rolling(20).std() * np.sqrt(252)
    df['volatility'] = df['volatility'].fillna(0.15)

  
    cum_returns = pd.Series(1.0, index=df.index)

    for i in range(1, len(df)):
        current_regime = df['regime'].iloc[i-1]
        current_volatility = df['volatility'].iloc[i-1]

        # Calculate current drawdown
        current_cum = cum_returns.iloc[i-1]
        running_max = cum_returns[:i].max()
        current_dd = (current_cum - running_max) / running_max if running_max > 0 else 0

        # Get position
        pos = calculate_position(
            current_regime, current_dd, current_volatility, params
        )

        # Calculate return
        ret = pos * df['asset_return'].iloc[i]
        cum_returns.iloc[i] = cum_returns.iloc[i-1] * (1 + ret)

        # Store position
        df.loc[df.index[i], 'position'] = pos

    # Calculate returns with transaction costs
    df['strategy_return'] = df['position'].shift(1) * df['asset_return']
    df['turnover'] = df['position'].diff().abs()
    df['transaction_cost'] = df['turnover'] * TRANSACTION_COST
    df['net_return'] = df['strategy_return'] - df['transaction_cost']
    df['cum_strategy'] = (1 + df['net_return']).cumprod()
    df['cum_buyhold'] = (1 + df['asset_return']).cumprod()

    return df

# Performance Metrics

def calculate_metrics(df):
    """Calculate comprehensive performance metrics"""

    clean = df.dropna(subset=['net_return'])

    # Total returns
    total_strategy = (clean['cum_strategy'].iloc[-1] - 1) * 100
    total_buyhold = (clean['cum_buyhold'].iloc[-1] - 1) * 100

    # Annual returns
    years = len(clean) / 252
    ann_strategy = ((1 + total_strategy/100) ** (1/years) - 1) * 100

    # Volatility
    vol_strategy = clean['net_return'].std() * np.sqrt(252) * 100

    # Sharpe ratio
    sharpe = ann_strategy / vol_strategy * 100 if vol_strategy > 0 else 0

    # Maximum drawdown
    cum = clean['cum_strategy']
    running_max = cum.expanding().max()
    drawdown = (cum - running_max) / running_max
    max_dd = drawdown.min() * 100

    # Win rate
    wins = (clean['net_return'] > 0).sum()
    active_days = (clean['position'] != 0).sum()
    win_rate = wins / active_days * 100 if active_days > 0 else 0

    # Calmar ratio
    calmar = total_strategy / abs(max_dd) if max_dd != 0 else 0

    # Average position
    avg_position = clean['position'].abs().mean()
    max_position = clean['position'].abs().max()
    avg_turnover = clean['turnover'].mean() * 100

    return {
        'total_return': total_strategy,
        'annual_return': ann_strategy,
        'volatility': vol_strategy,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'win_rate': win_rate,
        'calmar': calmar,
        'avg_position': avg_position,
        'max_position': max_position,
        'avg_turnover': avg_turnover,
        'excess_return': total_strategy - total_buyhold
    }

# Running all startegies

print("\n" + "="*70)
print(" RUNNING STRATEGIES")
print("="*70)

results = []
results_df = {}

for name, params in BEST_STRATEGIES.items():
    print(f"\n   Testing: {params['description']}")

    df = run_backtest(oos_df, name, params)
    metrics = calculate_metrics(df)
    metrics['strategy'] = name
    results.append(metrics)
    results_df[name] = df

# Create comparison table
comparison_df = pd.DataFrame(results)
comparison_df = comparison_df.set_index('strategy')

# Reorder columns
cols = ['bull', 'chop', 'bear', 'total_return', 'annual_return',
        'volatility', 'sharpe', 'max_drawdown', 'win_rate',
        'calmar', 'avg_position', 'max_position', 'avg_turnover', 'excess_return']

# Add parameters
for name, params in BEST_STRATEGIES.items():
    comparison_df.loc[name, 'bull'] = params['bull']
    comparison_df.loc[name, 'chop'] = params['chop']
    comparison_df.loc[name, 'bear'] = params['bear']

comparison_df = comparison_df[cols]

# Display Results

print("\n" + "="*70)
print(" STRATEGY COMPARISON TABLE")
print("="*70)

# Format for display
display_df = comparison_df.copy()
display_df['total_return'] = display_df['total_return'].round(2)
display_df['annual_return'] = display_df['annual_return'].round(2)
display_df['volatility'] = display_df['volatility'].round(2)
display_df['sharpe'] = display_df['sharpe'].round(2)
display_df['max_drawdown'] = display_df['max_drawdown'].round(2)
display_df['win_rate'] = display_df['win_rate'].round(2)
display_df['calmar'] = display_df['calmar'].round(2)
display_df['excess_return'] = display_df['excess_return'].round(2)

print(display_df.to_string())

# Visualisation

print("\n" + "="*70)
print("📊 GENERATING VISUALIZATION")
print("="*70)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

colors = {
    'aggressive': '#E74C3C',
    'balanced': '#3498DB',
    'conservative': '#2ECC71',
    'advanced_max': '#9B59B6',
    'advanced_safest': '#F39C12'
}

# 1. Total Return
ax = axes[0, 0]
bars = ax.bar(comparison_df.index, comparison_df['total_return'],
              color=[colors.get(i, '#95A5A6') for i in comparison_df.index])
ax.axhline(y=637.88, color='black', linestyle='--', label='Buy & Hold: 637.88%')
ax.set_title('Total Return by Strategy', fontsize=12, fontweight='bold')
ax.set_ylabel('Return (%)')
ax.legend()
ax.grid(True, alpha=0.3)


# value labels
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 10,
            f'{height:.0f}%', ha='center', va='bottom', fontsize=8)

# 2. Max Drawdown
ax = axes[0, 1]
bars = ax.bar(comparison_df.index, comparison_df['max_drawdown'],
              color=[colors.get(i, '#95A5A6') for i in comparison_df.index])
ax.axhline(y=-33.72, color='black', linestyle='--', label='Buy & Hold: -33.72%')
ax.set_title('Maximum Drawdown', fontsize=12, fontweight='bold')
ax.set_ylabel('Drawdown (%)')
ax.legend()
ax.grid(True, alpha=0.3)

for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height - 2,
            f'{height:.1f}%', ha='center', va='top', fontsize=8)

# 3. Sharpe Ratio
ax = axes[0, 2]
bars = ax.bar(comparison_df.index, comparison_df['sharpe'],
              color=[colors.get(i, '#95A5A6') for i in comparison_df.index])
ax.axhline(y=85.00, color='black', linestyle='--', label='Buy & Hold: 85.00')
ax.set_title('Sharpe Ratio', fontsize=12, fontweight='bold')
ax.set_ylabel('Sharpe')
ax.legend()
ax.grid(True, alpha=0.3)

for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 2,
            f'{height:.1f}', ha='center', va='bottom', fontsize=8)

# 4. Calmar Ratio
ax = axes[1, 0]
bars = ax.bar(comparison_df.index, comparison_df['calmar'],
              color=[colors.get(i, '#95A5A6') for i in comparison_df.index])
ax.set_title('Calmar Ratio (Return / Drawdown)', fontsize=12, fontweight='bold')
ax.set_ylabel('Calmar Ratio')
ax.grid(True, alpha=0.3)

for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'{height:.1f}', ha='center', va='bottom', fontsize=8)

# 5. Win Rate
ax = axes[1, 1]
bars = ax.bar(comparison_df.index, comparison_df['win_rate'],
              color=[colors.get(i, '#95A5A6') for i in comparison_df.index])
ax.set_title('Win Rate', fontsize=12, fontweight='bold')
ax.set_ylabel('Win Rate (%)')
ax.grid(True, alpha=0.3)

for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
            f'{height:.1f}%', ha='center', va='bottom', fontsize=8)

# 6. Avg Position
ax = axes[1, 2]
bars = ax.bar(comparison_df.index, comparison_df['avg_position'],
              color=[colors.get(i, '#95A5A6') for i in comparison_df.index])
ax.set_title('Average Position Size', fontsize=12, fontweight='bold')
ax.set_ylabel('Position Size')
ax.grid(True, alpha=0.3)

for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
            f'{height:.2f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('best_strategies_comparison.png', dpi=150)
print("   ✅ Saved: best_strategies_comparison.png")
plt.show()

# Recommendations

print("\n" + "="*70)
print(" STRATEGY RECOMMENDATIONS")
print("="*70)

# Find best in each category
best_return = comparison_df.loc[comparison_df['total_return'].idxmax()]
best_sharpe = comparison_df.loc[comparison_df['sharpe'].idxmax()]
best_calmar = comparison_df.loc[comparison_df['calmar'].idxmax()]
best_dd = comparison_df.loc[comparison_df['max_drawdown'].idxmax()]

print(f"\n MAXIMUM RETURN:")
print(f"   Strategy: {best_return.name}")
print(f"   Return: {best_return['total_return']:.2f}%")
print(f"   Drawdown: {best_return['max_drawdown']:.2f}%")
print(f"   Sharpe: {best_return['sharpe']:.2f}")

print(f"\n  BEST RISK-ADJUSTED (Sharpe):")
print(f"   Strategy: {best_sharpe.name}")
print(f"   Return: {best_sharpe['total_return']:.2f}%")
print(f"   Drawdown: {best_sharpe['max_drawdown']:.2f}%")
print(f"   Sharpe: {best_sharpe['sharpe']:.2f}")

print(f"\n  LOWEST DRAWDOWN:")
print(f"   Strategy: {best_dd.name}")
print(f"   Return: {best_dd['total_return']:.2f}%")
print(f"   Drawdown: {best_dd['max_drawdown']:.2f}%")
print(f"   Sharpe: {best_dd['sharpe']:.2f}")

print(f"\n  BEST RETURN PER DRAWDOWN (Calmar):")
print(f"   Strategy: {best_calmar.name}")
print(f"   Return: {best_calmar['total_return']:.2f}%")
print(f"   Drawdown: {best_calmar['max_drawdown']:.2f}%")
print(f"   Calmar: {best_calmar['calmar']:.2f}")

# Save Results

comparison_df.to_csv('best_strategies_results.csv')
print("\n📁 Results saved to: best_strategies_results.csv")
print("📁 Visualization saved to: best_strategies_comparison.png")

print("\n" + "="*70)
print("✅ COMPLETE! Choose your strategy based on your goal:")
print("   • Maximum Return → aggressive (2157.19%)")
print("   • Best Sharpe → advanced_safest (104.55)")
print("   • Safest → conservative (-8.98% DD)")
print("   • Best Calmar → aggressive (74.1)")
print("="*70)

# Quick Startegy Loader

def load_best_strategy(strategy_name):
    """
    Quick function to load any best strategy with its parameters

    Usage:
        params = load_best_strategy('aggressive')
        # params = {'bull': 2.0, 'chop': 1.5, 'bear': 0.6}
    """
    strategies = {
        'aggressive': {'bull': 2.0, 'chop': 1.5, 'bear': 0.6},
        'balanced': {'bull': 1.0, 'chop': 0.7, 'bear': 0.3},
        'conservative': {'bull': 1.0, 'chop': 0.3, 'bear': 0.0},
        'advanced_max': {'bull': 2.5, 'chop': 2.0, 'bear': 1.0},
        'advanced_safest': {'bull': 1.0, 'chop': 0.5, 'bear': 0.2},
    }
    return strategies.get(strategy_name, strategies['balanced'])

print("\n✅ Use load_best_strategy('name') to get strategy parameters")
print("   Options: aggressive, balanced, conservative, advanced_max, advanced_safest")
