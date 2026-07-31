"""
Run These Commands Before:
pip install yfinance
pip install pandas_ta
"""
import yfinance as yf
import pandas as pd
import numpy as np

df = yf.download("SPY", start="2005-01-01", end="2024-12-31")
#The reason i took 19 years of data is that it covers 2008 crisis(bullish market) also covid period , etc. hence our HMM model will be trained on every type of regime.
df.head()

df = df[~df.index.duplicated(keep='first')]

df = df.sort_index()

(df['Close'] <= 0).sum()

(df['Volume'] <= 0).sum()

def yang_zhang(df, window=20):
    o = np.log(df['Open'] / df['Close'].shift(1))   # overnight return
    c = np.log(df['Close'] / df['Open'])             # close-to-open return
    h = np.log(df['High'] / df['Open'])
    l = np.log(df['Low'] / df['Open'])

    k = 0.34 / (1.34 + (window + 1) / (window - 1))

    cc = (h * (h - c) + l * (l - c)).rolling(window).mean()
    oc = o.rolling(window).var()
    co = c.rolling(window).var()

    return np.sqrt(oc + k * co + (1 - k) * cc)

# Log returns
df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))

#return kurt
df['return_kurt'] = df['log_return'].rolling(20).kurt()

# yang-zhang volatility
df['yz_vol'] = yang_zhang(df)

#return_skew
df['return_skew'] = df['log_return'].rolling(20).skew()

#momentum
df['momentum'] = df['log_return'].rolling(5).mean()

# Volume z-score
df['volume_zscore'] = (
    (df['Volume'] - df['Volume'].rolling(20).mean()) /
    df['Volume'].rolling(20).std()
)

df = df.dropna()
print(df.shape)

def parkinson_volatility(df, window=20):
    """Parkinson volatility estimator using only High and Low prices."""
    return np.sqrt(
        (1 / (4 * np.log(2))) *
        (np.log(df['High'] / df['Low'])**2).rolling(window).mean()
    ) * np.sqrt(252)

df['parkinson_vol'] = parkinson_volatility(df, window=20)

def rsi(df, window=14):
    """Calculate RSI for overbought/oversold detection."""
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

df['rsi'] = rsi(df, window=14)

def bollinger_width(df, window=20, num_std=2):
    """Bollinger Band width — measures volatility expansion/contraction."""
    ma = df['Close'].rolling(window).mean()
    std = df['Close'].rolling(window).std()
    upper = ma + num_std * std
    lower = ma - num_std * std
    return (upper - lower) / ma * 100  # Normalized by price

df['bb_width'] = bollinger_width(df, window=20, num_std=2)

def ma_crossover(df, fast=20, slow=50):
    """Moving average crossover — measures trend direction and strength."""
    ma_fast = df['Close'].rolling(fast).mean()
    ma_slow = df['Close'].rolling(slow).mean()
    return (ma_fast - ma_slow) / ma_slow * 100  # Percentage difference

df['ma_cross'] = ma_crossover(df, fast=20, slow=50)
print("   ✅ MA Crossover added")

df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()

def macd(df, fast=12, slow=26, signal=9):
    """MACD — trend and momentum indicator."""
    ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

df['macd_line'], df['macd_signal'], df['macd_hist'] = macd(df)

before = len(df)
df = df.dropna()
dropped = before - len(df)
print(f"   Dropped {dropped} rows with NaN values")

print(f"   Final shape: {df.shape}")

all_features = [
    'log_return',
    'yz_vol',
    'volume_zscore',
    'momentum',
    'parkinson_vol',
    'rsi',
    'bb_width',
    'ma_cross',
    'volume_ratio',
    'macd_line',
    'macd_signal',
    'macd_hist',
]
print(f"   Total features: {len(all_features)}")

# Set 1: Basic (your original)
basic_features = ['log_return', 'yz_vol', 'volume_zscore', 'momentum']
print(f"\n   Set 1 — Basic (4 features): {basic_features}")

# Set 2: Improved (add volatility + momentum)
improved_features = ['log_return', 'volume_zscore', 'momentum',
                      'parkinson_vol', 'rsi', 'ma_cross']
print(f"   Set 2 — Improved (7 features): {improved_features}")

# Set 3: Advanced (all non-redundant)
advanced_features = ['log_return', 'volume_zscore', 'momentum',
                      'parkinson_vol', 'rsi', 'bb_width', 'ma_cross', 'atr']
print(f"   Set 3 — Advanced (9 features): {advanced_features}")

# Set 4: Max (all features)
max_features = all_features
print(f"   Set 4 — Max ({len(max_features)} features): {max_features[:5]}...")

print("\n💾 Saving feature files...")

# Save ALL features (for exploration)
df.to_csv('all_features.csv')
print("   ✅ Saved: all_features.csv")

# Save the improved feature set (recommended)
improved_df = df[['Close'] + improved_features].copy()
improved_df.to_csv('features_improved.csv')
print(f"   ✅ Saved: features_improved.csv ({len(improved_features)} features)")

# Save your original feature set (for comparison)
original_df = df[['Close'] + basic_features].copy()
original_df.to_csv('features_original.csv')
print(f"   ✅ Saved: features_original.csv ({len(basic_features)} features)")

print("\n📊 Correlation Analysis...")

# Calculate correlation matrix
corr_matrix = df[improved_features].corr()

# Find highly correlated features (>0.8)
high_corr = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i, j]) > 0.8:
            high_corr.append((corr_matrix.columns[i], corr_matrix.columns[j],
                              corr_matrix.iloc[i, j]))

if high_corr:
    print("\n   ⚠️ Highly correlated features (>0.8):")
    for feat1, feat2, corr in high_corr:
        print(f"      {feat1} ↔ {feat2}: {corr:.3f}")
    print("\n   Consider dropping one from each pair to reduce redundancy.")
else:
    print("\n   ✅ No highly correlated features found.")

df.columns = df.columns.get_level_values(0)

from sklearn.preprocessing import RobustScaler
import joblib

basic_features = ['log_return', 'yz_vol', 'volume_zscore', 'momentum']

# Drop NaN before scaling
df_clean = df[basic_features].dropna().copy()

# Scale on clean data
scaler = RobustScaler()
df_scaled = pd.DataFrame(
    scaler.fit_transform(df_clean),
    columns=basic_features,
    index=df_clean.index
)

joblib.dump(scaler, "scaler.pkl")
pd.set_option('display.float_format', '{:.6f}'.format)
print(df_scaled.describe())

from sklearn.preprocessing import RobustScaler
import joblib

improved_features = ['log_return', 'volume_zscore', 'momentum',
                      'parkinson_vol', 'rsi', 'ma_cross']

# Drop NaN before scaling
df_clean = df[improved_features].dropna().copy()

# Scale on clean data
scaler = RobustScaler()
df_scaled = pd.DataFrame(
    scaler.fit_transform(df_clean),
    columns=improved_features,
    index=df_clean.index
)

joblib.dump(scaler, "scaler.pkl")
pd.set_option('display.float_format', '{:.6f}'.format)
print(df_scaled.describe())

from sklearn.preprocessing import RobustScaler
import joblib

advanced_features = ['log_return', 'volume_zscore', 'momentum',
                      'parkinson_vol', 'rsi', 'bb_width', 'ma_cross']

# Drop NaN before scaling
df_clean = df[advanced_features].dropna().copy()

# Scale on clean data
scaler = RobustScaler()
df_scaled = pd.DataFrame(
    scaler.fit_transform(df_clean),
    columns=advanced_features,
    index=df_clean.index
)

joblib.dump(scaler, "scaler.pkl")
pd.set_option('display.float_format', '{:.6f}'.format)
print(df_scaled.describe())
