import sys
import warnings
import logging
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder
from hmmlearn.hmm import GaussianHMM
import yfinance as yf

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*did not converge.*")
warnings.filterwarnings("ignore", message=".*Could not infer format.*")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)



PRICE_CSV          = "price_data.csv"
FEATURE_CSV        = "features_improved.csv"


FEATURES = [
    "log_return",
    "volume_zscore",
    "momentum",
    "parkinson_vol",
    "rsi",
    "ma_cross"
]

TRAIN_WINDOW       = 1260
STEP_SIZE          = 63

N_STATES           = 3
N_ITER             = 200
COVARIANCE_TYPE    = "diag"
RANDOM_STATE       = 42

TARGET_VOL         = 0.15
ROLLING_VOL_WINDOW = 20
ANNUALIZATION      = 252

STRATEGY_PARAMS = {
    'Bull': 2.0,
    'Chop': 1.5,
    'Bear': 0.6,
}

TRANS_COST_BPS     = 0.0002


def load_price_data(filepath: str) -> pd.DataFrame:
    log.info(f"Loading price data from '{filepath}' ...")
    try:
        df = pd.read_csv(
            filepath,
            skiprows=[1, 2],
            header=0,
            index_col=0,
            parse_dates=True,
        )
    except FileNotFoundError:
        log.error(f"Price file not found: {filepath}")
        raise

    df.columns = df.columns.str.strip().str.lower()
    close_candidates = [c for c in df.columns if "close" in c or "price" in c]
    if not close_candidates:
        raise ValueError(f"No close/price column found. Available: {list(df.columns)}")

    close_col = close_candidates[0]
    log.info(f"  Using column '{close_col}' as closing price.")

    df = df[[close_col]].rename(columns={close_col: "close"})
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["close"])
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df = df.dropna(subset=["log_return"])

    log.info(f"  Price data loaded: {len(df)} rows, "
             f"{df.index[0].date()} → {df.index[-1].date()}")
    return df


def load_feature_data(filepath: str) -> pd.DataFrame:
    log.info(f"Loading feature data from '{filepath}' ...")
    try:
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    except FileNotFoundError:
        log.error(f"Feature file not found: {filepath}")
        log.info("  Please run feature_engineering.py first!")
        raise

    df.columns = df.columns.str.strip().str.lower()
    df.index = pd.to_datetime(df.index, errors='coerce')
    df = df[df.index.notnull()]
    df = df.sort_index()

    log.info(f"  Feature data loaded: {len(df)} rows, columns={list(df.columns)}")
    return df


def fetch_vix(start: str, end: str) -> pd.Series:
    log.info(f"Fetching VIX data via yfinance ({start} → {end}) ...")
    try:
        vix = yf.download('^VIX', start=start, end=end, progress=False)

        # Handle different yfinance output formats
        if isinstance(vix.columns, pd.MultiIndex):
            vix_close = vix['Close']['^VIX']
        else:
            vix_close = vix['Close']

        vix_series = vix_close.rename("vix").ffill()
        log.info(f"  VIX loaded: {len(vix_series)} rows.")
        return vix_series
    except Exception as e:
        log.error(f"  VIX fetch failed: {e}")
        # Return empty series with proper index
        return pd.Series(name="vix")


def build_master_dataframe(
    price_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    vix_data: pd.Series,
) -> pd.DataFrame:
    log.info("Merging all data sources into master dataframe ...")

    master = price_df.copy()

 
    cols_to_drop = []
    for col in ['close', 'log_return']:
        if col in feature_df.columns:
            cols_to_drop.append(col)

    if cols_to_drop:
        feature_df = feature_df.drop(columns=cols_to_drop)
        log.info(f"  Dropped duplicate columns from feature data: {cols_to_drop}")

    # --- Merge technical features ---
    master = master.join(feature_df, how="left")

    # --- Merge VIX ---
    master = master.join(vix_data, how="left")
    master["vix"] = master["vix"].ffill().bfill()

    # --- Check available features ---
    available_features = [f for f in FEATURES if f in master.columns]
    missing = [f for f in FEATURES if f not in master.columns]

    if missing:
        log.warning(f"  Missing features: {missing}")
        if available_features:
            log.info(f"  Using available features: {available_features}")
        else:
            raise KeyError(f"Missing required features: {missing}")

    # --- Drop any rows where features are NaN ---
    initial_len = len(master)
    master = master.dropna(subset=available_features)
    dropped = initial_len - len(master)
    if dropped:
        log.info(f"  Dropped {dropped} rows containing NaN in features.")

    log.info(
        f"  Master dataframe ready: {len(master)} rows, "
        f"{master.index[0].date()} → {master.index[-1].date()}"
    )
    log.info(f"  Features: {available_features}")
    return master



def winsorize_fit(data: np.ndarray, lower_pct: float = 1.0, upper_pct: float = 99.0):
    lower_bounds = np.percentile(data, lower_pct, axis=0)
    upper_bounds = np.percentile(data, upper_pct, axis=0)
    return lower_bounds, upper_bounds


def winsorize_transform(
    data: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
) -> np.ndarray:
    return np.clip(data, lower_bounds, upper_bounds)


def label_regimes(model: GaussianHMM) -> dict:
    mean_returns = model.means_[:, 0]
    sorted_states = np.argsort(mean_returns)

    regime_map = {
        int(sorted_states[0]):  "Bear",
        int(sorted_states[1]):  "Chop",
        int(sorted_states[-1]): "Bull",
    }
    return regime_map


def run_walk_forward(master: pd.DataFrame) -> pd.DataFrame:
    n = len(master)
    oos_results = []

    n_steps = (n - TRAIN_WINDOW) // STEP_SIZE
    if n_steps <= 0:
        raise ValueError(f"Not enough data. Need >{TRAIN_WINDOW} rows, have {n}.")

    log.info(f"Starting walk-forward: {n_steps} steps, "
             f"train_window={TRAIN_WINDOW}, step_size={STEP_SIZE}")

    # Use only features that exist
    actual_features = [f for f in FEATURES if f in master.columns]
    if not actual_features:
        raise ValueError("No features found in master dataframe!")

    log.info(f"  Using features: {actual_features}")
    feature_array = master[actual_features].values

    for step in range(n_steps):
        train_end = TRAIN_WINDOW + step * STEP_SIZE
        oos_start = train_end
        oos_end = min(oos_start + STEP_SIZE, n)

        train_data = feature_array[:train_end]
        oos_data = feature_array[oos_start:oos_end]

        if len(oos_data) == 0:
            break

        lower_bounds, upper_bounds = winsorize_fit(train_data, 1.0, 99.0)
        train_wins = winsorize_transform(train_data, lower_bounds, upper_bounds)
        oos_wins = winsorize_transform(oos_data, lower_bounds, upper_bounds)

        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_wins)
        oos_scaled = scaler.transform(oos_wins)

        try:
            model = GaussianHMM(
                n_components=N_STATES,
                covariance_type=COVARIANCE_TYPE,
                n_iter=N_ITER,
                random_state=RANDOM_STATE,
                verbose=False,
            )
            model.fit(train_scaled)
        except Exception as exc:
            log.warning(f"  Step {step+1}/{n_steps}: HMM fit failed — {exc}")
            continue

        try:
            oos_states = model.predict(oos_scaled)
        except Exception as exc:
            log.warning(f"  Step {step+1}/{n_steps}: HMM predict failed — {exc}")
            continue

        regime_map = label_regimes(model)
        oos_regimes = [regime_map[s] for s in oos_states]

        oos_idx = master.index[oos_start:oos_end]
        oos_chunk = master.iloc[oos_start:oos_end][actual_features].copy()
        oos_chunk["state"] = oos_states
        oos_chunk["regime"] = oos_regimes

        oos_results.append(oos_chunk)

        if (step + 1) % 10 == 0 or step == n_steps - 1:
            log.info(f"  Completed step {step+1}/{n_steps} — "
                    f"OOS: {oos_idx[0].date()} → {oos_idx[-1].date()}")

    if not oos_results:
        raise RuntimeError("Walk-forward loop produced no OOS results.")

    # Print final model emission means
    print("\n" + "="*65)
    print("  FINAL MODEL EMISSION MEANS (Z-SCORES)")
    print("="*65)

    feature_names = actual_features
    header = f"{'Regime':<10} |"
    for feat in feature_names:
        header += f" {feat[:10]:>12} |"
    print("-" * (20 + 14 * len(feature_names)))
    print(header)
    print("-" * (20 + 14 * len(feature_names)))

    for target_regime in ["Bear", "Chop", "Bull"]:
        state_idx = next(state for state, name in regime_map.items() if name == target_regime)
        means = model.means_[state_idx]
        row = f"{target_regime:<10} |"
        for i in range(len(feature_names)):
            row += f" {means[i]:>12.2f} |"
        print(row)
    print("-" * (20 + 14 * len(feature_names)))

    oos_df = pd.concat(oos_results)
    log.info(f"Walk-forward complete. OOS: {len(oos_df)} rows")
    return oos_df




def compute_positions(oos_df: pd.DataFrame) -> pd.DataFrame:
    df = oos_df.copy()

    df["realized_vol"] = (
        df["log_return"]
        .rolling(window=ROLLING_VOL_WINDOW, min_periods=5)
        .std()
        * np.sqrt(ANNUALIZATION)
    )

    df["realized_vol"] = df["realized_vol"].replace(0.0, np.nan)
    df["realized_vol"] = df["realized_vol"].ffill().bfill()
    df["base_exposure"] = TARGET_VOL / df["realized_vol"]

    def size_position(row):
        base = row["base_exposure"]
        regime = row["regime"]
        if pd.isna(base):
            return 0.0
        if regime == "Bull":
            return float(np.clip(base, 0.0, STRATEGY_PARAMS['Bull']))
        elif regime == "Chop":
            return float(np.clip(base, 0.0, STRATEGY_PARAMS['Chop']))
        else:
            return STRATEGY_PARAMS['Bear']

    df["raw_position"] = df.apply(size_position, axis=1)
    df["position"] = df["raw_position"].shift(1).fillna(0.0)

    log.info("Position sizing complete.")
    log.info(f"  Avg Position: {df['position'].abs().mean():.3f}")
    log.info(f"  Max Position: {df['position'].abs().max():.3f}")
    return df


# Backtesting

def run_backtest(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["strategy_gross_ret"] = df["position"] * df["log_return"]
    df["turnover"] = df["position"].diff().abs().fillna(0.0)
    df["trans_cost"] = TRANS_COST_BPS * df["turnover"]
    df["strategy_net_ret"] = df["strategy_gross_ret"] - df["trans_cost"]
    df["bah_ret"] = df["log_return"]

    df["strategy_cum"] = (1 + df["strategy_net_ret"]).cumprod()
    df["bah_cum"] = (1 + df["bah_ret"]).cumprod()

    log.info("Backtest complete.")
    return df


# Performance metrics

def max_drawdown(cum_returns: pd.Series) -> float:
    running_max = cum_returns.cummax()
    drawdown = (cum_returns - running_max) / running_max
    return float(drawdown.min())


def compute_performance_metrics(
    daily_returns: pd.Series,
    cum_wealth: pd.Series,
    label: str,
) -> dict:
    n_days = len(daily_returns)

    ann_return = (cum_wealth.iloc[-1] ** (ANNUALIZATION / n_days)) - 1.0
    ann_vol = daily_returns.std() * np.sqrt(ANNUALIZATION)
    sharpe = ann_return / ann_vol if ann_vol != 0 else np.nan
    mdd = max_drawdown(cum_wealth)

    return {
        "label": label,
        "annualised_return": ann_return,
        "annualised_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": mdd,
    }


def print_performance_report(results: list[dict], oos_df: pd.DataFrame) -> None:
    separator = "=" * 65

    print()
    print(separator)
    print("  OOS PERFORMANCE REPORT (Walk-Forward, No In-Sample Data)")
    print(separator)
    print(f"  {'Metric':<28} {'HMM Strategy':>16} {'Buy & Hold':>16}")
    print("-" * 65)

    metrics = [
        ("Annualised Return",     "annualised_return",  ".2%"),
        ("Annualised Volatility", "annualised_vol",     ".2%"),
        ("Sharpe Ratio",          "sharpe",             ".4f"),
        ("Maximum Drawdown",      "max_drawdown",       ".2%"),
    ]

    strategy_r, bah_r = results[0], results[1]

    for metric_label, key, fmt in metrics:
        s_val = f"{strategy_r[key]:{fmt}}" if not np.isnan(strategy_r[key]) else "N/A"
        b_val = f"{bah_r[key]:{fmt}}" if not np.isnan(bah_r[key]) else "N/A"
        print(f"  {metric_label:<26} {s_val:>16} {b_val:>16}")

    print(separator)

    # Regime occupancy
    print("\n  REGIME OCCUPANCY (OOS)")
    print("-" * 65)

    regime_counts = oos_df["regime"].value_counts()
    regime_pcts = oos_df["regime"].value_counts(normalize=True) * 100

    for regime in ["Bull", "Chop", "Bear"]:
        if regime in regime_counts:
            days = regime_counts[regime]
            pct = regime_pcts[regime]
            print(f"  {regime:<10} : {days:>5} days ({pct:>5.1f}%)")
        else:
            print(f"  {regime:<10} :     0 days (  0.0%)")

    print(separator)

    # Silhouette Score
    print("\n  CLUSTERING METRICS")
    print("-" * 65)

    try:
        actual_features = [f for f in FEATURES if f in oos_df.columns]
        if actual_features:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(oos_df[actual_features])
            le = LabelEncoder()
            labels = le.fit_transform(oos_df['regime'])
            sil_score = silhouette_score(X_scaled, labels)
            print(f"  Silhouette Score              : {sil_score:>7.4f}")
        else:
            print("  Silhouette Score              : N/A")
    except Exception as e:
        print(f"  Silhouette Score              : Error")

    # Directional Accuracy
    oos_df_eval = oos_df.copy()
    oos_df_eval['next_day_return'] = oos_df_eval['log_return'].shift(-1)

    bull_correct = ((oos_df_eval['regime'] == 'Bull') & (oos_df_eval['next_day_return'] > 0)).sum()
    bear_correct = ((oos_df_eval['regime'] == 'Bear') & (oos_df_eval['next_day_return'] < 0)).sum()
    total_directional_calls = (oos_df_eval['regime'] == 'Bull').sum() + (oos_df_eval['regime'] == 'Bear').sum()

    dir_accuracy = (bull_correct + bear_correct) / total_directional_calls if total_directional_calls > 0 else 0
    print(f"  Directional Hit Rate (Proxy) : {dir_accuracy:>7.2%}")

    # Trade Win Rate
    winning_days = (oos_df['strategy_net_ret'] > 0).sum() if 'strategy_net_ret' in oos_df.columns else 0
    active_days = (oos_df['position'] != 0).sum() if 'position' in oos_df.columns else 0
    win_rate = winning_days / active_days if active_days > 0 else 0
    print(f"  Trade Win Rate               : {win_rate:>7.2%}")

    print(separator)

    # Features used
    print("\n  FEATURES USED")
    print("-" * 65)
    actual_features = [f for f in FEATURES if f in oos_df.columns]
    for i, feat in enumerate(actual_features, 1):
        print(f"  {i}. {feat}")
    print(separator)

    # Strategy Parameters
    print("\n  STRATEGY PARAMETERS")
    print("-" * 65)
    for regime, cap in STRATEGY_PARAMS.items():
        print(f"  {regime}: {cap:.1f}x")
    print(f"  Transaction Cost: {TRANS_COST_BPS*100:.2f}% per trade")
    print(separator)
    print()

# Main Entry Point

def main():
    log.info("=" * 65)
    log.info("  HMM REGIME TRADING STRATEGY — PIPELINE START")
    log.info("=" * 65)

    log.info("\n[STAGE 1] Data Ingestion & Feature Engineering")
    log.info("-" * 50)

    price_df = load_price_data(PRICE_CSV)
    feature_df = load_feature_data(FEATURE_CSV)

    start_date = str(price_df.index[0].date())
    end_date = str(price_df.index[-1].date())

    vix_data = fetch_vix(start_date, end_date)

    master = build_master_dataframe(price_df, feature_df, vix_data)

    min_rows = TRAIN_WINDOW + STEP_SIZE
    if len(master) < min_rows:
        raise ValueError(
            f"Master dataframe has only {len(master)} rows; "
            f"need at least {min_rows} for walk-forward validation."
        )

    log.info("\n[STAGE 2] Walk-Forward Validation (Expanding Window)")
    log.info("-" * 50)
    oos_df = run_walk_forward(master)

    log.info("\n[STAGE 3] Volatility-Targeting Position Sizing")
    log.info("-" * 50)
    positioned_df = compute_positions(oos_df)

    log.info("\n[STAGE 4] Backtesting & Transaction Cost Application")
    log.info("-" * 50)
    results_df = run_backtest(positioned_df)

    log.info("\n[STAGE 5] Performance Metrics Computation")
    log.info("-" * 50)

    clean = results_df.dropna(subset=["strategy_net_ret", "bah_ret"])

    strategy_metrics = compute_performance_metrics(
        daily_returns=clean["strategy_net_ret"],
        cum_wealth=clean["strategy_cum"].dropna(),
        label="HMM Strategy",
    )

    bah_metrics = compute_performance_metrics(
        daily_returns=clean["bah_ret"],
        cum_wealth=clean["bah_cum"].dropna(),
        label="Buy & Hold",
    )

    print_performance_report([strategy_metrics, bah_metrics], clean)

    # Save results
    output_path = "hmm_oos_results.csv"
    results_df.to_csv(output_path)
    log.info(f"Full OOS results saved to '{output_path}'")

    log.info("Pipeline complete. ✓")


if __name__ == "__main__":
    main()

