"""
Usage:
      Type "python Cross_Asset_Calculations.py SPY/GLD/TLT" on terminal for different assest such as SPY/TLT/GLD use different keywords after typing python Cross_Asset_Calculations.
Produces:
    price_data.csv     — raw OHLCV (unchanged from before)
    feature_clean.csv  — RAW features, no scaling applied
"""

import sys
import logging

import numpy as np
import pandas as pd
import yfinance as yf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

# Usage: python feature_engineering.py [TICKER] [START_DATE]
# Defaults per-asset: BTC-USD actually has usable history back to
# 2014-09-17 on Yahoo Finance (not "2018" or "2005" -- that's just
# boilerplate copied from an equity-ETF script). Using the fuller history
# roughly doubles BTC's usable sample versus starting at 2018.
DEFAULT_START = {
    "SPY": "2005-01-01",
    "TLT": "2005-01-01",
    "GLD": "2005-01-01",
    "BTC-USD": "2014-09-17",
    "BTC": "2014-09-17",
}
END_DATE = "2024-12-31"

TICKER = sys.argv[1].upper() if len(sys.argv) > 1 else "SPY"
YF_TICKER = "BTC-USD" if TICKER == "BTC" else TICKER
START_DATE = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_START.get(TICKER, "2005-01-01")

ROLL_WINDOW_VOL = 20
ROLL_WINDOW_MOM = 5
YZ_WINDOW = 20

FEATURE_COLS = [
    "log_return",
    "vol_zscore",
    "volume_zscore",
    "momentum",
]


def yang_zhang(df: pd.DataFrame, window: int = YZ_WINDOW) -> pd.Series:
    """Yang-Zhang volatility estimator. Trailing rolling windows only — causal."""
    o = np.log(df["Open"] / df["Close"].shift(1))   # overnight return
    c = np.log(df["Close"] / df["Open"])             # close-to-open return
    h = np.log(df["High"] / df["Open"])
    l = np.log(df["Low"] / df["Open"])

    k = 0.34 / (1.34 + (window + 1) / (window - 1))

    cc = (h * (h - c) + l * (l - c)).rolling(window).mean()
    oc = o.rolling(window).var()
    co = c.rolling(window).var()

    return np.sqrt(oc + k * co + (1 - k) * cc)


def download_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame:
    log.info(f"Downloading {ticker} {start} -> {end} ...")
    df = yf.download(ticker, start=start, end=end, progress=False)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[~df.index.duplicated(keep="first")]
    df = df.sort_index()

    bad_close = (df["Close"] <= 0).sum()
    bad_vol = (df["Volume"] <= 0).sum()
    if bad_close or bad_vol:
        log.warning(f"  Found {bad_close} non-positive close rows, "
                    f"{bad_vol} non-positive volume rows.")

    log.info(f"  Downloaded {len(df)} rows, {df.index[0].date()} -> {df.index[-1].date()}")
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    All windows below are TRAILING (backward-looking) only. No center=True.
    No global fit/transform of any kind happens in this function — every
    value at row t is computable using only data up to and including t.
    """
    df = df.copy()

    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    df["yz_vol"] = yang_zhang(df, YZ_WINDOW)
    vol_window = df["yz_vol"].rolling(ROLL_WINDOW_VOL)
    vol_mean = vol_window.mean()
    vol_std = vol_window.std().replace(0, np.nan)
    df["vol_zscore"] = (df["yz_vol"] - vol_mean) / vol_std
    df["momentum"] = df["log_return"].rolling(ROLL_WINDOW_MOM).mean()
    df["volume_zscore"] = (
        (df["Volume"] - df["Volume"].rolling(ROLL_WINDOW_VOL).mean())
        / df["Volume"].rolling(ROLL_WINDOW_VOL).std()
    )

    before = len(df)
    df = df.dropna(subset=FEATURE_COLS)
    log.info(f"  Dropped {before - len(df)} warm-up rows with NaN features "
             f"({len(df)} rows remain).")
    return df


def stationarity_check(df: pd.DataFrame) -> None:
    """Informational only — does not feed the model. Full-sample ADF is fine
    here purely as an EDA sanity check, never as a modeling input."""
    try:
        from statsmodels.tsa.stattools import adfuller
    except ImportError:
        log.warning("  statsmodels not available, skipping ADF check.")
        return

    log.info("  ADF stationarity check (informational only):")
    for feature in FEATURE_COLS:
        result = adfuller(df[feature].dropna())
        log.info(f"    {feature}: p-value = {result[1]:.4f}")


def main():
    log.info("=" * 65)
    log.info(f"  FEATURE ENGINEERING — FIXED (no global scaling) — {TICKER} "
             f"(yfinance symbol: {YF_TICKER}), start={START_DATE}")
    log.info("=" * 65)

    raw = download_ohlcv(YF_TICKER, START_DATE, END_DATE)
    feat_df = build_features(raw)
    stationarity_check(feat_df)

    # RAW features only. No scaler.fit anywhere in this file.
    out = feat_df[FEATURE_COLS].copy()
    out.to_csv(f"feature_clean_{TICKER}.csv")
    log.info(f"  Wrote feature_clean_{TICKER}.csv (RAW, unscaled): shape={out.shape}")

    raw[["Open", "High", "Low", "Close", "Volume"]].to_csv(f"price_data_{TICKER}.csv")
    log.info(f"  Wrote price_data_{TICKER}.csv: shape={raw.shape}")

    log.info("Done. No global scaler fit — scaling happens per-fold in "
             "hmm_production.py, never here.")


if __name__ == "__main__":
    main()
"""
hmm_production.py — FIXED VERSION
==================================

CHANGES FROM THE ORIGINAL, AND WHY
-----------------------------------
1. FEATURES list now matches what feature_engineering.py actually produces
   (log_return, vol_zscore, volume_zscore, momentum). The old list asked
   for parkinson_vol/rsi/ma_cross, which never existed, and it also kept
   noisy return-skew/kurtosis inputs that were destabilizing the HMM.

2. model.predict() (Viterbi decode over the whole 63-row OOS block at once)
   is REPLACED by causal_forward_filter(): a hand-rolled forward algorithm
   that computes the filtered state at time t using ONLY observations
   0..t. Viterbi decodes the most probable JOINT path across the full
   block, meaning day 1's assigned state could depend on day 63's
   observation. That is look-ahead leakage. The forward filter cannot see
   forward by construction.

3. The EM convergence warning is no longer suppressed. Each fold now fits
   N_INIT restarts (different seeds) and keeps the best log-likelihood fit,
   and logs whether that fit converged. A summary of convergence failures
   across all folds is printed at the end. Previously this was silently
   swallowed by a warnings.filterwarnings() call and there was zero
   visibility into it.

4. label_regimes() / regime naming is now K-agnostic (generate_regime_labels)
   so this script and the new K-sweep tool work for N_STATES in {2,3,4,...},
   not just exactly 3. The old code would throw or mislabel with K != 3.

5. run_model_order_selection() sweeps K in {2,3,4} with BIC/AIC + held-out
   log-likelihood on two representative windows (early vs. late history) as
   a PRE-FLIGHT diagnostic. It reports a recommendation; it does NOT
   silently change N_STATES for you. You decide.

6. Per-fold silhouette scores are now reported alongside the old pooled
   silhouette. The pooled number mixes labels from ~59 independently-fit
   models with different per-fold feature scaling into one clustering
   metric — that conflates "do regimes separate" with "are regimes
   consistent across refits", which are different questions.

7. Winsorize + StandardScaler fit-on-train-only logic is UNCHANGED — that
   part of the original script was already correct and is preserved as-is.

STILL NOT FIXED (out of scope for this pass, flagged for later):
- Transaction costs still flat bps, no financing cost on leveraged notional.
- Bear regime sizing is still a flat exposure (not vol-scaled), while
  Bull/Chop are vol-scaled and capped — this asymmetry was in the original
  code and is preserved here unchanged. Worth a deliberate decision, not a
  silent fix.
- Posterior margin/entropy are computed and exposed as diagnostic columns
  but NOT yet wired into position sizing. That's the anomaly-detection /
  de-leveraging step from the roadmap — next after this.
"""

import sys
import warnings
import logging

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score
from hmmlearn.hmm import GaussianHMM
import yfinance as yf

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*Could not infer format.*")
# NOTE: the "did not converge" warning is intentionally NOT suppressed anymore.

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Asset selection: `python hmm_production.py TICKER`, defaults to SPY.
# All output filenames are suffixed by ticker so runs for different assets
# never overwrite each other. BTC trades every calendar day (crypto has no
# weekends/holidays), unlike equity/bond/commodity ETFs -- annualization
# factor must reflect that or Sharpe/vol comparisons across assets are
# apples-to-oranges.
# ---------------------------------------------------------------------------
TICKER = sys.argv[1].upper() if len(sys.argv) > 1 else "SPY"
ASSET_ANNUALIZATION = {"BTC": 365, "BTC-USD": 365}
ASSET_CAL_DAYS_PER_YEAR = ASSET_ANNUALIZATION.get(TICKER, 252)

PRICE_CSV   = f"price_data_{TICKER}.csv"
FEATURE_CSV = f"feature_clean_{TICKER}.csv"

# Aligned with feature_engineering.py's actual output columns.
FEATURES = [
    "log_return",
    "vol_zscore",
    "volume_zscore",
    "momentum",
]

TRAIN_WINDOW = 1260
STEP_SIZE    = 63

N_STATES        = 3        # Run run_model_order_selection() before trusting this.
N_INIT          = 5        # EM restarts per fold; keep best log-likelihood fit.
N_ITER          = 200
COVARIANCE_TYPE = "diag"
BASE_SEED       = 42

TARGET_VOL         = 0.15
ROLLING_VOL_WINDOW = 20
ANNUALIZATION       = ASSET_CAL_DAYS_PER_YEAR

LEVERAGE_BEAR = 0.6
LEVERAGE_CHOP = 1.5   # applied to "Chop" and any "Chop_i" (K > 3)
LEVERAGE_BULL = 2.0

TRANS_COST_BPS = 0.0002


# ---------------------------------------------------------------------------
# Data loading (unchanged from original)
# ---------------------------------------------------------------------------

def load_price_data(filepath: str) -> pd.DataFrame:
    log.info(f"Loading price data from '{filepath}' ...")
    df = pd.read_csv(filepath, skiprows=[1, 2], header=0, index_col=0, parse_dates=True)
    df.columns = df.columns.str.strip().str.lower()
    close_candidates = [c for c in df.columns if "close" in c or "price" in c]
    if not close_candidates:
        raise ValueError(f"No close/price column found. Available: {list(df.columns)}")
    close_col = close_candidates[0]
    df = df[[close_col]].rename(columns={close_col: "close"})
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["close"]).sort_index()
    df.index = pd.to_datetime(df.index)
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df = df.dropna(subset=["log_return"])
    log.info(f"  Price data loaded: {len(df)} rows, {df.index[0].date()} -> {df.index[-1].date()}")
    return df


def load_feature_data(filepath: str) -> pd.DataFrame:
    log.info(f"Loading feature data from '{filepath}' ...")
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    df.columns = df.columns.str.strip().str.lower()
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[df.index.notnull()].sort_index()
    log.info(f"  Feature data loaded: {len(df)} rows, columns={list(df.columns)}")
    return df


def build_master_dataframe(price_df: pd.DataFrame, feature_df: pd.DataFrame) -> pd.DataFrame:
    log.info("Merging data sources into master dataframe ...")
    master = price_df.copy()

    cols_to_drop = [c for c in ["close", "log_return"] if c in feature_df.columns]
    if cols_to_drop:
        feature_df = feature_df.drop(columns=cols_to_drop)
        log.info(f"  Dropped duplicate columns from feature data: {cols_to_drop}")

    master = master.join(feature_df, how="left")

    available = [f for f in FEATURES if f in master.columns]
    missing = [f for f in FEATURES if f not in master.columns]
    if missing:
        log.warning(f"  Missing features (will not be used): {missing}")
    if not available:
        raise KeyError(f"None of the required features found: {FEATURES}")

    before = len(master)
    master = master.dropna(subset=available)
    if before - len(master):
        log.info(f"  Dropped {before - len(master)} rows with NaN in features.")

    log.info(f"  Master dataframe ready: {len(master)} rows, "
             f"{master.index[0].date()} -> {master.index[-1].date()}. Features: {available}")
    return master


def winsorize_fit(data: np.ndarray, lower_pct=1.0, upper_pct=99.0):
    return np.percentile(data, lower_pct, axis=0), np.percentile(data, upper_pct, axis=0)


def winsorize_transform(data, lower, upper):
    return np.clip(data, lower, upper)


# ---------------------------------------------------------------------------
# Causal forward filter — replaces model.predict() (Viterbi block decode)
# ---------------------------------------------------------------------------

def _gaussian_diag_logpdf(X: np.ndarray, means: np.ndarray, covars: np.ndarray) -> np.ndarray:
    """
    Log-likelihood of independent (diagonal covariance) Gaussian emissions.
    X: (T, D) observations
    means: (K, D)
    covars: (K, D) diagonal variances
    Returns (T, K) log emission probability matrix.
    """
    T, D = X.shape
    K = means.shape[0]
    logB = np.empty((T, K))
    for k in range(K):
        var = np.maximum(covars[k], 1e-10)
        diff = X - means[k]
        logB[:, k] = -0.5 * np.sum(np.log(2 * np.pi * var) + (diff ** 2) / var, axis=1)
    return logB


def causal_forward_filter(model: GaussianHMM, X: np.ndarray):
    """
    Strictly causal state filtering via the forward algorithm.

    State estimate at time t depends ONLY on X[0..t]. This is the fix for
    the leakage bug: model.predict(X) runs Viterbi over the WHOLE array at
    once, which is a joint decode that lets early timesteps "see" later
    observations in the same block. The forward recursion below cannot,
    by construction — log_alpha[t] is built only from log_alpha[t-1] and
    the observation at t.

    Returns:
        states: (T,) argmax filtered state at each t
        probs:  (T, K) normalized filtered posterior P(state_t | X[0:t+1])
    """
    T = X.shape[0]
    K = model.n_components

    log_startprob = np.log(np.clip(model.startprob_, 1e-300, 1.0))
    log_transmat = np.log(np.clip(model.transmat_, 1e-300, 1.0))

    covars = model.covars_
    if covars.ndim == 3:  # some hmmlearn versions store diag as (K, D, D)
        covars = np.array([np.diag(c) for c in covars])

    logB = _gaussian_diag_logpdf(X, model.means_, covars)

    log_alpha = np.empty((T, K))
    log_alpha[0] = log_startprob + logB[0]
    for t in range(1, T):
        log_alpha[t] = logsumexp(log_alpha[t - 1][:, None] + log_transmat, axis=0) + logB[t]

    log_norm = logsumexp(log_alpha, axis=1, keepdims=True)
    probs = np.exp(log_alpha - log_norm)
    states = np.argmax(log_alpha, axis=1)
    return states, probs


# ---------------------------------------------------------------------------
# Regime labeling — generalized to arbitrary K (was hardcoded to K=3)
# ---------------------------------------------------------------------------

def generate_regime_labels(k: int) -> list:
    if k == 1:
        return ["All"]
    if k == 2:
        return ["Bear", "Bull"]
    if k == 3:
        return ["Bear", "Chop", "Bull"]
    n_mid = k - 2
    return ["Bear"] + [f"Chop_{i+1}" for i in range(n_mid)] + ["Bull"]


def label_regimes(model: GaussianHMM, return_idx: int = 0) -> dict:
    mean_returns = model.means_[:, return_idx]
    sorted_states = np.argsort(mean_returns)
    labels = generate_regime_labels(len(sorted_states))
    return {int(sorted_states[i]): labels[i] for i in range(len(sorted_states))}


def get_leverage_cap(regime_label: str) -> float:
    if regime_label == "Bull":
        return LEVERAGE_BULL
    if regime_label == "Bear":
        return LEVERAGE_BEAR
    return LEVERAGE_CHOP  # Chop / Chop_i


# ---------------------------------------------------------------------------
# Model fitting with multi-restart + convergence tracking
# ---------------------------------------------------------------------------

def _genuinely_converged(model: GaussianHMM) -> bool:
    """
    hmmlearn's own model.monitor_.converged property is misleading:
        return (self.iter == self.n_iter or
                (len(self.history) >= 2 and
                 self.history[-1] - self.history[-2] < self.tol))
    The first clause means "ran out of iterations" ALSO reports as
    converged=True, even if the tolerance criterion was never satisfied.
    Verified directly: a model fit with n_iter=1 reports .converged == True
    after a single EM step, purely because iter (1) == n_iter (1) -- there
    is no basis in a single log-likelihood value to claim convergence.

    This checks the real signal instead: did the log-likelihood improvement
    on the last step fall below tol, independent of whether the iteration
    budget was also exhausted at the same time.
    """
    mon = model.monitor_
    if mon.iter < mon.n_iter:
        return True  # stopped early -- only happens if EM's own internal
                      # check already found the true-convergence clause
    return len(mon.history) >= 2 and (mon.history[-1] - mon.history[-2]) < mon.tol


def fit_best_of_n(X: np.ndarray, n_states: int, n_init: int = N_INIT,
                   n_iter: int = N_ITER, covariance_type: str = COVARIANCE_TYPE,
                   base_seed: int = BASE_SEED):
    """
    Fits n_init random restarts, keeps the highest-log-likelihood model.
    Returns (best_model, best_score, n_converged_out_of_n_init).
    """
    best_model, best_score, n_converged = None, -np.inf, 0
    for i in range(n_init):
        m = GaussianHMM(n_components=n_states, covariance_type=covariance_type,
                         n_iter=n_iter, random_state=base_seed + i, verbose=False)
        try:
            m.fit(X)
            score = m.score(X)
        except Exception as exc:
            log.warning(f"    init {i}: fit failed — {exc}")
            continue
        if _genuinely_converged(m):
            n_converged += 1
        if score > best_score:
            best_score, best_model = score, m
    return best_model, best_score, n_converged


# ---------------------------------------------------------------------------
# Model order selection (BIC/AIC) — pre-flight diagnostic, K-agnostic
# ---------------------------------------------------------------------------

def count_hmm_params(k: int, d: int) -> int:
    """Free parameters for a GaussianHMM with covariance_type='diag'."""
    startprob_params = k - 1
    transmat_params = k * (k - 1)
    means_params = k * d
    var_params = k * d
    return startprob_params + transmat_params + means_params + var_params


def run_model_order_selection(feature_array: np.ndarray, k_values=(2, 3, 4),
                               n_init: int = N_INIT) -> None:
    """
    Fits each K on an EARLY window and a LATE window (to check whether
    regime structure / optimal K is stable across history), scores each on
    the immediately-following held-out block, and prints BIC/AIC + held-out
    log-likelihood. This is advisory only — it recommends, it does not set
    N_STATES automatically.
    """
    n = len(feature_array)
    d = feature_array.shape[1]

    windows = {}
    if n >= TRAIN_WINDOW + STEP_SIZE:
        windows["EARLY"] = (feature_array[:TRAIN_WINDOW], feature_array[TRAIN_WINDOW:TRAIN_WINDOW + STEP_SIZE])
    late_train_end = n - STEP_SIZE
    if late_train_end >= TRAIN_WINDOW:
        windows["LATE"] = (feature_array[late_train_end - TRAIN_WINDOW:late_train_end], feature_array[late_train_end:n])

    print("\n" + "=" * 78)
    print("  MODEL ORDER SELECTION (pre-flight diagnostic — advisory only)")
    print("=" * 78)
    print(f"  {'Window':<8} {'K':>3} {'LogL(train)':>14} {'BIC':>12} {'AIC':>12} "
          f"{'LogL(holdout)':>15} {'Converged':>10}")
    print("-" * 78)

    for window_name, (train_data, holdout_data) in windows.items():
        lb, ub = winsorize_fit(train_data)
        train_w = winsorize_transform(train_data, lb, ub)
        holdout_w = winsorize_transform(holdout_data, lb, ub)
        scaler = StandardScaler()
        train_s = scaler.fit_transform(train_w)
        holdout_s = scaler.transform(holdout_w)

        for k in k_values:
            model, logL, n_conv = fit_best_of_n(train_s, k, n_init=n_init)
            if model is None:
                print(f"  {window_name:<8} {k:>3}   FIT FAILED")
                continue
            n_params = count_hmm_params(k, d)
            bic = -2 * logL + n_params * np.log(len(train_s))
            aic = -2 * logL + 2 * n_params
            try:
                holdout_logL = model.score(holdout_s)
            except Exception:
                holdout_logL = np.nan
            print(f"  {window_name:<8} {k:>3} {logL:>14.1f} {bic:>12.1f} {aic:>12.1f} "
                  f"{holdout_logL:>15.1f} {n_conv:>7}/{n_init}")
    print("=" * 78)
    print("  Lower BIC/AIC = better. Higher holdout LogL = better generalization.")
    print("  If EARLY and LATE disagree on the best K, regime structure is not")
    print("  stable across your sample -- a fixed N_STATES may be misspecified")
    print("  for part of your history.")
    print("=" * 78 + "\n")


# ---------------------------------------------------------------------------
# Walk-forward loop
# ---------------------------------------------------------------------------

def run_walk_forward(master: pd.DataFrame) -> pd.DataFrame:
    n = len(master)
    n_steps = (n - TRAIN_WINDOW) // STEP_SIZE
    if n_steps <= 0:
        raise ValueError(f"Not enough data. Need >{TRAIN_WINDOW} rows, have {n}.")

    actual_features = [f for f in FEATURES if f in master.columns]
    log.info(f"Starting walk-forward: {n_steps} steps, train_window={TRAIN_WINDOW}, "
             f"step_size={STEP_SIZE}, features={actual_features}")
    feature_array = master[actual_features].values

    oos_results = []
    fold_silhouettes = []
    fold_diagnostics = []   # per-fold emission means / self-transition / variances
    n_folds_fit = 0
    n_folds_converged = 0
    last_model, last_regime_map = None, None

    for step in range(n_steps):
        train_end = TRAIN_WINDOW + step * STEP_SIZE
        oos_start, oos_end = train_end, min(train_end + STEP_SIZE, n)
        if oos_end <= oos_start:
            break

        train_data = feature_array[:train_end]
        oos_data = feature_array[oos_start:oos_end]

        lb, ub = winsorize_fit(train_data)
        train_wins = winsorize_transform(train_data, lb, ub)
        oos_wins = winsorize_transform(oos_data, lb, ub)

        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_wins)
        oos_scaled = scaler.transform(oos_wins)

        model, _, n_conv = fit_best_of_n(train_scaled, N_STATES, n_init=N_INIT)
        if model is None:
            log.warning(f"  Step {step+1}/{n_steps}: all {N_INIT} restarts failed, skipping.")
            continue

        n_folds_fit += 1
        if n_conv == 0:
            log.warning(f"  Step {step+1}/{n_steps}: NONE of {N_INIT} restarts converged "
                        f"within N_ITER={N_ITER}.")
        else:
            n_folds_converged += 1

        oos_states, oos_probs = causal_forward_filter(model, oos_scaled)
        regime_map = label_regimes(model)
        oos_regimes = [regime_map[s] for s in oos_states]

        top_prob = oos_probs.max(axis=1)
        sorted_probs = np.sort(oos_probs, axis=1)
        margin = sorted_probs[:, -1] - sorted_probs[:, -2]
        with np.errstate(divide="ignore", invalid="ignore"):
            entropy = -np.sum(oos_probs * np.log(np.clip(oos_probs, 1e-12, 1)), axis=1)

        oos_chunk = master.iloc[oos_start:oos_end][actual_features].copy()
        oos_chunk["state"] = oos_states
        oos_chunk["regime"] = oos_regimes
        oos_chunk["posterior_top_prob"] = top_prob
        oos_chunk["posterior_margin"] = margin
        oos_chunk["posterior_entropy"] = entropy
        oos_results.append(oos_chunk)

        # Per-fold silhouette (skip folds with <2 distinct states present)
        if len(set(oos_states)) >= 2 and len(oos_states) > len(set(oos_states)):
            try:
                fold_silhouettes.append(silhouette_score(oos_scaled, oos_states))
            except Exception:
                pass

        last_model, last_regime_map = model, regime_map

        # Structural diagnostics: capture emission means/variances and
        # self-transition probability per regime for THIS fold's model.
        # Must be captured here — the fitted model object is not otherwise
        # retained after the loop moves on.
        oos_idx_fold = master.index[oos_start:oos_end]
        covars = model.covars_
        for state_idx, regime_name in regime_map.items():
            var_row = np.diag(covars[state_idx]) if covars.ndim == 3 else covars[state_idx]
            row = {
                "fold": step,
                "start_date": oos_idx_fold[0],
                "end_date": oos_idx_fold[-1],
                "regime": regime_name,
                "self_transition_prob": float(model.transmat_[state_idx, state_idx]),
            }
            for feat_name, mean_val, var_val in zip(actual_features, model.means_[state_idx], var_row):
                row[f"mean_{feat_name}"] = float(mean_val)
                row[f"var_{feat_name}"] = float(var_val)
            fold_diagnostics.append(row)

        if (step + 1) % 10 == 0 or step == n_steps - 1:
            oos_idx = master.index[oos_start:oos_end]
            log.info(f"  Step {step+1}/{n_steps} done — OOS {oos_idx[0].date()} -> {oos_idx[-1].date()}")

    if not oos_results:
        raise RuntimeError("Walk-forward loop produced no OOS results.")

    log.info(f"Convergence summary: {n_folds_converged}/{n_folds_fit} folds had >=1 "
             f"converged restart out of {N_INIT} attempted.")
    if n_folds_converged < n_folds_fit:
        log.warning(f"  {n_folds_fit - n_folds_converged} fold(s) had ZERO converged "
                    f"restarts -- treat those states with caution.")

    # Final model emission means (from the LAST fold's best model)
    print("\n" + "=" * 65)
    print("  FINAL MODEL EMISSION MEANS (Z-SCORES) — last fold")
    print("=" * 65)
    feature_names = actual_features
    labels_ordered = [name for _, name in sorted(last_regime_map.items(),
                                                   key=lambda kv: last_model.means_[kv[0], 0])]
    header = f"{'Regime':<10} |" + "".join(f" {f[:10]:>12} |" for f in feature_names)
    print("-" * len(header)); print(header); print("-" * len(header))
    for regime_name in labels_ordered:
        state_idx = next(s for s, name in last_regime_map.items() if name == regime_name)
        means = last_model.means_[state_idx]
        row = f"{regime_name:<10} |" + "".join(f" {m:>12.2f} |" for m in means)
        print(row)
    print("-" * len(header))

    oos_df = pd.concat(oos_results)
    oos_df.attrs["fold_silhouettes"] = fold_silhouettes
    oos_df.attrs["fold_diagnostics"] = fold_diagnostics
    oos_df.attrs["n_folds_fit"] = n_folds_fit
    oos_df.attrs["n_folds_converged"] = n_folds_converged
    log.info(f"Walk-forward complete. OOS: {len(oos_df)} rows")
    return oos_df


# ---------------------------------------------------------------------------
# Position sizing (logic preserved from original — Bear stays flat exposure,
# Bull/Chop are vol-target-scaled and capped. See module docstring.)
# ---------------------------------------------------------------------------

def compute_positions(oos_df: pd.DataFrame) -> pd.DataFrame:
    df = oos_df.copy()
    df["realized_vol"] = (
        df["log_return"].rolling(window=ROLLING_VOL_WINDOW, min_periods=5).std() * np.sqrt(ANNUALIZATION)
    )
    df["realized_vol"] = df["realized_vol"].replace(0.0, np.nan).ffill().bfill()
    df["base_exposure"] = TARGET_VOL / df["realized_vol"]

    def size_position(row):
        base = row["base_exposure"]
        regime = row["regime"]
        if pd.isna(base):
            return 0.0
        if regime == "Bear":
            return LEVERAGE_BEAR
        cap = get_leverage_cap(regime)
        return float(np.clip(base, 0.0, cap))

    df["raw_position"] = df.apply(size_position, axis=1)
    df["position"] = df["raw_position"].shift(1).fillna(0.0)

    log.info(f"Position sizing complete. Avg Position: {df['position'].abs().mean():.3f} | "
             f"Max Position: {df['position'].abs().max():.3f}")
    return df


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


def max_drawdown(cum_returns: pd.Series) -> float:
    running_max = cum_returns.cummax()
    return float(((cum_returns - running_max) / running_max).min())


def compute_performance_metrics(daily_returns: pd.Series, cum_wealth: pd.Series, label: str) -> dict:
    n_days = len(daily_returns)
    ann_return = (cum_wealth.iloc[-1] ** (ANNUALIZATION / n_days)) - 1.0
    ann_vol = daily_returns.std() * np.sqrt(ANNUALIZATION)
    sharpe = ann_return / ann_vol if ann_vol != 0 else np.nan
    return {"label": label, "annualised_return": ann_return, "annualised_vol": ann_vol,
            "sharpe": sharpe, "max_drawdown": max_drawdown(cum_wealth)}


def print_performance_report(results: list, oos_df: pd.DataFrame) -> dict:
    summary = {}
    sep = "=" * 65
    print(f"\n{sep}\n  OOS PERFORMANCE REPORT (Walk-Forward, Causal Filter)\n{sep}")
    print(f"  {'Metric':<28} {'HMM Strategy':>16} {'Buy & Hold':>16}")
    print("-" * 65)
    metrics = [("Annualised Return", "annualised_return", ".2%"),
               ("Annualised Volatility", "annualised_vol", ".2%"),
               ("Sharpe Ratio", "sharpe", ".4f"),
               ("Maximum Drawdown", "max_drawdown", ".2%")]
    s_r, b_r = results[0], results[1]
    for label, key, fmt in metrics:
        s_val = f"{s_r[key]:{fmt}}" if not np.isnan(s_r[key]) else "N/A"
        b_val = f"{b_r[key]:{fmt}}" if not np.isnan(b_r[key]) else "N/A"
        print(f"  {label:<26} {s_val:>16} {b_val:>16}")
    print(sep)
    summary["strategy_metrics"] = {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in s_r.items()}
    summary["bah_metrics"] = {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in b_r.items()}

    print("\n  REGIME OCCUPANCY (OOS)"); print("-" * 65)
    counts = oos_df["regime"].value_counts()
    pcts = oos_df["regime"].value_counts(normalize=True) * 100
    for regime in counts.index:
        print(f"  {regime:<10} : {counts[regime]:>5} days ({pcts[regime]:>5.1f}%)")
    print(sep)
    summary["regime_occupancy_pct"] = pcts.to_dict()
    summary["regime_occupancy_days"] = counts.to_dict()

    print("\n  CLUSTERING METRICS"); print("-" * 65)
    try:
        actual_features = [f for f in FEATURES if f in oos_df.columns]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(oos_df[actual_features])
        le = LabelEncoder()
        labels = le.fit_transform(oos_df["regime"])
        pooled_sil = silhouette_score(X_scaled, labels)
        print(f"  Silhouette Score (pooled, all folds)  : {pooled_sil:>7.4f}")
        summary["silhouette_pooled"] = float(pooled_sil)
    except Exception:
        print("  Silhouette Score (pooled)              : Error")
        summary["silhouette_pooled"] = None

    fold_sils = oos_df.attrs.get("fold_silhouettes", [])
    if fold_sils:
        print(f"  Silhouette Score (mean of per-fold)    : {np.mean(fold_sils):>7.4f}")
        print(f"  Silhouette Score (median of per-fold)  : {np.median(fold_sils):>7.4f}")
        print(f"  Per-fold silhouette range              : [{np.min(fold_sils):.4f}, {np.max(fold_sils):.4f}]")
        print(f"  Folds with valid silhouette             : {len(fold_sils)}/{oos_df.attrs.get('n_folds_fit', '?')}")
        summary["silhouette_fold_mean"] = float(np.mean(fold_sils))
        summary["silhouette_fold_median"] = float(np.median(fold_sils))
        summary["silhouette_fold_min"] = float(np.min(fold_sils))
        summary["silhouette_fold_max"] = float(np.max(fold_sils))
        summary["n_folds_with_valid_silhouette"] = len(fold_sils)
    else:
        print("  Per-fold silhouette: no fold had >=2 distinct states to score.")

    oos_eval = oos_df.copy()
    oos_eval["next_day_return"] = oos_eval["log_return"].shift(-1)
    bull_correct = ((oos_eval["regime"] == "Bull") & (oos_eval["next_day_return"] > 0)).sum()
    bear_correct = ((oos_eval["regime"] == "Bear") & (oos_eval["next_day_return"] < 0)).sum()
    total_calls = (oos_eval["regime"] == "Bull").sum() + (oos_eval["regime"] == "Bear").sum()
    dir_acc = (bull_correct + bear_correct) / total_calls if total_calls > 0 else 0
    print(f"  Directional Hit Rate (Proxy)            : {dir_acc:>7.2%}")
    summary["directional_hit_rate"] = float(dir_acc)

    winning = (oos_df["strategy_net_ret"] > 0).sum() if "strategy_net_ret" in oos_df.columns else 0
    active = (oos_df["position"] != 0).sum() if "position" in oos_df.columns else 0
    win_rate = winning / active if active > 0 else 0
    print(f"  Trade Win Rate                          : {win_rate:>7.2%}")
    print(sep)
    summary["trade_win_rate"] = float(win_rate)

    print("\n  MODEL ROBUSTNESS"); print("-" * 65)
    n_folds_fit = oos_df.attrs.get("n_folds_fit", None)
    n_folds_converged = oos_df.attrs.get("n_folds_converged", None)
    print(f"  Folds fit                               : {n_folds_fit}")
    print(f"  Folds with >=1 converged EM restart      : {n_folds_converged}")
    print(f"  Mean posterior confidence (top state)    : {oos_df['posterior_top_prob'].mean():>7.2%}")
    print(f"  Mean posterior margin (top - runner-up)  : {oos_df['posterior_margin'].mean():>7.2%}")
    print(f"  Mean posterior entropy                   : {oos_df['posterior_entropy'].mean():>7.3f}")
    print(sep)
    summary["n_folds_fit"] = n_folds_fit
    summary["n_folds_converged"] = n_folds_converged
    summary["mean_posterior_top_prob"] = float(oos_df["posterior_top_prob"].mean())
    summary["mean_posterior_margin"] = float(oos_df["posterior_margin"].mean())
    summary["mean_posterior_entropy"] = float(oos_df["posterior_entropy"].mean())

    print("\n  STRATEGY PARAMETERS"); print("-" * 65)
    print(f"  Bull: {LEVERAGE_BULL:.1f}x | Chop: {LEVERAGE_CHOP:.1f}x | Bear: {LEVERAGE_BEAR:.1f}x (flat, not vol-scaled)")
    print(f"  Transaction Cost: {TRANS_COST_BPS*100:.2f}% per trade")
    print(sep + "\n")
    summary["leverage"] = {"Bull": LEVERAGE_BULL, "Chop": LEVERAGE_CHOP, "Bear": LEVERAGE_BEAR}
    summary["trans_cost_bps"] = TRANS_COST_BPS

    return summary


def main(run_order_selection: bool = True):
    log.info("=" * 65)
    log.info(f"  HMM REGIME TRADING STRATEGY — PIPELINE START — TICKER={TICKER} "
             f"(annualization={ANNUALIZATION})")
    log.info("=" * 65)

    price_df = load_price_data(PRICE_CSV)
    feature_df = load_feature_data(FEATURE_CSV)
    master = build_master_dataframe(price_df, feature_df)

    min_rows = TRAIN_WINDOW + STEP_SIZE
    if len(master) < min_rows:
        raise ValueError(f"Master dataframe has only {len(master)} rows; need >= {min_rows}.")

    if run_order_selection:
        actual_features = [f for f in FEATURES if f in master.columns]
        run_model_order_selection(master[actual_features].values)

    oos_df = run_walk_forward(master)
    positioned_df = compute_positions(oos_df)
    results_df = run_backtest(positioned_df)
    results_df.attrs = positioned_df.attrs

    clean = results_df.dropna(subset=["strategy_net_ret", "bah_ret"])
    strategy_metrics = compute_performance_metrics(clean["strategy_net_ret"], clean["strategy_cum"].dropna(), "HMM Strategy")
    bah_metrics = compute_performance_metrics(clean["bah_ret"], clean["bah_cum"].dropna(), "Buy & Hold")
    clean.attrs = results_df.attrs
    summary = print_performance_report([strategy_metrics, bah_metrics], clean)

    results_df.to_csv(f"hmm_oos_results_{TICKER}.csv")
    log.info(f"Full OOS results saved to 'hmm_oos_results_{TICKER}.csv'")

    fold_diag_df = pd.DataFrame(oos_df.attrs.get("fold_diagnostics", []))
    fold_diag_df.to_csv(f"fold_diagnostics_{TICKER}.csv", index=False)
    log.info(f"Per-fold structural diagnostics saved to 'fold_diagnostics_{TICKER}.csv' ({len(fold_diag_df)} rows)")

    import json
    with open(f"performance_summary_{TICKER}.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log.info(f"Performance summary saved to 'performance_summary_{TICKER}.json'")

    log.info(f"Pipeline complete for {TICKER}.")


if __name__ == "__main__":
    main()
