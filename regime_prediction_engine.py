# ==============================================================================
# FORWARD-LOOKING HMM REGIME PREDICTION & BACKTESTING ENGINE
#
# REQUIREMENTS:
#   1. Python 3.8+
#   2. Dependencies: pip install numpy pandas scikit-learn hmmlearn
#   3. Required Input Files (in same directory):
#      - 'price_data.csv'         : Historical daily pricing data.
#      - 'features_improved.csv'   : Pre-computed technical features matching FEATURES list.
#
# HOW TO RUN:
#   - Standalone Terminal Script : python script_name.py
# ==============================================================================
import sys
import logging
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from hmmlearn.hmm import GaussianHMM

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("forward_engine")


FEATURES = [
    "log_return",
    "volume_zscore",
    "momentum",
    "parkinson_vol",
    "rsi",
    "ma_cross",
]

TRAIN_WINDOW    = 1260
STEP_SIZE       = 63
N_STATES        = 3
N_ITER          = 200
COVARIANCE_TYPE = "diag"
RANDOM_STATE    = 42
TARGET_VOL      = 0.15
ANNUALIZATION   = 252
TRANS_COST_BPS  = 0.0002

# Strategy caps — same as your STRATEGY_PARAMS
CAPS = {
    'Bull': 2.0,
    'Chop': 1.5,
    'Bear': 0.6,
}

# Forward-looking risk threshold:
# If P(Bear tomorrow) exceeds this, flag a transition warning
BEAR_WARNING_THRESHOLD = 0.30

# Stability threshold:
# If P(staying in current regime) drops below this, flag instability
STABILITY_THRESHOLD = 0.70



# SECTION 1 — CORE MATH: The three operations that make this
#             forward-looking rather than just label-today


def get_today_state_distribution(model: GaussianHMM,
                                  scaled_sequence: np.ndarray) -> np.ndarray:
    """
    Returns the POSTERIOR probability distribution over hidden states
    for the LAST observation in the sequence.

    model.predict()      → hard label (single most likely state)
    model.predict_proba()→ soft distribution [P(Bear), P(Chop), P(Bull)]

    The soft distribution captures UNCERTAINTY. On a day where the model
    is 51% Bull / 49% Chop, predict() returns "Bull" — but predict_proba()
    correctly shows you're barely leaning Bull, not confidently there.

    Parameters
    ----------
    model          : fitted GaussianHMM (from your walk-forward step)
    scaled_sequence: the full scaled feature sequence for this OOS block,
                     shape (T, n_features)

    Returns
    -------
    pi_t : shape (n_states,)  — today's state probability distribution
           e.g. [0.05, 0.15, 0.80] means 5% Bear, 15% Chop, 80% Bull
    """
    # predict_proba runs the forward-backward algorithm and returns
    # the SMOOTHED posterior P(s_t | all observations), shape (T, n_states)
    all_posteriors = model.predict_proba(scaled_sequence)

    # We want the LAST timestep — that's "today" in the OOS block
    pi_today = all_posteriors[-1]  # shape (n_states,)

    return pi_today


def forecast_tomorrow_distribution(pi_today: np.ndarray,
                                    transition_matrix: np.ndarray) -> np.ndarray:
  
    pi_tomorrow = pi_today @ transition_matrix   # matrix-vector multiply
    return pi_tomorrow


def map_probs_to_regimes(state_probs: np.ndarray,
                          regime_map: dict) -> dict:
  
    return {
        regime_map[state_idx]: float(state_probs[state_idx])
        for state_idx in range(len(state_probs))
    }



# SECTION 2 — DERIVED SIGNALS from the forecast distribution

def compute_forward_position(tomorrow_regime_probs: dict,
                              caps: dict = CAPS) -> float:
 
    position = sum(
        tomorrow_regime_probs.get(regime, 0.0) * cap
        for regime, cap in caps.items()
    )
    # Apply the same hard safety bounds as your current pipeline
    return float(np.clip(position, 0.0, 3.0))


def compute_transition_risk_score(tomorrow_regime_probs: dict) -> float:
   
    return tomorrow_regime_probs.get('Bear', 0.0)


def compute_regime_stability(pi_today: np.ndarray,
                              transition_matrix: np.ndarray,
                              current_state: int) -> float:
  
    # P(stay in current regime) = A[current_state, current_state]
    # Weighted by our confidence in being in current_state today
    raw_stability = transition_matrix[current_state, current_state]
    confidence_today = float(pi_today[current_state])
    return float(raw_stability * confidence_today + (1 - confidence_today) * 0.5)


def compute_n_step_forecast(pi_today: np.ndarray,
                              transition_matrix: np.ndarray,
                              regime_map: dict,
                              n_steps: int = 5) -> pd.DataFrame:
    
    forecasts = []
    pi_current = pi_today.copy()

    for day in range(1, n_steps + 1):
        pi_current = pi_current @ transition_matrix
        regime_probs = map_probs_to_regimes(pi_current, regime_map)
        forecasts.append({
            'day_ahead': day,
            'Bull': regime_probs.get('Bull', 0.0),
            'Chop': regime_probs.get('Chop', 0.0),
            'Bear': regime_probs.get('Bear', 0.0),
            'dominant_regime': max(regime_probs, key=regime_probs.get),
        })

    return pd.DataFrame(forecasts)


# SECTION 3 — WALK-FORWARD ENGINE with forward-looking signals


def winsorize_fit(data, lower_pct=1.0, upper_pct=99.0):
    return np.percentile(data, lower_pct, axis=0), np.percentile(data, upper_pct, axis=0)

def winsorize_transform(data, lower_bounds, upper_bounds):
    return np.clip(data, lower_bounds, upper_bounds)

def label_regimes(model: GaussianHMM) -> dict:
    """Same logic as your HMMstrategy.py label_regimes()"""
    mean_returns = model.means_[:, 0]
    sorted_states = np.argsort(mean_returns)
    return {
        int(sorted_states[0]):  "Bear",
        int(sorted_states[1]):  "Chop",
        int(sorted_states[-1]): "Bull",
    }


def run_forward_walk_forward(master: pd.DataFrame) -> pd.DataFrame:
   
    n = len(master)
    actual_features = [f for f in FEATURES if f in master.columns]
    if not actual_features:
        raise ValueError("No features found in master dataframe.")

    feature_array = master[actual_features].values
    n_steps = (n - TRAIN_WINDOW) // STEP_SIZE
    log.info(f"Forward walk-forward: {n_steps} steps over {n} rows")
    log.info(f"Features: {actual_features}")

    all_results = []

    for step in range(n_steps):
        train_end  = TRAIN_WINDOW + step * STEP_SIZE
        oos_start  = train_end
        oos_end    = min(oos_start + STEP_SIZE, n)

        train_data = feature_array[:train_end]
        oos_data   = feature_array[oos_start:oos_end]

        if len(oos_data) == 0:
            break

        # ---- Preprocessing (train only) ----
        lower_bounds, upper_bounds = winsorize_fit(train_data)
        train_wins = winsorize_transform(train_data, lower_bounds, upper_bounds)
        oos_wins   = winsorize_transform(oos_data,   lower_bounds, upper_bounds)

        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_wins)
        oos_scaled   = scaler.transform(oos_wins)

        # ---- Fit HMM ----
        try:
            model = GaussianHMM(
                n_components=N_STATES,
                covariance_type=COVARIANCE_TYPE,
                n_iter=N_ITER,
                random_state=RANDOM_STATE,
                verbose=False,
            )
            model.fit(train_scaled)
        except Exception as e:
            log.warning(f"Step {step+1}: HMM fit failed — {e}")
            continue

        # Extract the transition matrix — this is the key addition
        A = model.transmat_           # shape (n_states, n_states)
        regime_map = label_regimes(model)

        # ---- For each OOS day, compute forward-looking signals ----
        try:
            # predict_proba on the full OOS sequence
            # Shape: (len(oos_data), n_states)
            all_posteriors = model.predict_proba(oos_scaled)

            # Hard labels (same as your current code, for comparison)
            hard_states = model.predict(oos_scaled)

        except Exception as e:
            log.warning(f"Step {step+1}: predict failed — {e}")
            continue

        oos_idx = master.index[oos_start:oos_end]

        for t in range(len(oos_data)):
            # ── TODAY's signals ──────────────────────────────────────
            pi_today        = all_posteriors[t]           # shape (n_states,)
            hard_state_idx  = int(hard_states[t])
            today_regime    = regime_map[hard_state_idx]

            today_regime_probs = map_probs_to_regimes(pi_today, regime_map)

            # ── TOMORROW's forecast ───────────────────────────────────
            # π_{t+1} = π_t @ A  — the core forecasting equation
            pi_tomorrow           = pi_today @ A
            tomorrow_regime_probs = map_probs_to_regimes(pi_tomorrow, regime_map)

            # ── DERIVED SIGNALS ───────────────────────────────────────
            # 1. Forward-weighted position (uses tomorrow's forecast)
            fwd_position = compute_forward_position(tomorrow_regime_probs)

            # 2. Transition risk score = P(Bear tomorrow)
            risk_score = compute_transition_risk_score(tomorrow_regime_probs)

            # 3. Regime stability
            stability = compute_regime_stability(pi_today, A, hard_state_idx)

            # 4. Most likely regime tomorrow
            tomorrow_regime = max(tomorrow_regime_probs, key=tomorrow_regime_probs.get)

            # 5. 5-step forecast dominant regimes
            # Only compute for the LAST day of each OOS block
            # (computing for every day is redundant and slow)
            fcast_5d = None
            if t == len(oos_data) - 1:
                fcast_df = compute_n_step_forecast(pi_today, A, regime_map, n_steps=5)
                fcast_5d = list(fcast_df['dominant_regime'])

            # ── ASSEMBLE ROW ──────────────────────────────────────────
            row = {
                # Index
                'date': oos_idx[t],

                # Original features
                **{f: oos_data[t, i] for i, f in enumerate(actual_features)},

                # Hard label (same as current code)
                'state':               hard_state_idx,
                'regime':              today_regime,

                # Today's soft probabilities
                'p_bull_today':        today_regime_probs.get('Bull', 0.0),
                'p_chop_today':        today_regime_probs.get('Chop', 0.0),
                'p_bear_today':        today_regime_probs.get('Bear', 0.0),

                # Tomorrow's forecasted probabilities (KEY NEW SIGNAL)
                'p_bull_tomorrow':     tomorrow_regime_probs.get('Bull', 0.0),
                'p_chop_tomorrow':     tomorrow_regime_probs.get('Chop', 0.0),
                'p_bear_tomorrow':     tomorrow_regime_probs.get('Bear', 0.0),
                'regime_tomorrow':     tomorrow_regime,

                # Derived position signals
                'fwd_weighted_pos':    fwd_position,
                'transition_risk':     risk_score,
                'regime_stability':    stability,

                # Warning flags
                'bear_warning':        risk_score > BEAR_WARNING_THRESHOLD,
                'low_stability':       stability  < STABILITY_THRESHOLD,
            }
            all_results.append(row)

        if (step + 1) % 10 == 0 or step == n_steps - 1:
            log.info(f"  Step {step+1}/{n_steps} — "
                     f"OOS: {oos_idx[0].date()} → {oos_idx[-1].date()}")

    if not all_results:
        raise RuntimeError("Forward walk-forward produced no results.")

    oos_df = pd.DataFrame(all_results).set_index('date')
    log.info(f"Forward walk-forward complete: {len(oos_df)} rows")
    return oos_df



# SECTION 4 — FORWARD-LOOKING BACKTEST
# Compares three position sizing methods:
#   A. Hard-cap (your current method)
#   B. Soft today (use today's proba-weighted position)
#   C. Forward (use tomorrow's predicted proba-weighted position)

def run_forward_backtest(oos_df: pd.DataFrame) -> pd.DataFrame:
    """
    Runs three backtests in parallel to directly compare the benefit
    of using transition probability forecasting.

    Method A — Hard-cap (current):
        position = regime_cap, lag 1 day

    Method B — Soft today:
        position = P(Bull|today)×2.0 + P(Chop|today)×1.5 + P(Bear|today)×0.6
        lag 1 day

    Method C — Forward (new):
        position = P(Bull|tomorrow)×2.0 + P(Chop|tomorrow)×1.5 + P(Bear|tomorrow)×0.6
        NO lag needed — tomorrow's prediction is already forward-looking
    """
    df = oos_df.copy()
    log_return = df['log_return']

    # ── Method A: Hard-cap position (replicates your current code) ──
    cap_map = {'Bull': CAPS['Bull'], 'Chop': CAPS['Chop'], 'Bear': CAPS['Bear']}
    df['pos_hardcap'] = df['regime'].map(cap_map).shift(1).fillna(0.0)

    # ── Method B: Soft today (proba-weighted, lag 1) ──
    soft_today = (
        df['p_bull_today'] * CAPS['Bull'] +
        df['p_chop_today'] * CAPS['Chop'] +
        df['p_bear_today'] * CAPS['Bear']
    )
    df['pos_soft_today'] = soft_today.shift(1).fillna(0.0)

    # ── Method C: Forward prediction (NO lag — this IS the forward signal) ──
    # fwd_weighted_pos already uses tomorrow's predicted distribution.
    # We shift by 1 because even the forward prediction is made at EOD
    # using data available through close of day t.
    df['pos_forward'] = df['fwd_weighted_pos'].shift(1).fillna(0.0)

    # ── Compute net returns for each method ──
    for method, pos_col in [
        ('hardcap', 'pos_hardcap'),
        ('soft_today', 'pos_soft_today'),
        ('forward', 'pos_forward'),
    ]:
        gross = df[pos_col] * log_return
        turnover = df[pos_col].diff().abs().fillna(0.0)
        cost = TRANS_COST_BPS * turnover
        net = gross - cost
        cum = (1 + net).cumprod()

        df[f'ret_{method}']  = net
        df[f'cum_{method}']  = cum

    # Buy & Hold
    df['ret_bah'] = log_return
    df['cum_bah'] = (1 + log_return).cumprod()

    log.info("Forward backtest complete.")
    return df



# SECTION 5 — PERFORMANCE COMPARISON REPORT


def compute_metrics(returns: pd.Series, cum: pd.Series, label: str) -> dict:
    n = len(returns.dropna())
    ann_ret  = (cum.iloc[-1] ** (ANNUALIZATION / n)) - 1.0
    ann_vol  = returns.std() * np.sqrt(ANNUALIZATION)
    sharpe   = ann_ret / ann_vol if ann_vol > 0 else np.nan
    running_max = cum.cummax()
    mdd      = float(((cum - running_max) / running_max).min())
    total    = (cum.iloc[-1] - 1.0) * 100
    return {
        'label': label, 'total_return_pct': total,
        'ann_return': ann_ret, 'ann_vol': ann_vol,
        'sharpe': sharpe, 'max_drawdown': mdd,
    }


def print_comparison_report(results_df: pd.DataFrame,
                              oos_df: pd.DataFrame) -> None:
    sep = "=" * 72

    # ── Method comparison ──
    methods = [
        ('ret_hardcap',   'cum_hardcap',   'A. Hard-Cap (current)     '),
        ('ret_soft_today','cum_soft_today', 'B. Soft Today (proba)     '),
        ('ret_forward',   'cum_forward',   'C. Forward Prediction (NEW)'),
        ('ret_bah',       'cum_bah',       'D. Buy & Hold             '),
    ]

    all_metrics = []
    for ret_col, cum_col, label in methods:
        clean = results_df[[ret_col, cum_col]].dropna()
        m = compute_metrics(clean[ret_col], clean[cum_col], label)
        all_metrics.append(m)

    print()
    print(sep)
    print("  FORWARD-LOOKING ENGINE — METHOD COMPARISON")
    print(sep)
    print(f"  {'Method':<35} {'Total Ret':>10} {'Sharpe':>8} {'MaxDD':>8} {'Ann Vol':>8}")
    print("-" * 72)
    for m in all_metrics:
        print(f"  {m['label']:<35} "
              f"{m['total_return_pct']:>9.1f}% "
              f"{m['sharpe']:>8.4f} "
              f"{m['max_drawdown']:>7.1%} "
              f"{m['ann_vol']:>7.1%}")
    print(sep)

    # ── Transition risk signal analysis ──
    print("\n  TRANSITION RISK SIGNAL ANALYSIS")
    print("-" * 72)

    risk = oos_df['transition_risk']
    print(f"  P(Bear tomorrow) — distribution:")
    print(f"    Mean   : {risk.mean():.4f}")
    print(f"    Median : {risk.median():.4f}")
    print(f"    Max    : {risk.max():.4f}")
    print(f"    > {BEAR_WARNING_THRESHOLD} (warning days)  : "
          f"{(risk > BEAR_WARNING_THRESHOLD).sum()} / {len(risk)} "
          f"({(risk > BEAR_WARNING_THRESHOLD).mean():.1%})")

    # How accurate is 'regime_tomorrow' vs actual next day regime?
    if 'regime' in oos_df.columns:
        oos_eval = oos_df.copy()
        oos_eval['actual_tomorrow_regime'] = oos_eval['regime'].shift(-1)
        correct = (oos_eval['regime_tomorrow'] == oos_eval['actual_tomorrow_regime']).sum()
        total_eval = oos_eval['actual_tomorrow_regime'].notna().sum()
        accuracy = correct / total_eval if total_eval > 0 else 0.0
        print(f"\n  Next-day regime prediction accuracy:")
        print(f"    Predicted correct : {correct} / {total_eval} days ({accuracy:.2%})")

    # ── Stability signal analysis ──
    print(f"\n  REGIME STABILITY SIGNAL ANALYSIS")
    print("-" * 72)
    stab = oos_df['regime_stability']
    low_stab_days = (oos_df['low_stability'] == True).sum()
    print(f"  Stability — distribution:")
    print(f"    Mean   : {stab.mean():.4f}")
    print(f"    Median : {stab.median():.4f}")
    print(f"    Low stability days (< {STABILITY_THRESHOLD}): "
          f"{low_stab_days} ({low_stab_days/len(stab):.1%})")

    print(sep)

    # ── Warning signal effectiveness ──
    print("\n  BEAR WARNING SIGNAL EFFECTIVENESS")
    print("-" * 72)
    if 'log_return' in oos_df.columns:
        oos_eval = oos_df.copy()
        oos_eval['next_return'] = oos_eval['log_return'].shift(-1)
        oos_eval = oos_eval.dropna(subset=['next_return'])

        warned_days   = oos_eval[oos_eval['bear_warning'] == True]
        unwarned_days = oos_eval[oos_eval['bear_warning'] == False]

        if len(warned_days) > 0:
            avg_ret_warned   = warned_days['next_return'].mean() * 100
            avg_ret_unwarned = unwarned_days['next_return'].mean() * 100
            print(f"  Avg next-day return when WARNING raised : {avg_ret_warned:>7.4f}%")
            print(f"  Avg next-day return when no warning     : {avg_ret_unwarned:>7.4f}%")
            print(f"  Warning days correctly predicted down   : "
                  f"{(warned_days['next_return'] < 0).mean():.2%}")

    print(sep)

    # ── 5-day forward forecast for the LAST date in OOS ──
    print("\n  LATEST 5-DAY FORWARD REGIME FORECAST")
    print("-" * 72)
    last_row = oos_df.iloc[-1]
    last_date = oos_df.index[-1]

    print(f"  As of: {last_date.date()}")
    print(f"  Today's regime   : {last_row['regime']}")
    print(f"  Today's probs    : "
          f"Bull={last_row['p_bull_today']:.1%}  "
          f"Chop={last_row['p_chop_today']:.1%}  "
          f"Bear={last_row['p_bear_today']:.1%}")
    print(f"  Tomorrow forecast: {last_row['regime_tomorrow']}  "
          f"(Bear risk={last_row['transition_risk']:.1%})")
    print(f"  Stability score  : {last_row['regime_stability']:.4f} "
          f"({'LOW — consider reducing' if last_row['low_stability'] else 'STABLE'})")
    print(f"\n  Recommended forward position:")
    print(f"    Hard-cap method (current) : {CAPS.get(last_row['regime'], 0.0):.2f}x")
    print(f"    Forward-weighted (new)    : {last_row['fwd_weighted_pos']:.2f}x")
    if last_row['bear_warning']:
        print(f"\n  ⚠  BEAR WARNING: P(Bear tomorrow) = {last_row['transition_risk']:.1%}")
        print(f"     Consider reducing to {last_row['fwd_weighted_pos']*0.7:.2f}x as precaution")
    print(sep)
    print()


# ============================================================
# SECTION 6 — NEXT DAY TERMINAL REPORT (matches screenshot style)
# ============================================================

def print_next_day_prediction(oos_df: pd.DataFrame) -> None:
    """
    Prints the NEXT DAY REGIME PREDICTION terminal output
    matching the style of the screenshot you showed — but now
    showing BOTH today's posterior AND tomorrow's forecast.
    """
    last = oos_df.iloc[-1]
    width = 62
    bar_width = 40

    def bar(pct: float) -> str:
        filled = int(round(pct * bar_width))
        return "█" * filled

    print("\n" + "=" * width)
    print("  NEXT DAY REGIME PREDICTION  (Forward-Looking Engine)")
    print("=" * width)
    print(f"  As of date       : {oos_df.index[-1].date()}")
    print(f"  Current Regime   : {last['regime']}")
    print(f"  Predicted Next   : {last['regime_tomorrow']}")
    print("-" * width)
    print("  TOMORROW'S PREDICTED PROBABILITIES (via π_{t+1} = π_t @ A)")
    print("-" * width)

    probs_tomorrow = [
        ('Bear', last['p_bear_tomorrow']),
        ('Bull', last['p_bull_tomorrow']),
        ('Chop', last['p_chop_tomorrow']),
    ]
    probs_tomorrow.sort(key=lambda x: -x[1])

    for regime, p in probs_tomorrow:
        print(f"  {regime:<8}: {p*100:>6.2f}%  {bar(p)}")

    print("-" * width)
    print("  TODAY'S POSTERIOR (for reference)")
    print("-" * width)
    probs_today = [
        ('Bear', last['p_bear_today']),
        ('Bull', last['p_bull_today']),
        ('Chop', last['p_chop_today']),
    ]
    probs_today.sort(key=lambda x: -x[1])
    for regime, p in probs_today:
        print(f"  {regime:<8}: {p*100:>6.2f}%  {bar(p)}")

    print("-" * width)
    print(f"  Transition Risk Score : {last['transition_risk']:.4f}  "
          f"{'⚠  ELEVATED' if last['bear_warning'] else '✓  Normal'}")
    print(f"  Regime Stability      : {last['regime_stability']:.4f}  "
          f"{'⚠  LOW' if last['low_stability'] else '✓  Stable'}")
    print("-" * width)
    print(f"  Forward Position (NEW): {last['fwd_weighted_pos']:.2f}x capital")
    print(f"  Hard-Cap Pos (current): "
          f"{CAPS.get(last['regime'], 0.0):.2f}x capital")
    print("=" * width)
    print()



# SECTION 7 — MAIN ENTRY POINT


def run_forward_engine(master: pd.DataFrame) -> pd.DataFrame:
    """
    Main function. Accepts the `master` dataframe from your
    HMMstrategy.py pipeline and returns the full OOS dataframe
    with all forward-looking signals added.

    Parameters
    ----------
    master : pd.DataFrame — the merged price + features dataframe
             from build_master_dataframe() in HMMstrategy.py

    Returns
    -------
    results_df : pd.DataFrame — full results with all signals
    """
    log.info("=" * 65)
    log.info("  FORWARD-LOOKING REGIME PREDICTION ENGINE")
    log.info("=" * 65)

    # Step 1: Walk-forward with forward-looking signals
    log.info("\n[STAGE 1] Forward Walk-Forward Validation")
    log.info("-" * 50)
    oos_df = run_forward_walk_forward(master)

    # Step 2: Run the three-way backtest comparison
    log.info("\n[STAGE 2] Three-Method Backtest Comparison")
    log.info("-" * 50)
    results_df = run_forward_backtest(oos_df)

    # Step 3: Print reports
    log.info("\n[STAGE 3] Performance Reports")
    log.info("-" * 50)
    print_comparison_report(results_df, oos_df)
    print_next_day_prediction(oos_df)

    # Step 4: Save
    output_path = "forward_regime_results.csv"
    results_df.to_csv(output_path)
    log.info(f"Results saved to '{output_path}'")

    return results_df


# STANDALONE EXECUTION
# Run this file directly after HMMstrategy.py has run and
# saved hmm_oos_results.csv — OR call run_forward_engine(master)
# from the same session.

if __name__ == "__main__":
    import os
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from hmmlearn.hmm import GaussianHMM

    log.info("Standalone mode — loading from saved CSVs...")

    # ── Configuration — hardcoded for SPY ──
    PRICE_CSV   = "price_data.csv"
    FEATURE_CSV = "features_improved.csv"

    # ── Check files exist ──
    missing_files = []
    for f in [PRICE_CSV, FEATURE_CSV]:
        if not os.path.exists(f):
            missing_files.append(f)

    if missing_files:
        log.error(f"Missing files: {missing_files}")
        log.info("Make sure price_data.csv and features_improved.csv are in the same folder")
        sys.exit(1)

    # ── Load price data ──
    def load_price(filepath):
        try:
            df = pd.read_csv(filepath, skiprows=[1, 2],
                           index_col=0, parse_dates=True)
        except:
            df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        df.columns = df.columns.str.strip().str.lower()
        close_col = [c for c in df.columns if 'close' in c or 'price' in c][0]
        df = df[[close_col]].rename(columns={close_col: 'close'})
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        df = df.dropna(subset=['close']).sort_index()
        df.index = pd.to_datetime(df.index)
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        df = df.dropna(subset=['log_return'])
        log.info(f"Price data loaded: {len(df)} rows")
        return df

    # ── Load feature data ──
    def load_features(filepath):
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        df.columns = df.columns.str.strip().str.lower()
        df.index = pd.to_datetime(df.index, errors='coerce')
        df = df[df.index.notnull()].sort_index()
        log.info(f"Feature data loaded: {len(df)} rows, columns={list(df.columns)}")
        return df

    # ── Build master dataframe ──
    def build_master(price_df, feature_df):
        master = price_df.copy()
        cols_to_drop = [c for c in ['close', 'log_return']
                       if c in feature_df.columns]
        if cols_to_drop:
            feature_df = feature_df.drop(columns=cols_to_drop)
        master = master.join(feature_df, how='left')
        available = [f for f in FEATURES if f in master.columns]
        missing = [f for f in FEATURES if f not in master.columns]
        if missing:
            log.warning(f"Missing features: {missing}")
        if not available:
            raise KeyError(f"No features found: {FEATURES}")
        before = len(master)
        master = master.dropna(subset=available)
        log.info(f"Master dataframe: {len(master)} rows, "
                f"{master.index[0].date()} -> {master.index[-1].date()}")
        log.info(f"Features available: {available}")
        return master

    # ── Run pipeline ──
    try:
        price_df   = load_price(PRICE_CSV)
        feature_df = load_features(FEATURE_CSV)
        master     = build_master(price_df, feature_df)

        min_rows = TRAIN_WINDOW + STEP_SIZE
        if len(master) < min_rows:
            raise ValueError(
                f"Not enough data: {len(master)} rows, need {min_rows}"
            )

        results = run_forward_engine(master)
        log.info("Pipeline complete for SPY.")

    except FileNotFoundError as e:
        log.error(f"File not found: {e}")
    except Exception as e:
        log.error(f"Pipeline failed: {e}")
        raise
