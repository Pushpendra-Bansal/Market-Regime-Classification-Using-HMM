# -*- coding: utf-8 -*-
# ==============================================================================
# HMM REGIME POSTERIOR ANOMALY DETECTION & UNCERTAINTY VISUALIZATION
#
# OVERVIEW:
#   This module analyzes out-of-sample (OOS) HMM market regime probabilities 
#   to detect high-uncertainty or low-confidence classification days. It evaluates
#   three posterior metrics (top state probability, margin between top two states, 
#   and entropy) to generate an overall anomaly score and flags regime transitions/anomalies.
#
# SYSTEM & ENVIRONMENT REQUIREMENTS:
#   1. Python Version: Python 3.8 or higher
#   2. Core Libraries: pip install numpy pandas matplotlib
#
# REQUIRED INPUT FILES (Must reside in the working directory):
#   1. 'hmm_oos_results_SPY.csv' (or asset equivalent): Must be installed from Anomaly Detection Folder in our repo.Must contain date indices and columns for:
#      ['posterior_top_prob', 'posterior_margin', 'posterior_entropy', 'regime'].
#      Optionally includes price/returns columns: ['cum_strategy', 'strategy_cum', 'close'].
#
# HOW TO RUN:
#   - Command Line Exec : python anomaly_detection.py
#   - Jupyter Notebook  : Run cell directly after generating the walk-forward results CSV.
#
# CONFIGURATION THRESHOLDS:
#   - PROB_THRESHOLD    (Default: 0.70) : Flags days where top state probability < 70%.
#   - MARGIN_THRESHOLD  (Default: 0.50) : Flags days where top state margin < 50%.
#   - ENTROPY_THRESHOLD (Default: 0.50) : Flags days where posterior entropy > 0.50.
#
# GENERATED OUTPUTS & ARTIFACTS:
#   1. 'hmm_oos_results_with_anomalies.csv': Updated OOS DataFrame containing 
#                                           ['is_anomaly', 'anomaly_score'] columns.
#   2. 'anomaly_chart.png'                 : Two-panel plot showing anomaly flags overlaid 
#                                           on performance/price charts alongside the 
#                                           continuous anomaly score.
# ==============================================================================

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# --- Config ---
INPUT_CSV = "hmm_oos_results_SPY.csv"   # <-- change to your actual results file

PROB_THRESHOLD    = 0.70   # below this = low confidence
MARGIN_THRESHOLD  = 0.50   # below this = uncertain between states
ENTROPY_THRESHOLD = 0.50   # above this = high uncertainty

REQUIRED_COLUMNS = ["posterior_top_prob", "posterior_margin", "posterior_entropy", "regime"]

# Candidate column names for the "value line" to plot anomalies against.
# The script will use the first one it finds in the CSV.
VALUE_COLUMN_CANDIDATES = ["cum_strategy", "strategy_cum", "cum_hardcap", "close"]


def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        print("=" * 70)
        print(f"ERROR: Required file '{path}' was not found.")
        print(f"You need to add '{path}' to this folder before running this script.")
        print("=" * 70)
        sys.exit(1)

    df = pd.read_csv(path, index_col=0, parse_dates=True)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        print(f"ERROR: '{path}' is missing required columns: {missing}")
        print(f"Available columns: {df.columns.tolist()}")
        sys.exit(1)

    return df


def flag_anomalies(oos_df: pd.DataFrame) -> pd.DataFrame:
    # flag anomalies
    oos_df["is_anomaly"] = (
        (oos_df["posterior_top_prob"] < PROB_THRESHOLD) |
        (oos_df["posterior_margin"] < MARGIN_THRESHOLD) |
        (oos_df["posterior_entropy"] > ENTROPY_THRESHOLD)
    )

    # anomaly severity score (0 to 1, higher = more anomalous)
    oos_df["anomaly_score"] = (
        (1 - oos_df["posterior_top_prob"]) * 0.4 +
        (1 - oos_df["posterior_margin"].clip(0, 1)) * 0.3 +
        (oos_df["posterior_entropy"] / oos_df["posterior_entropy"].max()) * 0.3
    )

    return oos_df


def report(oos_df: pd.DataFrame) -> None:
    print(f"Total anomaly days: {oos_df['is_anomaly'].sum()}")
    print(f"Anomaly percentage: {oos_df['is_anomaly'].mean():.1%}")

    print(f"\nRegime breakdown of anomaly days:")
    print(oos_df[oos_df["is_anomaly"]]["regime"].value_counts())

    print(f"\nTop 15 most anomalous days:")
    print(
        oos_df[oos_df["is_anomaly"]][
            ["anomaly_score", "regime", "posterior_top_prob",
             "posterior_margin", "posterior_entropy"]
        ].nlargest(15, "anomaly_score")
    )


def find_value_column(df: pd.DataFrame) -> str | None:
    for col in VALUE_COLUMN_CANDIDATES:
        if col in df.columns:
            return col
    return None


def visualize(oos_df: pd.DataFrame, out_path: str = "anomaly_chart.png") -> None:
    value_col = find_value_column(oos_df)

    fig, axes = plt.subplots(
        2, 1, figsize=(13, 8), sharex=True,
        gridspec_kw={"height_ratios": [3, 1.2]}
    )
    ax_main, ax_score = axes

    # --- Regime shading (if regime is string-labeled Bull/Chop/Bear) ---
    regime_colors = {"Bull": "#e8f5e9", "Chop": "#fff8e1", "Bear": "#ffebee"}
    if oos_df["regime"].isin(regime_colors.keys()).any():
        regime_vals = oos_df["regime"].values
        dates = oos_df.index
        start_idx = 0
        for i in range(1, len(regime_vals) + 1):
            if i == len(regime_vals) or regime_vals[i] != regime_vals[start_idx]:
                color = regime_colors.get(regime_vals[start_idx], "#ffffff")
                ax_main.axvspan(dates[start_idx], dates[min(i, len(dates) - 1)],
                                 color=color, alpha=0.5, lw=0)
                start_idx = i

    # --- Main line (cumulative return or price, whichever is available) ---
    if value_col is not None:
        ax_main.plot(oos_df.index, oos_df[value_col], color="#1f2937", lw=1.1,
                     label=value_col, zorder=2)
        # Mark anomaly days on top of the line
        anomalies = oos_df[oos_df["is_anomaly"]]
        ax_main.scatter(anomalies.index, anomalies[value_col],
                         color="#d62728", s=18, zorder=3,
                         label=f"Anomaly days (n={len(anomalies)})")
        ax_main.set_ylabel(value_col)
    else:
        # No known value column — just mark anomaly days on a flat line
        print("NOTE: No cumulative return / price column found "
              f"(looked for {VALUE_COLUMN_CANDIDATES}). "
              "Plotting anomaly flags only, without a value line.")
        anomalies = oos_df[oos_df["is_anomaly"]]
        ax_main.scatter(anomalies.index, [1] * len(anomalies),
                         color="#d62728", s=18, label=f"Anomaly days (n={len(anomalies)})")
        ax_main.set_yticks([])

    ax_main.set_title("HMM Posterior Anomalies", fontsize=14, fontweight="bold")
    ax_main.legend(loc="upper left", frameon=True)
    ax_main.grid(alpha=0.2)

    # --- Anomaly score subplot ---
    ax_score.plot(oos_df.index, oos_df["anomaly_score"], color="#9b2226", lw=0.8)
    ax_score.axhline(0.5, color="gray", ls="--", lw=0.8, alpha=0.6)
    ax_score.set_ylabel("Anomaly score")
    ax_score.set_xlabel("Date")
    ax_score.grid(alpha=0.2)

    ax_score.xaxis.set_major_locator(mdates.YearLocator(2))
    ax_score.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved chart to: {out_path}")
    plt.show()


def main():
    oos_df = load_data(INPUT_CSV)
    oos_df = flag_anomalies(oos_df)
    report(oos_df)

    out_csv = "hmm_oos_results_with_anomalies.csv"
    oos_df.to_csv(out_csv)
    print(f"\nSaved results with anomaly flags to: {out_csv}")

    visualize(oos_df)


if __name__ == "__main__":
    main()
