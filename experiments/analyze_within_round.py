"""
Round-controlled analysis of heterogeneity vs personalization.

This script analyzes the subject-partition observations one communication
round at a time, so comparisons are made between clients at the same
training stage.

Inputs:
    results/joined_subject.csv

Outputs:
    results/within_round_correlations_subject.csv
"""

from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr


INPUT_PATH = Path("results/joined_subject.csv")
OUTPUT_PATH = Path("results/within_round_correlations_subject.csv")

HETEROGENEITY_METRICS = [
    "representation_distance",
    "update_cosine_distance",
    "temporal_acf_distance",
]

OUTCOMES = [
    "personalization_gain",
    "alpha_star",
]


def spearman_pair(df, x_col, y_col):
    """Return Spearman rho and p-value, handling too-small/constant inputs."""
    pair = df[[x_col, y_col]].dropna()

    if len(pair) < 3:
        return float("nan"), float("nan"), len(pair)

    if pair[x_col].nunique() < 2 or pair[y_col].nunique() < 2:
        return float("nan"), float("nan"), len(pair)

    result = spearmanr(pair[x_col], pair[y_col])
    return float(result.statistic), float(result.pvalue), len(pair)


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {INPUT_PATH}. "
            "Run the heterogeneity/personalization analysis first."
        )

    df = pd.read_csv(INPUT_PATH)

    required = {
        "partition",
        "round",
        "client_id",
        *HETEROGENEITY_METRICS,
        *OUTCOMES,
    }

    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.sort_values(["round", "client_id"]).reset_index(drop=True)

    print("\nSubject partition: within-round correlations")
    print(f"Number of observations: {len(df)}")
    print(f"Rounds: {sorted(df['round'].unique().tolist())}")
    print(f"Clients per round: {df.groupby('round')['client_id'].nunique().to_dict()}")

    rows = []

    for round_id, round_df in df.groupby("round", sort=True):
        print(f"\n--- Round {round_id} ---")
        print(
            round_df[
                [
                    "client_id",
                    "representation_distance",
                    "update_cosine_distance",
                    "temporal_acf_distance",
                    "alpha_star",
                    "personalization_gain",
                ]
            ].to_string(index=False)
        )

        for metric in HETEROGENEITY_METRICS:
            for outcome in OUTCOMES:
                rho, p_value, n = spearman_pair(round_df, metric, outcome)

                rows.append(
                    {
                        "round": round_id,
                        "metric": metric,
                        "outcome": outcome,
                        "spearman_rho": rho,
                        "p_value": p_value,
                        "n_clients": n,
                    }
                )

                if pd.isna(rho):
                    print(
                        f"{metric:25s} -> {outcome:22s}: "
                        "rho=NaN (constant/tiny sample)"
                    )
                else:
                    print(
                        f"{metric:25s} -> {outcome:22s}: "
                        f"rho={rho:+.3f}, p={p_value:.4f}, n={n}"
                    )

    result_df = pd.DataFrame(rows)

    # Compact summary across rounds. This is descriptive only because each
    # round has only four clients.
    summary_rows = []
    for metric in HETEROGENEITY_METRICS:
        for outcome in OUTCOMES:
            subset = result_df[
                (result_df["metric"] == metric)
                & (result_df["outcome"] == outcome)
            ].dropna(subset=["spearman_rho"])

            summary_rows.append(
                {
                    "metric": metric,
                    "outcome": outcome,
                    "mean_round_rho": subset["spearman_rho"].mean(),
                    "median_round_rho": subset["spearman_rho"].median(),
                    "min_round_rho": subset["spearman_rho"].min(),
                    "max_round_rho": subset["spearman_rho"].max(),
                    "rounds_with_valid_rho": len(subset),
                }
            )

    summary_df = pd.DataFrame(summary_rows)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(OUTPUT_PATH, index=False)

    print("\n=== Descriptive summary across rounds ===")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
