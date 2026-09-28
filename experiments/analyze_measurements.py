"""Analyze heterogeneity measurements and personalization results.

Run after the two experiments:
    python -m experiments.analyze_measurements --partition iid
    python -m experiments.analyze_measurements --partition subject
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS_DIR = Path("results")
METRICS = [
    "label_js",
    "temporal_acf_distance",
    "representation_distance",
    "update_cosine_distance",
    "parameter_distance",
]


def spearman(a, b):
    """Spearman correlation without scipy."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ra = pd.Series(a).rank(method="average").to_numpy()
    rb = pd.Series(b).rank(method="average").to_numpy()
    if np.std(ra) == 0 or np.std(rb) == 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def run(partition):
    h_path = RESULTS_DIR / f"heterogeneity_{partition}.csv"
    p_path = RESULTS_DIR / f"personalization_{partition}.csv"

    h = pd.read_csv(h_path)
    p = pd.read_csv(p_path)

    # For each client/round, estimate alpha* from validation accuracy.
    best = (
        p.sort_values("validation_accuracy", ascending=False)
         .groupby(["partition", "round", "client_id"], as_index=False)
         .first()[["partition", "round", "client_id", "alpha", "validation_accuracy"]]
         .rename(columns={"alpha": "best_alpha", "validation_accuracy": "best_val_accuracy"})
    )

    merged = h.merge(best, on=["partition", "round", "client_id"], how="inner")
    merged.to_csv(RESULTS_DIR / f"combined_measurements_{partition}.csv", index=False)

    print("\n=== Heterogeneity vs estimated personalization need ===")
    for metric in METRICS:
        r = spearman(merged[metric], merged["best_alpha"])
        print(f"{metric:30s} Spearman rho = {r:+.3f}")

    print("\n=== Client/round observations ===")
    print(
        merged[[
            "round", "client_id", *METRICS,
            "best_alpha", "best_val_accuracy"
        ]].to_string(index=False)
    )

    print("\nSaved:")
    print(RESULTS_DIR / f"combined_measurements_{partition}.csv")
    print("\nInterpretation: treat these correlations as preliminary evidence only; the current experiment has 4 clients x 5 rounds and should be repeated across seeds before making a research claim.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition", choices=["iid", "subject"], required=True)
    args = parser.parse_args()
    run(args.partition)


if __name__ == "__main__":
    main()
