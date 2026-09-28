"""Run the within-round analysis across the five seeded replications.

Inputs per seed:
    results/seed_replication/seed_<seed>/heterogeneity_subject.csv
    results/seed_replication/seed_<seed>/personalization_subject.csv

Outputs:
    results/seed_replication/within_round_all_seeds.csv
    results/seed_replication/seed_summary.csv
"""

from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

SEEDS = [42, 123, 456, 789, 2026]
BASE_RESULTS = Path("results") / "seed_replication"

METRICS = [
    "representation_distance",
    "update_cosine_distance",
    "temporal_acf_distance",
]
OUTCOMES = ["personalization_gain", "alpha_star"]


def safe_spearman(x, y):
    x = pd.Series(x).dropna()
    y = pd.Series(y).dropna()
    if len(x) < 3 or len(y) < 3 or x.nunique() < 2 or y.nunique() < 2:
        return float("nan"), float("nan")
    r = spearmanr(x, y)
    return float(r.statistic), float(r.pvalue)


def load_seed(seed):
    d = BASE_RESULTS / f"seed_{seed}"
    h = pd.read_csv(d / "heterogeneity_subject.csv")
    p = pd.read_csv(d / "personalization_subject.csv")

    # Collapse the alpha sweep to alpha* and gain, exactly as in the earlier
    # analysis: alpha* is the alpha with maximum local validation accuracy;
    # gain is best validation accuracy minus alpha=0 validation accuracy.
    p = p.sort_values(["round", "client_id", "alpha"])
    best_rows = []
    for (rnd, cid), g in p.groupby(["round", "client_id"], sort=True):
        baseline = g.loc[g["alpha"].eq(0.0), "validation_accuracy"]
        if baseline.empty:
            raise ValueError(f"No alpha=0 baseline for seed={seed}, round={rnd}, client={cid}")
        baseline_acc = float(baseline.iloc[0])
        # idxmax gives the first alpha on ties, matching the original sweep.
        best = g.loc[g["validation_accuracy"].idxmax()]
        best_rows.append({
            "round": int(rnd),
            "client_id": int(cid),
            "alpha_star": float(best["alpha"]),
            "personalization_gain": float(best["validation_accuracy"] - baseline_acc),
        })
    p_best = pd.DataFrame(best_rows)

    h = h[[
        "round", "client_id", "representation_distance",
        "update_cosine_distance", "temporal_acf_distance"
    ]]

    return h.merge(p_best, on=["round", "client_id"], how="inner")


def analyze_seed(seed, df):
    rows = []
    for rnd, g in df.groupby("round", sort=True):
        for metric in METRICS:
            for outcome in OUTCOMES:
                rho, p = safe_spearman(g[metric], g[outcome])
                rows.append({
                    "seed": seed,
                    "round": int(rnd),
                    "metric": metric,
                    "outcome": outcome,
                    "spearman_rho": rho,
                    "p_value": p,
                    "n_clients": len(g),
                })
    return pd.DataFrame(rows)


def main():
    all_results = []
    seed_summaries = []

    for seed in SEEDS:
        df = load_seed(seed)
        result = analyze_seed(seed, df)
        all_results.append(result)

        target = result[
            (result["metric"] == "update_cosine_distance")
            & (result["outcome"] == "alpha_star")
        ].dropna(subset=["spearman_rho"])

        seed_summaries.append({
            "seed": seed,
            "update_to_alpha_mean_rho": target["spearman_rho"].mean(),
            "update_to_alpha_median_rho": target["spearman_rho"].median(),
            "positive_rounds": int((target["spearman_rho"] > 0).sum()),
            "valid_rounds": len(target),
            "all_rounds_positive": bool((target["spearman_rho"] > 0).all()) if len(target) else False,
        })

    combined = pd.concat(all_results, ignore_index=True)
    summary = pd.DataFrame(seed_summaries)

    BASE_RESULTS.mkdir(parents=True, exist_ok=True)
    combined.to_csv(BASE_RESULTS / "within_round_all_seeds.csv", index=False)
    summary.to_csv(BASE_RESULTS / "seed_summary.csv", index=False)

    target_all = combined[
        (combined["metric"] == "update_cosine_distance")
        & (combined["outcome"] == "alpha_star")
    ].dropna(subset=["spearman_rho"])

    print("\nUpdate cosine distance -> alpha* by seed")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\nRound-level values")
    print(
        target_all[["seed", "round", "spearman_rho", "p_value"]]
        .to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )

    print("\nOverall replication summary")
    print(f"Seeds with all 5 round correlations positive: {int(summary['all_rounds_positive'].sum())}/{len(summary)}")
    print(f"Positive round correlations: {int((target_all['spearman_rho'] > 0).sum())}/{len(target_all)}")
    print(f"Mean round rho across all seeds/rounds: {target_all['spearman_rho'].mean():+.3f}")
    print(f"Median round rho across all seeds/rounds: {target_all['spearman_rho'].median():+.3f}")

    print(f"\nSaved: {BASE_RESULTS / 'within_round_all_seeds.csv'}")
    print(f"Saved: {BASE_RESULTS / 'seed_summary.csv'}")


if __name__ == "__main__":
    main()
