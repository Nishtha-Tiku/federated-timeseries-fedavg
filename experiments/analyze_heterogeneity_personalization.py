import os
import pandas as pd
from scipy.stats import spearmanr


RESULTS_DIR = "results"

PARTITIONS = ["iid", "subject"]

HETEROGENEITY_METRICS = [
    "label_js",
    "temporal_acf_distance",
    "representation_distance",
    "update_cosine_distance",
    "parameter_distance",
]


def load_and_join(partition):
    heterogeneity_path = os.path.join(
        RESULTS_DIR, f"heterogeneity_{partition}.csv"
    )
    personalization_path = os.path.join(
        RESULTS_DIR, f"personalization_{partition}.csv"
    )

    heterogeneity = pd.read_csv(heterogeneity_path)
    personalization = pd.read_csv(personalization_path)

    # Keep only columns needed for the analysis
    heterogeneity = heterogeneity[
        ["round", "client_id"] + HETEROGENEITY_METRICS
    ].copy()

    personalization = personalization[
        ["round", "client_id", "alpha", "validation_accuracy"]
    ].copy()

    # Find the best alpha* for each client and round
    idx = personalization.groupby(
        ["round", "client_id"]
    )["validation_accuracy"].idxmax()

    best = personalization.loc[idx].copy()

    best = best.rename(
        columns={
            "alpha": "alpha_star",
            "validation_accuracy": "best_validation_accuracy",
        }
    )

    # Validation accuracy at alpha = 0
    alpha_zero = personalization[
        personalization["alpha"] == 0.0
    ][["round", "client_id", "validation_accuracy"]].rename(
        columns={"validation_accuracy": "alpha_zero_validation_accuracy"}
    )

    best = best.merge(
        alpha_zero,
        on=["round", "client_id"],
        how="left",
    )

    # Personalization gain over the global model
    best["personalization_gain"] = (
        best["best_validation_accuracy"]
        - best["alpha_zero_validation_accuracy"]
    )

    # Join heterogeneity metrics with personalization results
    merged = heterogeneity.merge(
        best[
            [
                "round",
                "client_id",
                "alpha_star",
                "best_validation_accuracy",
                "alpha_zero_validation_accuracy",
                "personalization_gain",
            ]
        ],
        on=["round", "client_id"],
        how="inner",
    )

    merged.insert(0, "partition", partition)

    return merged


def compute_correlations(df, partition):
    rows = []

    for metric in HETEROGENEITY_METRICS:
        # Heterogeneity metric vs alpha*
        valid_alpha = df[[metric, "alpha_star"]].dropna()

        if len(valid_alpha) >= 3:
            rho_alpha, p_alpha = spearmanr(
                valid_alpha[metric],
                valid_alpha["alpha_star"],
            )
        else:
            rho_alpha, p_alpha = float("nan"), float("nan")

        # Heterogeneity metric vs personalization gain
        valid_gain = df[[metric, "personalization_gain"]].dropna()

        if len(valid_gain) >= 3:
            rho_gain, p_gain = spearmanr(
                valid_gain[metric],
                valid_gain["personalization_gain"],
            )
        else:
            rho_gain, p_gain = float("nan"), float("nan")

        rows.append(
            {
                "partition": partition,
                "metric": metric,
                "n_alpha": len(valid_alpha),
                "spearman_rho_alpha_star": rho_alpha,
                "p_value_alpha_star": p_alpha,
                "n_gain": len(valid_gain),
                "spearman_rho_personalization_gain": rho_gain,
                "p_value_personalization_gain": p_gain,
            }
        )

    return pd.DataFrame(rows)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_data = []
    all_correlations = []

    for partition in PARTITIONS:
        print(f"\nAnalyzing {partition} partition...")

        merged = load_and_join(partition)

        merged_path = os.path.join(
            RESULTS_DIR,
            f"joined_{partition}.csv",
        )

        merged.to_csv(merged_path, index=False)

        correlations = compute_correlations(
            merged,
            partition,
        )

        all_data.append(merged)
        all_correlations.append(correlations)

        print("\nCorrelations:")
        print(
            correlations[
                [
                    "metric",
                    "spearman_rho_alpha_star",
                    "p_value_alpha_star",
                    "spearman_rho_personalization_gain",
                    "p_value_personalization_gain",
                ]
            ].to_string(index=False)
        )

        print(f"\nSaved joined data to: {merged_path}")

    # Combined results across IID + subject partitions
    combined_data = pd.concat(all_data, ignore_index=True)
    combined_correlations = pd.concat(
        all_correlations,
        ignore_index=True,
    )

    combined_data_path = os.path.join(
        RESULTS_DIR,
        "joined_all_partitions.csv",
    )

    correlations_path = os.path.join(
        RESULTS_DIR,
        "heterogeneity_personalization_correlations.csv",
    )

    combined_data.to_csv(
        combined_data_path,
        index=False,
    )

    combined_correlations.to_csv(
        correlations_path,
        index=False,
    )

    print("\n" + "=" * 70)
    print("SAVED RESULTS")
    print("=" * 70)
    print(f"Joined data:    {combined_data_path}")
    print(f"Correlations:   {correlations_path}")


if __name__ == "__main__":
    main()