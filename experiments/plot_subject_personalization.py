import os

import pandas as pd
import matplotlib.pyplot as plt


RESULTS_DIR = "results"
INPUT_FILE = os.path.join(
    RESULTS_DIR,
    "joined_subject.csv",
)

OUTPUT_DIR = os.path.join(
    RESULTS_DIR,
    "plots",
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data():
    df = pd.read_csv(INPUT_FILE)

    required_columns = [
        "round",
        "client_id",
        "representation_distance",
        "temporal_acf_distance",
        "update_cosine_distance",
        "alpha_star",
        "personalization_gain",
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns in {INPUT_FILE}: {missing}"
        )

    return df


def add_labels(ax, df):
    for _, row in df.iterrows():
        ax.annotate(
            f"r{int(row['round'])}/c{int(row['client_id'])}",
            (
                row.iloc[0],
                row.iloc[1],
            ),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )


def plot_scatter(
    df,
    x,
    y,
    xlabel,
    ylabel,
    filename,
    title,
):
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(
        df[x],
        df[y],
        s=60,
        alpha=0.8,
    )

    for _, row in df.iterrows():
        ax.annotate(
            f"r{int(row['round'])}/c{int(row['client_id'])}",
            (
                row[x],
                row[y],
            ),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.grid(
        True,
        alpha=0.25,
    )

    fig.tight_layout()

    output_path = os.path.join(
        OUTPUT_DIR,
        filename,
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {output_path}")


def main():
    df = load_data()

    print("\nSubject partition observations:")
    print(f"Number of observations: {len(df)}")
    print(f"Clients: {sorted(df['client_id'].unique())}")
    print(f"Rounds: {sorted(df['round'].unique())}")

    print("\nAlpha* by round/client:")
    print(
        df[
            [
                "round",
                "client_id",
                "representation_distance",
                "alpha_star",
                "personalization_gain",
            ]
        ].sort_values(
            ["round", "client_id"]
        ).to_string(index=False)
    )

    # 1. Representation distance vs personalization gain
    plot_scatter(
        df,
        x="representation_distance",
        y="personalization_gain",
        xlabel="Representation distance",
        ylabel="Personalization gain",
        filename="subject_representation_vs_gain.png",
        title="Subject partition: Representation distance vs personalization gain",
    )

    # 2. Representation distance vs alpha*
    plot_scatter(
        df,
        x="representation_distance",
        y="alpha_star",
        xlabel="Representation distance",
        ylabel="Best alpha*",
        filename="subject_representation_vs_alpha.png",
        title="Subject partition: Representation distance vs alpha*",
    )

    # 3. Temporal distance vs personalization gain
    plot_scatter(
        df,
        x="temporal_acf_distance",
        y="personalization_gain",
        xlabel="Temporal ACF distance",
        ylabel="Personalization gain",
        filename="subject_temporal_vs_gain.png",
        title="Subject partition: Temporal ACF distance vs personalization gain",
    )

    # 4. Update distance vs personalization gain
    plot_scatter(
        df,
        x="update_cosine_distance",
        y="personalization_gain",
        xlabel="Update cosine distance",
        ylabel="Personalization gain",
        filename="subject_update_vs_gain.png",
        title="Subject partition: Update cosine distance vs personalization gain",
    )

    print("\nAll plots saved to:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()