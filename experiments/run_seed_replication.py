"""Run the existing heterogeneity + personalization experiments for multiple seeds.

This reuses the project's existing experiment implementations without changing
those files. It patches only their Config() factory and RESULTS_DIR at runtime,
so each seed gets isolated outputs.

Run from project root:
    python -m experiments.run_seed_replication

Seeds:
    42, 123, 456, 789, 2026

After the runs, execute:
    python -m experiments.analyze_seed_replication
"""

from pathlib import Path

from src.config import Config as BaseConfig

import experiments.measure_heterogeneity as mh
import experiments.personalization_sweep as ps

SEEDS = [42, 123, 456, 789, 2026]
PARTITION = "subject"
BASE_RESULTS = Path("results") / "seed_replication"


def make_config(seed):
    return BaseConfig(seed=seed)


def run():
    BASE_RESULTS.mkdir(parents=True, exist_ok=True)

    for seed in SEEDS:
        seed_dir = BASE_RESULTS / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        # The existing experiment functions instantiate Config() internally.
        # Replace that symbol with a zero-argument factory returning the
        # requested seeded configuration.
        mh.Config = lambda seed=seed: make_config(seed)
        ps.Config = lambda seed=seed: make_config(seed)

        mh.RESULTS_DIR = seed_dir
        ps.RESULTS_DIR = seed_dir

        print("\n" + "=" * 72)
        print(f"SEED {seed}: heterogeneity")
        print("=" * 72)
        mh.run(PARTITION)

        print("\n" + "=" * 72)
        print(f"SEED {seed}: personalization")
        print("=" * 72)
        ps.run(PARTITION)

    print("\nAll seed experiments completed.")
    print(f"Results: {BASE_RESULTS}")


if __name__ == "__main__":
    run()
