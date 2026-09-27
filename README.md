# Federated Time-Series Learning under Heterogeneous Clients

A research-oriented FedAvg project using the UCI Human Activity Recognition dataset.

**Research question:** How does FedAvg behave when clients hold different distributions of time-series data?

## Current implementation

- UCI HAR raw inertial-signal data loader (128 timesteps × 9 sensor channels)
- LSTM classifier
- Centralized training baseline
- Sample-weighted FedAvg aggregation
- IID client partitioning
- Subject-based heterogeneous client partitioning
- Communication-round evaluation
- Partition inspection and unit tests
- Reproducible experiment scripts under `experiments/`

## Experimental setup

Four simulated clients are used for federated training. Two client-partitioning settings are compared:

1. **IID:** training samples are distributed approximately uniformly across clients.
2. **Subject-based:** clients are formed using subject-based partitions from the UCI HAR training data to introduce client-level distribution differences.

The FedAvg experiments use **5 communication rounds** with **1 local training epoch per round**. Global performance is evaluated on the held-out UCI HAR test set after each round.

## Results

The preliminary FedAvg runs show improving global test performance over communication rounds under both partitioning settings. At round 5, IID FedAvg reached **49.34% test accuracy** with a **1.2029 test loss**, while subject-based FedAvg reached **50.56% test accuracy** with a **1.2039 test loss**.

The trajectories are not strictly monotonic: both settings show a temporary accuracy drop at round 2, and the subject-based run shows additional variation at round 4. With only one run per condition, these results are treated as preliminary observations rather than evidence of a statistically meaningful difference between the two partitioning strategies.

### Test accuracy by communication round

![FedAvg test accuracy comparison](results/plots/test_accuracy_comparison.png)

### Test loss by communication round

![FedAvg test loss comparison](results/plots/test_loss_comparison.png)

### Round-level results

| Round | IID test accuracy | Subject-based test accuracy | IID test loss | Subject-based test loss |
|---:|---:|---:|---:|---:|
| 1 | 40.65% | 40.58% | 1.5730 | 1.5672 |
| 2 | 34.82% | 34.85% | 1.4165 | 1.4174 |
| 3 | 41.77% | 42.89% | 1.3764 | 1.3122 |
| 4 | 47.47% | 40.35% | 1.2376 | 1.2670 |
| 5 | 49.34% | 50.56% | 1.2029 | 1.2039 |

## Limitations and next steps

- Results currently represent a single run for each partitioning condition; repeated runs with controlled random seeds are needed to assess variability.
- The centralized baseline should be compared using a training budget aligned with the federated setting before drawing convergence or performance conclusions.
- Additional rounds and local-epoch ablations can be used to study convergence more thoroughly.
- The subject-based partition should be analyzed using client-level class distributions and performance variation to quantify the degree of heterogeneity.

Do not claim experimental results on a CV until the experiments have been run and the methodology/results are finalized.