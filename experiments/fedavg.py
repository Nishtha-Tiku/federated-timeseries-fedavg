import argparse
import csv
from pathlib import Path

import torch

from federated.client import train_client
from federated.partition import iid_partition, subject_partition
from federated.server import fedavg
from src.config import Config
from src.data import load_data
from src.model import LSTMClassifier
from src.train import evaluate
from src.utils import set_seed

RESULTS_DIR = Path("results")


def weighted_mean(values, weights):
    total = sum(weights)
    return sum(value * weight for value, weight in zip(values, weights)) / total


def run(partition_name: str):
    cfg = Config()
    set_seed(cfg.seed)

    (x_train, y_train, subjects_train), (x_test, y_test, _) = load_data()

    if partition_name == "iid":
        partitions = iid_partition(y_train.numpy(), cfg.num_clients, cfg.seed)
    elif partition_name == "subject":
        partitions = subject_partition(subjects_train.numpy(), cfg.num_clients)
    else:
        raise ValueError(f"Unknown partition: {partition_name}")

    global_model = LSTMClassifier(
        input_size=x_train.shape[-1],
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        num_classes=6,
    ).to(cfg.device)
    global_state = {name: tensor.detach().cpu().clone() for name, tensor in global_model.state_dict().items()}

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / f"fedavg_{partition_name}.csv"
    fieldnames = [
        "partition",
        "round",
        "num_clients",
        "local_epochs",
        "total_samples",
        "train_loss",
        "train_accuracy",
        "test_loss",
        "test_accuracy",
        "client_accuracy_mean",
        "client_accuracy_std",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for round_number in range(1, cfg.num_rounds + 1):
            client_results = []
            for client_id in range(cfg.num_clients):
                indices = partitions[client_id]
                result = train_client(
                    client_id=client_id,
                    global_state=global_state,
                    x=x_train[indices],
                    y=y_train[indices],
                    hidden_size=cfg.hidden_size,
                    num_layers=cfg.num_layers,
                    learning_rate=cfg.learning_rate,
                    batch_size=cfg.batch_size,
                    local_epochs=cfg.local_epochs,
                    device=cfg.device,
                )
                client_results.append(result)

            global_state = fedavg(client_results)
            global_model.load_state_dict(global_state)
            test_loss, test_acc = evaluate(
                global_model,
                x_test,
                y_test,
                batch_size=cfg.batch_size,
                device=cfg.device,
            )

            sample_counts = [result.num_samples for result in client_results]
            train_losses = [result.train_loss for result in client_results]
            train_accs = [result.train_acc for result in client_results]
            client_acc_mean = weighted_mean(train_accs, sample_counts)
            variance = weighted_mean(
                [(acc - client_acc_mean) ** 2 for acc in train_accs], sample_counts
            )

            row = {
                "partition": partition_name,
                "round": round_number,
                "num_clients": cfg.num_clients,
                "local_epochs": cfg.local_epochs,
                "total_samples": sum(sample_counts),
                "train_loss": weighted_mean(train_losses, sample_counts),
                "train_accuracy": client_acc_mean,
                "test_loss": test_loss,
                "test_accuracy": test_acc,
                "client_accuracy_mean": client_acc_mean,
                "client_accuracy_std": variance ** 0.5,
            }
            writer.writerow(row)
            handle.flush()

            print(
                f"partition={partition_name} round={round_number} "
                f"train_loss={row['train_loss']:.4f} "
                f"train_acc={row['train_accuracy']:.4f} "
                f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}"
            )

    print(f"Saved metrics to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run sample-weighted FedAvg on UCI HAR.")
    parser.add_argument(
        "--partition",
        choices=("iid", "subject"),
        required=True,
        help="Client partitioning strategy.",
    )
    args = parser.parse_args()
    run(args.partition)


if __name__ == "__main__":
    main()
