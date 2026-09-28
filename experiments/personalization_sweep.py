"""Measure local/global personalization need on the existing FedAvg setup.

For every round/client, train locally from the same global model, then evaluate
interpolated models:
    w(alpha) = (1-alpha) * w_global + alpha * w_local

A client-specific alpha that maximizes LOCAL VALIDATION accuracy is reported.
The official UCI-HAR test set is not used to select alpha.

Run from project root:
    python -m experiments.personalization_sweep --partition iid
    python -m experiments.personalization_sweep --partition subject
"""

import argparse
import csv
from pathlib import Path

import numpy as np
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
ALPHAS = np.round(np.linspace(0.0, 1.0, 11), 1)


def split_client(indices, seed, val_fraction=0.2):
    rng = np.random.default_rng(seed)
    indices = np.asarray(indices)
    shuffled = rng.permutation(indices)
    n_val = max(1, int(len(shuffled) * val_fraction))
    return shuffled[n_val:].tolist(), shuffled[:n_val].tolist()


def interpolate_state(global_state, local_state, alpha):
    out = {}
    for name in global_state:
        if torch.is_floating_point(global_state[name]):
            out[name] = (1.0 - alpha) * global_state[name] + alpha * local_state[name]
        else:
            out[name] = global_state[name].clone()
    return out


def run(partition_name):
    cfg = Config()
    set_seed(cfg.seed)
    (x_train, y_train, subjects_train), (x_test, y_test, _) = load_data()

    if partition_name == "iid":
        partitions = iid_partition(y_train.numpy(), cfg.num_clients, cfg.seed)
    else:
        partitions = subject_partition(subjects_train.numpy(), cfg.num_clients)

    # Fixed client train/validation splits. Validation is only for estimating alpha.
    splits = {
        cid: split_client(indices, cfg.seed + cid)
        for cid, indices in partitions.items()
    }

    global_model = LSTMClassifier(
        input_size=9, hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers, num_classes=6,
    ).to(cfg.device)
    global_state = {k: v.detach().cpu().clone() for k, v in global_model.state_dict().items()}

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"personalization_{partition_name}.csv"
    fields = [
        "partition", "round", "client_id", "alpha",
        "validation_loss", "validation_accuracy",
        "official_test_accuracy_at_alpha0",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for rnd in range(1, cfg.num_rounds + 1):
            client_results = []
            for cid in range(cfg.num_clients):
                train_idx, _ = splits[cid]
                result = train_client(
                    client_id=cid,
                    global_state=global_state,
                    x=x_train[train_idx], y=y_train[train_idx],
                    hidden_size=cfg.hidden_size,
                    num_layers=cfg.num_layers,
                    learning_rate=cfg.learning_rate,
                    batch_size=cfg.batch_size,
                    local_epochs=cfg.local_epochs,
                    device=cfg.device,
                )
                client_results.append(result)

            # Measure alpha on local validation data BEFORE aggregation.
            for result in client_results:
                cid = result.client_id
                _, val_idx = splits[cid]
                local_state = result.state_dict
                best_alpha = None
                best_acc = -1.0

                for alpha in ALPHAS:
                    state = interpolate_state(global_state, local_state, float(alpha))
                    model = LSTMClassifier(
                        input_size=9, hidden_size=cfg.hidden_size,
                        num_layers=cfg.num_layers, num_classes=6,
                    ).to(cfg.device)
                    model.load_state_dict(state)
                    val_loss, val_acc = evaluate(
                        model, x_train[val_idx], y_train[val_idx],
                        cfg.batch_size, cfg.device,
                    )
                    writer.writerow({
                        "partition": partition_name,
                        "round": rnd,
                        "client_id": cid,
                        "alpha": float(alpha),
                        "validation_loss": val_loss,
                        "validation_accuracy": val_acc,
                        "official_test_accuracy_at_alpha0": "",
                    })
                    if val_acc > best_acc:
                        best_acc = val_acc
                        best_alpha = float(alpha)

                # Print best alpha for this client/round.
                print(
                    f"partition={partition_name} round={rnd} client={cid} "
                    f"best_alpha={best_alpha:.1f} val_acc={best_acc:.4f}"
                )

            # FedAvg proceeds exactly as in the existing experiment.
            global_state = fedavg(client_results)
            global_model.load_state_dict(global_state)
            _, global_test_acc = evaluate(
                global_model, x_test, y_test,
                cfg.batch_size, cfg.device,
            )
            print(
                f"partition={partition_name} round={rnd} "
                f"global_test_acc={global_test_acc:.4f}"
            )
            f.flush()

    print(f"Saved {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition", choices=["iid", "subject"], required=True)
    args = parser.parse_args()
    run(args.partition)


if __name__ == "__main__":
    main()
