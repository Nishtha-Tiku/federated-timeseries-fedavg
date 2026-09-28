"""Measure client heterogeneity during the existing custom FedAvg experiment.

Run from the project root:
    python -m experiments.measure_heterogeneity --partition iid
    python -m experiments.measure_heterogeneity --partition subject

Outputs:
    results/heterogeneity_<partition>.csv

Metrics:
    label_js              JS divergence of client/global label distributions
    temporal_acf_distance L2 distance between normalized ACF profiles
    representation_distance Euclidean distance between mean LSTM embeddings
    update_cosine_distance 1 - cosine(local_update, aggregated_update)
    parameter_distance      L2(local_model - global_model)
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from federated.client import train_client
from federated.partition import iid_partition, subject_partition
from federated.server import fedavg
from src.config import Config
from src.data import load_data
from src.model import LSTMClassifier
from src.train import evaluate
from src.utils import set_seed

RESULTS_DIR = Path("results")
EPS = 1e-12


def label_js(labels, indices, n_classes=6):
    p = np.bincount(np.asarray(labels)[indices], minlength=n_classes).astype(np.float64)
    p /= max(p.sum(), 1.0)
    return p


def js_divergence(p, q):
    p = np.asarray(p, dtype=np.float64) + EPS
    q = np.asarray(q, dtype=np.float64) + EPS
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    return 0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m))


def flatten_state_difference(a, b):
    parts = []
    for name in a:
        if torch.is_floating_point(a[name]):
            parts.append((a[name].float() - b[name].float()).reshape(-1))
    return torch.cat(parts)


def cosine_distance(a, b):
    denom = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    if denom.item() < EPS:
        return 0.0
    return float(1.0 - torch.dot(a, b).item() / denom.item())


def acf_profile(x, max_lag=20):
    """Mean normalized autocorrelation over samples/channels."""
    # x: [N, T, C]
    x = x.astype(np.float64)
    x = x - x.mean(axis=1, keepdims=True)
    denom = np.sum(x * x, axis=1) + EPS
    profiles = []
    for lag in range(1, max_lag + 1):
        numerator = np.sum(x[:, :-lag, :] * x[:, lag:, :], axis=1)
        # denominator uses all timesteps, giving a stable normalized profile.
        acf = numerator / denom
        profiles.append(acf.mean(axis=(0, 1)))
    return np.asarray(profiles)


def temporal_acf_distance(client_x, global_x, max_lag=20):
    a = acf_profile(client_x, max_lag=max_lag)
    b = acf_profile(global_x, max_lag=max_lag)
    return float(np.linalg.norm(a - b) / np.sqrt(len(a)))


def get_embeddings(model, x, batch_size, device):
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = x[start:start + batch_size].to(device)
            output, _ = model.lstm(xb)
            chunks.append(output[:, -1, :].cpu())
    return torch.cat(chunks, dim=0)


def mean_representation_distance(model, client_x, global_x, batch_size, device):
    client_emb = get_embeddings(model, client_x, batch_size, device).mean(dim=0)
    global_emb = get_embeddings(model, global_x, batch_size, device).mean(dim=0)
    return float(torch.linalg.vector_norm(client_emb - global_emb).item())


def run(partition_name):
    cfg = Config()
    set_seed(cfg.seed)
    (x_train, y_train, subjects_train), (x_test, y_test, _) = load_data()

    if partition_name == "iid":
        partitions = iid_partition(y_train.numpy(), cfg.num_clients, cfg.seed)
    else:
        partitions = subject_partition(subjects_train.numpy(), cfg.num_clients)

    global_model = LSTMClassifier(
        input_size=9,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        num_classes=6,
    ).to(cfg.device)
    global_state = {k: v.detach().cpu().clone() for k, v in global_model.state_dict().items()}

    # Global training-set label distribution and temporal reference.
    global_label_dist = np.bincount(y_train.numpy(), minlength=6).astype(np.float64)
    global_label_dist /= global_label_dist.sum()
    global_x_np = x_train.numpy()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"heterogeneity_{partition_name}.csv"
    fields = [
        "partition", "round", "client_id", "samples",
        "label_js", "temporal_acf_distance", "representation_distance",
        "update_cosine_distance", "parameter_distance",
        "client_train_accuracy", "global_test_accuracy",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for rnd in range(1, cfg.num_rounds + 1):
            client_results = []
            for cid in range(cfg.num_clients):
                idx = partitions[cid]
                result = train_client(
                    client_id=cid,
                    global_state=global_state,
                    x=x_train[idx], y=y_train[idx],
                    hidden_size=cfg.hidden_size,
                    num_layers=cfg.num_layers,
                    learning_rate=cfg.learning_rate,
                    batch_size=cfg.batch_size,
                    local_epochs=cfg.local_epochs,
                    device=cfg.device,
                )
                client_results.append(result)

            new_global_state = fedavg(client_results)
            global_model.load_state_dict(global_state)
            global_update = flatten_state_difference(new_global_state, global_state)

            global_model.load_state_dict(new_global_state)
            global_test_loss, global_test_acc = evaluate(
                global_model, x_test, y_test, cfg.batch_size, cfg.device
            )

            for result in client_results:
                cid = result.client_id
                idx = partitions[cid]
                client_label = label_js(y_train.numpy(), idx)
                label_js_value = js_divergence(client_label, global_label_dist)

                temporal_value = temporal_acf_distance(
                    x_train[idx].numpy(), global_x_np, max_lag=20
                )

                client_model = LSTMClassifier(
                    input_size=9, hidden_size=cfg.hidden_size,
                    num_layers=cfg.num_layers, num_classes=6,
                ).to(cfg.device)
                client_model.load_state_dict(result.state_dict)
                rep_value = mean_representation_distance(
                    client_model, x_train[idx], x_train, cfg.batch_size, cfg.device
                )

                client_update = flatten_state_difference(result.state_dict, global_state)
                update_cos = cosine_distance(client_update, global_update)
                param_dist = float(torch.linalg.vector_norm(client_update).item())

                writer.writerow({
                    "partition": partition_name,
                    "round": rnd,
                    "client_id": cid,
                    "samples": result.num_samples,
                    "label_js": label_js_value,
                    "temporal_acf_distance": temporal_value,
                    "representation_distance": rep_value,
                    "update_cosine_distance": update_cos,
                    "parameter_distance": param_dist,
                    "client_train_accuracy": result.train_acc,
                    "global_test_accuracy": global_test_acc,
                })

            f.flush()
            print(
                f"partition={partition_name} round={rnd} "
                f"global_test_acc={global_test_acc:.4f}"
            )
            global_state = new_global_state

    print(f"Saved {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition", choices=["iid", "subject"], required=True)
    args = parser.parse_args()
    run(args.partition)


if __name__ == "__main__":
    main()
