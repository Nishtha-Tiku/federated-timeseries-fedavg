import numpy as np
from collections import Counter


def iid_partition(labels, num_clients, seed=42):
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(labels))
    return {cid: indices[cid::num_clients].tolist() for cid in range(num_clients)}


def dirichlet_partition(labels, num_clients, concentration, seed=42, min_size=1):
    """Partition samples with controllable label-distribution heterogeneity.

    Larger concentration values produce client label distributions closer to
    IID; smaller values produce stronger statistical heterogeneity.
    """
    if concentration <= 0:
        raise ValueError("concentration must be > 0")

    labels = np.asarray(labels)
    classes = np.unique(labels)
    rng = np.random.default_rng(seed)

    for _ in range(100):
        client_indices = [[] for _ in range(num_clients)]

        for cls in classes:
            cls_indices = np.where(labels == cls)[0]
            rng.shuffle(cls_indices)

            proportions = rng.dirichlet(
                np.full(num_clients, concentration, dtype=np.float64)
            )
            counts = rng.multinomial(len(cls_indices), proportions)

            start = 0
            for cid, count in enumerate(counts):
                client_indices[cid].extend(
                    cls_indices[start:start + count].tolist()
                )
                start += count

        sizes = [len(indices) for indices in client_indices]
        if min(sizes) >= min_size:
            for cid in range(num_clients):
                rng.shuffle(client_indices[cid])
            return {cid: client_indices[cid] for cid in range(num_clients)}

    raise RuntimeError(
        f"Could not create a valid Dirichlet partition after 100 attempts "
        f"(concentration={concentration}, min_size={min_size})."
    )


def subject_partition(subjects, num_clients):
    unique_subjects = sorted(np.unique(subjects))
    groups = np.array_split(unique_subjects, num_clients)
    return {
        cid: np.where(np.isin(subjects, group))[0].tolist()
        for cid, group in enumerate(groups)
    }


def summarize_partition(labels, partitions):
    labels = np.asarray(labels)
    result = {}
    for cid, indices in partitions.items():
        counts = Counter(labels[indices].tolist())
        result[cid] = {
            "samples": len(indices),
            "class_counts": dict(sorted(counts.items())),
        }
    return result
