import numpy as np
from collections import Counter

def iid_partition(labels, num_clients, seed=42):
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(labels))
    return {cid: indices[cid::num_clients].tolist() for cid in range(num_clients)}

def subject_partition(subjects, num_clients):
    unique_subjects = sorted(np.unique(subjects))
    groups = np.array_split(unique_subjects, num_clients)
    return {cid: np.where(np.isin(subjects, group))[0].tolist() for cid, group in enumerate(groups)}

def summarize_partition(labels, partitions):
    labels = np.asarray(labels)
    result = {}
    for cid, indices in partitions.items():
        counts = Counter(labels[indices].tolist())
        result[cid] = {"samples": len(indices), "class_counts": dict(sorted(counts.items()))}
    return result
