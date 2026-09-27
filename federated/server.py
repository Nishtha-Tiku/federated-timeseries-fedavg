from typing import Dict, Iterable
import torch
from federated.client import ClientResult

def fedavg(results: Iterable[ClientResult]) -> Dict[str, torch.Tensor]:
    results = list(results)
    if not results:
        raise ValueError("FedAvg requires at least one client result")
    total_samples = sum(result.num_samples for result in results)
    if total_samples <= 0:
        raise ValueError("Total client sample count must be positive")
    aggregated = {}
    for name in results[0].state_dict:
        first = results[0].state_dict[name]
        if torch.is_floating_point(first):
            aggregated[name] = sum(result.state_dict[name] * (result.num_samples / total_samples) for result in results)
        else:
            aggregated[name] = first.clone()
    return aggregated
