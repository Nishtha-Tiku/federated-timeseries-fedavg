import torch
from federated.client import ClientResult
from federated.server import fedavg

def test_fedavg_is_sample_weighted():
    state_a={"weight":torch.tensor([1.0])}; state_b={"weight":torch.tensor([3.0])}
    results=[ClientResult(0,1,state_a,0.0,0.0),ClientResult(1,3,state_b,0.0,0.0)]
    aggregated=fedavg(results)
    assert torch.allclose(aggregated["weight"],torch.tensor([2.5]))
