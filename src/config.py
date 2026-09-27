from dataclasses import dataclass
import torch

@dataclass(frozen=True)
class Config:
    seed: int = 42
    num_clients: int = 4
    batch_size: int = 64
    hidden_size: int = 64
    num_layers: int = 1
    learning_rate: float = 1e-3
    num_rounds: int = 5
    local_epochs: int = 1

    @property
    def device(self):
        return "cuda" if torch.cuda.is_available() else "cpu"
