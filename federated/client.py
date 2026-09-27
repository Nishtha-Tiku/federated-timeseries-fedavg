from dataclasses import dataclass
from typing import Dict
import torch
from src.model import LSTMClassifier
from src.train import train_one_epoch

@dataclass
class ClientResult:
    client_id: int
    num_samples: int
    state_dict: Dict[str, torch.Tensor]
    train_loss: float
    train_acc: float

def train_client(client_id, global_state, x, y, hidden_size, num_layers, learning_rate, batch_size, local_epochs, device):
    model = LSTMClassifier(input_size=x.shape[-1], hidden_size=hidden_size, num_layers=num_layers, num_classes=6).to(device)
    model.load_state_dict(global_state)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    train_loss = train_acc = 0.0
    for _ in range(local_epochs):
        train_loss, train_acc = train_one_epoch(model, x, y, optimizer, criterion, batch_size=batch_size, device=device)
    state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
    return ClientResult(client_id, len(x), state, train_loss, train_acc)
