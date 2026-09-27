import torch
from torch import nn


class LSTMClassifier(nn.Module):
    """LSTM classifier for UCI HAR raw inertial sequences.

    Expected input shape: (batch, 128 timesteps, 9 sensor channels).
    """

    def __init__(self, input_size=9, hidden_size=64, num_layers=1, num_classes=6):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        if x.ndim != 3 or x.shape[-1] != self.lstm.input_size:
            raise ValueError(f"Expected input shape (batch, timesteps, {self.lstm.input_size}), got {tuple(x.shape)}")
        output, _ = self.lstm(x)
        return self.classifier(output[:, -1, :])
