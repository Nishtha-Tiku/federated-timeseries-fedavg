import torch
from src.model import LSTMClassifier

def test_lstm_accepts_uci_har_raw_signal_shape():
    torch.set_num_threads(1)
    x=torch.randn(2,128,9)
    model=LSTMClassifier(input_size=9,hidden_size=8,num_layers=1,num_classes=6)
    model.eval()
    with torch.no_grad(): logits=model(x)
    assert logits.shape==(2,6)
