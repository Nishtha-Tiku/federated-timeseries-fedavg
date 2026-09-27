from src.config import Config
from src.data import load_data
from src.model import LSTMClassifier
from src.train import train_one_epoch, evaluate
from src.utils import set_seed
import torch

def main():
    cfg = Config(); set_seed(cfg.seed)
    (x_train, y_train, _), (x_test, y_test, _) = load_data()
    model = LSTMClassifier(input_size=x_train.shape[-1], hidden_size=cfg.hidden_size, num_layers=cfg.num_layers).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    print(f"Device: {cfg.device}")
    print(f"Train samples: {len(x_train)}")
    print(f"Test samples: {len(x_test)}")
    for epoch in range(3):
        tr_loss, tr_acc = train_one_epoch(model, x_train, y_train, optimizer, criterion, cfg.batch_size, cfg.device)
        te_loss, te_acc = evaluate(model, x_test, y_test, cfg.batch_size, cfg.device)
        print(f"epoch={epoch+1} train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} test_loss={te_loss:.4f} test_acc={te_acc:.4f}")

if __name__ == "__main__":
    main()
