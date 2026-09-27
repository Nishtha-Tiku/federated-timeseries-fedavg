import torch
from src.data import make_loader

def train_one_epoch(model, x, y, optimizer, criterion, batch_size=64, device="cpu"):
    model.train()
    loader = make_loader(x, y, batch_size, True)
    total_loss = total_correct = total = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(yb)
        total_correct += (logits.argmax(1) == yb).sum().item()
        total += len(yb)
    return total_loss / total, total_correct / total

def evaluate(model, x, y, batch_size=64, device="cpu"):
    model.eval()
    loader = make_loader(x, y, batch_size, False)
    criterion = torch.nn.CrossEntropyLoss()
    total_loss = total_correct = total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            total_loss += loss.item() * len(yb)
            total_correct += (logits.argmax(1) == yb).sum().item()
            total += len(yb)
    return total_loss / total, total_correct / total
