# train_lstm_only.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import configs as cfg
from dataset import IMUDataset
from model_lstm_only import LSTMOnly

LSTM_ONLY_MODEL_PATH = "model_lstm_only.pth"


def unpack_batch(batch):
    """
    Supports both:
      - (x, y)
      - (x, y, sid)
    """
    if len(batch) == 2:
        data, target = batch
    elif len(batch) == 3:
        data, target, _sid = batch
    else:
        raise ValueError(f"Unexpected batch format with length {len(batch)}")
    return data, target


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in loader:
            data, target = unpack_batch(batch)
            data, target = data.to(device), target.to(device)

            logits = model(data)
            loss = criterion(logits, target)

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == target).sum().item()
            total_samples += target.size(0)

    avg_loss = total_loss / max(len(loader), 1)
    acc = total_correct / max(total_samples, 1)
    return avg_loss, acc


def train_lstm_only():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = IMUDataset(cfg.TRAIN_METADATA, cfg.SESSION_CSV, cfg.WINDOW_SIZE, cfg.STEP_SIZE)
    val_ds = IMUDataset(cfg.VAL_METADATA, cfg.SESSION_CSV, cfg.WINDOW_SIZE, cfg.STEP_SIZE)

    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False)

    model = LSTMOnly(
        input_channels=cfg.INPUT_CHANNELS,
        num_classes=cfg.NUM_CLASSES,
        lstm_units=cfg.LSTM_UNITS,
        num_layers=2,
        dropout=cfg.DROPOUT,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)

    best_val_acc = 0.0

    for epoch in range(cfg.EPOCHS):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            data, target = unpack_batch(batch)
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            logits = model(data)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= max(len(train_loader), 1)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(
            f"[LSTM-ONLY] Epoch {epoch+1}/{cfg.EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc*100:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), LSTM_ONLY_MODEL_PATH)

    print(f"[LSTM-ONLY] Best model saved to {LSTM_ONLY_MODEL_PATH} | Best Val Acc: {best_val_acc*100:.2f}%")


if __name__ == "__main__":
    train_lstm_only()