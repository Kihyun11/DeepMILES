# train_acc_only.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import configs as cfg
from model import DeepConvLSTM
from dataset_acc_only import IMUDatasetAccOnly

ACC_ONLY_MODEL_PATH = "model_acc_only.pth"

def train_acc_only():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = IMUDatasetAccOnly(cfg.TRAIN_METADATA, cfg.SESSION_CSV, cfg.WINDOW_SIZE, cfg.STEP_SIZE)
    val_ds   = IMUDatasetAccOnly(cfg.VAL_METADATA,   cfg.SESSION_CSV, cfg.WINDOW_SIZE, cfg.STEP_SIZE)

    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.BATCH_SIZE, shuffle=False)  # (optional)

    model = DeepConvLSTM(
        input_channels=3,
        num_classes=cfg.NUM_CLASSES,
        conv_kernels=cfg.CONV_KERNELS,
        lstm_units=cfg.LSTM_UNITS
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)

    for epoch in range(cfg.EPOCHS):
        model.train()
        train_loss = 0.0

        for data, target, _sid in train_loader:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            logits = model(data)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        print(f"[ACC-ONLY] Epoch {epoch+1}/{cfg.EPOCHS} | Loss: {train_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), ACC_ONLY_MODEL_PATH)
    print(f"[ACC-ONLY] Model saved to {ACC_ONLY_MODEL_PATH}")

if __name__ == "__main__":
    train_acc_only()