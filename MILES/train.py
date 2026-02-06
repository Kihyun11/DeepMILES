import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import configs as cfg  # Import your settings
from model import DeepConvLSTM
from dataset import IMUDataset

def train():
    # 1. Setup Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 2. Initialize Datasets using config values
    train_ds = IMUDataset(cfg.TRAIN_METADATA, cfg.SESSION_CSV, cfg.WINDOW_SIZE, cfg.STEP_SIZE)
    val_ds = IMUDataset(cfg.VAL_METADATA, cfg.SESSION_CSV, cfg.WINDOW_SIZE, cfg.STEP_SIZE)
    
    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False)

    # 3. Initialize Model
    model = DeepConvLSTM(
        input_channels=cfg.INPUT_CHANNELS, 
        num_classes=cfg.NUM_CLASSES,
        conv_kernels=cfg.CONV_KERNELS,
        lstm_units=cfg.LSTM_UNITS
    ).to(device)

    # 4. Optimizer and Loss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)

    # 5. Training Loop
    for epoch in range(cfg.EPOCHS):
        model.train()
        train_loss = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        print(f"Epoch {epoch+1}/{cfg.EPOCHS} | Loss: {train_loss/len(train_loader):.4f}")

    # 6. Save Model
    torch.save(model.state_dict(), cfg.MODEL_SAVE_PATH)
    print(f"Model saved to {cfg.MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()

# import torch.optim as optim
# from model import DeepConvLSTM
# from dataset import IMUDataset

# # Hyperparameters
# WINDOW_SIZE = 128
# BATCH_SIZE = 32
# LR = 0.001
# EPOCHS = 20

# # Initialize Datasets
# train_ds = IMUDataset('metadata_train.csv', 'session.csv', window_size=WINDOW_SIZE)
# train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

# # Initialize Model (Assuming 6 input sensors and N classes)
# num_classes = len(train_ds.label_map)
# model = DeepConvLSTM(input_channels=6, num_classes=num_classes).cuda()

# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=LR)

# # Training Loop
# for epoch in range(EPOCHS):
#     model.train()
#     total_loss = 0
#     for batch_idx, (data, target) in enumerate(train_loader):
#         data, target = data.cuda(), target.cuda()
        
#         optimizer.zero_grad()
#         output = model(data)
#         loss = criterion(output, target)
#         loss.backward()
#         optimizer.step()
        
#         total_loss += loss.item()
        
#     print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(train_loader):.4f}")

# # Save the model for stability checking later
# torch.save(model.state_dict(), 'har_model.pth')