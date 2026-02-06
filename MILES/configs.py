# config.py

# File Paths
SESSION_CSV = 'session.csv'
TRAIN_METADATA = 'metadata_train.csv'
VAL_METADATA = 'metadata_val.csv'
MODEL_SAVE_PATH = 'har_model.pth'

# Data Processing
WINDOW_SIZE = 128
STEP_SIZE = 64
INPUT_CHANNELS = 6  # acc_x,y,z + gyro_x,y,z

# Training Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 30
DROPOUT = 0.5

# Model Architecture
CONV_KERNELS = 64
LSTM_UNITS = 128
NUM_CLASSES = 6  # Update this based on your actual label count