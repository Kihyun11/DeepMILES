import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

class IMUDataset(Dataset):
    def __init__(self, metadata_csv, session_csv, window_size=128, step_size=64):
        """
        Args:
            metadata_csv: Path to metadata_train.csv (or val/test)
            session_csv: Path to original session.csv (mapping session_id to file_path)
            window_size: Number of time steps per sample for the LSTM
            step_size: Overlap between windows
        """
        self.metadata = pd.read_csv(metadata_csv)
        self.sessions = pd.read_csv(session_csv)
        self.window_size = window_size
        self.step_size = step_size
        
        self.data_windows = []
        self.labels = []
        
        # Mapping labels to integers
        self.label_map = {label: i for i, label in enumerate(self.metadata['label'].unique())}
        
        self._prepare_data()

    def _prepare_data(self):
        for _, row in self.metadata.iterrows():
            # 1. Find the file path from session.csv
            session_info = self.sessions[self.sessions['session_id'] == row['session_id']].iloc[0]
            file_path = session_info['file_path']
            
            # 2. Read the raw IMU data
            raw_df = pd.read_csv(file_path)
            
            # 3. Trim based on start_time and end_time
            mask = (raw_df['seconds_elapsed'] >= row['start_time']) & (raw_df['seconds_elapsed'] <= row['end_time'])
            trimmed_data = raw_df.loc[mask, ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']].values
            
            # 4. Sliding Window Extraction
            for i in range(0, len(trimmed_data) - self.window_size, self.step_size):
                window = trimmed_data[i : i + self.window_size]
                self.data_windows.append(window)
                self.labels.append(self.label_map[row['label']])

    def __len__(self):
        return len(self.data_windows)

    def __getitem__(self, idx):
        # Convert to tensor and add channel dimension for Conv2D: (1, window_size, 6)
        x = torch.tensor(self.data_windows[idx], dtype=torch.float32).unsqueeze(0)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y