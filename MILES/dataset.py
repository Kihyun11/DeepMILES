import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class IMUDataset(Dataset):
    def __init__(self, metadata_csv, session_csv, window_size=128, step_size=64):
        self.metadata = pd.read_csv(metadata_csv)
        self.sessions = pd.read_csv(session_csv)

        # Robust: strip accidental spaces in headers (you hit this already)
        self.metadata.columns = self.metadata.columns.str.strip()
        self.sessions.columns = self.sessions.columns.str.strip()

        self.window_size = window_size
        self.step_size = step_size

        self.data_windows = []
        self.labels = []
        self.window_session_ids = []   # NEW: keep which session produced each window

        # Mapping labels to integers
        if "label" not in self.metadata.columns:
            raise KeyError(f"metadata_csv missing 'label' column. Found: {list(self.metadata.columns)}")
        self.label_map = {label: i for i, label in enumerate(self.metadata["label"].unique())}

        self._prepare_data()

    def _clean_path(self, p: str) -> str:
        # robust: handle accidental quotes/spaces in session.csv paths
        p = str(p).strip()
        p = p.strip('"').strip("'").strip()
        return p

    def _prepare_data(self):
        required_session_cols = {"session_id", "file_path"}
        if not required_session_cols.issubset(set(self.sessions.columns)):
            raise KeyError(f"session_csv must contain {required_session_cols}. Found: {list(self.sessions.columns)}")

        for _, row in self.metadata.iterrows():
            sid = row["session_id"]

            # 1) Find file_path from session.csv
            session_info = self.sessions[self.sessions["session_id"] == sid]
            if len(session_info) == 0:
                raise KeyError(f"session_id={sid} not found in session.csv")
            file_path = self._clean_path(session_info.iloc[0]["file_path"])

            # 2) Read raw IMU
            raw_df = pd.read_csv(file_path)
            raw_df.columns = raw_df.columns.str.strip()

            # 3) Trim by time
            # Your pipeline uses seconds_elapsed in plotting & later dataset version
            # (keep as-is for your current data format)
            if "seconds_elapsed" not in raw_df.columns:
                raise KeyError(f"Raw IMU CSV '{file_path}' missing 'seconds_elapsed'. Found: {list(raw_df.columns)}")

            needed_cols = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
            for c in needed_cols:
                if c not in raw_df.columns:
                    raise KeyError(f"Raw IMU CSV '{file_path}' missing '{c}'. Found: {list(raw_df.columns)}")

            start_t = float(row["start_time"])
            end_t = float(row["end_time"])

            mask = (raw_df["seconds_elapsed"] >= start_t) & (raw_df["seconds_elapsed"] <= end_t)
            trimmed = raw_df.loc[mask, needed_cols].values

            # 4) Sliding windows
            for i in range(0, len(trimmed) - self.window_size, self.step_size):
                window = trimmed[i:i + self.window_size]
                self.data_windows.append(window)
                self.labels.append(self.label_map[row["label"]])
                self.window_session_ids.append(sid)

    def __len__(self):
        return len(self.data_windows)

    def __getitem__(self, idx):
        x = torch.tensor(self.data_windows[idx], dtype=torch.float32).unsqueeze(0)  # (1, T, 6)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        sid = self.window_session_ids[idx]
        return x, y, sid


#import torch
# from torch.utils.data import Dataset, DataLoader
# import pandas as pd
# import numpy as np

# class IMUDataset(Dataset):
#     def __init__(self, metadata_csv, session_csv, window_size=128, step_size=64):
#         """
#         Args:
#             metadata_csv: Path to metadata_train.csv (or val/test)
#             session_csv: Path to original session.csv (mapping session_id to file_path)
#             window_size: Number of time steps per sample for the LSTM
#             step_size: Overlap between windows
#         """
#         self.metadata = pd.read_csv(metadata_csv)
#         self.sessions = pd.read_csv(session_csv)
#         self.window_size = window_size
#         self.step_size = step_size
        
#         self.data_windows = []
#         self.labels = []
        
#         # Mapping labels to integers
#         self.label_map = {label: i for i, label in enumerate(self.metadata['label'].unique())}
        
#         self._prepare_data()

#     def _prepare_data(self):
#         for _, row in self.metadata.iterrows():
#             # 1. Find the file path from session.csv
#             session_info = self.sessions[self.sessions['session_id'] == row['session_id']].iloc[0]
#             file_path = session_info['file_path']
            
#             # 2. Read the raw IMU data
#             raw_df = pd.read_csv(file_path)
            
#             # 3. Trim based on start_time and end_time
#             mask = (raw_df['seconds_elapsed'] >= row['start_time']) & (raw_df['seconds_elapsed'] <= row['end_time'])
#             trimmed_data = raw_df.loc[mask, ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']].values
            
#             # 4. Sliding Window Extraction
#             for i in range(0, len(trimmed_data) - self.window_size, self.step_size):
#                 window = trimmed_data[i : i + self.window_size]
#                 self.data_windows.append(window)
#                 self.labels.append(self.label_map[row['label']])

#     def __len__(self):
#         return len(self.data_windows)

#     def __getitem__(self, idx):
#         # Convert to tensor and add channel dimension for Conv2D: (1, window_size, 6)
#         x = torch.tensor(self.data_windows[idx], dtype=torch.float32).unsqueeze(0)
#         y = torch.tensor(self.labels[idx], dtype=torch.long)
#         return x, y