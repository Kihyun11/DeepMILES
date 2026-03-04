# dataset_acc_only.py
import torch
from torch.utils.data import Dataset
import pandas as pd

class IMUDatasetAccOnly(Dataset):
    """
    Acc-only dataset (acc_x, acc_y, acc_z) from the same raw CSV.
    Output shape: x = (1, T, 3)
    """
    def __init__(self, metadata_csv, session_csv, window_size=128, step_size=64):
        self.metadata = pd.read_csv(metadata_csv)
        self.sessions = pd.read_csv(session_csv)

        self.metadata.columns = self.metadata.columns.str.strip()
        self.sessions.columns = self.sessions.columns.str.strip()

        self.window_size = window_size
        self.step_size = step_size

        self.data_windows = []
        self.labels = []
        self.window_session_ids = []

        if "label" not in self.metadata.columns:
            raise KeyError(f"metadata_csv missing 'label'. Found: {list(self.metadata.columns)}")
        self.label_map = {label: i for i, label in enumerate(self.metadata["label"].unique())}

        self._prepare_data()

    def _clean_path(self, p: str) -> str:
        p = str(p).strip()
        p = p.strip('"').strip("'").strip()
        return p

    def _prepare_data(self):
        required_session_cols = {"session_id", "file_path"}
        if not required_session_cols.issubset(set(self.sessions.columns)):
            raise KeyError(f"session_csv must contain {required_session_cols}. Found: {list(self.sessions.columns)}")

        for _, row in self.metadata.iterrows():
            sid = row["session_id"]

            session_info = self.sessions[self.sessions["session_id"] == sid]
            if len(session_info) == 0:
                raise KeyError(f"session_id={sid} not found in session.csv")
            file_path = self._clean_path(session_info.iloc[0]["file_path"])

            raw_df = pd.read_csv(file_path)
            raw_df.columns = raw_df.columns.str.strip()

            if "seconds_elapsed" not in raw_df.columns:
                raise KeyError(f"Raw IMU CSV '{file_path}' missing 'seconds_elapsed'. Found: {list(raw_df.columns)}")

            needed_cols = ["acc_x", "acc_y", "acc_z"]
            for c in needed_cols:
                if c not in raw_df.columns:
                    raise KeyError(f"Raw IMU CSV '{file_path}' missing '{c}'. Found: {list(raw_df.columns)}")

            start_t = float(row["start_time"])
            end_t = float(row["end_time"])
            mask = (raw_df["seconds_elapsed"] >= start_t) & (raw_df["seconds_elapsed"] <= end_t)

            trimmed = raw_df.loc[mask, needed_cols].values  # (N, 3)

            for i in range(0, len(trimmed) - self.window_size, self.step_size):
                window = trimmed[i:i + self.window_size]  # (T, 3)
                self.data_windows.append(window)
                self.labels.append(self.label_map[row["label"]])
                self.window_session_ids.append(sid)

    def __len__(self):
        return len(self.data_windows)

    def __getitem__(self, idx):
        x = torch.tensor(self.data_windows[idx], dtype=torch.float32).unsqueeze(0)  # (1, T, 3)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        sid = self.window_session_ids[idx]
        return x, y, sid