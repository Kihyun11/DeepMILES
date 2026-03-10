import pandas as pd

# =========================
# Configuration
# =========================
SAMPLING_RATE = 100  # Hz
WINDOW_SIZE = 128    # samples
STEP_SIZE = 64       # samples

CSV_FILE = "action_labels.csv"


# =========================
# Function to compute windows
# =========================
def compute_windows(duration_seconds):
    
    total_samples = duration_seconds * SAMPLING_RATE
    
    if total_samples < WINDOW_SIZE:
        return 0
    
    n_windows = int((total_samples - WINDOW_SIZE) / STEP_SIZE) + 1
    
    return n_windows


# =========================
# Load CSV
# =========================
df = pd.read_csv(CSV_FILE)

# duration
df["duration"] = df["end_time"] - df["start_time"]

# compute windows per session
df["num_windows"] = df["duration"].apply(compute_windows)


# =========================
# Print per-session results
# =========================
print("\n===== WINDOWS PER SESSION =====")
print(df[["session_id", "label", "duration", "num_windows"]])


# =========================
# Class statistics
# =========================
stats = df.groupby("label")["num_windows"].agg(
    total_windows="sum",
    avg_windows="mean",
    num_sessions="count"
).reset_index()


print("\n===== CLASS STATISTICS =====")
print(stats)


# =========================
# Total dataset windows
# =========================
total_windows = df["num_windows"].sum()

print("\n===== DATASET SUMMARY =====")
print("Total sessions:", len(df))
print("Total windows:", total_windows)