import pandas as pd
import matplotlib.pyplot as plt

def analyze_imu_data_stacked(file_path):
    df = pd.read_csv(file_path)
    t = df["seconds_elapsed"]

    channels = [
        ("acc_x", "Acc X", "m/s² (or g)"),
        ("acc_y", "Acc Y", "m/s² (or g)"),
        ("acc_z", "Acc Z", "m/s² (or g)"),
        ("gyro_x", "Gyro X", "deg/s (or rad/s)"),
        ("gyro_y", "Gyro Y", "deg/s (or rad/s)"),
        ("gyro_z", "Gyro Z", "deg/s (or rad/s)"),
    ]

    fig, axes = plt.subplots(len(channels), 1, figsize=(12, 12), sharex=True)

    for ax, (col, title, ylab) in zip(axes, channels):
        if col not in df.columns:
            raise KeyError(f"Missing column '{col}'. Available columns: {list(df.columns)}")

        ax.plot(t, df[col], alpha=0.8)
        ax.set_title(title)
        ax.set_ylabel(ylab)
        ax.grid(True, linestyle="--", alpha=0.6)

    axes[-1].set_xlabel("Elapsed Time (s)")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    analyze_imu_data_stacked(
        r"C:\Users\User\Documents\GitHub\Deep-MILES-Personalized-Performance-Evaluation-AI-Model-for-Next-Gen-KCTC\dataset\walk5\walk5_acc_gyro.csv"
    )
