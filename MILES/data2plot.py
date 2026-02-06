import pandas as pd
import matplotlib.pyplot as plt

def analyze_imu_data(file_path):
    # Load the dataset
    # Assuming the first row contains the headers you mentioned
    df = pd.read_csv(file_path)

    # Create the figure with two subplots (Accelerometer and Gyroscope)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Plot Accelerometer Data
    ax1.plot(df['elapsed time'], df['acc_x'], label='Acc X', alpha=0.7)
    ax1.plot(df['elapsed time'], df['acc_y'], label='Acc Y', alpha=0.7)
    ax1.plot(df['elapsed time'], df['acc_z'], label='Acc Z', alpha=0.7)
    ax1.set_title('Accelerometer Data')
    ax1.set_ylabel('m/s^2 (or g)')
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Plot Gyroscope Data
    ax2.plot(df['elapsed time'], df['gyro_x'], label='Gyro X', alpha=0.7)
    ax2.plot(df['elapsed time'], df['gyro_y'], label='Gyro Y', alpha=0.7)
    ax2.plot(df['elapsed time'], df['gyro_z'], label='Gyro Z', alpha=0.7)
    ax2.set_title('Gyroscope Data')
    ax2.set_ylabel('deg/s (or rad/s)')
    ax2.set_xlabel('Elapsed Time (s)')
    ax2.legend(loc='upper right')
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()

# Replace 'data.csv' with your actual filename
analyze_imu_data('data.csv')