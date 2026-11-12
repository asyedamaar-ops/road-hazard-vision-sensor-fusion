import pandas as pd
import os

# 1. File Paths (Make sure these match your folder structure exactly)
data_path = os.path.join("Data", "Raw_Data", "PVS 1", "dataset_gps_mpu_left.csv")
labels_path = os.path.join("Data", "Raw_Data", "PVS 1", "dataset_labels.csv")

print("Loading data... This might take a few seconds.")

try:
    # 2. Load the data
    df_data = pd.read_csv(data_path)
    df_labels = pd.read_csv(labels_path)

    # 3. MERGE the datasets side-by-side
    # The PVS dataset is designed so row 1 in data matches row 1 in labels perfectly.
    df = pd.concat([df_data, df_labels], axis=1)

    # 4. Filter for Anomalies (Find the "Bad Roads")
    # We look for rows where the left OR right side of the car hit a "Bad Road"
    bad_road_mask = (df['bad_road_left'] == 1) | (df['bad_road_right'] == 1)
    df_anomalies = df[bad_road_mask].copy()

    print(f"\nFound {len(df_anomalies)} rows of 'Bad Road' data out of {len(df)} total rows.")

    # 5. Apply the "Sinkhole" Severity Threshold
    # We use the sensor mounted directly to the wheel/suspension for maximum accuracy
    target_sensor = 'acc_z_below_suspension'

    # Calculate baseline gravity (usually around 9.8 m/s^2)
    baseline_z = df[target_sensor].median()
    print(f"Baseline Z-Axis (Gravity) is approx: {baseline_z:.2f} m/s²")

    # A sinkhole drop is a sudden freefall. We look for a drop of at least 3.8 m/s^2 below baseline
    severe_drop_threshold = baseline_z - 3.8 

    # Filter the anomalies to find ONLY the severe drops
    sinkhole_proxies = df_anomalies[df_anomalies[target_sensor] < severe_drop_threshold]

    print("-" * 40)
    print(f" SINKHOLE PROXIES FOUND: {len(sinkhole_proxies)} rows ")
    print("-" * 40)

    # 6. Save the results
    output_folder = os.path.join("Data", "Processed_Data")
    os.makedirs(output_folder, exist_ok=True) 

    output_path = os.path.join(output_folder, "sinkhole_events_session1.csv")
    sinkhole_proxies.to_csv(output_path, index=False)
    print(f"\n Saved severe drop events to: {output_path}")

    if len(sinkhole_proxies) > 0:
        print("\nSample Timestamps of Severe Drops (For Video Extraction later):")
        print(sinkhole_proxies[['timestamp', target_sensor, 'speed']].head())

except FileNotFoundError:
    print(" ERROR: Could not find one of the CSV files. Check your Data folder paths!")
except Exception as e:
    print(f" An error occurred: {e}")