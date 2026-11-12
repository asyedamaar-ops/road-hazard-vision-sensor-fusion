import pandas as pd
import cv2
import os

# 1. Using the Environment video to simulate a Dashcam
video_filename = "video_environment_dataset_left.mp4"

# 2. File Paths
video_path = os.path.join("Data", "Raw_Data", "PVS 1", video_filename)
events_path = os.path.join("Data", "Processed_Data", "sinkhole_events_session1.csv")
full_data_path = os.path.join("Data", "Raw_Data", "PVS 1", "dataset_gps_mpu_left.csv")

output_folder = os.path.join("Data", "Processed_Data", "Extracted_Frames")
os.makedirs(output_folder, exist_ok=True)

print(f"Loading data to sync sensor drops with {video_filename}...")

try:
    # 3. Load the data
    df_events = pd.read_csv(events_path)
    df_full = pd.read_csv(full_data_path)
    
    # Get "Time Zero" (The exact moment the sensors started recording)
    t0 = df_full['timestamp'].min()
    print(f"Sensor Start Time (T0): {t0}")
    
    # 4. Open the Video File
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f" ERROR: Could not open {video_filename}. Make sure it is in the PVS 1 folder!")
        exit()
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video loaded! FPS: {fps}, Total Frames: {total_frames}")
    
    # 5. Extract the Frames!
    # We grab every 50th severe drop to avoid extracting identical, consecutive frames of the same hole
    sample_events = df_events.iloc[::50] 
    
    print(f"\nExtracting {len(sample_events)} unique dashcam frames for YOLO training...")
    
    count = 0
    for index, row in sample_events.iterrows():
        event_time = row['timestamp']
        
        # Calculate how many seconds into the video the event happened
        seconds_in = event_time - t0
        
        # Calculate the exact frame number
        target_frame = int(seconds_in * fps)
        
        # Fast-forward the video to that exact frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        
        # Read the frame
        # Read the frame
        success, frame = cap.read()
        
        if success:
            #  NEW CODE: Crop the frame to keep only the left 55% (the road) 
            height, width, _ = frame.shape
            cropped_frame = frame[:, :int(width * 0.50)] 
            
            # Save the CROPPED image
            image_name = f"dashcam_sinkhole_proxy_{target_frame}.jpg"
            save_path = os.path.join(output_folder, image_name)
            cv2.imwrite(save_path, cropped_frame) # Make sure this says cropped_frame!
            count += 1
            
    cap.release()
    print("-" * 40)
    print(f" SUCCESS! Extracted {count} images to: {output_folder}")
    print("-" * 40)

except FileNotFoundError:
    print(" ERROR: Could not find one of the CSV files. Did you move them?")
except Exception as e:
    print(f" An error occurred: {e}")