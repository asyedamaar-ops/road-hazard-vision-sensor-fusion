import cv2
import pandas as pd
from ultralytics import YOLO
import os

print(" Initiating Final Protocol: Vision + Sensor Fusion...")

# --- EXACT PATHS FROM YOUR WORKSPACE ---
model_path = r"E:\Vehicle-Damage-Detection\runs\detect\train4\weights\best.pt"
video_path = os.path.join("Data", "Raw_Data", "PVS 1", "video_environment_dataset_left.mp4")
sensor_data_path = os.path.join("Data", "Raw_Data", "PVS 1", "dataset_gps_mpu_left.csv")

try:
    # --- 1. LOAD AI AND SENSOR DATA ---
    print(" Loading YOLOv8 Brain...")
    model = YOLO(model_path)

    print(" Loading Accelerometer Data...")
    df = pd.read_csv(sensor_data_path)
    t0 = df['timestamp'].min() # Establish Time Zero

    print(" Opening Dashcam Feed...")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame = int(5 * 60 * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_count = start_frame
    print("\n System Active. Press 'q' in the video window to safely shut down.")

    # --- 2. MAIN REAL-TIME LOOP ---
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print(" End of video reached.")
            break

        # A. The Crop: Remove the old raw graphs from the right side of the video
        height, width, _ = frame.shape
        frame = frame[:, :int(width * 0.50)]

        # B. The Sync: Find the exact sensor reading for this specific video frame
        current_video_time = frame_count / fps
        current_timestamp = t0 + current_video_time
        
        # Grab the single closest row of sensor data
        sensor_row = df.iloc[(df['timestamp'] - current_timestamp).abs().argsort()[:1]]
        current_az = sensor_row['acc_z_below_suspension'].values[0]

        # C. The Vision: Let the RTX 4060 scan the frame for hazards
        results = model(frame, conf=0.5, verbose=False)
        
        # plot() draws the beautiful bounding boxes automatically
        annotated_frame = results[0].plot() 

        # D. The Fusion Logic
        detected_classes = [model.names[int(c)] for c in results[0].boxes.cls]
        warning_triggered = False
        
        if "Concentric_Crack" in detected_classes:
            # Gravity is ~9.8. If it drops past -5.0, the car is falling into a hole.
            if current_az < -5.0: 
                warning_triggered = True

        # --- 3. HEADS UP DISPLAY (HUD) ---
        # Print the live Z-Axis gravity in the top left corner
        cv2.putText(annotated_frame, f"Live Z-Accel: {current_az:.2f} m/s^2", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        # Flash the massive red warning if fusion conditions are met!
        if warning_triggered:
            cv2.putText(annotated_frame, "CRITICAL: SINKHOLE IMMINENT", (20, 100),
                        cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 3)

        # Display the final fused video on your screen
        cv2.imshow("Sinkhole Vision Fusion AI", annotated_frame)

        # Safety escape key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

except Exception as e:
    print(f"\n CRITICAL ERROR: {e}")
    print("Make sure your hard drive is plugged in and the file paths are correct!")

finally:
    # Always clean up windows when done
    if 'cap' in locals() and cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    print(" System Shut Down Successfully.")
