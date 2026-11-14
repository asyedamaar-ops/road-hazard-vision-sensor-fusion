"""All paths and thresholds in one place.

Every value can be changed without editing code, either with an environment
variable (for example ROAD_HAZARD_DATA_DIR) or with a command-line option of
the script you run.
"""
import os
from pathlib import Path

# folder that holds the PVS driving dataset (not included in this repository)
DATA_DIR = Path(os.environ.get('ROAD_HAZARD_DATA_DIR', 'Data'))
PVS_SESSION_DIR = DATA_DIR / 'Raw_Data' / 'PVS 1'
PROCESSED_DIR = DATA_DIR / 'Processed_Data'

SENSOR_CSV = PVS_SESSION_DIR / 'dataset_gps_mpu_left.csv'
LABELS_CSV = PVS_SESSION_DIR / 'dataset_labels.csv'
VIDEO_PATH = PVS_SESSION_DIR / 'video_environment_dataset_left.mp4'
EVENTS_CSV = PROCESSED_DIR / 'sinkhole_events_session1.csv'
FRAMES_DIR = PROCESSED_DIR / 'Extracted_Frames'

# YOLO dataset (Roboflow export) and trained weights
DATASET_YAML = Path(os.environ.get('ROAD_HAZARD_DATASET_YAML', 'Sinkhole_Vision_Fusion-1/data.yaml'))
WEIGHTS_PATH = Path(os.environ.get('ROAD_HAZARD_WEIGHTS', 'runs/detect/train/weights/best.pt'))

# sensor column: accelerometer mounted below the suspension, vertical axis
ACC_COLUMN = 'acc_z_below_suspension'

# event extraction: a severe drop is this far below the median reading (m/s^2)
SEVERE_DROP_BELOW_MEDIAN = 3.8
# live loop: warn when the reading falls below this absolute value (m/s^2).
# NOTE: this is a different rule from the one above and has not been validated.
LIVE_ACC_THRESHOLD = -5.0

# detector settings
CONFIDENCE = 0.5
HAZARD_CLASS = 'Concentric_Crack'

# the right half of the PVS video shows plotted graphs, so only the left part is used
KEEP_LEFT_FRACTION = 0.50
# the live demo starts this many minutes into the video
START_MINUTE = 5.0
# keep every Nth severe-drop row, to avoid many frames of the same spot
EVENT_SAMPLE_STEP = 50
