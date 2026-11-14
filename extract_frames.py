"""Step 2: save the video frame for each severe-drop event, for labelling.

    python extract_frames.py

Converts each event timestamp to a frame number, reads that frame and keeps
the road half of the picture.
"""
import argparse
import os

import cv2
import pandas as pd

import config
from fusion import crop_road, timestamp_to_frame


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--video', default=config.VIDEO_PATH)
    parser.add_argument('--events-csv', default=config.EVENTS_CSV)
    parser.add_argument('--sensor-csv', default=config.SENSOR_CSV)
    parser.add_argument('--output-dir', default=config.FRAMES_DIR)
    parser.add_argument('--step', type=int, default=config.EVENT_SAMPLE_STEP,
                        help='keep every Nth event, to avoid many frames of the same spot')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    df_events = pd.read_csv(args.events_csv)
    # "time zero": the moment the sensors started recording
    t0 = pd.read_csv(args.sensor_csv, usecols=['timestamp'])['timestamp'].min()
    print(f'Sensor start time (t0): {t0}')

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f'Could not open the video: {args.video}')
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f'Video loaded. FPS: {fps}, frames: {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}')

    sample_events = df_events.iloc[::args.step]
    print(f'Extracting {len(sample_events)} frames...')

    count = 0
    for _, row in sample_events.iterrows():
        target_frame = timestamp_to_frame(row['timestamp'], t0, fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        success, frame = cap.read()
        if success:
            image_name = f'dashcam_sinkhole_proxy_{target_frame}.jpg'
            cv2.imwrite(os.path.join(str(args.output_dir), image_name),
                        crop_road(frame, config.KEEP_LEFT_FRACTION))
            count += 1

    cap.release()
    print(f'Extracted {count} images to: {args.output_dir}')


if __name__ == '__main__':
    main()
