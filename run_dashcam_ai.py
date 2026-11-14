"""Step 4: run the detector on the dashcam video and fuse it with the accelerometer.

    python run_dashcam_ai.py --weights runs/detect/train/weights/best.pt

For every frame the script finds the accelerometer reading taken at the same
moment. A warning is drawn only when the camera sees a crack AND the vertical
acceleration is below the threshold (see fusion.should_warn).

    --no-display        run without a window (for servers)
    --save out.mp4      write the annotated video to a file
"""
import argparse

import cv2
import pandas as pd

import config
from fusion import SensorLookup, crop_road, frame_to_timestamp, should_warn

WARNING_TEXT = 'WARNING: POSSIBLE ROAD HAZARD'


def yolo_detector(weights, conf):
    """Wrap a YOLO model as detect(frame) -> (class names found, annotated frame)."""
    from ultralytics import YOLO

    model = YOLO(str(weights))

    def detect(frame):
        result = model(frame, conf=conf, verbose=False)[0]
        names = [model.names[int(c)] for c in result.boxes.cls]
        return names, result.plot()

    return detect


def run(video_path, sensor, detect, start_minute=config.START_MINUTE,
        acc_threshold=config.LIVE_ACC_THRESHOLD, hazard_class=config.HAZARD_CLASS,
        display=True, save_path=None, max_frames=None):
    """Process the video. Returns the list of frame numbers on which a warning was raised.

    `sensor` is a fusion.SensorLookup and `detect` is a function like the one
    yolo_detector() returns. Passing them in keeps this loop testable with a
    fake detector and a tiny synthetic video.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f'Could not open the video: {video_path}')
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Jump ahead, and start the counter at the SAME frame. The counter is used to
    # look up the sensor reading, so it must be the position in the whole video.
    start_frame = int(start_minute * 60 * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_index = start_frame

    writer = None
    warnings = []
    try:
        while max_frames is None or frame_index - start_frame < max_frames:
            success, frame = cap.read()
            if not success:
                break

            frame = crop_road(frame, config.KEEP_LEFT_FRACTION)
            acc_z = sensor.nearest(frame_to_timestamp(frame_index, sensor.t0, fps))
            detected, annotated = detect(frame)

            cv2.putText(annotated, f'Z-accel: {acc_z:.2f} m/s^2', (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            if should_warn(detected, acc_z, acc_threshold, hazard_class):
                warnings.append(frame_index)
                cv2.putText(annotated, WARNING_TEXT, (20, 100),
                            cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 3)

            if save_path:
                if writer is None:
                    height, width = annotated.shape[:2]
                    writer = cv2.VideoWriter(str(save_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
                writer.write(annotated)
            if display:
                cv2.imshow('Road Hazard Vision Fusion', annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            frame_index += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if display:
            cv2.destroyAllWindows()
    return warnings


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--weights', default=config.WEIGHTS_PATH)
    parser.add_argument('--video', default=config.VIDEO_PATH)
    parser.add_argument('--sensor-csv', default=config.SENSOR_CSV)
    parser.add_argument('--start-minute', type=float, default=config.START_MINUTE)
    parser.add_argument('--acc-threshold', type=float, default=config.LIVE_ACC_THRESHOLD)
    parser.add_argument('--conf', type=float, default=config.CONFIDENCE)
    parser.add_argument('--no-display', action='store_true')
    parser.add_argument('--save', default=None, help='write the annotated video to this file')
    parser.add_argument('--max-frames', type=int, default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.sensor_csv)
    sensor = SensorLookup(df['timestamp'], df[config.ACC_COLUMN])
    detect = yolo_detector(args.weights, args.conf)

    print("Running. Press 'q' in the video window to stop.")
    warnings = run(args.video, sensor, detect, start_minute=args.start_minute,
                   acc_threshold=args.acc_threshold, display=not args.no_display,
                   save_path=args.save, max_frames=args.max_frames)
    print(f'Finished. Warnings raised on {len(warnings)} frames.')


if __name__ == '__main__':
    main()
