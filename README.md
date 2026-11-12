# Road-Hazard Detection with Vision and Sensor Fusion

Prototype that combines a YOLOv8n road-hazard detector with suspension accelerometer data. A warning is raised only when the camera sees a crack and the vertical acceleration passes a threshold.

Built as a team of 3 in the Renault Nissan Technology and Business Centre India collaboration program at SRM (Jan to Apr 2026).

## My contribution
The pipeline was originally built by my project team (team of 3). My part:
- Re-trained the YOLOv8n detector and evaluated it on a held-out test split (results below).
- Fixed a frame-counter bug in `run_dashcam_ai.py` (the video was seeked to minute 5 but the counter restarted at 0, so the sensor lookup was about 300 seconds off).
- Wrote this documentation.

## Pipeline
1. `extract_sinkholes.py` merges the PVS GPS/IMU data with bad-road labels and keeps severe suspension-drop events.
2. `extract_frames.py` converts event timestamps to video frames and crops the road area.
3. `train_model.py` fine-tunes YOLOv8n (50 epochs, 640 px) on the Sinkhole_Vision_Fusion dataset (187 images, classes Concentric_Crack and Pothole).
4. `run_dashcam_ai.py` runs the detector on the video, looks up the matching accelerometer reading and shows a warning when both conditions hold.

## Results (my re-training run, test split)
| Metric | Value |
|---|---|
| mAP50 | 0.910 |
| mAP50-95 | 0.651 |
| Precision | 0.853 |
| Recall | 0.875 |

Test split: 15 images, 16 Concentric_Crack instances.

## Limitations
- The validation and test splits contain no Pothole boxes, so pothole detection was not evaluated.
- The test set is very small, so scores are noisy. Near-duplicate frames across splits may inflate them.
- "Sinkhole" events are hard suspension drops on bad roads (a proxy). No real sinkhole data was used.
- The acceleration threshold in the live loop (-5.0) differs from the one used for event extraction (median minus 3.8) and has not been validated.
- The fix to the frame counter was not tested against the PVS video.

## Dataset and data
- Detector data: Roboflow, Sinkhole_Vision_Fusion v1, CC BY 4.0.
- Sensor and video data: PVS driving dataset (not included).
