"""End-to-end test of the live loop with a tiny synthetic video and a fake detector."""
import cv2
import numpy as np
import pytest

from fusion import SensorLookup
from run_dashcam_ai import run

FPS = 10.0


@pytest.fixture
def video(tmp_path):
    """3 seconds of video. The red channel of every frame stores its own frame number."""
    path = tmp_path / 'drive.avi'
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), FPS, (64, 48))
    for i in range(30):
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        frame[:, :, 2] = i * 8
        writer.write(frame)
    writer.release()
    return path


def test_warning_only_when_crack_and_drop_coincide(video):
    # sensor: one reading per frame, starting at t0 = 500 s. Hard drops at frames 12 and 20.
    timestamps = 500.0 + np.arange(30) / FPS
    acc = np.full(30, 9.8)
    acc[[12, 20]] = -7.0
    sensor = SensorLookup(timestamps, acc)

    seen = []

    def fake_detector(frame):
        index = int(round(frame[0, 0, 2] / 8))            # read the frame number back
        seen.append((index, frame.shape[1]))
        # the "camera" sees a crack on frames 10..15 only
        return (['Concentric_Crack'] if 10 <= index <= 15 else []), frame.copy()

    # start one second in (frame 10), like the real script starts at minute 5
    warnings = run(video, sensor, fake_detector, start_minute=1 / 60, display=False)

    assert [i for i, _ in seen][:3] == [10, 11, 12]       # playback really started at frame 10
    assert all(width == 32 for _, width in seen)          # only the left half reaches the detector
    # frame 12: crack + drop -> warning. Frame 20: drop but no crack -> no warning.
    assert warnings == [12]


def test_saving_the_annotated_video(video, tmp_path):
    sensor = SensorLookup([0.0, 1.0], [9.8, 9.8])
    out = tmp_path / 'out.mp4'
    warnings = run(video, sensor, lambda f: ([], f.copy()), start_minute=0,
                   display=False, save_path=out, max_frames=5)
    assert warnings == []
    assert out.exists() and out.stat().st_size > 0


def test_missing_video_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError):
        run(tmp_path / 'nope.mp4', SensorLookup([0.0], [9.8]), lambda f: ([], f), display=False)
