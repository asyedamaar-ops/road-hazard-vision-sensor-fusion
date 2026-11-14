"""Core logic of the pipeline, kept free of file paths, video windows and the YOLO model.

These small functions are what the scripts call, and what tests/ checks:
  - which sensor rows count as severe suspension drops
  - how a sensor timestamp maps to a video frame, and back
  - which sensor reading belongs to a video frame
  - when the camera and the accelerometer together raise a warning
"""
import numpy as np


def find_severe_drops(df, acc_column, drop_below_median):
    """Rows on a labelled bad road where the vertical acceleration falls far below normal.

    Returns (events, baseline, threshold). The baseline is the median reading
    of the whole session and the threshold is baseline - drop_below_median.
    """
    bad_road = (df['bad_road_left'] == 1) | (df['bad_road_right'] == 1)
    baseline = df[acc_column].median()
    threshold = baseline - drop_below_median
    events = df[bad_road & (df[acc_column] < threshold)].copy()
    return events, baseline, threshold


def timestamp_to_frame(timestamp, t0, fps):
    """Video frame that was on screen at a sensor timestamp. t0 is the first sensor timestamp."""
    return int((timestamp - t0) * fps)


def frame_to_timestamp(frame_index, t0, fps):
    """Sensor timestamp for a video frame.

    frame_index must be the position in the WHOLE video. When playback starts
    at minute 5, the counter has to start at that frame and not at 0. Starting
    at 0 was the bug that shifted the sensor lookup by about 300 seconds.
    """
    return t0 + frame_index / fps


class SensorLookup:
    """Finds the sensor reading closest in time to a given timestamp.

    The timestamps are sorted once, then every lookup is a binary search.
    The first version sorted the whole table again for every video frame.
    """

    def __init__(self, timestamps, values):
        timestamps = np.asarray(timestamps, dtype=float)
        values = np.asarray(values, dtype=float)
        if len(timestamps) == 0:
            raise ValueError('sensor data is empty')
        order = np.argsort(timestamps, kind='stable')
        self.timestamps = timestamps[order]
        self.values = values[order]
        self.t0 = float(self.timestamps[0])

    def nearest(self, timestamp):
        i = int(np.searchsorted(self.timestamps, timestamp))
        if i == 0:
            return float(self.values[0])
        if i == len(self.timestamps):
            return float(self.values[-1])
        before, after = self.timestamps[i - 1], self.timestamps[i]
        # on a tie the earlier reading wins
        return float(self.values[i - 1] if timestamp - before <= after - timestamp else self.values[i])


def crop_road(frame, keep_left_fraction):
    """Keep the left part of the frame (the road). The right part of the PVS video shows graphs."""
    width = frame.shape[1]
    return frame[:, :int(width * keep_left_fraction)]


def should_warn(detected_classes, acc_z, acc_threshold, hazard_class):
    """The fusion rule: warn only when BOTH sensors agree.

    The camera must see the hazard class in this frame AND the vertical
    acceleration must be below the threshold. A crack seen on a smooth ride,
    or a bump with no crack in view, does not raise a warning.
    """
    return hazard_class in detected_classes and acc_z < acc_threshold
