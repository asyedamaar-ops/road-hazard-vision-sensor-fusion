"""Tests for the pipeline logic. They use small synthetic data, so they need
neither the PVS dataset nor the trained weights."""
import numpy as np
import pandas as pd
import pytest

from fusion import (SensorLookup, crop_road, find_severe_drops, frame_to_timestamp,
                    should_warn, timestamp_to_frame)

ACC = 'acc_z_below_suspension'


def sensor_table():
    """10 readings around 9.8 m/s^2 with two deep drops, one of them on a good road."""
    return pd.DataFrame({
        'timestamp': np.arange(10) * 0.5 + 1000.0,
        ACC: [9.8, 9.9, 9.7, 3.0, 9.8, 9.8, 2.5, 9.9, 9.8, 9.7],
        'bad_road_left':  [0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
        'bad_road_right': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    })


def test_severe_drop_needs_bad_road_label_and_low_reading():
    events, baseline, threshold = find_severe_drops(sensor_table(), ACC, 3.8)
    assert baseline == pytest.approx(9.8)
    assert threshold == pytest.approx(6.0)
    # row 3: bad road + deep drop. Row 6 drops on a good road, row 8 is a bad road without a drop.
    assert list(events.index) == [3]


def test_timestamp_and_frame_conversions_are_inverse():
    t0, fps = 1000.0, 30.0
    assert timestamp_to_frame(1010.0, t0, fps) == 300
    assert frame_to_timestamp(300, t0, fps) == pytest.approx(1010.0)


def test_frame_counter_must_include_the_start_offset():
    """The bug that was fixed: after seeking to minute 5 the counter restarted at 0."""
    t0, fps = 1000.0, 30.0
    start_frame = int(5 * 60 * fps)
    right = frame_to_timestamp(start_frame, t0, fps)
    wrong = frame_to_timestamp(0, t0, fps)
    assert right - wrong == pytest.approx(300.0)          # the lookup was 300 seconds off


def test_nearest_reading_matches_a_full_search():
    rng = np.random.default_rng(0)
    timestamps = np.sort(rng.uniform(0, 100, 500))
    values = rng.normal(9.8, 2.0, 500)
    lookup = SensorLookup(timestamps, values)
    for t in rng.uniform(-5, 105, 200):
        expected = values[np.abs(timestamps - t).argmin()]   # the slow way the first version did it
        assert lookup.nearest(t) == pytest.approx(expected)


def test_nearest_reading_handles_unsorted_input_and_the_edges():
    lookup = SensorLookup([3.0, 1.0, 2.0], [30.0, 10.0, 20.0])
    assert lookup.t0 == 1.0
    assert lookup.nearest(0.0) == 10.0
    assert lookup.nearest(2.1) == 20.0
    assert lookup.nearest(99.0) == 30.0
    with pytest.raises(ValueError):
        SensorLookup([], [])


def test_crop_keeps_the_left_part():
    frame = np.zeros((10, 100, 3), dtype=np.uint8)
    assert crop_road(frame, 0.5).shape == (10, 50, 3)


@pytest.mark.parametrize('classes, acc_z, expected', [
    (['Concentric_Crack'], -6.0, True),       # crack seen and a hard drop
    (['Concentric_Crack'], 9.8, False),       # crack seen, smooth ride
    ([], -6.0, False),                        # hard drop, nothing seen
    (['Pothole'], -6.0, False),               # only the crack class triggers the warning
    (['Pothole', 'Concentric_Crack'], -5.1, True),
    (['Concentric_Crack'], -5.0, False),      # exactly at the threshold is not below it
])
def test_warning_needs_both_camera_and_accelerometer(classes, acc_z, expected):
    assert should_warn(classes, acc_z, -5.0, 'Concentric_Crack') is expected
