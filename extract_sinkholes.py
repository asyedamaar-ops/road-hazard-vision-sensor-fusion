"""Step 1: find severe suspension-drop events in the PVS sensor data.

    python extract_sinkholes.py

Reads the GPS/IMU table and the road labels, keeps rows on a labelled bad road
whose vertical acceleration is far below the median, and saves them as a CSV.
These events are a proxy for a sinkhole. No real sinkhole data was used.
"""
import argparse
import os

import pandas as pd

import config
from fusion import find_severe_drops


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--sensor-csv', default=config.SENSOR_CSV)
    parser.add_argument('--labels-csv', default=config.LABELS_CSV)
    parser.add_argument('--output', default=config.EVENTS_CSV)
    parser.add_argument('--drop', type=float, default=config.SEVERE_DROP_BELOW_MEDIAN,
                        help='how far below the median reading counts as severe (m/s^2)')
    args = parser.parse_args()

    print('Loading data... This might take a few seconds.')
    df_data = pd.read_csv(args.sensor_csv)
    df_labels = pd.read_csv(args.labels_csv)

    # The PVS dataset is designed so row 1 in data matches row 1 in labels.
    df = pd.concat([df_data, df_labels], axis=1)

    events, baseline, threshold = find_severe_drops(df, config.ACC_COLUMN, args.drop)
    print(f'Baseline Z-axis reading (median): {baseline:.2f} m/s^2')
    print(f'Severe drop threshold: {threshold:.2f} m/s^2')
    print(f'Severe drop events found: {len(events)} rows out of {len(df)}')

    os.makedirs(os.path.dirname(str(args.output)) or '.', exist_ok=True)
    events.to_csv(args.output, index=False)
    print(f'Saved to: {args.output}')

    if len(events) > 0:
        print('\nSample events:')
        print(events[['timestamp', config.ACC_COLUMN, 'speed']].head())


if __name__ == '__main__':
    main()
