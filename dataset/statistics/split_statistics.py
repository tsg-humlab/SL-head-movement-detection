import argparse
from pathlib import Path

import pandas as pd


def main(frames_csv):
    df_frames = pd.read_csv(frames_csv)

    grouped_stats = {}

    for _, row in df_frames.iterrows():
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('frames_csv', metavar='frames-csv', type=Path)
    args = parser.parse_args()

    main(args.frames_csv)
