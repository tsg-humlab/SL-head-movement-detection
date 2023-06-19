import argparse
from pathlib import Path

import pandas as pd

from utils.frames_csv import get_splits


def main(frames_csv):
    df_frames = pd.read_csv(frames_csv)
    n_folds = get_splits(df_frames)

    for fold in n_folds:
        df_val = df_frames[df_frames['split'] == fold]
        df_train = df_frames[df_frames['split'].str.contains('fold')].drop(df_val.index)

        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('frames_csv', type=Path)
    args = parser.parse_args()

    main(args.frames_csv)
