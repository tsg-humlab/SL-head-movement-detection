from pathlib import Path

import pandas as pd


def get_splits(df_frames):
    return sorted(set(df_frames[df_frames['split'].str.contains('fold')]['split']))


def get_n_splits(df_frames):
    return len(get_splits(df_frames))


def load_df(frames_csv):
    if isinstance(frames_csv, Path) or type(frames_csv) == str:
        return pd.read_csv(frames_csv)
    else:
        return frames_csv
