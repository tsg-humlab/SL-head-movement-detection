from pathlib import Path

import numpy as np
import pandas as pd

def get_splits(df_frames):
    return sorted(set(df_frames[df_frames['split'].str.contains('fold')]['split']))


def get_n_splits(df_frames):
    return len(get_splits(df_frames))


def load_all_labels(df_frames, shift=0, window=None):
    """
    Loads all labels from the dataframe
    """    
    if window:
        from models.simple.detector import verify_window_size
        window = verify_window_size(window)
        cut = int((window - 1) / 2)
    else:
        cut = 0

    labels = list(df_frames.apply(
        lambda x: load_label(x['labels_path'],
                             int(x['start_frame']) + cut,
                             int(x['end_frame']) - cut,
                             shift=shift),
                             axis=1)
    )

    return labels


def load_label(path, start, end, shift=0):
    labels = np.load(path)[start + shift:end]
    return labels


def load_df(frames_csv):
    if isinstance(frames_csv, Path) or type(frames_csv) == str:
        return pd.read_csv(frames_csv)
    else:
        return frames_csv
