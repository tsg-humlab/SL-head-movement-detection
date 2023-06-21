import argparse
from pathlib import Path

import pandas as pd

from models.rule_based.rule_based_shake_detector import RuleBasedShakeDetector
from utils.frames_csv import get_splits


def cross_validate(frames_csv):
    df_frames = pd.read_csv(frames_csv)
    splits = get_splits(df_frames)

    for fold in splits:
        df_val = df_frames[df_frames['split'] == fold]
        df_train = df_frames[df_frames['split'].str.contains('fold')].drop(df_val.index)


def validate_fold(frames_csv, results_dir, fold='fold_1'):
    df_frames = pd.read_csv(frames_csv)

    df_val = df_frames[df_frames['split'] == fold]
    df_train = df_frames[df_frames['split'].str.contains('fold')].drop(df_val.index)
    assert len(df_val) > 0

    detector = RuleBasedShakeDetector(0.1, movement_threshold=0.5)
    detector.fit(df_train, results_dir)
    detector.plot_hmm_distributions()


def validate_untrained(frames_csv):
    df_frames = pd.read_csv(frames_csv)
    df_val = df_frames[df_frames['split'].str.contains('fold')]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('frames_csv', type=Path)
    parser.add_argument('results_dir', type=Path)
    args = parser.parse_args()

    validate_fold(args.frames_csv, args.results_dir)
