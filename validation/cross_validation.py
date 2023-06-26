import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from models.hmm.facial_movement import compute_hmm_vectors
from models.rule_based.rule_based_shake_detector import RuleBasedShakeDetector
from utils.frames_csv import get_splits


def cross_validate(frames_csv):
    df_frames = pd.read_csv(frames_csv)
    splits = get_splits(df_frames)

    matrix = np.zeros((2, 2))

    for fold in splits:
        matrix = matrix + validate_fold(df_frames, fold=fold)

    disp = ConfusionMatrixDisplay(matrix, display_labels=['background', 'head-shake'])

    disp.plot()
    plt.subplots_adjust(left=0.25)
    plt.show()


def validate_fold(frames_csv, fold='fold_1'):
    if type(frames_csv) == str:
        df_frames = pd.read_csv(frames_csv)
    else:
        df_frames = frames_csv

    df_val = df_frames[df_frames['split'] == fold]
    df_train = df_frames[df_frames['split'].str.contains('fold')].drop(df_val.index)
    assert len(df_val) > 0

    detector = RuleBasedShakeDetector(window_size=48, deviance_threshold=0.1, movement_threshold=0.5)
    detector.fit(df_train)

    val_sequences = compute_hmm_vectors(df_val, detector.movement_threshold)
    cut_size = int((detector.window_size - 1) / 2)
    i = 0
    pred_list = []
    label_list = []

    for _, row in df_val.iterrows():
        pred_list.append(detector.predict(val_sequences[i]))
        labels = np.load(row['labels_path'])[row['start_frame']:row['end_frame']]
        labels = labels[cut_size+1:-cut_size]
        label_list.append(labels)

        i += 1

    preds = np.concatenate(pred_list)
    labels = np.concatenate(label_list)

    return confusion_matrix(labels, preds, labels=[0, 1])


def validate_untrained(frames_csv):
    df_frames = pd.read_csv(frames_csv)
    df_val = df_frames[df_frames['split'].str.contains('fold')]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('frames_csv', type=Path)
    args = parser.parse_args()

    cross_validate(args.frames_csv)
