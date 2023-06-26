import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from models.hmm.facial_movement import compute_hmm_vectors
from models.rule_based.rule_based_shake_detector import RuleBasedShakeDetector
from utils.frames_csv import get_splits, load_df


def cross_validate_hmm_preload(frames_csv, window_size=48, deviance_threshold=0.1, movement_threshold=0.5):
    df_frames = load_df(frames_csv)
    vectors = compute_hmm_vectors(df_frames, threshold=movement_threshold)

    cross_validate(
        frames_csv,
        data=vectors,
        window_size=window_size,
        deviance_threshold=deviance_threshold,
        movement_threshold=movement_threshold
    )


def cross_validate(frames_csv, data=None, window_size=48, deviance_threshold=0.1, movement_threshold=0.5):
    df_frames = load_df(frames_csv)
    splits = get_splits(df_frames)
    matrix = np.zeros((2, 2))

    for fold in splits:
        detector = RuleBasedShakeDetector(
            window_size=window_size,
            deviance_threshold=deviance_threshold,
            movement_threshold=movement_threshold
        )
        if data:
            matrix = matrix + validate_fold(df_frames, detector, fold=fold, data=data)
        else:
            matrix = matrix + validate_fold(df_frames, detector, fold=fold)

    disp = ConfusionMatrixDisplay(matrix, display_labels=['background', 'head-shake'])

    disp.plot()
    plt.subplots_adjust(left=0.25)
    plt.show()


def validate_fold(frames_csv, model, fold='fold_1', data=None):
    df_frames = load_df(frames_csv)
    df_val = df_frames[df_frames['split'] == fold]
    df_train = df_frames[df_frames['split'].str.contains('fold')].drop(df_val.index)
    assert len(df_val) > 0

    if data:
        train_indices = list(df_train.index)
        val_indices = list(df_val.index)
        model.fit(df_train, data=[data[index] for index in train_indices])
        val_sequences = [data[index] for index in val_indices]
    else:
        model.fit(df_train)
        val_sequences = compute_hmm_vectors(df_val, model.movement_threshold)

    cut_size = int((model.window_size - 1) / 2)
    i = 0
    pred_list = []
    label_list = []

    for _, row in df_val.iterrows():
        pred_list.append(model.predict(val_sequences[i]))
        labels = np.load(row['labels_path'])[row['start_frame']:row['end_frame']]
        labels = labels[cut_size + 1:-cut_size]
        label_list.append(labels)

        i += 1

    preds = np.concatenate(pred_list)
    labels = np.concatenate(label_list)

    return confusion_matrix(labels, preds, labels=[0, 1])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('frames_csv', type=Path)
    args = parser.parse_args()

    import time

    start = time.time()
    cross_validate_hmm_preload(args.frames_csv)
    end = time.time()
    print(end - start)
