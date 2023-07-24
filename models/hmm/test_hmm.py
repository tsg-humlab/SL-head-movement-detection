import argparse
import pickle
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix

from models.hmm.facial_movement import derivatives_from_csv
from models.hmm.train_hmm import BACKGROUND_HMM_FILENAME, SHAKE_HMM_FILENAME
from models.simple.detector import verify_window_size
from utils.frames_csv import load_df, load_all_labels


def predict_hmm(frames_csv, models_dir, window_size=48):
    window_size = verify_window_size(window_size)

    df_frames = load_df(frames_csv)
    vectors = derivatives_from_csv(df_frames)
    labels = load_all_labels(df_frames, shift=1, window=window_size)

    with open(models_dir/BACKGROUND_HMM_FILENAME, 'rb') as input_handle:
        hmm_bg = pickle.load(input_handle)
    with open(models_dir/SHAKE_HMM_FILENAME, 'rb') as input_handle:
        hmm_shake = pickle.load(input_handle)

    predictions = []

    for vector in vectors:
        pred_len = len(vector) - window_size + 1

        windows = []
        for i in range(pred_len):
            windows.append(vector[i:i + window_size])

        windows = np.stack(windows)
        pred_bg = hmm_bg.forward_backward(windows)
        pred_shake = hmm_shake.forward_backward(windows)
        log_probs = np.stack([pred_bg[4].cpu().detach().numpy(), pred_shake[4].cpu().detach().numpy()])

        predictions.append(np.argmax(log_probs, axis=0))

    return labels, predictions


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('frames_csv', type=Path)
    parser.add_argument('models_dir', type=Path)
    parser.add_argument('-o', '--overwrite', action='store_true')
    args = parser.parse_args()

    predict_hmm(args.frames_csv, args.models_dir)
