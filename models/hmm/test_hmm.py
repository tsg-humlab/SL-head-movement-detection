import argparse
import pickle
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix

from models.processing.facial_movement import derivatives_from_csv
from models.simple.detector import verify_window_size
from utils.frames_csv import load_df, load_all_labels

BACKGROUND_HMM_FILENAME = 'background_hmm.p'
SHAKE_HMM_FILENAME = 'shake_hmm.p'
NOD_HMM_FILENAME = 'nod_hmm.p'

def predict_hmm(frames_csv, models_dir, window_size=36, only_nods = False, only_shakes = False):

    if only_nods and only_shakes:
        raise ValueError("Cannot predict only nods and only shakes at the same time")
    
    window_size = verify_window_size(window_size)
    print("Verified window size: ", window_size)

    df_frames = load_df(frames_csv)
    vectors = derivatives_from_csv(df_frames, take_diff=True)
    labels = load_all_labels(df_frames, shift=1, window=window_size)

    print("Removing files that do not contain nods")
    indexes_with_nods = [index for index, array in enumerate(labels) if 2 in array]
    print("Nr of files in validation/test data that contain nods: ", len(indexes_with_nods), "/", len(labels), ", namely the indexes: ", indexes_with_nods)
    labels = [labels[index] for index in indexes_with_nods]
    vectors = [vectors[index] for index in indexes_with_nods]

    with open(models_dir/BACKGROUND_HMM_FILENAME, 'rb') as input_handle:
        hmm_bg = pickle.load(input_handle)
    if not only_nods:
        with open(models_dir/SHAKE_HMM_FILENAME, 'rb') as input_handle:
            hmm_shake = pickle.load(input_handle)
    if not only_shakes:
        with open(models_dir/NOD_HMM_FILENAME, 'rb') as input_handle:
            hmm_nod = pickle.load(input_handle)

    predictions, new_labels = [], []

    for vector_i, vector in enumerate(vectors):
        pred_len = len(vector) - window_size + 1

        windows, new_labs = [], []
        for i in range(pred_len):
            if (labels[vector_i][i] == 1 and only_nods) or (labels[vector_i][i] == 2 and only_shakes):
                continue
            else:
                if only_shakes or only_nods:
                    if labels[vector_i][i] != 0:
                        new_labs.append(1)
                    else:
                        new_labs.append(0)
                windows.append(vector[i:i + window_size])

        windows = np.stack(windows)
        pred_bg = hmm_bg.forward_backward(windows)
        if not only_nods:
            pred_shake = hmm_shake.forward_backward(windows)
        if not only_shakes:
            pred_nod = hmm_nod.forward_backward(windows)
        if only_nods:
            log_probs = np.stack([pred_bg[4].cpu().detach().numpy(), pred_nod[4].cpu().detach().numpy()])
        elif only_shakes:
            log_probs = np.stack([pred_bg[4].cpu().detach().numpy(), pred_shake[4].cpu().detach().numpy()])
        else:
            log_probs = np.stack([pred_bg[4].cpu().detach().numpy(), pred_shake[4].cpu().detach().numpy(), pred_nod[4].cpu().detach().numpy()])

        predictions.append(np.argmax(log_probs, axis=0))

        if only_nods or only_shakes:
            old_preds = predictions[-1]
            nr_nods_or_shakes = new_labs.count(1)
            import random
            random_background_indexes = random.sample(range(new_labs.count(0)), nr_nods_or_shakes)
            nls, preds = [], []
            nr_bgs = 0
            for nl_i, nl in enumerate(new_labs):
                if nl != 0:
                    nls.append(1)
                    preds.append(old_preds[nl_i])
                if nl == 0 and nr_bgs in random_background_indexes:
                    nls.append(0)
                    preds.append(old_preds[nl_i])
                if nl == 0:
                    nr_bgs += 1

            new_labs, predictions[-1] = nls, preds
            new_labels.append(np.array(new_labs))

    if only_nods or only_shakes:
        labels = new_labels

    print("Label len, pred len: ", len(labels[0]), len(predictions[0]))
    print("Label len, pred len: ", len(labels[1]), len(predictions[1]))
    print("Label len, pred len: ", len(labels[2]), len(predictions[2]))
    
    return labels, predictions


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('frames_csv', type=Path)
    parser.add_argument('models_dir', type=Path)
    parser.add_argument('-o', '--overwrite', action='store_true')
    args = parser.parse_args()

    predict_hmm(args.frames_csv, args.models_dir)
