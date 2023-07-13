import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
from pomegranate.distributions import Normal
from pomegranate.hmm import DenseHMM

from models.hmm.facial_movement import derivatives_from_csv
from utils.array_manipulation import get_change_indices
from utils.frames_csv import load_df, load_all_labels


BACKGROUND_HMM_FILENAME = 'background_hmm.p'
SHAKE_HMM_FILENAME = 'shake_hmm.p'
MINIMUM_SEQUENCE_LENGTH = 2


def fit_hmms(frames_csv, model_dir, overwrite=False, debug=False, epochs=100):
    if ((model_dir/BACKGROUND_HMM_FILENAME).exists() or (model_dir/SHAKE_HMM_FILENAME).exists()) and not overwrite:
        if not debug:
            raise FileExistsError('The models already exist in this directory, you can explicitly allow overwriting')

    print(f'Loading dataframe from {frames_csv}')
    df_frames = load_df(frames_csv)

    labels = load_all_labels(df_frames, shift=1)

    print('Calculating vector derivatives')
    vectors = derivatives_from_csv(df_frames)
    background_vectors, shake_vectors = separate_seqs(vectors, labels)

    print('Moving data to tensors')
    background_encodings = [np.expand_dims(vector, axis=0) for vector in background_vectors]
    shake_encodings = [np.expand_dims(vector, axis=0) for vector in shake_vectors]
    background_tensors = [torch.tensor(encoding) for encoding in background_encodings]
    shake_tensors = [torch.tensor(encoding) for encoding in shake_encodings]

    print('Initializing HMMs')
    background_hmm = DenseHMM([Normal(), Normal(), Normal(), Normal(), Normal()], max_iter=epochs, verbose=True)
    shake_hmm = DenseHMM([Normal(), Normal(), Normal()], max_iter=epochs, verbose=True)

    print('Fitting Shake HMM')
    shake_hmm.fit(shake_tensors)
    print(f'Shake HMM completed, storing to: {model_dir/SHAKE_HMM_FILENAME}')
    with open(model_dir/SHAKE_HMM_FILENAME, 'wb') as output_handle:
        pickle.dump(shake_hmm, output_handle)

    print('Fitting Background HMM')
    background_hmm.fit(background_tensors)
    print(f'Background HMM completed, storing to: {model_dir / BACKGROUND_HMM_FILENAME}')
    with open(model_dir/BACKGROUND_HMM_FILENAME, 'wb') as output_handle:
        pickle.dump(background_hmm, output_handle)

    print('Training program complete!')


def separate_seqs(sequence_list, labels_list):
    assert len(sequence_list) == len(labels_list)

    background_seqs = []
    shake_seqs = []

    for seq_index in range(len(sequence_list)):
        assert len(sequence_list[seq_index]) == len(labels_list[seq_index])

        on_background = labels_list[seq_index][0] == 0

        change_indices = get_change_indices(labels_list[seq_index])
        change_indices = np.insert(change_indices, 0, 0)
        if change_indices[-1] != len(labels_list[seq_index]):
            change_indices = np.append(change_indices, len(labels_list[seq_index]))

        for change_index in range(len(change_indices) - 1):
            if (change_indices[change_index + 1] - change_indices[change_index]) >= MINIMUM_SEQUENCE_LENGTH:
                sub_sequence = sequence_list[seq_index][change_indices[change_index]:change_indices[change_index + 1]]

                if on_background:
                    background_seqs.append(sub_sequence)
                else:
                    shake_seqs.append(sub_sequence)

            on_background = not on_background

    return background_seqs, shake_seqs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('frames_csv', type=Path)
    parser.add_argument('model_dir', type=Path)
    parser.add_argument('-o', '--overwrite', action='store_true')
    args = parser.parse_args()

    fit_hmms(args.frames_csv, args.model_dir, args.overwrite)
