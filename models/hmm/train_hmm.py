import argparse
import pickle
from pathlib import Path

import pickle
import numpy as np
import torch
from pomegranate.distributions import Normal
from pomegranate.hmm import DenseHMM

from models.processing.facial_movement import derivatives_from_csv
from models.processing.preparation import separate_seqs, remove_nan_values_from_tensors
from models.hmm.test_hmm import predict_hmm
from utils.array_manipulation import get_change_indices
from utils.frames_csv import load_df, load_all_labels
from sklearn.metrics import confusion_matrix
from validation.validate_model import plot_confusion_matrix
from models.hmm.test_hmm import BACKGROUND_HMM_FILENAME, SHAKE_HMM_FILENAME, NOD_HMM_FILENAME

MINIMUM_SEQUENCE_LENGTH = 2

def fit_hmms(df_train, model_dir, load_values=False, epochs=100):
    
    if load_values and (Path(model_dir)/'shake_hmm.pkl').exists() \
            and (Path(model_dir)/'nod_hmm.pkl').exists() and (Path(model_dir)/'background_hmm.pkl').exists():
        print('Loading train and val data from saved')
        
        with open(Path(model_dir)/'background_hmm.pkl', 'rb') as f:
            background_vectors = pickle.load(f)
        with open(Path(model_dir)/'shake_hmm.pkl', 'rb') as f:
            shake_vectors = pickle.load(f)
        with open(Path(model_dir)/'nod_hmm.pkl', 'rb') as f:
            nod_vectors = pickle.load(f)
    else:
        print(f'Loading dataframes')
        df_train = load_df(df_train)

        print("Using difference values as input")
        labels = load_all_labels(df_train, shift=1)
        
        print('Calculating vector derivatives')
        vectors = derivatives_from_csv(df_train, take_diff=True)
        
        print('All files for learning shakes')
        _, shake_vectors, _ = separate_seqs(vectors, labels, MINIMUM_SEQUENCE_LENGTH)

        print("Only files with nods for learning nods and background")
        indexes_with_nods = [index for index, array in enumerate(labels) if 2 in array]
        print("Nr of files in training data that contain nods: ", len(indexes_with_nods), "/", len(labels), ", namely the indexes: ", indexes_with_nods)
        labels = [labels[index] for index in indexes_with_nods]
        vectors = [vectors[index] for index in indexes_with_nods]
        background_vectors, _, nod_vectors = separate_seqs(vectors, labels, MINIMUM_SEQUENCE_LENGTH)
        
        print(f'Found {len(background_vectors)} background sequences')
        print(f'Found {len(shake_vectors)} shake sequences')
        print(f'Found {len(nod_vectors)} nod sequences')
        
        with open(Path(model_dir)/'background_hmm.pkl', 'wb') as f:
            pickle.dump(background_vectors, f)
        with open(Path(model_dir)/'shake_hmm.pkl', 'wb') as f:
            pickle.dump(shake_vectors, f)
        with open(Path(model_dir)/'nod_hmm.pkl', 'wb') as f:
            pickle.dump(nod_vectors, f)

    print('Moving data to tensors')
    # [(X, 3), ...] to [(1, X, 3), ...]
    background_encodings = [np.expand_dims(vector, axis=0) for vector in background_vectors]
    shake_encodings = [np.expand_dims(vector, axis=0) for vector in shake_vectors]
    nod_encodings = [np.expand_dims(vector, axis=0) for vector in nod_vectors]
    # [(1, X, 3), ...] to [Tensor([[[X,X,X], [X,X,X], ...]]), ...]
    background_tensors = remove_nan_values_from_tensors([torch.tensor(encoding) for encoding in background_encodings])
    shake_tensors = remove_nan_values_from_tensors([torch.tensor(encoding) for encoding in shake_encodings])
    nod_tensors = remove_nan_values_from_tensors([torch.tensor(encoding) for encoding in nod_encodings])

    print('Initializing HMMs')
    background_hmm = DenseHMM([Normal(), Normal(), Normal(), Normal(), Normal()], max_iter=epochs, verbose=True)
    shake_hmm = DenseHMM([Normal(), Normal(), Normal()], max_iter=epochs, verbose=True)
    nod_hmm = DenseHMM([Normal(), Normal(), Normal()], max_iter=epochs, verbose=True)

    print('Fitting Shake HMM')
    shake_hmm.fit(shake_tensors)
    print(f'Shake HMM completed, storing to: {model_dir/SHAKE_HMM_FILENAME}')
    with open(model_dir/SHAKE_HMM_FILENAME, 'wb') as output_handle:
        pickle.dump(shake_hmm, output_handle)

    print('Fitting Nod HMM')
    nod_hmm.fit(nod_tensors)
    print(f'Nod HMM completed, storing to: {model_dir/NOD_HMM_FILENAME}')
    with open(model_dir/NOD_HMM_FILENAME, 'wb') as output_handle:
        pickle.dump(nod_hmm, output_handle)

    print('Fitting Background HMM')
    background_hmm.fit(background_tensors)
    print(f'Background HMM completed, storing to: {model_dir / BACKGROUND_HMM_FILENAME}')
    with open(model_dir/BACKGROUND_HMM_FILENAME, 'wb') as output_handle:
        pickle.dump(background_hmm, output_handle)

    print('Training program complete!')

    # print('Validation')
    # labels, predictions = predict_hmm(val_csv, model_dir, window_size=36, only_shakes = True)
    # conc_labels = np.concatenate(labels)
    # conc_predictions = np.concatenate(predictions)
    # cm = confusion_matrix(conc_labels, conc_predictions)
    # plot_confusion_matrix(cm, labels = ['Background', 'Shake'])
    # print("Accuracy only shakes: ", np.sum(conc_labels == conc_predictions) / len(conc_labels))
    # labels, predictions = predict_hmm(val_csv, model_dir, window_size=36, only_nods = True)
    # conc_labels = np.concatenate(labels)
    # conc_predictions = np.concatenate(predictions)
    # cm = confusion_matrix(conc_labels, conc_predictions)
    # plot_confusion_matrix(cm, labels = ['Background', 'Nod'])
    # print("Accuracy only nods: ", np.sum(conc_labels == conc_predictions) / len(conc_labels))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('train_csv', type=Path)
    parser.add_argument('val_csv', type=Path)
    parser.add_argument('model_dir', type=Path)
    parser.add_argument('-o', '--overwrite', action='store_true')
    parser.add_argument('-lv', '--load_values', action='store_true')
    args = parser.parse_args()

    fit_hmms(args.train_csv, args.val_csv, args.model_dir, args.overwrite, args.load_values)
