import argparse
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from models.hmm.facial_movement import ordinal_from_csv, derivatives_from_csv
from models.simple.memory_detector import MemoryBasedShakeDetector
from models.simple.random_detector import RandomShakeDetector
from models.simple.rule_detector import RuleBasedShakeDetector, majority_vote
from utils.frames_csv import get_splits, load_df

import torch
from pomegranate.distributions import Normal
from pomegranate.hmm import DenseHMM
from sklearn.preprocessing import OneHotEncoder


def cross_validate_hmm(frames_csv, window_size=48):
    df_frames = load_df(frames_csv)
    vectors = derivatives_from_csv(df_frames)
    encodings = [np.expand_dims(vector, axis=0) for vector in vectors]
    tensors = [torch.tensor(encoding) for encoding in encodings]
    splits = get_splits(df_frames)

    model = DenseHMM([Normal(), Normal(), Normal()], max_iter=100, verbose=True)
    model.fit(tensors)

    return

    for fold in splits:
        model = model_class(**params)
        matrix = matrix + validate_fold(frames_csv=df_frames, model=model, fold=fold, data=data)

    return matrix


def cross_validate_random_preload(frames_csv, window_size=48, movement_threshold=0.5):
    df_frames = load_df(frames_csv)
    vectors = ordinal_from_csv(df_frames, threshold=movement_threshold)

    matrix = cross_validate(
        frames_csv,
        data=vectors,
        model_class=RandomShakeDetector,
        params={
            'window_size': window_size,
            'movement_threshold': movement_threshold
        }
    )
    plot_confusion_matrix(matrix,
                          title='Confusion matrix for random classifier')


def cross_validate_rule_preload(frames_csv, window_size=48, movement_threshold=0.5):
    df_frames = load_df(frames_csv)
    vectors = ordinal_from_csv(df_frames, threshold=movement_threshold)

    matrix = cross_validate(
        frames_csv,
        data=vectors,
        model_class=RuleBasedShakeDetector,
        params={
            'window_size': window_size,
            'movement_threshold': movement_threshold,
            'rule_func': majority_vote
        }
    )
    plot_confusion_matrix(matrix,
                          title=f'Confusion matrix for rule-based classifier with threshold={movement_threshold}')


def cross_validate_memory_preload(frames_csv, window_size=48, movement_threshold=0.5):
    """Cross validation wrapper specifically designed for the rule based classifier using HMM vectors.

    :param frames_csv: Dataframe with frame annotations, labels and any other information used by your model
    :param window_size: Window size to use during prediction of new sequences
    :param movement_threshold: Movement threshold in pixels to determine when a person is considered to move
    """
    df_frames = load_df(frames_csv)
    vectors = ordinal_from_csv(df_frames, threshold=movement_threshold)

    matrix = cross_validate(
        frames_csv,
        data=vectors,
        model_class=MemoryBasedShakeDetector,
        params={
            'window_size': window_size,
            'movement_threshold': movement_threshold
        }
    )
    plot_confusion_matrix(matrix,
                          title=f'Confusion matrix for memory-based classifier with threshold={movement_threshold}')


def cross_validate(frames_csv, data, model_class, params):
    """Do cross validation with an arbitrary model and number of folds.

    The model will be created using any parameters given in params and should be able to fit using a Pandas dataframe
    and index-able collection of data sequences, which length may be variable.

    The code assumes your folds are defined in the Dataframe under a 'split' column, where every fold has a unique
    identifier for the rows to use during validation. This identifier may be anything, so long as it includes the word
    'fold'. Any rows that don't include this word will be ignored (e.g. testing sets or any other collections of
    datapoints you don't wish to use at this time).

    :param frames_csv: Dataframe with frame annotations, labels and any other information used by your model
    :param data: Collection of datapoints, the indices of these datapoints must correspond to the Dataframe
    :param model_class: Reference to the model class to use
    :param params: Parameters for creating models
    :return: Confusion matrix of the results
    """
    df_frames = load_df(frames_csv)
    splits = get_splits(df_frames)
    matrix = np.zeros((2, 2))

    for fold in splits:
        model = model_class(**params)
        matrix = matrix + validate_fold(frames_csv=df_frames, model=model, fold=fold, data=data)

    return matrix


def validate_fold(frames_csv, model, data, fold, classes=None):
    """Validate a single fold. This function assumes a two-class classification problem is being solved, but this can
    be explicitly adjusted using the 'classes' argument.

    :param frames_csv: Dataframe with frame annotations, labels and any other information used by your model
    :param model: Instance of the model that will be trained
    :param data: Collection of datapoints, the indices of these datapoints must correspond to the Dataframe
    :param fold: Label of the fold used for evaluation.
    :param classes: List of integers representing classes.
    :return: Confusion matrix of the validation results.
    """
    if classes is None:
        classes = [0, 1]

    df_frames = load_df(frames_csv)
    df_val = df_frames[df_frames['split'] == fold]
    df_train = df_frames[df_frames['split'].str.contains('fold')].drop(df_val.index)
    assert len(df_val) > 0

    train_indices = list(df_train.index)
    val_indices = list(df_val.index)
    model.fit(df_train, data=[data[index] for index in train_indices])
    val_sequences = [data[index] for index in val_indices]

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

    return confusion_matrix(labels, preds, labels=classes)


def plot_confusion_matrix(matrix, title=None):
    disp = ConfusionMatrixDisplay(matrix, display_labels=['background', 'head-shake'])

    disp.plot()
    plt.subplots_adjust(left=0.25)
    if title:
        plt.title(title, fontsize=15)
        plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('frames_csv', type=Path)
    args = parser.parse_args()

    cross_validate_hmm(args.frames_csv,
                       window_size=48,
                       )
