import argparse
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score

from models.hmm.prediction_visualisation import load_predictions
# from models.lstm.test_lstm import predict_lstm
from models.hmm.test_hmm import predict_hmm
from models.lstm.test_lstm import predict_lstm
from models.hmm.train_hmm import fit_hmms
from models.lstm.train_lstm import fit_lstms
from models.lstm.train_lstm_class import fit_lstms_class
from models.lstm.train_lstm_class_windows import fit_lstms_class_windows

from models.lstm.train_lstm_class_windows_middle_all import fit_lstms_class_windows_middle_all
from models.lstm.train_lstm_class_windows_middle_shake import fit_lstms_class_windows_middle_shake
from models.lstm.train_lstm_class_windows_middle_nod import fit_lstms_class_windows_middle_nod

from models.lstm.test_lstm_class import predict_lstm_class
from models.lstm.test_lstm_class_windows import predict_lstm_class_windows
from models.lstm.test_lstm_class_windows_middle_all import predict_lstm_class_windows_middle_all
from models.lstm.test_lstm_class_windows_middle_shake import predict_lstm_class_windows_middle_shake
from models.lstm.test_lstm_class_windows_middle_nod import predict_lstm_class_windows_middle_nod
from models.simple.memory_detector import MemoryBasedShakeNodDetector
from models.simple.random_detector import RandomShakeDetector
from models.simple.rule_detector import RuleBasedShakeDetector
from utils.draw import set_seaborn_theme
from utils.frames_csv import get_splits, load_df, load_all_labels
from utils.io import mkdir_if_not_exists
from validation.event_based import evaluate_events
from models.processing.filters import majority_filter
from models.processing.facial_movement import ordinal_from_csv
from validation.validate_model import plot_confusion_matrix
from dataset.preparation.inter_annotator_agreement import *

import json

from seqeval.metrics import classification_report
from nervaluate import Evaluator

from sklearn.metrics import classification_report


def hmm_filter_search(frames_csv, folds_dir, results_dir, method='event', metric='f1'):
    # sizes = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39]
    # sizes = [11, 21, 31, 41, 51, 61, 71, 81, 91, 101]
    sizes = [11, 51, 101, 151, 201]
    results = []

    for size in sizes:
        results.append(validate_hmm_folds(frames_csv, folds_dir, filter_size=size,
                                          window_size=36, method=method, metric=metric,
                                          results_dir=results_dir, load_previous=True))

    set_seaborn_theme()
    plt.title('F1 score for head-shake detector on different filter sizes', fontsize=15)
    plt.locator_params(axis='x', nbins=10)
    plt.plot(sizes, results, 'o-')
    plt.xticks(sizes, sizes)
    plt.xlabel('Filter size (frames)')
    plt.ylabel('F1 score')
    plt.show()


def hmm_window_search(frames_csv, folds_dir, method='frame', metric='f1'):
    sizes = [12, 24, 36, 48, 60, 72, 84, 96, 108, 120]
    labels = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
    results = []

    for size in sizes:
        results.append(validate_hmm_folds(frames_csv, folds_dir, window_size=size, 
                                          method=method, metric=metric))

    set_seaborn_theme()
    plt.title('F1 score for head-shake detector on different window sizes', fontsize=15)
    plt.locator_params(axis='x', nbins=10)
    plt.plot(sizes, results, 'o-')
    plt.xticks(sizes, labels)
    plt.xlabel('Window size (seconds)')
    plt.ylabel('F1 score')
    plt.show()


def numerical_to_iob(array, codes = ['background', 'shake', 'nod']):
    """
    Convert a numerical array to IOB notation. This is useful for visualising the results of a model.
    """
    iob = []

    cur_value = 0
    for _, value in enumerate(array):
        if value == 0:
            iob.append('O')
        elif value != 0:
            if value == cur_value:
                iob.append('I-' + codes[value])
            else:
                iob.append('B-' + codes[value])
        cur_value = value

    return iob


def evaluate_seqeval(labels, predictions):
    """
    Evaluate the sequences using seqeval.
    """
    bio_labels = []
    for label in labels:
        bio_labels.extend(numerical_to_iob(label))
    bio_predictions = []
    for prediction in predictions:
        bio_predictions.extend(numerical_to_iob(prediction))
    print(classification_report(bio_labels, bio_predictions))

def evaluate_nervaluate(labels, predictions):
    """
    Evaluate the sequences using nervaluate.
    """
    bio_labels = []
    for label in labels:
        bio_labels.append(numerical_to_iob(label))
    bio_predictions = []
    for prediction in predictions:
        bio_predictions.append(numerical_to_iob(prediction))
    evaluator = Evaluator(bio_labels, bio_predictions, tags=['shake', 'nod'], loader="list")
    results, results_by_tag = evaluator.evaluate()
    print(json.dumps(results, indent=2))
    print(json.dumps(results_by_tag, indent=2))


def flatten_concatenation(list):
    flat_list = []
    for row in list:
        flat_list.extend(row)
    return flat_list


def calculate_iou_per_label(predictions, labels, target_labels=[0, 1, 2], target_label_names=['background', 'shake', 'nod']):
    """
    Calculate Intersection over Union (IoU) for each label separately.

    Parameters:
    - predictions: Predicted sequence labels.
    - labels: Ground truth sequence labels.
    - target_labels: List of labels for which IoU should be calculated.

    Returns:
    - Dictionary containing IoU scores for each target label.
    """
    assert len(predictions) == len(labels), "Input arrays must have the same length."

    iou_per_label = {}

    # Convert predictions and labels to target label names
    predictions_new, labels_new = [], []
    for i, prediction in enumerate(predictions):
        predictions_new.append(target_label_names[prediction])
        labels_new.append(target_label_names[labels[i]])
    predictions, labels = predictions_new, labels_new

    for target_label in target_labels:
        # Find the positions of the target label in predictions and labels
        pred_positions = [i for i, label in enumerate(predictions) if label == target_label_names[target_label]]
        gt_positions = [i for i, label in enumerate(labels) if label == target_label_names[target_label]]

        # Find the intersection positions
        intersection_positions = set(pred_positions) & set(gt_positions)

        # Calculate IoU for the target label
        iou = len(intersection_positions) / float(len(pred_positions) + len(gt_positions) - len(intersection_positions))

        iou_per_label[target_label_names[target_label]] = iou

    return iou_per_label


def validate_folds(frames_csv, folds_dir, window_size=36, val_windows=[35], results_dir=None, filter_size=None,
                       method='frame', model="hmm", load_previous=False, metric=None, test=False, load_data=False, only_nods=False):
    
    print(f'Loading samples from {frames_csv}')
    df_frames = load_df(frames_csv)

    # Get splits
    if not test:
        splits = get_splits(df_frames)[4:5]
    if test:
        splits = ['test']

    all_predictions = []
    label_index = 0

    if metric == 'f1':
        all_labels = []

    if load_previous and not test:
        pred_file_name = 'predictions_'+model+'.npz'
        all_predictions = load_predictions(results_dir / pred_file_name)
    if load_previous and test:
        pred_file_name = 'predictions_test_'+model+'.npz'
        all_predictions = load_predictions(results_dir / pred_file_name)

    # Initialize an empty confusion matrix
    if method == 'frame':
        matrix = np.zeros((3, 3))
    elif method == 'event':
        matrix_shake = np.zeros((2, 2))
        matrix_nod = np.zeros((2, 2))
        matrix_all = np.zeros((3, 3))

    conc_all_labels, conc_all_predictions = [], []

    for fold in splits:
        if test:
            print(f'Validating test')
        else:
            print(f'Validating {fold}/{len(splits)}')

        df_val = df_frames[df_frames['split'] == fold]

        if load_previous:
            labels = load_all_labels(df_val, shift=0, window=window_size)
            predictions = all_predictions[label_index:label_index+len(df_val)]
            label_index += len(df_val)
        else:
            if model == 'hmm':
                if not test:
                    labels, predictions = predict_hmm(df_val, Path(folds_dir) / fold, window_size=window_size)
                    print("Nr of validation samples: ", len(predictions))
                else:
                    labels, predictions = predict_hmm(df_val, Path(folds_dir) / 'fold_5', window_size=window_size)
                    print("Nr of test samples: ", len(predictions))
            # elif model == 'lstm':
            #     if not test:
            #         labels, predictions = predict_lstm(df_val, Path(folds_dir) / fold)
            #         print("Nr of validation samples: ", len(predictions))
            #     else:
            #         labels, predictions = predict_lstm(df_val, Path(folds_dir) / 'fold_5')
            #         print("Nr of test samples: ", len(predictions))
            #     # Remove padded values
            #     prediction_no_pad = []
            #     for i, prediction in enumerate(predictions):
            #         single_prediction_no_pad = []
            #         for pred_i, pred in enumerate(prediction):
            #             if pred_i<len(labels[i]):
            #                 single_prediction_no_pad.append(pred)
            #         prediction_no_pad.append(single_prediction_no_pad)
            #     predictions = prediction_no_pad
            # elif model == 'lstm_class':
            #     if not test:
            #         labels, predictions = predict_lstm_class(df_val, Path(folds_dir) / fold, window_size=window_size, val_windows=val_windows)
            #     else:
            #         labels, predictions = predict_lstm_class(df_val, Path(folds_dir) / 'fold_5', window_size=window_size, val_windows=val_windows)
            #         print("Nr of test samples: ", len(predictions))
            elif model == 'lstm_class_windows':
                if not test:
                    labels, predictions = predict_lstm_class_windows(df_val, Path(folds_dir) / fold, window_size=window_size, load_data=load_data, only_nods=only_nods)
                    print("Nr of validation samples: ", len(predictions))
                else:
                    labels, predictions = predict_lstm_class_windows(df_val, Path(folds_dir) / 'fold_5', window_size=window_size, load_data=load_data, only_nods=only_nods)
                    print("Nr of test samples: ", len(predictions))
            elif model == 'lstm_class_windows_middle_all':
                if not test:
                    labels, predictions = predict_lstm_class_windows_middle_all(df_val, Path(folds_dir) / fold, window_size=window_size)
                    print("Nr of validation samples: ", len(predictions))
                else:
                    labels, predictions = predict_lstm_class_windows_middle_all(df_val, Path(folds_dir) / 'fold_5', window_size=window_size)
                    print("Nr of test samples: ", len(predictions))
            elif model == 'lstm_class_windows_middle_shake':
                if not test:
                    labels, predictions = predict_lstm_class_windows_middle_shake(df_val, Path(folds_dir) / fold, window_size=window_size)
                    print("Nr of validation samples: ", len(predictions))
                else:
                    labels, predictions = predict_lstm_class_windows_middle_shake(df_val, Path(folds_dir) / 'fold_5', window_size=window_size)
                    print("Nr of test samples: ", len(predictions))
            elif model == 'lstm_class_windows_middle_nod':
                if not test:
                    labels, predictions = predict_lstm_class_windows_middle_nod(df_val, Path(folds_dir) / fold, window_size=window_size)
                    print("Nr of validation samples: ", len(predictions))
                else:
                    labels, predictions = predict_lstm_class_windows_middle_nod(df_val, Path(folds_dir) / 'fold_5', window_size=window_size)
                    print("Nr of test samples: ", len(predictions))
            else:
                raise ValueError('Invalid model!')

        if filter_size:
            predictions = [majority_filter(prediction, filter_size) for prediction in predictions]

        conc_labels = np.concatenate(labels)
        conc_predictions = np.concatenate(predictions)
        
        assert(len(conc_labels) == len(conc_predictions))

        if method == 'frame':
            conc_all_labels.append(conc_labels)
            conc_all_predictions.append(conc_predictions)
            matrix = matrix + confusion_matrix(conc_labels, conc_predictions, labels=[0, 1, 2])
        # elif method == 'event':
            # evals = evaluate_events(conc_labels, conc_predictions)
            # matrix_shake = matrix_shake + evals[0]
            # matrix_nod = matrix_nod + evals[1]
        # else:
        #     raise ValueError('Invalid method!')

        if (results_dir and not load_previous) or (metric == 'f1'):
            all_predictions.extend(predictions)
        if metric == 'f1':
            all_labels.extend(labels)

    if results_dir and not load_previous:
        mkdir_if_not_exists(results_dir)
        if not test:
            pred_file_name = 'predictions_'+model+'.npz'
            np.savez(Path(results_dir) / pred_file_name, *all_predictions)
        else:
            pred_file_name = 'predictions_test_'+model+'.npz'
            np.savez(Path(results_dir) / pred_file_name, *all_predictions)

    if metric == 'f1':
        if method == 'frame':
            return f1_score(np.concatenate(all_labels), np.concatenate(all_predictions))
        # if method == 'event':
            # evals = evaluate_events(np.concatenate(all_labels), np.concatenate(all_predictions))
            # shake_f1 = evals[0][3] / (evals[0][3] + (0.5 * (evals[0][1] + evals[0][2])))
            # nod_f1 = evals[1][3] / (evals[1][3] + (0.5 * (evals[1][1] + evals[1][2])))
            # return [shake_f1, nod_f1]

    if method == 'frame':
        
        # list of lists to list of np arrays
        conc_all_labels_t = [np.array(x).astype(int) for x in conc_all_labels]
        conc_all_predictions_t = [np.array(x).astype(int) for x in conc_all_predictions]
        # evaluate_seqeval(conc_all_labels_t, conc_all_predictions_t)
        # evaluate_nervaluate(conc_all_labels_t, conc_all_predictions_t)

        conc_all_predictions = np.array(flatten_concatenation(conc_all_predictions)).astype(int)
        conc_all_labels = np.array(flatten_concatenation(conc_all_labels)).astype(int)

        print("IoU: ", calculate_iou_per_label(conc_all_predictions, conc_all_labels))
        target_names = ['background', 'shake', 'nod']
        print(classification_report(conc_all_labels, conc_all_predictions, target_names=target_names))

        plot_confusion_matrix(matrix, title='Frame-level confusion matrix for head-shake detection')
    elif method == 'event':
        if model == 'lstm_class_windows_middle_nod':
            for i in range(len(predictions)):
                predictions[i] = [1 if x == 2 else x for x in predictions[i]]
            for i in range(len(labels)):
                labels[i] = [0 if x == 1 else x for x in labels[i]]
                labels[i] = [1 if x == 2 else x for x in labels[i]]
            run_event_level_agreement(predictions, labels, labels_dict = {'background': 0, 'nod': 1})
        elif model == 'lstm_class_windows_middle_shake':
            for i in range(len(labels)):
                labels[i] = [0 if x == 2 else x for x in labels[i]]
            run_event_level_agreement(predictions, labels, labels_dict = {'background': 0, 'shake': 1})
        else:
            run_event_level_agreement(predictions, labels)
        

def fit_folds(frames_csv, folds_dir, model, window_size=36, load_values=False):
    """
    Fit models for every fold in the dataset. This function assumes the folds are defined in the Dataframe under a
    'split' column, where every fold has a unique identifier for the rows to use during validation. This identifier may
    be anything, so long as it includes the word 'fold'. Any rows that don't include this word will be ignored (e.g.
    testing sets or any other collections of datapoints you don't wish to use at this time).
    """
    df_frames = load_df(frames_csv)
    splits = get_splits(df_frames)

    for fold in splits[4:5]:
        print(f'Fitting for {fold} as validation set')
        df_val = df_frames[df_frames['split'] == fold]
        df_train = df_frames[df_frames['split'].str.contains('fold')].drop(df_val.index)
        
        # Training three models (bg, shake, nod) on their sequences of variable length
        if model == 'hmm':
            fit_hmms(df_train, Path(folds_dir) / fold, load_values=load_values)
        # Windows but training for frame-by-frame scores
        elif model == 'lstm_class_windows':
            fit_lstms_class_windows(df_train, df_val, Path(folds_dir) / fold, window_size=window_size, load_values=load_values)
        # window classification based on the middle frame
        elif model == 'lstm_class_windows_middle_all':
            fit_lstms_class_windows_middle_all(df_train, df_val, Path(folds_dir) / fold, window_size=window_size, load_values=load_values)
        # window classification based on the middle frame, only shakes
        elif model == 'lstm_class_windows_middle_shake':
            fit_lstms_class_windows_middle_shake(df_train, df_val, Path(folds_dir) / fold, window_size=window_size, load_values=load_values)
        # window classification based on the middle frame, only nods
        elif model == 'lstm_class_windows_middle_nod':
            fit_lstms_class_windows_middle_nod(df_train, df_val, Path(folds_dir) / fold, window_size=window_size, load_values=load_values)
        else:
            raise ValueError('Invalid model!')


def cross_validate_random_preload(frames_csv, window_size=36, movement_threshold=0.5):
    """
    Show a confusion matrix for a random classifier
    """
    from models.simple.detector import verify_window_size
    window_size = verify_window_size(window_size)
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
    plot_confusion_matrix(matrix, title='Confusion matrix for random classifier')


def cross_validate_rule_preload(frames_csv, window_size=48, movement_threshold=0.5):
    """
    Show a confusion matrix for a rule-based classifier
    """
    from models.simple.detector import verify_window_size
    window_size = verify_window_size(window_size)
    df_frames = load_df(frames_csv)
    vectors = ordinal_from_csv(df_frames, threshold=movement_threshold)

    matrix = cross_validate(
        frames_csv,
        data=vectors,
        model_class=RuleBasedShakeDetector,
        params={
            'window_size': window_size,
            'movement_threshold': movement_threshold,
            'rule_func': RuleBasedShakeDetector.majority_vote
        }
    )
    plot_confusion_matrix(matrix,
                title=f'Confusion matrix for rule-based classifier with threshold={movement_threshold}')


def cross_validate_memory_preload(frames_csv, window_size=48, movement_threshold=0.5):
    """
    Cross validation wrapper specifically designed for the memory based classifier using HMM vectors.

    :param frames_csv: Dataframe with frame annotations, labels and any other information used by your model
    :param window_size: Window size to use during prediction of new sequences
    :param movement_threshold: Movement threshold in pixels to determine when a person is considered to move
    """
    from models.simple.detector import verify_window_size
    window_size = verify_window_size(window_size)
    df_frames = load_df(frames_csv)
    vectors = ordinal_from_csv(df_frames, threshold=movement_threshold)

    matrix = cross_validate(
        frames_csv,
        data=vectors,
        model_class=MemoryBasedShakeNodDetector,
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
    matrix = np.zeros((3, 3))

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
    pred_list, label_list = [], []

    for _, row in df_val.iterrows():
        pred_list.append(model.predict(val_sequences[i]))
        labels = np.load(row['labels_path'])[row['start_frame']:row['end_frame']]
        labels = labels[cut_size + 1:-cut_size]
        label_list.append(labels)

        i += 1

    preds = np.concatenate(pred_list)
    labels = np.concatenate(label_list)

    return confusion_matrix(labels, preds, labels=classes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('frames_csv', type=Path)
    parser.add_argument('-w', '--window_size', default=36 , type=int)
    parser.add_argument('-t', '--movement_threshold', default=0.5, type=float)
    subparsers = parser.add_subparsers(dest='subparser')

    hmm_parser = subparsers.add_parser('hmm')
    hmm_parser.add_argument('folds_dir', type=Path)
    hmm_parser.add_argument('-s', '--save_dir', type=Path)

    window_parser = subparsers.add_parser('window')
    window_parser.add_argument('folds_dir', type=Path)

    filter_parser = subparsers.add_parser('filter')
    filter_parser.add_argument('folds_dir', type=Path)
    filter_parser.add_argument('-s', '--save_dir', type=Path)

    hmm_fit_parser = subparsers.add_parser('hmm_fit')
    hmm_fit_parser.add_argument('folds_dir', type=Path)

    memory_parser = subparsers.add_parser('memory')
    rule_parser = subparsers.add_parser('rule')
    random_parser = subparsers.add_parser('random')

    args = parser.parse_args()

    if args.subparser == 'hmm':
        validate_hmm_folds(args.frames_csv,
                           args.folds_dir,
                           window_size=args.window_size,
                           results_dir=args.save_dir,
                           filter_size=None,
                           method='event',
                           load_previous=True)
    elif args.subparser == 'window':
        hmm_window_search(args.frames_csv,
                          args.fols_dir,
                          method='frame',
                          metric='f1')
    elif args.subparser == 'filter':
        hmm_filter_search(args.frames_csv,
                          args.folds_dir,
                          results_dir=args.save_dir,
                          method='event',
                          metric='f1')
    elif args.subparser == 'hmm_fit':
        fit_hmm_folds(args.frames_csv,
                      args.folds_dir)
    elif args.subparser == 'memory':
        cross_validate_memory_preload(args.frames_csv,
                                      window_size=args.window_size,
                                      movement_threshold=args.movement_threshold)
    elif args.subparser == 'rule':
        cross_validate_rule_preload(args.frames_csv,
                                    window_size=args.window_size,
                                    movement_threshold=args.movement_threshold)
    elif args.subparser == 'random':
        cross_validate_random_preload(args.frames_csv,
                                      window_size=args.window_size,
                                      movement_threshold=args.movement_threshold)
    else:
        raise RuntimeError('Please choose a subcommand')
