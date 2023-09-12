import argparse
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

from models.hmm.filters import majority_filter
from models.hmm.prediction_visualisation import load_predictions
from models.hmm.test_hmm import predict_hmm
from utils.draw import set_seaborn_theme
from utils.frames_csv import load_df, get_splits, load_all_labels
from utils.io import mkdir_if_not_exists

from scipy.stats import f_oneway


def validate_hmm_folds(frames_csv,
                       folds_dir,
                       window_size=36,
                       results_dir=None,
                       filter_size=None,
                       load_previous=False):
    print(f'Loading samples from {frames_csv}')
    df_frames = load_df(frames_csv)
    splits = get_splits(df_frames)
    all_predictions = []
    label_index = 0

    if load_previous:
        all_predictions = load_predictions(results_dir / 'predictions.npz')

    signer_to_labels = {}
    signer_to_predictions = {}

    for fold in splits:
        print(f'Validating {fold}/{len(splits)}')
        df_val = df_frames[df_frames['split'] == fold]

        if load_previous:
            labels = load_all_labels(df_val, shift=1, window=window_size)
            predictions = all_predictions[label_index:label_index + len(df_val)]
            label_index += len(df_val)
        else:
            labels, predictions = predict_hmm(df_val, folds_dir / fold, window_size=window_size)

        if filter_size:
            predictions = [majority_filter(prediction, filter_size) for prediction in predictions]

        for i, row in enumerate(df_val.iterrows()):
            row = row[1]
            signer = row['video_id'].split('_')[1]

            try:
                signer_to_predictions[signer].append(predictions[i])
                signer_to_labels[signer].append(labels[i])
            except KeyError:
                signer_to_predictions[signer] = [predictions[i]]
                signer_to_labels[signer] = [labels[i]]

        if results_dir and not load_previous:
            all_predictions.extend(predictions)

    precision = []
    precision_sep = []
    recall = []
    recall_sep = []
    f1 = []
    f1_sep = []
    signers = list(signer_to_labels.keys())

    for signer in signers:
        precision.append(precision_score(np.concatenate(signer_to_labels[signer]),
                                         np.concatenate(signer_to_predictions[signer])))
        recall.append(recall_score(np.concatenate(signer_to_labels[signer]),
                                   np.concatenate(signer_to_predictions[signer])))
        f1.append(f1_score(np.concatenate(signer_to_labels[signer]),
                           np.concatenate(signer_to_predictions[signer])))

        precision_tmp = []
        recall_tmp = []
        f1_tmp = []
        for i in range(len(signer_to_labels[signer])):
            precision_tmp.append(precision_score(signer_to_labels[signer][i],
                                                 signer_to_predictions[signer][i]))
            recall_tmp.append(recall_score(signer_to_labels[signer][i],
                                           signer_to_predictions[signer][i]))
            f1_tmp.append(f1_score(signer_to_labels[signer][i],
                                   signer_to_predictions[signer][i]))
        precision_sep.append(precision_tmp)
        recall_sep.append(recall_tmp)
        f1_sep.append(f1_tmp)

    set_seaborn_theme()

    precision_f = f_oneway(*precision_sep)
    recall_f = f_oneway(*recall_sep)
    f1_f = f_oneway(*f1_sep)

    fig, ax = plt.subplots()
    plt.title(f'Precision per signer, ANOVA p={round(precision_f[1], 3)}')
    plt.bar(signers, precision)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.show()

    fig, ax = plt.subplots()
    plt.title(f'Recall per signer, ANOVA p={round(recall_f[1], 3)}')
    plt.bar(signers, recall)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.show()

    fig, ax = plt.subplots()
    plt.title(f'F1 per signer, ANOVA p={round(f1_f[1], 3)}')
    plt.bar(signers, f1)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.show()

    if results_dir and not load_previous:
        mkdir_if_not_exists(results_dir)
        np.savez(results_dir / 'predictions.npz', *all_predictions)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('frames_csv', type=Path)
    parser.add_argument('-w', '--window_size', default=36, type=int)
    parser.add_argument('-t', '--movement_threshold', default=0.5, type=float)
    subparsers = parser.add_subparsers(dest='subparser')

    hmm_parser = subparsers.add_parser('hmm')
    hmm_parser.add_argument('folds_dir', type=Path)
    hmm_parser.add_argument('-s', '--save_dir', type=Path)

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
                           filter_size=11,
                           load_previous=True)
