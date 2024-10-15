import numpy as np
import argparse
from pathlib import Path

from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from utils.draw import set_seaborn_theme, reset_seaborn_theme

from Levenshtein import ratio, distance


def main(frames_csv, predictions_path, fold):
    """Show the results of the head-shake detection on the CNGT videos.

    :param frames_csv: Path to the CSV with the frame ranges
    :param predictions_path: Path to the predictions file 
    """
    from utils.frames_csv import load_all_labels, load_df, get_splits
    df_frames = load_df(frames_csv)
    predictions = load_predictions(predictions_path)

    splits = get_splits(df_frames)
    if fold == "val":
        splits = splits[4:5]
    elif fold == "test":
        splits = ['test']

    labels = []
    for fold in splits:
        df_val = df_frames[df_frames['split'] == fold]
        labels.extend(load_all_labels(df_val, shift=1, window=36))
    
    # Only keep the labels that contain nods
    new_labels = []
    for l in labels:
        if 2 in l:
            new_labels.append(l)
    labels = new_labels

    # Check the length of the predictions and labels
    for i in range(len(predictions)):
        if len(predictions[i]) != len(labels[i]):
            print(f'Length mismatch: {len(predictions[i])} vs {len(labels[i])}')

    statistics = {'flip_diff': np.array(list(map(flips, predictions))) / np.array(list(map(flips, labels))),
                  'edit_ratio': np.array([ratio(predictions[i], labels[i]) for i in range(len(predictions))]),
                  'edit_dist': np.array([distance(predictions[i], labels[i]) for i in range(len(predictions))]),
                  'acc': np.array([accuracy_score(predictions[i], labels[i]) for i in range(len(predictions))]),
                  'precision': np.array([precision_score(predictions[i], labels[i], average='weighted') for i in range(len(predictions))]),
                  'recall': np.array([recall_score(predictions[i], labels[i], average='weighted') for i in range(len(predictions))]),
                  'f1': np.array([f1_score(predictions[i], labels[i], average='weighted') for i in range(len(predictions))])
                  }

    set_seaborn_theme()
    fig, axs = plt.subplots(nrows=4, ncols=2, figsize=(10, 8))
    fig.tight_layout()
    
    plt.title('Performance of head-shake detection on CNGT videos')

    print(f'Range for flip ratio: '
          f'{round(min(statistics["flip_diff"]), 2)}-{round(max(statistics["flip_diff"]), 2)}')
    print(f'Mean for flip ratio: {round(float(np.mean(statistics["flip_diff"])), 2)}')
    print(f'Std for flip ratio: {round(float(np.std(statistics["flip_diff"])), 2)}')
    print('#' * 40)
    axs[0, 0].hist(statistics['flip_diff'], bins=40)
    axs[0, 0].set_title('Histogram of flip ratio', fontsize=10)

    print(f'Range for Levenshtein ratio: '
          f'{round(min(statistics["edit_ratio"]), 2)}-{round(max(statistics["edit_ratio"]), 2)}')
    print(f'Mean for Levenshtein ratio: {round(float(np.mean(statistics["edit_ratio"])), 2)}')
    print(f'Std for Levenshtein ratio: {round(float(np.std(statistics["edit_ratio"])), 2)}')
    print('#' * 40)
    axs[1, 0].hist(statistics['edit_ratio'], bins=40)
    axs[1, 0].set_title('Histogram of Levenshtein ratio', fontsize=10)

    print(f'Range for Levenshtein distance: '
          f'{round(min(statistics["edit_dist"]), 2)}-{round(max(statistics["edit_dist"]), 2)}')
    print(f'Mean for Levenshtein distance: {round(float(np.mean(statistics["edit_dist"])), 2)}')
    print(f'Std for Levenshtein distance: {round(float(np.std(statistics["edit_dist"])), 2)}')
    print('#' * 40)
    axs[1, 1].hist(statistics['edit_dist'], bins=40)
    axs[1, 1].set_title('Histogram of Levenshtein distance', fontsize=10)

    print(f'Range for accuracy: '
          f'{round(min(statistics["acc"]), 2)}-{round(max(statistics["acc"]), 2)}')
    print(f'Mean for accuracy: {round(float(np.mean(statistics["acc"])), 2)}')
    print(f'Std for accuracy: {round(float(np.std(statistics["acc"])), 2)}')
    print('#' * 40)
    axs[2, 0].hist(statistics['acc'], bins=40)
    axs[2, 0].set_title('Histogram of Accuracy', fontsize=10)

    print(f'Range for precision: '
          f'{round(min(statistics["precision"]), 2)}-{round(max(statistics["precision"]), 2)}')
    print(f'Mean for precision: {round(float(np.mean(statistics["precision"])), 2)}')
    print(f'Std for precision: {round(float(np.std(statistics["precision"])), 2)}')
    print('#' * 40)
    axs[2, 1].hist(statistics['precision'], bins=40)
    axs[2, 1].set_title('Histogram of precision', fontsize=10)

    print(f'Range for recall: '
          f'{round(min(statistics["recall"]), 2)}-{round(max(statistics["recall"]), 2)}')
    print(f'Mean for recall: {round(float(np.mean(statistics["recall"])), 2)}')
    print(f'Std for recall: {round(float(np.std(statistics["recall"])), 2)}')
    print('#' * 40)
    axs[3, 0].hist(statistics['recall'], bins=40)
    axs[3, 0].set_title('Histogram of recall', fontsize=10)

    print(f'Range for f1 score: '
          f'{round(min(statistics["f1"]), 2)}-{round(max(statistics["f1"]), 2)}')
    print(f'Mean for f1: {round(float(np.mean(statistics["f1"])), 2)}')
    print(f'Std for f1: {round(float(np.std(statistics["f1"])), 2)}')
    print('#' * 40)
    axs[3, 1].hist(statistics['f1'], bins=40)
    axs[3, 1].set_title('Histogram of f1 score', fontsize=10)

    plt.tight_layout()

    plt.show()

    reset_seaborn_theme()

def load_predictions(path):
    preds = np.load(path)
    arrays = []

    # noinspection PyUnresolvedReferences
    for file in preds.files:
        arrays.append(preds[file])

    # noinspection PyUnresolvedReferences
    preds.close()

    return arrays

def flips_indices(sequence):
    return np.diff(sequence) != 0


def flips(sequence):
    """Count the number of times the value in a time sequence changes from one step to the next.

    :param sequence: Numpy array with time data
    :return: Number of value changes over the sequence
    """
    return flips_indices(sequence).sum()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('frames_csv', type=Path)
    parser.add_argument('predictions_csv', type=Path)
    parser.add_argument('fold', type=str)

    args = parser.parse_args()
    main(args.frames_csv, args.predictions_csv, args.fold)
