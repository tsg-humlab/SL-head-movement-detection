import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_score

from utils.draw import set_seaborn_theme
from utils.frames_csv import load_all_labels, load_df, get_splits

from Levenshtein import ratio, distance

from models.hmm.filters import majority_filter


def main(frames_csv, predictions_path):
    df_frames = load_df(frames_csv)
    df_frames = df_frames[df_frames['split'].str.contains('fold')]
    predictions = load_predictions(predictions_path)

    splits = get_splits(df_frames)
    labels = []
    for fold in splits:
        df_val = df_frames[df_frames['split'] == fold]

        labels.extend(load_all_labels(df_val, shift=1, window=48))

    statistics = {'flip_diff': np.array(list(map(flips, predictions))) / np.array(list(map(flips, labels))),
                  'edit_ratio': np.array([ratio(predictions[i], labels[i]) for i in range(len(predictions))]),
                  'edit_dist': np.array([distance(predictions[i], labels[i]) for i in range(len(predictions))]),
                  'acc': np.array([accuracy_score(predictions[i], labels[i]) for i in range(len(predictions))]),
                  'precision': np.array([precision_score(predictions[i], labels[i]) for i in range(len(predictions))]),
                  }

    set_seaborn_theme()

    print(f'Range for flip ratio: '
          f'{round(min(statistics["flip_diff"]), 2)}-{round(max(statistics["flip_diff"]), 2)}')
    print(f'Mean for flip ratio: {round(float(np.mean(statistics["flip_diff"])), 2)}')
    print(f'Std for flip ratio: {round(float(np.std(statistics["flip_diff"])), 2)}')
    print('#' * 40)
    plt.hist(statistics['flip_diff'], bins=40)
    plt.title('Histogram of flip ratio', fontsize=15)
    plt.show()

    print(f'Range for Levenshtein ratio: '
          f'{round(min(statistics["edit_ratio"]), 2)}-{round(max(statistics["edit_ratio"]), 2)}')
    print(f'Mean for Levenshtein ratio: {round(float(np.mean(statistics["edit_ratio"])), 2)}')
    print(f'Std for Levenshtein ratio: {round(float(np.std(statistics["edit_ratio"])), 2)}')
    print('#' * 40)
    plt.hist(statistics['edit_ratio'], bins=40)
    plt.title('Histogram of Levenshtein ratio', fontsize=15)
    plt.show()

    print(f'Range for Levenshtein distance: '
          f'{round(min(statistics["edit_dist"]), 2)}-{round(max(statistics["edit_dist"]), 2)}')
    print(f'Mean for Levenshtein distance: {round(float(np.mean(statistics["edit_dist"])), 2)}')
    print(f'Std for Levenshtein distance: {round(float(np.std(statistics["edit_dist"])), 2)}')
    print('#' * 40)
    plt.hist(statistics['edit_dist'], bins=40)
    plt.title('Histogram of Levenshtein distance', fontsize=15)
    plt.show()

    print(f'Range for accuracy: '
          f'{round(min(statistics["acc"]), 2)}-{round(max(statistics["acc"]), 2)}')
    print(f'Mean for accuracy: {round(float(np.mean(statistics["acc"])), 2)}')
    print(f'Std for accuracy: {round(float(np.std(statistics["acc"])), 2)}')
    print('#' * 40)
    plt.hist(statistics['acc'], bins=40)
    plt.title('Histogram of Accuracy', fontsize=15)
    plt.show()

    print(f'Range for precision: '
          f'{round(min(statistics["precision"]), 2)}-{round(max(statistics["precision"]), 2)}')
    print(f'Mean for precision: {round(float(np.mean(statistics["precision"])), 2)}')
    print(f'Std for precision: {round(float(np.std(statistics["precision"])), 2)}')
    print('#' * 40)
    plt.hist(statistics['precision'], bins=40)
    plt.title('Histogram of precision', fontsize=15)
    plt.show()

    for i in range(len(predictions)):
        barcode_and_truth(predictions[i], labels[i])
        input('Press Enter to continue...')


def investigate_filter(frames_csv, predictions_path):
    df_frames = load_df(frames_csv)
    df_frames = df_frames[df_frames['split'].str.contains('fold')]
    predictions = load_predictions(predictions_path)

    splits = get_splits(df_frames)
    labels = []
    for fold in splits:
        df_val = df_frames[df_frames['split'] == fold]

        labels.extend(load_all_labels(df_val, shift=1, window=48))

    set_seaborn_theme()

    for i in range(len(predictions)):
        sequence = predictions[i]
        sequence_filtered = majority_filter(sequence, 11)

        double_barcode_label(sequence, sequence_filtered, labels[i])
        input('Press Enter to continue...')


def load_predictions(path):
    preds = np.load(path)
    arrays = []

    # noinspection PyUnresolvedReferences
    for file in preds.files:
        arrays.append(preds[file])

    # noinspection PyUnresolvedReferences
    preds.close()

    return arrays


def barcode(sequence):
    set_seaborn_theme()

    pixel_per_bar = 4
    dpi = 100

    fig = plt.figure(figsize=(len(sequence) * pixel_per_bar / dpi, 2), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.imshow(sequence.reshape(1, -1), cmap='Paired', aspect='auto', interpolation='nearest')

    plt.show()


def barcode_and_truth(sequence, labels):
    sequence = sequence + np.where((sequence == labels) & (sequence == 1), 1, 0)

    set_seaborn_theme()

    pixel_per_bar = 4
    dpi = 100

    fig, axs = plt.subplots(2, figsize=(len(sequence) * pixel_per_bar / dpi, 2), dpi=dpi)
    axs[0].set_axis_off()
    axs[0].imshow(sequence.reshape(1, -1), cmap='brg', aspect='auto', interpolation='nearest')
    axs[1].set_axis_off()
    axs[1].imshow(labels.reshape(1, -1), cmap='binary', aspect='auto', interpolation='nearest')

    plt.show()


def double_barcode(sequence1, sequence2):
    set_seaborn_theme()

    pixel_per_bar = 4
    dpi = 100

    fig, axs = plt.subplots(2, figsize=(len(sequence1) * pixel_per_bar / dpi, 2), dpi=dpi)
    axs[0].set_axis_off()
    axs[0].imshow(sequence1.reshape(1, -1), cmap='brg', aspect='auto', interpolation='nearest')
    axs[1].set_axis_off()
    axs[1].imshow(sequence2.reshape(1, -1), cmap='brg', aspect='auto', interpolation='nearest')

    plt.show()


def double_barcode_label(sequence1, sequence2, labels):
    sequence1 = sequence1 + np.where((sequence1 == labels) & (sequence1 == 1), 1, 0)
    sequence2 = sequence2 + np.where((sequence2 == labels) & (sequence2 == 1), 1, 0)

    set_seaborn_theme()

    pixel_per_bar = 4
    dpi = 100

    fig, axs = plt.subplots(3, figsize=(len(sequence1) * pixel_per_bar / dpi, 2), dpi=dpi)
    axs[0].set_axis_off()
    axs[0].imshow(sequence1.reshape(1, -1), cmap='brg', aspect='auto', interpolation='nearest')
    axs[1].set_axis_off()
    axs[1].imshow(sequence2.reshape(1, -1), cmap='brg', aspect='auto', interpolation='nearest')
    axs[2].set_axis_off()
    axs[2].imshow(labels.reshape(1, -1), cmap='binary', aspect='auto', interpolation='nearest')

    plt.show()


def flips_indices(sequence):
    return np.diff(sequence) != 0


def flips(sequence):
    """Count the number of times the value in a time sequence changes from one step to the next.

    :param sequence: Numpy array with time data
    :return: Number of value changes over the sequence
    """
    return flips_indices(sequence).sum()


if __name__ == '__main__':
    investigate_filter(r"E:\Data\CNGT_pose\frames_pose_fixed_v2.csv",
         r"E:\Experiments\hmm\cross_val_fixed_v2\val_preds\predictions.npz")
