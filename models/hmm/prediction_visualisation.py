import numpy as np
from matplotlib import pyplot as plt

from utils.draw import set_seaborn_theme
from utils.frames_csv import load_all_labels, load_df, get_splits


def main(frames_csv, predictions_path):
    df_frames = load_df(frames_csv)
    df_frames = df_frames[df_frames['split'].str.contains('fold')]
    predictions = load_predictions(predictions_path)

    splits = get_splits(df_frames)
    labels = []
    for fold in splits:
        df_val = df_frames[df_frames['split'] == fold]

        labels.extend(load_all_labels(df_val, shift=1, window=48))

    statistics = {'flips': list(map(flips, predictions))}

    for i in range(len(predictions)):
        barcode_and_truth(predictions[i], labels[i])
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
    ax = fig.add_axes([0, 0, 1, 1])  # span the whole figure
    ax.set_axis_off()
    ax.imshow(sequence.reshape(1, -1), cmap='Paired', aspect='auto', interpolation='nearest')

    plt.show()


def barcode_and_truth(sequence, labels):
    set_seaborn_theme()

    pixel_per_bar = 4
    dpi = 100

    fig, axs = plt.subplots(2, figsize=(len(sequence) * pixel_per_bar / dpi, 2), dpi=dpi)
    # ax = fig.add_axes([0, 0, 1, 1])  # span the whole figure
    axs[0].set_axis_off()
    axs[0].imshow(sequence.reshape(1, -1), cmap='Paired', aspect='auto', interpolation='nearest')
    axs[1].set_axis_off()
    axs[1].imshow(labels.reshape(1, -1), cmap='binary', aspect='auto', interpolation='nearest')

    plt.show()


def flips(sequence):
    return (np.diff(sequence) != 0).sum() / len(sequence)


if __name__ == '__main__':
    main(r"E:\Data\CNGT_pose\frames_pose.csv",
         r"E:\Experiments\hmm\cross_val\val_preds\predictions.npz")
