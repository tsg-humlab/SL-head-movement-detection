import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from utils.helper_functions import sort_dict


def main(frames_csv):
    df_frames = pd.read_csv(frames_csv)

    background_counts = {}
    shake_counts = {}
    split_to_signers = {}

    for _, row in df_frames.iterrows():
        labels = np.load(row['labels_path'])

        shake_count = np.sum(labels)
        background_count = int(row['end_frame']) - int(row['start_frame']) - shake_count

        try:
            shake_counts[row['split']] += shake_count
            background_counts[row['split']] += background_count
            split_to_signers[row['split']].add(row['video_id'].split('_')[1])
        except KeyError:
            shake_counts[row['split']] = shake_count
            background_counts[row['split']] = background_count
            split_to_signers[row['split']] = {row['video_id'].split('_')[1]}

    background_counts = sort_dict(background_counts)
    shake_counts = sort_dict(shake_counts)
    split_to_signers = sort_dict(split_to_signers)

    sns.set_theme()
    sns.set_style('whitegrid')
    sns.set_context('paper')

    plt.title(f'Frame counts for the different splits', fontsize=15)
    plt.bar(list(shake_counts.keys()),
            list(shake_counts.values()),
            align='center')
    plt.bar(list(background_counts.keys()),
            list(background_counts.values()),
            align='center',
            bottom=list(shake_counts.values()))
    plt.show()

    plt.title(f'Signers per split', fontsize=15)
    plt.bar(list(split_to_signers.keys()),
            list([len(value) for value in split_to_signers.values()]),
            align='center')
    plt.show()

    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('frames_csv', metavar='frames-csv', type=Path)
    args = parser.parse_args()

    main(args.frames_csv)
