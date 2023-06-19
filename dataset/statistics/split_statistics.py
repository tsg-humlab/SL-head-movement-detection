import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def main(frames_csv):
    df_frames = pd.read_csv(frames_csv)

    background_counts = {}
    shake_counts = {}

    for _, row in df_frames.iterrows():
        labels = np.load(row['labels_path'])

        shake_count = np.sum(labels)
        background_count = int(row['end_frame']) - int(row['start_frame']) - shake_count

        try:
            shake_counts[row['split']] += shake_count
        except KeyError:
            shake_counts[row['split']] = shake_count
        try:
            background_counts[row['split']] += background_count
        except KeyError:
            background_counts[row['split']] = background_count

    background_counts = {key: value for key, value in sorted(background_counts.items())}
    shake_counts = {key: value for key, value in sorted(shake_counts.items())}

    sns.set_theme()
    sns.set_style('whitegrid')
    sns.set_context('paper')

    plt.title(f'Frame counts for the different datasets', fontsize=15)
    plt.bar(list(shake_counts.keys()),
            list(shake_counts.values()),
            align='center')
    plt.bar(list(background_counts.keys()),
            list(background_counts.values()),
            align='center',
            bottom=list(shake_counts.values()))
    plt.show()

    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('frames_csv', metavar='frames-csv', type=Path)
    args = parser.parse_args()

    main(args.frames_csv)
