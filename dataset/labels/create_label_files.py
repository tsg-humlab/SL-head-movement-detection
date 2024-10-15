import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from dataset.labels.eaf_parser import Video
from utils.media import get_n_frames_from_keypoints
from utils.frames_csv import load_df


def set_start_end_frames(frames_csv, start_frames=75, end_frames=25):
    """Add to the frames csv the frame count of every video from start to end without the license frames 
    (default frames match that of NGT dataset).

    :param overview_csv: Path to the frames CSV with media paths
    :param output_csv: Path to write the frames CSV to
    """
    df_frames = load_df(frames_csv)
    starts, ends = [], []

    for _, row in df_frames.iterrows():

        video_id = row['video_id']
        keypoints_path = row['keypoints_path']

        try:
            n_frames = get_n_frames_from_keypoints(keypoints_path)
        except:
            print(keypoints_path+" doesn't exits")

        if 'CNGT' in video_id:
            starts.append(start_frames)
            ends.append(n_frames - end_frames)
        else:
            starts.append(0)
            ends.append(n_frames)

    df_frames['start_frame'] = starts
    df_frames['end_frame'] = ends
    df_frames.to_csv(frames_csv, index=False)


def make_pretty_histogram(data, type="Shake"):
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Create a figure composed of two matplotlib.Axes objects (ax_box and ax_hist)
    sns.set_theme(style="darkgrid")
    f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)}, figsize=(20, 8))

    # Make box plot
    box_plot = sns.boxplot(x=data, orient="h", ax=ax_box, showmeans=True, 
        meanprops={"marker": ".", "markeredgecolor": "black", "markersize": "10"})
    ax_box.set_xlim(left=0, right=200)
    median_value = np.median(data)
    mean_value = np.mean(data)
    ax_box.text(median_value+0.30, 0.15, int(median_value), verticalalignment='center', size=10, alpha=0.8, color='black')
    ax_box.text(mean_value+0.30, 0.15, int(mean_value), verticalalignment='center', size=10, alpha=0.8, color='black')

    # Make hist plot
    counts, bins, patches = ax_hist.hist(data, bins=np.arange(data.min(), data.max() + 1))
    ax_hist.set_xlim(left=0, right=200)
    max_count = counts.max()
    max_count_index = counts.argmax()
    max_bin_center = (bins[max_count_index] + bins[max_count_index + 1]) / 2
    ax_hist.text(max_bin_center, max_count + 0.5, f"{int(max_count)}, {int(max_bin_center)}", ha='center', va='bottom', color='black', fontsize=10)

    plt.xlabel(type+' length (number of frames)')  # Add x-axis label
    plt.ylabel('Frequency')     # Add y-axis label
    plt.title('Histogram of '+type.lower()+' lengths')
    plt.show()

def label_statistics(eaf_dir, frames_path):
    """ Calculate statistics for the nodding and shaking movements """

    df_frames = load_df(frames_path)

    nr_shakes, nr_nods = 0,0
    shake_lengths, nod_lengths = [], []
    unsure_shakes, unsure_nods = 0,0

    for _, row in df_frames.iterrows():
        unique_id = row['video_id']
        signer_id = unique_id.split('_')[1]

        eaf_path = eaf_dir+"/"+f'{unique_id}.eaf'
        video = Video.from_eaf(eaf_path)
        annotations = video[signer_id].annotations

        keypoints_path = row['keypoints_path']
        if not Path(keypoints_path).exists():
            print(f'No keypoints found for {unique_id}')
            df_frames.drop(df_frames[df_frames['keypoints_path'] == str(keypoints_path)].index, inplace=True)
            continue
        labels = np.zeros(get_n_frames_from_keypoints(keypoints_path))

        for annotation in annotations:
            orig_label = annotation.label.lower()
            label = orig_label.replace("?", "")

            if label in ['n', 'nf', 'nx', 'nn', 'nn:l', 'nnf', 'shake']:
                if "?" in orig_label:
                    unsure_shakes += 1
                shake_lengths.append(annotation.end-annotation.start)
                labels[annotation.start:annotation.end] = 1
                nr_shakes += 1

            if label in ['d', 'df', 'db', 'nod:l', 'nod', 'nod:n']:
                if "?" in orig_label:
                    unsure_nods += 1
                nod_lengths.append(annotation.end-annotation.start)
                labels[annotation.start:annotation.end] = 2
                nr_nods += 1
        
    print(f'Number of shakes: {nr_shakes}')
    print(f'Number of nods: {nr_nods}')
    print(f'Mean shake length: {np.mean(shake_lengths)}')
    print(f'Mean nod length: {np.mean(nod_lengths)}')
    print(f'Std shake length: {np.std(shake_lengths)}')
    print(f'Std nod length: {np.std(nod_lengths)}')
    print(f'Unsure shakes: {unsure_shakes}')
    print(f'Unsure nods: {unsure_nods}')
    
    import matplotlib.pyplot as plt
    print(len(shake_lengths))
    print("Most common shake length: ", np.argmax(np.bincount(np.array(shake_lengths))))
    make_pretty_histogram(np.array(shake_lengths), "Shake")
    print(len(nod_lengths))
    print("Most common nod length: ", np.argmax(np.bincount(np.array(nod_lengths))))
    make_pretty_histogram(np.array(nod_lengths), "Nod")


def main(eaf_dir, output_dir, frames_path, add_starts_ends):
    """ Create the labels for the nodding and shaking movements """

    if add_starts_ends:
        set_start_end_frames(frames_path)

    df_frames = load_df(frames_path)

    nr_shakes, nr_nods = 0,0
    labels_paths = []

    for _, row in df_frames.iterrows():
        unique_id = row['video_id']
        signer_id = unique_id.split('_')[1]

        eaf_path = eaf_dir+"/"+f'{unique_id}.eaf'
        video = Video.from_eaf(eaf_path)
        annotations = video[signer_id].annotations

        keypoints_path = row['keypoints_path']
        if not Path(keypoints_path).exists():
            print(f'No keypoints found for {unique_id}')
            df_frames.drop(df_frames[df_frames['keypoints_path'] == str(keypoints_path)].index, inplace=True)
            continue
        labels = np.zeros(get_n_frames_from_keypoints(keypoints_path))
        labels_paths.append(Path(output_dir) / Path(f'{unique_id}.npy'))

        for annotation in annotations:
            label = annotation.label.lower().replace("?", "")

            if label in ['n', 'nf', 'nx', 'nn', 'nn:l', 'nnf', 'shake']:
                labels[annotation.start:annotation.end] = 1
                nr_shakes += 1

            if label in ['d', 'df', 'db', 'nod:l', 'nod', 'nod:n']:
                labels[annotation.start:annotation.end] = 2
                nr_nods += 1

        np.save(Path(output_dir) / Path(f'{unique_id}.npy'), labels)
        
    print("Labels saved!")
    
    df_frames['labels_path'] = labels_paths
    df_frames.to_csv(frames_path, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('eaf_dir', metavar='eaf-dir', type=Path)
    parser.add_argument('output_dir', metavar='output-dir', type=Path)
    parser.add_argument('frames_path', metavar='frames-path', type=Path)
    parser.add_argument('add_starts_ends', metavar='add-starts-ends', type=bool, default=False)
    args = parser.parse_args()

    main(args.eaf_dir, args.output_dir, args.frames_path, args.add_starts_ends)
