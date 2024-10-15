import random
from collections import defaultdict
from pathlib import Path
import argparse

import pandas as pd
import numpy as np

from utils.frames_csv import load_df


def main(frames_csv, num_splits=4):
    """
    For the DGS dataset, all files are split into 4 folds (the training folds, fold 5 is used for validation
    and only contains full NGT videos)
    The files contain only videos that are shakes and nods, so we take the label of the video's middle frame
    as the label for the video. We then distribute the videos into the folds such that each fold has an equal
    number of nods and shakes.
    """

    df_frames = load_df(frames_csv)
    files = list(df_frames['video_id']+"_"+df_frames['speaker_id'])
    labels = list(df_frames['labels_path'].apply(np.load))

    # Extract middle value from each numpy array and put it in a new list
    labels_maxs = [int(max(arr)) for arr in labels.copy()]
    for l_i, _ in enumerate(labels):
        if labels_maxs[l_i] == 0:
            print("Zero label found")
            print(files[l_i])
    labels = labels_maxs

    # Create a dictionary to store files for each video and speaker combination
    video_speaker_files = defaultdict(list)
    for file, label in zip(files, labels):
        video_id = file.split('-')[0]
        speaker_id = file.split('_')[1]
        video_speaker_files[(video_id, speaker_id)].append((file, label))

    # Initialize splits
    splits = [[] for _ in range(num_splits)]
    nod_counts = [0] * num_splits
    shake_counts = [0] * num_splits

    nod_in_split = {i: 0 for i in range(num_splits)}
    shake_in_split = {i: 0 for i in range(num_splits)}

    # Sort video_speaker_files by the number of files in each group
    video_speaker_files = dict(sorted(video_speaker_files.items(), key=lambda x: len(x[1]), reverse=True))

    # Iterate over video and speaker combinations and distribute files into splits
    for video_speaker, files in video_speaker_files.items():

        # Count the number of nods and shakes for each combination
        nod_count = sum(1 for file, label in files if label == 2)
        shake_count = sum(1 for file, label in files if label == 1)
        if not (shake_count == (len(files) - nod_count)):
            print("Error: background label found in the files")

        # Find the split with the fewest nods or shakes
        if nod_count > shake_count:
            min_split = min(range(num_splits), key=lambda i: nod_in_split[i])
        else:
            min_split = min(range(num_splits), key=lambda i: shake_in_split[i])

        # Iterate over files for the current combination and distribute them into splits
        for file, label in files:

            # Add the file to the split
            splits[min_split].append((file, label))

            # Update nod and shake counts for the split
            if label == 2:
                nod_in_split[min_split] += 1
                nod_counts[min_split] += 1
            elif label == 1:
                shake_in_split[min_split] += 1
                shake_counts[min_split] += 1

    # Print the number of nods and shakes per split
    for i in range(num_splits):
        print(f"Split {i + 1}: Nr nods: {nod_counts[i]}, Nr shakes: {shake_counts[i]}")

    for i, split in enumerate(splits):
        print(f"Split {i + 1}: {split}")

    split_col = []
    for i in range(len(df_frames)):
        for j, split in enumerate(splits):
            if df_frames['video_id'][i]+"_"+df_frames['speaker_id'][i] in [file for file, _ in split]:
                split_col.append("fold_"+str(j+1))
                break
        else:
            raise ValueError("File not found in any split")
    
    df_frames['split'] = split_col

    df_frames = df_frames.reindex(columns=['video_id', 'media_path', 'start_frame', 'end_frame', 'labels_path', 'split', 'keypoints_path', 'boxes_path', 'headboxes_path', 'headposes_path'])

    df_frames.to_csv(frames_csv, index=False)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('frames_csv', metavar='frames_csv', type=Path)
    args = parser.parse_args()

    main(args.frames_csv)
