import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd

from utils.array_manipulation import get_uninterrupted
from utils.frames_csv import load_df


def main(frames_csv, folds=5, test_size=0.15, method='moves'):
    """
    This function splits the NGT dataset into training, validation and test sets.
    The dataset is split by signer, such that each signer is only in one of the splits.
    The test set is created by selecting a random subset of the signers and moving all their videos to the test set.
    The remaining videos are then split into folds, such that each fold has an equal number of nods and shakes.
    """
    df_frames = load_df(frames_csv)
    shake_counts, nod_counts = {}, {}
    total_count = 0

    unique_ids = list(df_frames['video_id'])
    labels = list(df_frames['labels_path'].apply(np.load))

    start_indices = list(df_frames['start_frame'])
    end_indices = list(df_frames['end_frame'])

    # Count the number of shakes and nods in each video
    # and remove videos with no shakes or nods
    for i in range(len(df_frames)):
        if method == 'frames':
            count_ones = np.count_nonzero(labels[i][start_indices[i]:end_indices[i]] == 1)
            count_twos = np.count_nonzero(labels[i][start_indices[i]:end_indices[i]] == 2) 
        elif method == 'moves':
            count_ones = len(get_uninterrupted(labels[i][start_indices[i]:end_indices[i]], 1))
            count_twos = len(get_uninterrupted(labels[i][start_indices[i]:end_indices[i]], 2))
        else:
            raise ValueError(f'{method} is not a valid argument (should be frames or moves)')

        shake_counts[unique_ids[i]] = count_ones
        total_count += count_ones
        nod_counts[unique_ids[i]] = count_twos
        total_count += count_twos

        if (count_ones + count_twos) == 0:
            print("Removing video: ", unique_ids[i])
            df_frames.drop(df_frames[df_frames['video_id'] == unique_ids[i]].index, inplace=True)

    print("Total count: ", total_count)
    shake_count = 0
    # Count shakes and nods
    for (_, s_v) in shake_counts.items():
        shake_count += s_v
    print("Shake count: ", shake_count)
    nod_count = 0
    for (_, n_v) in nod_counts.items():
        nod_count += n_v
    print("Nod count: ", nod_count)

    # group_videos(counts)
    clean_keys(nod_counts)
    signer_to_ids = create_signer_groups(nod_counts)
    df_frames['split'] = 'train'

    seeded_random = random.Random()
    ids = list(signer_to_ids.keys())
    seeded_random.shuffle(ids)

    test_count = 0
    target = nod_count * test_size

    for signer in ids:
        # adds all videos of the signer to the test set until the target is reached
        ids.remove(signer)

        for video in signer_to_ids[signer]:
            
            df_frames.loc[(df_frames['video_id']).str.contains(video), 'split'] = 'test'
            test_count += nod_counts[video]

        if test_count >= target:
            break

    fold_counts_nods = np.zeros(folds)
    fold_counts_shakes = np.zeros(folds)

    i = 0

    for signer in ids:
        # adds all videos of the signer to the fold with the least nods/shakes

        for video in signer_to_ids[signer]:

            # If there is a nod found, add it to the fold with the lease nods
            # Otherwise, add it to the fold of the least shakes
            if nod_counts[video] > 0:
                i = np.argmin(fold_counts_nods)
            else:
                i = np.argmin(fold_counts_shakes)

            fold_counts_nods[i] = fold_counts_nods[i] + nod_counts[video]
            fold_counts_shakes[i] = fold_counts_shakes[i] + shake_counts[video]
            df_frames.loc[df_frames['video_id'].str.contains(video), 'split'] = f'fold_{i + 1}'
    
    print("Nod counts per fold:", fold_counts_nods)
    print("Shake counts per fold:", fold_counts_shakes)

    df_frames = df_frames.reindex(columns=['video_id', 'media_path', 'start_frame', 'end_frame', 'labels_path', 'split', 'keypoints_path', 'boxes_path', 'headboxes_path', 'headposes_path'])

    df_frames.to_csv(frames_csv, index=False)


def clean_keys(count_dict):
    """ Removes the last part of the key if it is not a signer id """
    for key in list(count_dict):
        if len(key.split('_')) != 2:
            value = count_dict.pop(key)
            count_dict['_'.join(key.split('_')[:2])] = value


def create_signer_groups(count_dict):
    """ Creates a dictionary with the signer id as key and a list of video ids as value """
    keys = list(count_dict.keys())
    output = {}

    for key in keys:
        signer = key.split('_')[-1]

        try:
            output[signer].append(key)
        except KeyError:
            output[signer] = [key]

    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('frames_csv', metavar='frames-csv', type=Path)
    args = parser.parse_args()

    main(args.frames_csv)
