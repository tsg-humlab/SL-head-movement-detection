import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd

from utils.array_manipulation import get_uninterrupted_ones


def main(frames_csv, folds=5, test_size=0.1, seed=94, method='moves'):
    df_frames = pd.read_csv(frames_csv)
    shake_counts = {}
    total_count = 0

    video_ids = list(df_frames['video_id'])
    labels = list(df_frames['labels_path'].apply(np.load))
    start_indices = list(df_frames['start_frame'])
    end_indices = list(df_frames['end_frame'])

    for i in range(len(df_frames)):
        if method == 'frames':
            count = np.sum(labels[i][start_indices[i]:end_indices[i]])
        elif method == 'moves':
            count = len(get_uninterrupted_ones(labels[i][start_indices[i]:end_indices[i]]))
        else:
            raise ValueError(f'{method} is not a valid argument (should be frames or moves)')

        if count > 0:
            shake_counts[video_ids[i]] = count
            total_count += count
        else:
            df_frames.drop(df_frames[df_frames['video_id'] == video_ids[i]].index, inplace=True)

    group_fragments(shake_counts)
    clean_keys(shake_counts)
    signer_to_ids = create_singer_groups(shake_counts)
    df_frames['split'] = 'train'

    seeded_random = random.Random(seed)
    ids = list(signer_to_ids.keys())
    seeded_random.shuffle(ids)

    test_count = 0
    target = total_count * test_size

    for signer in ids:
        ids.remove(signer)

        for video in signer_to_ids[signer]:
            df_frames.loc[df_frames['video_id'].str.contains(video), 'split'] = 'test'
            test_count += shake_counts[video]

        if test_count >= target:
            break

    fold_counts = np.zeros(folds)
    i = 0

    for signer in ids:
        for video in signer_to_ids[signer]:
            fold_counts[i] = fold_counts[i] + shake_counts[video]
            df_frames.loc[df_frames['video_id'].str.contains(video), 'split'] = f'fold_{i + 1}'

        i = np.argmin(fold_counts)

    df_frames.to_csv(frames_csv, index=False)


def group_fragments(count_dict):
    keys = list(count_dict.keys())

    for key in keys:
        stripped_key = '_'.join(key.split('_')[:2])
        matches = [(k, v) for k, v in count_dict.items() if stripped_key in k]

        if len(matches) > 1:
            count_dict[stripped_key] = sum([match[1] for match in matches])

            for match in matches:
                del count_dict[match[0]]


def clean_keys(count_dict):
    for key in list(count_dict):
        if len(key.split('_')) != 2:
            value = count_dict.pop(key)
            count_dict['_'.join(key.split('_')[:2])] = value


def create_singer_groups(count_dict):
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
