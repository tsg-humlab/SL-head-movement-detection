import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from pose import BOXES, KEYPOINTS
from utils.array_manipulation import find_subject_video
from utils.media import get_session_from_cngt_file, get_signer_from_cngt_file


def process_all(frames_csv, results_dir):
    df_frames = pd.read_csv(frames_csv)
    avg_position_x = np.zeros(len(df_frames))
    avg_position_y = np.zeros(avg_position_x.shape)

    for i, row in df_frames.iterrows():
        media_path = Path(row['media_path'])
        session_id = get_session_from_cngt_file(media_path)
        signer_id = get_signer_from_cngt_file(media_path)

        keypoints_path = results_dir / KEYPOINTS / f'{session_id}_{signer_id}.npy'
        boxes_path = results_dir / BOXES / f'{session_id}_{signer_id}.npy'
        keypoints_arr = np.load(keypoints_path)
        boxes_arr = np.load(boxes_path)

        # process_array(keypoints_arr[row['start_frame']:row['end_frame']],
        #               boxes_arr[row['start_frame']:row['end_frame']])
        avg_position_x[i], avg_position_y[i] = compute_average_positions(
            keypoints_arr[row['start_frame']:row['end_frame']],
            boxes_arr[row['start_frame']:row['end_frame']]
        )

        pass

    pass

    plot_positions(avg_position_x, avg_position_y)


def process_array(keypoints, boxes):
    indices = find_subject_video(keypoints, boxes)
    keypoints = keypoints[indices]

    x_diff = np.diff(keypoints[:, 0, 0])
    y_diff = np.diff(keypoints[:, 0, 1])

    pass


def compute_average_positions(keypoints, boxes):
    indices = find_subject_video(keypoints, boxes)
    keypoints = keypoints[indices]

    average_x = np.average(keypoints[:, 0, 0])
    average_y = np.average(keypoints[:, 0, 1])

    return average_x, average_y


def plot_positions(positions_x, positions_y, n_bins=20):
    positions_x = (positions_x - np.mean(positions_x)) / np.std(positions_x)
    positions_y = (positions_y - np.mean(positions_y)) / np.std(positions_y)
    
    sns.set_theme()
    sns.set_style('whitegrid')
    sns.set_context('paper')

    plt.title('Average nose pixel coordinates per video', fontsize=15)
    plt.xlabel('X pixel')
    plt.ylabel('Y pixel')
    plt.scatter(positions_x, positions_y)
    plt.show()

    plt.title('Distribution of average x pixel coordinate', fontsize=15)
    plt.hist(positions_x, bins=n_bins)
    plt.show()

    plt.title('Distribution of average y pixel coordinate', fontsize=15)
    plt.hist(positions_y, bins=n_bins)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('frames_csv', type=Path)
    parser.add_argument('results_dir', type=Path)
    args = parser.parse_args()

    process_all(args.frames_csv, args.results_dir)
