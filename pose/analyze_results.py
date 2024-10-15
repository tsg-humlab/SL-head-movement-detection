import argparse
import os
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from pose import KEYPOINTS, BOXES, KEYPOINT_VIDEO, POSE_VIDEO, HEADPOSES


def calculate_statistics(results_dir):
    assert is_output_dir(results_dir)

    print(f'Looking for results in {results_dir}')
    np_files = glob(str(results_dir / KEYPOINTS / '*.npy'))
    print(f'{len(np_files)} results found')

    keypoints = []
    n_people = np.zeros(len(np_files))

    for i in range(len(np_files)):
        array = np.load(np_files[i])
        keypoints.append(array)
        n_people[i] = array.shape[1]

    people_labels, people_counts = np.unique(n_people, return_counts=True)

    sns.set_theme()
    sns.set_style('whitegrid')
    sns.set_context('paper')
    plt.title('Max number of people found in any frame', fontsize=15)
    plt.pie(people_counts,
            labels=[int(label) for label in people_labels],
            autopct='%1.1f%%')
    plt.show()


def is_output_dir(path):
    if not os.path.exists(path):
        return False

    subdirs = [BOXES, KEYPOINTS, KEYPOINT_VIDEO]

    for subdir in subdirs:
        if not os.path.exists(os.path.join(path, subdir)):
            return False

    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('results_dir', metavar='results-dir', type=Path)
    args = parser.parse_args()

    calculate_statistics(args.results_dir)
