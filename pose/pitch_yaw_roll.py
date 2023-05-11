import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pose.review_conflicts import show_frame
from utils.media import get_session_from_cngt_file, get_signer_from_cngt_file


def main(frames_csv, keypoints_dir):
    df_frames = pd.read_csv(frames_csv)

    for _, row in df_frames.iterrows():
        media_path = Path(row['media_path'])
        session_id = get_session_from_cngt_file(media_path)
        signer_id = get_signer_from_cngt_file(media_path)

        keypoints_path = keypoints_dir / f'{session_id}_{signer_id}.npy'
        keypoints_arr = np.load(keypoints_path)

        process_array(keypoints_arr[row['start_frame']:row['end_frame']])

        pass

    pass


def process_array(arr):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('frames_csv', type=Path)
    parser.add_argument('keypoints_dir', type=Path)
    args = parser.parse_args()

    main(args.frames_csv, args.keypoints_dir)
