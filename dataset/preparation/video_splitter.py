import os
import cv2
from pathlib import Path
from typing import Optional

import pandas as pd
import argparse

from pose.review_conflicts import show_frame
from utils.media import get_metadata, get_n_frames


def create_csv(overview_csv, output_csv):
    data = []
    df_overview = pd.read_csv(overview_csv)

    for _, row in df_overview.iterrows():
        media_path = Path(row['media_path'])
        filename = media_path.stem
        filesplit = filename.split('_')
        unique_id = f'{filesplit[0]}_{filesplit[1]}'

        data.append([
            unique_id,
            media_path,
            75,
            get_n_frames(media_path)
        ])

    df_frames = pd.DataFrame(data, columns=['video_id', 'media_path', 'start_frame', 'end_frame'])
    df_frames.to_csv(output_csv, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('overview_csv')
    parser.add_argument('output_csv', type=Path)
    args = parser.parse_args()

    create_csv(args.overview_csv, args.output_csv)
