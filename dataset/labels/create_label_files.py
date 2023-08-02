import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from pympi.Elan import Eaf

from dataset.labels.eaf_parser import Video
from utils.media import get_n_frames


def main(eaf_dir, output_dir, frames_path, position_csv=None):
    df_overview = pd.read_csv(frames_path)

    if position_csv:
        df_position = pd.read_csv(position_csv, index_col='video_id')

    paths = set(df_overview['media_path'])

    for path in paths:
        path = Path(path)
        ngt_id = path.stem.split('_')[0]
        signer_id = path.stem.split('_')[1]

        eaf = Eaf(eaf_dir / f'{ngt_id}.eaf', 'pympi')
        video = Video.from_eaf(ngt_id, eaf)
        annotations = video[signer_id].annotations

        assert len(annotations) > 0

        labels = np.zeros(get_n_frames(path))

        for annotation in annotations:
            label = annotation.label.lower()

            if label in ['n', 'nf', 'nx', 'nn']:
                labels[annotation.start:annotation.end] = 1

        if np.max(labels) != 1:
            df_overview.drop(df_overview[df_overview['media_path'] == str(path)].index, inplace=True)
        else:
            np.save(output_dir / f'{path.stem}.npy', labels)

    def transform_path(filepath, directory=output_dir, suffix='.npy'):
        return directory / Path(filepath).with_suffix(suffix).name

    df_overview['labels_path'] = df_overview['media_path'].apply(transform_path)
    df_overview.to_csv(frames_path, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('eaf_dir', metavar='eaf-dir', type=Path)
    parser.add_argument('output_dir', metavar='output-dir', type=Path)
    parser.add_argument('frames_path', metavar='frames-path', type=Path)
    parser.add_argument('-p', '--position_csv', type=Path)
    args = parser.parse_args()

    main(args.eaf_dir, args.output_dir, args.frames_path, position_csv=args.position_csv)
