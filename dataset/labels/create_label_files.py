import argparse
from pathlib import Path

import pandas as pd
from pympi.Elan import Eaf

from dataset.labels.eaf_parser import Video


def main(eaf_dir, output_dir, frames_path):
    df_overview = pd.read_csv(frames_path)

    ids = df_overview['video_id'].str.split('_')

    for ngt_id, signer_id in ids:
        eaf = Eaf(eaf_dir / f'{ngt_id}.eaf', 'pympi')
        video = Video.from_eaf(ngt_id, eaf)
        annotations = video[signer_id].annotations

        assert len(annotations) > 0

        pass

    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('eaf_dir', metavar='eaf-dir', type=Path)
    parser.add_argument('output_dir', metavar='output-dir', type=Path)
    parser.add_argument('frames_path', metavar='frames-path', type=Path)
    args = parser.parse_args()

    main(args.eaf_dir, args.output_dir, args.frames_path)
