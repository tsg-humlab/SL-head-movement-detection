import os

import pandas as pd

from dataset.labels import eaf_parser
from utils.config import Config


def main():
    """Creates an overview file at the config location"""
    write_overview()


def add_entry(df, session_id, speaker_id, position, path):
    """Add a new row to the overview dataframe

    :param df: overview dataframe
    :param session_id: CNGT id of the session
    :param speaker_id: Unique ID of the speaker
    :param position: position of the speaker (left or right)
    :param path: path to the media file
    :return: dataframe with the new row appended
    """

    return pd.concat(
        [df,
         pd.Series(
             {
                'session_id': session_id,
                'speaker_id': speaker_id,
                'position': position,
                'media_path': path
             }
         ).to_frame().T
         ], ignore_index=True
    )


def write_overview(output_path=None):
    """Load the dumped data labels and write an overview file.

    :param output_path: path to the
    """
    config = Config()
    if output_path is None:
        output_path = config.content['overview']
    videos = eaf_parser.main()

    print(f'Found {len(videos)} videos')

    df_overview = pd.DataFrame(columns=['session_id', 'speaker_id', 'position', 'media_path'])

    for video in videos:
        if len(video.signer_left.annotations) > 0:
            filepath = f'{config.content["media"]["body_720"]}{os.sep}{video.ngt_id}_{video.signer_left.signer_id}_b_720.mp4'
            assert(os.path.exists(filepath))
            df_overview = add_entry(df_overview, video.ngt_id, video.signer_left.signer_id, 'left', filepath)
        if len(video.signer_right.annotations) > 0:
            filepath = f'{config.content["media"]["body_720"]}{os.sep}{video.ngt_id}_{video.signer_right.signer_id}_b_720.mp4'
            assert(os.path.exists(filepath))
            df_overview = add_entry(df_overview, video.ngt_id, video.signer_right.signer_id, 'right', filepath)

    print(f'Writing results to {output_path}')
    df_overview.to_csv(output_path, index=False)
    print('Complete!')


if __name__ == '__main__':
    main()
