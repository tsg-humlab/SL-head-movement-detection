import os

import pandas as pd

from dataset.labels import eaf_parser
from utils.config import Config


def add_entry(df, video_id, speaker_id, path):
    """Add a new row to the overview dataframe

    :param df: overview dataframe
    :param video_id: Unique ID of the video
    :param speaker_id: Unique ID of the speaker
    :param path: path to the media file
    :return: dataframe with the new row appended
    """

    return pd.concat(
        [df,
         pd.Series(
             {
                'video_id': video_id+"_"+speaker_id,
                'media_path': path
             }
         ).to_frame().T
         ], ignore_index=True
    )


def main(output_path=None):
    """Load the dumped data labels and write an overview file.
    
    :param output_path: path to the output file
    """
    config = Config()
    if output_path is None:
        output_path = config.content['overview']
    videos = eaf_parser.main()

    print(f'Found {len(videos)} videos')

    df_overview = pd.DataFrame(columns=['video_id', 'media_path'])

    for video in videos:
        if len(video.signer.annotations) > 0:
            filepath = f'{config.content["media"]["video"]}{os.sep}{video.ngt_id}_{video.signer.signer_id}.mp4'
            if 'None' not in filepath:
                df_overview = add_entry(df_overview, video.ngt_id, video.signer.signer_id, filepath)

    print(f'Writing results to {output_path}')
    df_overview.to_csv(output_path, index=False)
    print('Complete!')


if __name__ == '__main__':
    main()
