from pathlib import Path

import pandas as pd

from utils.config import Config


def main():
    """Creates an overview file at the config location"""
    write_overview()


def add_entry(df, position, path):
    """Add a new row to the overview dataframe

    :param df: overview dataframe
    :param position: position of the speaker (left or right)
    :param path: path to the media file
    :return: dataframe with the new row appended
    """
    path = Path(path)
    filename = path.stem
    session_id, speaker_id = filename.split('_')[:2]

    return pd.concat(
        [df,
         pd.Series(
             {
                'session_id': session_id,
                'speaker_id': speaker_id,
                'position': position,
                'path': str(path)
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
    video_data = config.load_data_dump_labels()

    df_overview = pd.DataFrame(columns=['session_id', 'speaker_id', 'position', 'path'])

    for pair in video_data:
        if len(pair[3]) > 0:
            df_overview = add_entry(df_overview, 'left', pair[1])
        if len(pair[4]) > 0:
            df_overview = add_entry(df_overview, 'right', pair[2])

    df_overview.to_csv(output_path, index=False)


if __name__ == '__main__':
    main()
