from pathlib import Path

import pandas as pd


class Overview:
    def __init__(self, overview):
        if type(overview) == str:
            self.overview = pd.read_csv(Path(overview))
        elif type(overview) == pd.DataFrame:
            self.overview = overview
        else:
            raise TypeError(f'Should be dataframe like object, received {type(overview)}')

    def __getitem__(self, args):
        if len(args) == 2:
            ngt_id, speaker_id = args
        else:
            raise ValueError(f'Invalid number of arguments: {len(args)} (should be 2)')

        return Path(self.overview.loc[
            (self.overview['session_id'] == ngt_id) & (self.overview['speaker_id'] == speaker_id),
            'media_path'
        ].iloc[0])
