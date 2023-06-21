import shutil

import pandas as pd
from tqdm import tqdm

from utils.config import Config


def copy_subset(output_dir):
    config = Config()
    overview_df = pd.read_csv(config.overview)

    print(f'Copying {len(overview_df)} files to {output_dir}')

    for i in tqdm(range(len(overview_df))):
        shutil.copy2(overview_df['media_path'].values[i], output_dir)


if __name__ == '__main__':
    copy_subset(r'E:\CorpusNGT\CNGT_720p_annotated')
