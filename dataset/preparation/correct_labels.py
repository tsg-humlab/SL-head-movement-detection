import argparse
from pathlib import Path

from utils.frames_csv import load_df


def main(frames_csv, position_csv):
    df_frames = load_df(frames_csv)
    df_position = load_df(position_csv)

    for _, row in df_position.iterrows():
        if int(row['Reversed']) > 0:
            result = df_frames[df_frames['video_id'].str.contains(row['video_id'])]

            signer_to_labels = {}

            for _, result_row in result.iterrows():
                signer_to_labels[result_row['video_id'].split('_')[1]] = result_row['labels_path']

            if len(signer_to_labels) == 2:
                signers = list(signer_to_labels.keys())
                tmp = signer_to_labels[signers[0]]
                signer_to_labels[signers[0]] = signer_to_labels[signers[1]]
                signer_to_labels[signers[1]] = tmp

                for idx, result_row in result.iterrows():
                    df_frames.loc[idx, 'labels_path'] = signer_to_labels[result_row['video_id'].split('_')[1]]

                    pass
            else:
                for idx, result_row in result.iterrows():
                    df_frames.loc[idx, 'labels_path'] = 'SWAP TO OTHER SIGNER'

    df_frames.to_csv(frames_csv, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('frames_csv', type=Path)
    parser.add_argument('position_csv', type=Path)
    args = parser.parse_args()

    main(args.frames_csv, args.position_csv)