import argparse
from pathlib import Path

import pandas as pd

from utils.media import get_n_frames, get_fps


def create_csv(overview_csv, output_csv):
    """Create a default CSV that sets the frame count of every video from start to end without the license frames.

    :param overview_csv: Path to the overview CSV with media paths
    :param output_csv: Path to write the frames CSV to
    """
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
            get_n_frames(media_path) - 25
        ])

    df_frames = pd.DataFrame(data, columns=['video_id', 'media_path', 'start_frame', 'end_frame'])
    df_frames.to_csv(output_csv, index=False)


def split_video(frames_csv, video_id, timestamps):
    """Split a single video from a frames CSV into multiple videos based on timestamps.

    The timestamps should be given in pairs of two and be in ascending order.
    See the timestamp conversion function for accepted timestamp formats.

    :param frames_csv: CSV with the frame ranges, should include the video you want to split
    :param video_id: Unique video ID of the video you want to split
    :param timestamps: Timestamps you want to cut out of the video
    """
    df_frames = pd.read_csv(frames_csv)
    video_index = df_frames.index[df_frames['video_id'] == video_id].to_list()[0]
    video_row = df_frames.iloc[video_index]
    fps = get_fps(video_row['media_path'])

    timestamps = [timestamp_to_seconds(ts) * fps for ts in timestamps]
    assert(all(timestamps[i] < timestamps[i+1] for i in range(len(timestamps) - 1)))
    assert(timestamps[-1] < video_row['end_frame'])

    new_rows = []
    i = -2
    stop = video_row['start_frame']
    prev_start = stop

    for i in range(0, len(timestamps), 2):
        new_id = f'{video_id}_{int(i/2 + 1)}'
        start = timestamps[i]
        stop = timestamps[i+1]

        new_rows.append([new_id, video_row['media_path'], prev_start, start])
        prev_start = stop

    new_id = f'{video_id}_{int(i/2 + 2)}'
    new_rows.append([new_id, video_row['media_path'], stop, video_row['end_frame']])

    df_new = pd.DataFrame(new_rows, columns=df_frames.columns)
    df_frames_new = pd.concat([df_frames, df_new], ignore_index=True)
    df_frames_new = df_frames_new.drop(video_index)
    df_frames_new.to_csv(frames_csv, index=False)


def timestamp_to_seconds(timestamp):
    """Function to convert timestamps to seconds.

    A valid timestamp can be either on of these: HH:MM:SS/MM:SS/SS
    Single digits are also accepted.

    Credits to FMc: https://stackoverflow.com/questions/6402812/how-to-convert-an-hmmss-time-string-to-seconds-in-python

    :param timestamp: String with timestamp
    :return: Timestamp in seconds
    """
    return sum(int(x) * 60 ** i for i, x in enumerate(reversed(timestamp.split(':'))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')

    create_parser = subparsers.add_parser('create', help='Create a new csv with timeframes of the full videos')
    create_parser.add_argument('overview_csv', type=Path)
    create_parser.add_argument('output_csv', type=Path)

    split_parser = subparsers.add_parser('split', help='Split a single video in the timeframe csv into multiple videos')
    split_parser.add_argument('frame_csv', type=Path)
    split_parser.add_argument('video_id')
    split_parser.add_argument('timestamps', nargs='*')

    args = parser.parse_args()

    if args.command == 'create':
        create_csv(args.overview_csv, args.output_csv)
    elif args.command == 'split':
        split_video(args.frame_csv, args.video_id, args.timestamps)
