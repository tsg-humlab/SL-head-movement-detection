import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from moviepy.editor import VideoFileClip
from ultralytics import YOLO

from utils.config import Config


def process_dataset(output_dir, overview_path=None, log_file='log.txt'):
    if overview_path is None:
        config = Config()
        overview_path = config.content['overview']
    df_overview = pd.read_csv(overview_path)

    log_handle = open(log_file, 'w')

    for _, row in df_overview.iterrows():
        unique_id = f"{row['session_id']}_{row['speaker_id']}"

        # noinspection PyBroadException
        try:
            process_video(unique_id, row['media_path'], output_dir)
            log_handle.write(f'Successfully processed {unique_id}\n')
        except Exception as err:
            log_handle.write(f'Error processing {unique_id}\n')
            log_handle.write(str(err) + '\n')

    log_handle.close()


def create_subdirs(path):
    path = Path(path)
    if not os.path.exists(path / 'video'):
        os.mkdir(path / 'video')
    if not os.path.exists(path / 'boxes'):
        os.mkdir(path / 'boxes')
    if not os.path.exists(path / 'keypoints'):
        os.mkdir(path / 'keypoints')


def process_video(unique_id, video_path, output_dir):
    create_subdirs(output_dir)
    model = YOLO('yolov8n-pose.pt')
    video = VideoFileClip(video_path)

    duration = video.duration
    fps = round(video.fps)
    n_frames = round(duration * fps)

    capture = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_file = output_dir / 'video' / f'{unique_id}.mp4'
    output_video = cv2.VideoWriter(str(output_file), fourcc, fps, (int(capture.get(3)), int(capture.get(4))))

    idx = 1

    boxes_list = []
    keypoints_list = []

    while capture.isOpened():
        ret, frame = capture.read()

        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model(frame_rgb)

            boxes = results[0].boxes.data.numpy()
            keypoints = results[0].keypoints.data.numpy()
            n_results = boxes.shape[0]

            if n_results == 0:
                boxes_list.append(np.zeros((1, 6)))
                keypoints_list.append(np.zeros((1, 17, 3)))
            else:
                boxes_list.append(boxes)
                keypoints_list.append(keypoints)

            output_frame = results[0].plot()
            output_frame = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)
            output_video.write(output_frame)

            idx += 1
        else:
            break

    capture.release()
    output_video.release()

    boxes = stack_with_padding(boxes_list)
    keypoints = stack_with_padding(keypoints_list)

    np.save(output_dir / 'boxes' / f'{unique_id}.npy', boxes)
    np.save(output_dir / 'keypoints' / f'{unique_id}.npy', keypoints)

    assert n_frames == boxes.shape[0]
    assert n_frames == keypoints.shape[0]


def stack_with_padding(array_list, variable_dim=0):
    padded_list = []
    shape = list(array_list[0].shape)

    sizes = [array.shape[variable_dim] for array in array_list]
    max_size = max(sizes)

    for i in range(len(array_list)):
        padding_shape = shape
        padding_shape[0] = max_size - array_list[i].shape[0]
        padded_list.append(np.concatenate([array_list[i], np.zeros(padding_shape)]))

    return np.stack(padded_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output-dir', type=Path, required=True)
    parser.add_argument('-v', '--overview-path', type=Path)
    parser.add_argument('-l', '--log-file', type=Path, default='log.txt')
    args = parser.parse_args()

    process_dataset(output_dir=args.output_dir,
                    overview_path=args.overview_path,
                    log_file=args.log_file)
