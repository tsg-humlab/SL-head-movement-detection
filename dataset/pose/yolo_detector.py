import os
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO
from moviepy.editor import VideoFileClip

from utils.config import Config
import pandas as pd


def process_dataset(output_dir, log_file='log.txt'):
    config = Config()
    overview_path = config.content['overview']
    df_overview = pd.read_csv(overview_path)

    output_dir = Path(output_dir)
    if not os.path.exists(output_dir / 'video'):
        os.mkdir(output_dir / 'video')
    if not os.path.exists(output_dir / 'boxes'):
        os.mkdir(output_dir / 'boxes')
    if not os.path.exists(output_dir / 'keypoints'):
        os.mkdir(output_dir / 'keypoints')

    log_handle = open(log_file, 'w')

    for _, row in df_overview.iterrows():
        unique_id = f"{row['session_id']}_{row['speaker_id']}"

        # noinspection PyBroadException
        try:
            process_video(unique_id, row['media_path'], output_dir)
            log_handle.write(f'Successfully processed {unique_id}')
        except Exception as err:
            log_handle.write(f'Error processing {unique_id}\n')
            log_handle.write(str(err) + '\n')

    log_handle.close()


def process_video(unique_id, video_path, output_dir):
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
    process_dataset(r"E:\Results\pose_detection")
    # process_video('CNGT1590_S068', r"E:\CorpusNGT\CNGT_720p\CNGT1590_S068_b_720.mp4", Path("E:\Results\pose_detection"))