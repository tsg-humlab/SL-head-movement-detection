import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

from pose import KEYPOINTS, BOXES, VIDEO
from utils.array_manipulation import stack_with_padding
from utils.config import Config
from utils.media import get_metadata


def process_dataset(output_dir, overview_path=None, log_file='log.txt', model_name='yolov8n-pose.pt'):
    """Make pose predictions over an entire dataset using YOLO.

    The overview file can either be provided explicitly or it will be read out of the config.yml file.

    :param output_dir: Directory where the predictions should be stored
    :param overview_path: Path to the overview CSV
    :param log_file: Log file with success/error statements for every video in the overview (will be overwritten!)
    :param model_name: Name of the YOLO checkpoint that should be used (default nano)
    """
    if overview_path is None:
        config = Config()
        overview_path = config.content['overview']
    df_overview = pd.read_csv(overview_path)

    create_subdirs(output_dir)
    log_handle = open(log_file, 'w')

    for _, row in df_overview.iterrows():
        unique_id = f"{row['session_id']}_{row['speaker_id']}"

        # noinspection PyBroadException
        try:
            process_video(unique_id, row['media_path'], output_dir, model_name=model_name)
            log_handle.write(f'Successfully processed {unique_id}\n')
        except Exception as err:
            log_handle.write(f'Error processing {unique_id}\n')
            log_handle.write(str(err) + '\n')

    log_handle.close()


def create_subdirs(path):
    """Create (sub)directories for the prediction outputs.

    If any directories already exists, they will be ignored without raising an exception.

    :param path: Path to the output directory
    """
    path = Path(path)

    if not os.path.exists(path):
        os.mkdir(path)
    if not os.path.exists(path / VIDEO):
        os.mkdir(path / VIDEO)
    if not os.path.exists(path / BOXES):
        os.mkdir(path / BOXES)
    if not os.path.exists(path / KEYPOINTS):
        os.mkdir(path / KEYPOINTS)


def process_video(unique_id, video_path, output_dir, model_name='yolov8n-pose.pt'):
    """Make pose predictions on a single video.

    I recommend using a GPU for this process, although the nano model can still be run in a reasonable time window
    through only a CPU.

    :param unique_id: Unique identifier for the video (results with the same ID will be overwritten!)
    :param video_path: Path to the video file
    :param output_dir: Directory to store the results in (subdirs will be created if necessary)
    :param model_name: Name of the YOLO checkpoint that should be used (default nano)
    """
    create_subdirs(output_dir)
    model = YOLO(model_name)

    duration, fps = get_metadata(video_path)
    n_frames = round(duration * fps)

    capture = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_file = output_dir / VIDEO / f'{unique_id}.mp4'
    output_video = cv2.VideoWriter(str(output_file), fourcc, fps, (int(capture.get(3)), int(capture.get(4))))

    idx = 1

    boxes_list = []
    keypoints_list = []

    while capture.isOpened():
        ret, frame = capture.read()

        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model(frame_rgb)

            boxes = results[0].boxes.data.cpu().numpy()
            keypoints = results[0].keypoints.data.cpu().numpy()
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

    np.save(output_dir / BOXES / f'{unique_id}.npy', boxes)
    np.save(output_dir / KEYPOINTS / f'{unique_id}.npy', keypoints)

    assert n_frames == boxes.shape[0]
    assert n_frames == keypoints.shape[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output-dir', type=Path, required=True)
    parser.add_argument('-v', '--overview-path', type=Path)
    parser.add_argument('-l', '--log-file', type=Path, default='log.txt')
    args = parser.parse_args()

    process_dataset(output_dir=args.output_dir,
                    overview_path=args.overview_path,
                    log_file=args.log_file)
