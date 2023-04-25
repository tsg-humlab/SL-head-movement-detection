import os
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO
from moviepy.editor import VideoFileClip

from utils.config import Config
import pandas as pd


def process_dataset(output_dir):
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

    for _, row in df_overview.iterrows():
        unique_id = f"{row['session_id']}_{row['speaker_id']}"
        process_video(unique_id, row['media_path'], output_dir)

        break


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

    prev_nose_x = 0
    prev_nose_y = 0

    while capture.isOpened():
        ret, frame = capture.read()

        if ret:
            # TODO: remove
            if idx < 510:
                idx += 1
                continue
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model(frame_rgb)

            boxes = results[0].boxes.data.numpy()
            keypoints = results[0].keypoints.data.numpy()
            n_results = boxes.shape[0]

            if n_results == 0:
                boxes_list.append(np.zeros((1, 6)))
                keypoints_list.append(np.zeros((1, 17, 3)))

                # output_video.write(frame)
                # TODO: remove
                output_frame = results[0].plot()
                output_frame = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)
                cv2.imshow('no detections', output_frame)
                key = cv2.waitKey(0)
                while key not in [ord('q'), ord('k')]:
                    key = cv2.waitKey(0)
            else:
                if n_results == 1:
                    result_id = 0
                else:
                    cur_nose_x = keypoints[:, 0, 0]
                    cur_nose_y = keypoints[:, 0, 1]
                    diff_x = cur_nose_x - prev_nose_x
                    diff_y = cur_nose_y - prev_nose_y
                    diff = np.abs(diff_x + diff_y)
                    result_id = np.argmin(diff)

                    output_frame = results[0].plot()
                    output_frame = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)
                    cv2.imshow('2 detections', output_frame)
                    # TODO: remove
                    key = cv2.waitKey(0)
                    while key not in [ord('q'), ord('k')]:
                        key = cv2.waitKey(0)

                boxes_list.append(boxes[result_id, :])
                keypoints_list.append(keypoints[result_id, :, :])

                prev_nose_x = keypoints[result_id, 0, 0]
                prev_nose_y = keypoints[result_id, 0, 1]

            output_frame = results[0].plot()
            output_frame = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)
            output_video.write(output_frame)
            idx += 1
        else:
            break

    capture.release()
    output_video.release()

    boxes = np.stack(boxes_list)
    keypoints = np.stack(keypoints_list)

    np.save(output_dir / 'boxes' / f'{unique_id}.npy', boxes)
    np.save(output_dir / 'keypoints' / f'{unique_id}.npy', keypoints)

    assert n_frames == boxes.shape[0]
    assert n_frames == keypoints.shape[0]


if __name__ == '__main__':
    # process_dataset(r"E:\Results\pose_detection")
    # main(r"E:\CorpusNGT\CNGT_720p\CNGT2103_S084_b_720.mp4")
    # main(r"E:\CorpusNGT\CNGT_720p\CNGT0002_S004_b_720.mp4")
    process_video('CNGT1590_S068', r"E:\CorpusNGT\CNGT_720p\CNGT1590_S068_b_720.mp4", Path("E:\Results\pose_detection"))
