from pathlib import Path
import numpy as np

import cv2
from moviepy.video.io.VideoFileClip import VideoFileClip


def get_n_frames(media_file):
    cap = cv2.VideoCapture(str(media_file))

    return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

def get_n_frames_from_keypoints(keypoints_file):
    keypoints = np.load(keypoints_file)
    return(keypoints.shape[0])

def get_fps(media_file):
    cap = cv2.VideoCapture(str(media_file))
    return int(cap.get(cv2.CAP_PROP_FPS))


def get_metadata(media_file):
    video = VideoFileClip(str(media_file))

    duration = video.duration
    fps = video.fps
    print(duration, fps)

    return duration, fps


def get_session_from_cngt_file(filepath):
    filepath = Path(filepath)
    filename = filepath.stem

    filename_split = filename.split('_')

    return filename_split[0]


def get_signer_from_cngt_file(filepath):
    filepath = Path(filepath)
    filename = filepath.stem

    filename_split = filename.split('_')

    return filename_split[1]
