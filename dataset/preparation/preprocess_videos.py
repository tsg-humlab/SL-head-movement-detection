import os
from pathlib import Path

from cv2 import cv2
from glob import glob
import numpy as np

from utils.file_processing import ambiguous_to_numpy


def cut_license(video: str | Path | np.ndarray, visual=False, window_name="Video"):
    """Cuts the creative commons license out of a videos. The technique is simple, it takes the upper left pixel from
    the first frame of the video and removes every frame where the upper left pixel is exactly that colour value.

    Ofcourse this is only useful because we know for sure that all videos in our dataset have a dark background that
    won't match this colour, don't use this function on videos where you can't guarantee this.

    You can optionally inspect the process visually.

    :param video: Path to the CNGT file or the contents in numpy format.
    :param visual: True if you want to inspect the video before and after cutting
    :param window_name: Name of the inspection window
    :return: Numpy array with uncut video
    """
    video = ambiguous_to_numpy(video)

    license_background = video[0, 0, 0]
    license_frames = []

    for i in range(video.shape[0]):
        if visual:
            cv2.imshow(window_name, cv2.cvtColor(video[i], cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

        if np.array_equal(video[i, 0, 0], license_background):
            license_frames.append(i)

    video = np.delete(video, license_frames, axis=0)

    if visual:
        for i in range(video.shape[0]):
            cv2.imshow(window_name, cv2.cvtColor(video[i], cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

        cv2.destroyWindow(window_name)

    return video


def sample_frame_measurements(media_dir, n_videos=10):
    files = glob(f'{media_dir}{os.sep}*.mp4')
    selected = np.random.choice(files, size=n_videos, replace=False)

    start_frames = None
    end_frames = None

    for video_path in selected:
        result = measure_license_frames(video_path)

        if start_frames is None:
            start_frames = result[0]
            end_frames = result[1]
        else:
            if start_frames != result[0] or end_frames != result[1]:
                return False

    return True


def measure_license_frames(video):
    video = ambiguous_to_numpy(video)

    license_background = video[0, 0, 0]
    start_frames = 0

    for i in range(video.shape[0]):
        if np.array_equal(video[i, 0, 0], license_background):
            start_frames += 1
        else:
            break

    license_background = video[-1, 0, 0]
    end_frames = 0

    for i in range(video.shape[0]-1, -1, -1):
        if np.array_equal(video[i, 0, 0], license_background):
            end_frames += 1
        else:
            break

    return start_frames, end_frames


def main():
    # result = sample_frame_measurements(r'E:\CorpusNGT\CNGT_720p')
    result = measure_license_frames(r"E:\CorpusNGT\CNGT_720p\CNGT1901_S078_b_720.mp4")

    pass


if __name__ == '__main__':
    main()
