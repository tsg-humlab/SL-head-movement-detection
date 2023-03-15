from pathlib import Path

import cv2
import numpy as np

from utils.file_processing import ambiguous_to_numpy


def cut_license(video: str | Path | np.ndarray, visual=False, window_name="Video"):
    """Cuts the creative commons license out of a videos.

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

        cv2.destroyWindow("Video")

    return video
