from pathlib import Path

import cv2
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


import time


def main():
    path_full = r"E:\CorpusNGT\CNGT_numpy\CNGT0003_S004.npy"
    path_compressed = r"E:\CorpusNGT\CNGT_numpy\CNGT0003_S004.npz"

    video = cut_license(r"E:\CorpusNGT\CNGT 720p\CNGT0003_S004_b_720.mp4")

    start = time.time()
    # np.save(path_full, video)
    end = time.time()
    save_full = (end - start)

    start = time.time()
    # np.savez_compressed(path_compressed, video)
    end = time.time()
    save_compressed = (end - start)

    start = time.time()
    video = np.load(path_full)
    end = time.time()
    load_full = (end - start)

    start = time.time()
    video = np.load(path_compressed)['arr_0']
    end = time.time()
    load_compressed = (end - start)

    normalized_video = video / 255.0

    ...


if __name__ == '__main__':
    main()
