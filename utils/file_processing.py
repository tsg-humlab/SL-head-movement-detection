from pathlib import Path

import numpy as np
from skvideo.io import vread


def ambiguous_to_numpy(ambiguous):
    """Helper function for converting an ambiguous argument to a numpy video.

    Will read the video if a valid path was given. Will return the argument if it is already a video. Throws an error
    in all other cases.

    :param ambiguous: Ambiguous variable that should relate to a video file or it's contents
    :return: video: The video in numpy format
    """
    if type(ambiguous) == str or isinstance(ambiguous, Path):
        path = str(ambiguous)

        video = vread(path)
    elif type(ambiguous) is np.ndarray:
        assert(len(ambiguous.shape) == 4)
        assert(ambiguous.shape[-1] == 3)

        video = ambiguous
    else:
        raise ValueError("Couldn't recognise ambiguous argument as a video")

    return video


if __name__ == '__main__':
    ambiguous_to_numpy(vread(r"E:\CorpusNGT\CNGT 720p\CNGT1752_S071_b_720.mp4"))
