import abc

from models.processing.facial_movement import ordinal_from_csv


class ShakeNodDetector:
    def __init__(self, window_size, movement_threshold):
        self.movement_threshold = movement_threshold

        self.window_size = verify_window_size(window_size)

        self.data = None

    def __load_data(self, df_data, threshold):
        self.data = ordinal_from_csv(df_data, threshold=threshold)

    def fit(self, *args, **kwargs):
        """Detectors that don't need to fit any parameters to the training data can keep this method as-is. This way
        non-learning types of detectors can still go through the same code pipeline as models that have to be fit.
        """
        return

    @abc.abstractmethod
    def predict(self, sequence):
        """All shake detectors should have a predict method that takes a numpy array sequence as input, and an output
        sequence of the same length with frame level annotations for head-shakes.

        :param sequence: Numpy array of movements (see facial_movement.py)
        :return: Numpy array of zeros with ones for shake annotations
        """
        return


def verify_window_size(size):
    size = abs(int(size))
    if size % 2 == 0:
        size += 1

    return max(size, 3)
