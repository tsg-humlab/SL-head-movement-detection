import abc

from models.hmm.facial_movement import ordinal_from_csv


class ShakeDetector:
    def __init__(self, window_size, movement_threshold):
        self.movement_threshold = movement_threshold

        window_size = abs(int(window_size))
        if window_size % 2 == 0:
            window_size += 1
        self.window_size = max(window_size, 3)

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
