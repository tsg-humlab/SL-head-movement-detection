import numpy as np

from models.simple.detector import ShakeDetector


class RandomShakeDetector(ShakeDetector):
    def __init__(self, window_size, movement_threshold, seed=None):
        super().__init__(window_size, movement_threshold)

        if seed:
            self.random = np.random.default_rng(seed)
        else:
            self.random = np.random.default_rng()

    def predict(self, sequence):
        pred_len = len(sequence) - self.window_size + 1

        return self.random.integers(0, 2, size=pred_len)
