import numpy as np

from models.simple.detector import ShakeNodDetector


class RuleBasedShakeDetector(ShakeNodDetector):
    def __init__(self, window_size, movement_threshold, rule_func):
        super().__init__(window_size, movement_threshold)

        self.rule = rule_func

    def majority_vote(sequence):
        return np.count_nonzero(sequence == 0) > (len(sequence) / 2)
    
    def predict(self, sequence):
        pred_len = len(sequence) - self.window_size + 1
        result = np.zeros(pred_len)

        for i in range(pred_len):
            window = sequence[i:i + self.window_size]

            if self.rule(window):
                result[i] = 1

        return result



