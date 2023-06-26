import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from models.hmm import HMM_DECODER
from models.hmm.facial_movement import compute_hmm_vectors


class RuleBasedShakeDetector:
    def __init__(self, window_size, movement_threshold):
        self.movement_threshold = movement_threshold

        window_size = abs(int(window_size))
        if window_size % 2 == 0:
            window_size += 1
        self.window_size = max(window_size, 3)

        self.shakes_counts = {}
        self.background_counts = {}
        self.data = None

    def __load_data(self, df_data, threshold):
        self.data = compute_hmm_vectors(df_data, threshold=threshold)

    def fit(self, df_data, data=None):
        if data:
            self.data = data
        else:
            self.__load_data(df_data, self.movement_threshold)

        shake_list = []
        background_list = []
        i = 0

        for _, row in df_data.iterrows():
            labels = np.load(row['labels_path'])[row['start_frame'] + 1:row['end_frame']]

            shake_list.append(self.data[i][labels > 0])
            background_list.append(self.data[i][labels == 0])

            i += 1

        shakes = np.concatenate(shake_list)
        background = np.concatenate(background_list)

        self.shakes_counts = {}
        values, counts = np.unique(shakes, return_counts=True)
        for i in range(len(values)):
            self.shakes_counts[HMM_DECODER[values[i]]] = counts[i] / len(shakes)

        self.background_counts = {}
        values, counts = np.unique(background, return_counts=True)
        for i in range(len(values)):
            self.background_counts[HMM_DECODER[values[i]]] = counts[i] / len(background)

    def predict(self, sequence):
        pred_len = len(sequence) - self.window_size + 1
        result = np.zeros(pred_len)

        for i in range(pred_len):
            window = sequence[i:i + self.window_size]
            values, counts = np.unique(window, return_counts=True)

            shake_diff = 0
            bg_diff = 0

            for j in range(len(values)):
                label = HMM_DECODER[values[j]]
                ratio = counts[j] / self.window_size

                shake_diff += abs(self.shakes_counts[label] - ratio)
                bg_diff += abs(self.background_counts[label] - ratio)

            if shake_diff < bg_diff:
                result[i] = 1

        return result

    def plot_hmm_distributions(self):
        sns.set_theme()
        sns.set_style('whitegrid')
        sns.set_context('paper')

        plt.title(f'Distribution of sequences with alpha={self.movement_threshold}', fontsize=15)
        plt.ylabel('% of sequence category')
        plt.bar(
            [i for i in range(len(self.shakes_counts))],
            list(self.shakes_counts.values()),
            width=0.25,
            label='shake',
            align='center')
        plt.bar(
            [i + 0.25 for i in range(len(self.background_counts))],
            list(self.background_counts.values()),
            width=0.25,
            label='background',
            align='center')
        plt.xticks([i for i in range(len(self.background_counts))], list(self.background_counts.keys()))
        plt.legend()
        plt.show()
