import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from models.hmm import HMM_DECODER
from models.simple.detector import ShakeNodDetector


class MemoryBasedShakeNodDetector(ShakeNodDetector):
    def __init__(self, window_size, movement_threshold):
        super().__init__(window_size, movement_threshold)

        self.shakes_counts, self.nods_counts, self.background_counts = {}

    def __load_data(self, df_data, threshold):
        self.data = ordinal_from_csv(df_data, threshold=threshold)
        
    def fit(self, df_data, data=None):
        if data:
            self.data = data
        else:
            self.__load_data(df_data, self.movement_threshold)

        shake_list, nod_list, background_list = [], [], []
        i = 0

        for _, row in df_data.iterrows():
            labels = np.load(row['labels_path'])[row['start_frame'] + 1:row['end_frame']]

            shake_list.append(self.data[i][labels == 1])
            nod_list.append(self.data[i][labels == 2])
            background_list.append(self.data[i][labels == 0])

            i += 1

        shakes = np.concatenate(shake_list)
        nods = np.concatenate(nod_list)
        background = np.concatenate(background_list)

        self.shakes_counts, self.nods_counts, self.background_counts = {}, {}, {}
        values, counts = np.unique(shakes, return_counts=True)
        for i in range(len(values)):
            self.shakes_counts[HMM_DECODER[values[i]]] = counts[i] / len(shakes)

        values, counts = np.unique(nods, return_counts=True)
        for i in range(len(values)):
            self.nods_counts[HMM_DECODER[values[i]]] = counts[i] / len(nods)

        values, counts = np.unique(background, return_counts=True)
        for i in range(len(values)):
            self.background_counts[HMM_DECODER[values[i]]] = counts[i] / len(background)

    def predict(self, sequence):
        """
        Predicts the labels for a given sequence
        """
        pred_len = len(sequence) - self.window_size + 1
        result = np.zeros(pred_len)

        for i in range(pred_len):
            window = sequence[i:i + self.window_size]
            values, counts = np.unique(window, return_counts=True)

            shake_diff, nod_diff, bg_diff = 0, 0, 0

            for j in range(len(values)):
                # e.g. label: [none,   up,    down,   left,  right]
                # e.g. ratio: [9/10,   8/10,  9/10,   3/10,  2/10]
                label = HMM_DECODER[values[j]]
                ratio = counts[j] / self.window_size

                # e.g. shake_diff = abs(8/10 - 9/10) = 1/10
                shake_diff += abs(self.shakes_counts[label] - ratio)
                nod_diff += abs(self.nods_counts[label] - ratio)
                bg_diff += abs(self.background_counts[label] - ratio)

            if shake_diff < bg_diff and shake_diff < nod_diff:
                result[i] = 1
            elif nod_diff < bg_diff:
                result[i] = 2

        return result

    def plot_memory_distributions(self):
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
