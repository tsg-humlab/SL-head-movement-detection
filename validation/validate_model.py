import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

def plot_confusion_matrix(matrix, title=None, labels = ['background', 'head-shake', 'head-nod']):
    """
    Plot a confusion matrix.
    """
    disp = ConfusionMatrixDisplay(matrix, display_labels=labels)
    disp.plot(cmap="Blues", values_format='.0f', colorbar=False)
    if title:
        plt.title(title, fontsize=12, y=1.1)
    plt.show()
    