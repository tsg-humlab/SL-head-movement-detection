from collections import Counter
import numpy as np


def majority_filter(seq, width):
    """Filter that sets every element to the majority of their neighborhood, effectively closing gaps

    Credit to PM 2Ring
    https://stackoverflow.com/questions/39829716/majority-filter-python

    :param seq: Arbitrary sequence
    :param width: Width of the filter
    :return: Filtered sequence
    """
    offset = width // 2
    seq = [0] * offset + list(seq)

    return np.array([Counter(a).most_common(1)[0][0] for a in (seq[i:i + width] for i in range(len(seq) - offset))])
