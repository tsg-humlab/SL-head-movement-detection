import numpy as np


def stack_with_padding(array_list, variable_dim=0):
    """Stack the predictions made on a video with zero padding.

    This means that every time step in the video will have max(n_predictions) - n_predictions[time] added to the
    existing predictions.

    The dimensions of the input arrays have to match except for the variable dimension, which will be padded.

    :param array_list: List of numpy arrays with predictions
    :param variable_dim: The dimension that should be padded
    :return: Stacked and padded array created from the input arrays
    """
    padded_list = []
    shape = list(array_list[0].shape)

    sizes = [array.shape[variable_dim] for array in array_list]
    max_size = max(sizes)

    for i in range(len(array_list)):
        padding_shape = shape
        padding_shape[0] = max_size - array_list[i].shape[0]
        padded_list.append(np.concatenate([array_list[i], np.zeros(padding_shape)]))

    return np.stack(padded_list)
