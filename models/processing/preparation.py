from utils.array_manipulation import get_change_indices
import numpy as np
import torch
import math


def remove_nan_values_from_tensors(tensor_list):
    """
    Removes any tensors that contain NaN values
    Find another solution to this, this is a hacky fix
    Nan values are caused by the derivative function
    """
    tensor_list_new = []
    for tensor_i, tensor in enumerate(tensor_list):
        if torch.isnan(tensor).any():
            print(f'Found NaN in tensor_list[{tensor_i}]')
            continue
        else:
            tensor_list_new.append(tensor)
    return tensor_list_new


def smoothen_curve(array, window):
    """
    Window needs to be uneven
    """
    import math
    window_half = math.floor(window/2) 
    smoothened_values = []

    for i in range(len(array)):
        
        if i < window_half or i >= len(array)-window_half:
            smoothened_values.append(array[i])
            continue

        average = sum(array[i-window_half:i+window_half+1]) / window
        smoothened_values.append(average)

    return smoothened_values


def add_diff(list):
    """
    Makes the list values cumulative, starting at 0
    """
    list = np.diff(list)
    added_list = []
    added_list.append(0)
    for l in list:
        added_list.append(added_list[-1]+l)
    return added_list


def add_diffs_list(list):
    """
    Do add_diff for each lettered list [[a,b,c], [a,b,c], ...], so for each column
    """
    for i in range(len(list[0])):
        list[:,i] = add_diff(list[:,i])
    return list


def separate_seqs(sequence_list, labels_list, MINIMUM_SEQUENCE_LENGTH=2):
    """
    Separates the sequences into background, shake, and nod sequences
    Based on the exact changes in labels
    """
    assert len(sequence_list) == len(labels_list)
    background_seqs, shake_seqs, nod_seqs = [], [], []

    for seq_index in range(len(sequence_list)):
        assert len(sequence_list[seq_index]) == len(labels_list[seq_index])

        # find the indices where the labels change
        change_indices = np.where(labels_list[seq_index][1:] != labels_list[seq_index][:-1])[0]+1
        change_indices = np.insert(change_indices, 0, 1)
        if change_indices[-1] != len(labels_list[seq_index]):
            change_indices = np.append(change_indices, len(labels_list[seq_index])-1)
        label_indices = np.array(labels_list[seq_index])[change_indices]

        # if a label repeats for more than MINIMUM_SEQUENCE_LENGTH frames, add it to the corresponding list
        for change_index in range(len(change_indices) - 1):
            if (change_indices[change_index + 1] - change_indices[change_index]) >= MINIMUM_SEQUENCE_LENGTH:
                sub_sequence = sequence_list[seq_index][change_indices[change_index]:change_indices[change_index + 1]]

                # shake
                if label_indices[change_index] == 1:
                    shake_seqs.append(sub_sequence)
                # nod
                elif label_indices[change_index] == 2:
                    nod_seqs.append(sub_sequence)   
                # background
                else:
                    background_seqs.append(sub_sequence)

    return background_seqs, shake_seqs, nod_seqs

def separate_seqs_all(sequence_list, labels_list, window_size=36):
    """
    Separates the sequences into background, shake, and nod sequences
    Based on the middle frame within a given the window size sliding over the training data
    One label per window
    """
    assert len(sequence_list) == len(labels_list)
    background_seqs, shake_seqs, nod_seqs = [], [], []

    for seq_index in range(len(sequence_list)):
        assert len(sequence_list[seq_index]) == len(labels_list[seq_index])

        for window_index in range(len(sequence_list[seq_index])-window_size):
            sub_sequence = sequence_list[seq_index][window_index : (window_index + window_size)].copy()
            # sub_sequence = add_diffs_list(sub_sequence)
            assert(len(sub_sequence) == window_size)

            label = labels_list[seq_index][window_index + math.floor(window_size/2)]

            # shake
            if label == 1:
                shake_seqs.append(sub_sequence)
            # nod
            if label == 2:
                nod_seqs.append(sub_sequence)
            # background
            else:
                background_seqs.append(sub_sequence)

    print(f'Found {len(background_seqs)} background sequences')
    print(f'Found {len(shake_seqs)} shake sequences')
    print(f'Found {len(nod_seqs)} nod sequences')
    return background_seqs, shake_seqs, nod_seqs

def separate_seqs_all_with_return_seq(sequence_list, labels_list, window_size=36):
    """
    Separates the sequences into background, shake, and nod sequences
    Based on the middle frame within a given the window size sliding over the training data
    List of labels per window
    """
    assert len(sequence_list) == len(labels_list)
    background_seqs, shake_seqs, nod_seqs, shake_label_seqs, nod_label_seqs, background_label_seqs = [], [], [], [], [], []

    # For each video
    for seq_index in range(len(sequence_list)):
        assert len(sequence_list[seq_index]) == len(labels_list[seq_index])

        # For each window
        for window_index in range(len(sequence_list[seq_index])-window_size):
            sub_sequence = sequence_list[seq_index][window_index : (window_index + window_size)].copy()
            # sub_sequence = add_diffs_list(sub_sequence).tolist()
            sub_seq_labels = labels_list[seq_index][window_index : (window_index + window_size)].copy().tolist()
            sub_seq_labels = [int(ssl) for ssl in sub_seq_labels]
            label = sub_seq_labels[math.floor(window_size/2)]

            assert(len(sub_sequence) == window_size)

            # shake
            if label == 1:
                shake_seqs.append(sub_sequence)
                shake_label_seqs.append(sub_seq_labels)
            # nod
            if label == 2:
                nod_seqs.append(sub_sequence)
                nod_label_seqs.append(sub_seq_labels)
            # background
            else:
                background_seqs.append(sub_sequence)
                background_label_seqs.append(sub_seq_labels)

    print(f'Found {len(background_seqs)} background sequences')
    print(f'Found {len(shake_seqs)} shake sequences')
    print(f'Found {len(nod_seqs)} nod sequences')
    return background_seqs, shake_seqs, nod_seqs, shake_label_seqs, nod_label_seqs, background_label_seqs

def separate_seqs_all_middle_frame_all(sequence_list, labels_list, window_size=36):
    """
    Separates the sequences into background, shake, and nod sequences
    Based on the middle frame within a given the window size sliding over the training data
    List of labels per window
    """
    assert len(sequence_list) == len(labels_list)
    background_seqs, shake_seqs, nod_seqs, shake_labels, nod_labels, background_labels = [], [], [], [], [], []

    # For each video
    for seq_index in range(len(sequence_list)):
        assert len(sequence_list[seq_index]) == len(labels_list[seq_index])

        # For each window
        for window_index in range(len(sequence_list[seq_index])-window_size):
            sub_sequence = sequence_list[seq_index][window_index : (window_index + window_size)].copy()
            # sub_sequence = add_diffs_list(sub_sequence).tolist()
            sub_seq_labels = labels_list[seq_index][window_index : (window_index + window_size)].copy().tolist()
            sub_seq_labels = [int(ssl) for ssl in sub_seq_labels]
            label = sub_seq_labels[math.floor(window_size/2)]

            assert(len(sub_sequence) == window_size)

            
            if label == 0:
                background_seqs.append(sub_sequence)
                background_labels.append(label)
            # shake
            if label == 1:
                shake_seqs.append(sub_sequence)
                shake_labels.append(label)
            # nod
            if label == 2:
                nod_seqs.append(sub_sequence)
                nod_labels.append(label)
            
    print(f'Found {len(background_seqs)} background sequences')
    print(f'Found {len(shake_seqs)} shake sequences')
    print(f'Found {len(nod_seqs)} nod sequences')
    return background_seqs, shake_seqs, nod_seqs, shake_labels, background_labels, nod_labels


def shuffle_data(data_x, data_y, nr_samples = -1):
    """
    Shuffle the data
    """
    c = list(zip(data_x, data_y))
    import random
    random.shuffle(c)
    data_x, data_y = zip(*c)
    data_x = list(data_x)
    data_y = list(data_y)

    # If a nr_samples is given, only take that many samples
    if nr_samples != -1:
        data_x = data_x[:nr_samples]
        data_y = data_y[:nr_samples]

    return data_x, data_y


def plot_pyr_graphs(vector, number=2, add_diff=False, diff_pix=False, diff_ang=False, movement_type='Movement', ang_norm=False):
    """Plot the pitch, yaw, roll, and shoulder angles in a graph for a number of random sequences from the vector list"""
    import random
    randoms = random.sample(range(0, len(vector)), number)
    for ra in randoms:
        p, y, r, s = [], [], [], []
        if add_diff:
            vector_ra = add_diffs_list(vector[ra])
        else:
            vector_ra = vector[ra]
        for xi, xx in enumerate(vector_ra):
            if not diff_pix:
                p.append(xx[0])
                y.append(xx[1])
                r.append(xx[2])
                s.append(xx[3])
            if diff_pix:
                if xi == 0:
                    p.append(0)
                    y.append(0)
                    r.append(0)
                    s.append(0)
                p.append(xx[0])
                y.append(xx[1])
                r.append(xx[2])
                s.append(xx[3])
            if xi != 0 and diff_pix:
                p[-1] = p[-1]+p[-2]
                y[-1] = y[-1]+y[-2]
                r[-1] = r[-1]+r[-2]
                s[-1] = s[-1]+s[-2]
        from matplotlib import pyplot as plt
        plt.figsize=(30,6)
        plt.plot(p, label='pitch')
        plt.plot(y, label='yaw')
        plt.plot(r, label='roll')
        plt.plot(s, label='shoulder')
        plt.ylabel('angles')
        plt.xlabel('frame')
        plt.title(movement_type + " angles")
        plt.legend()
        plt.show()
