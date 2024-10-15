import statsmodels
from statistics import mean
import krippendorff as kd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from pympi.Elan import Eaf
import itertools
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.metrics import cohen_kappa_score

DICT_LABELS = {'background' : 0, 'shake': 1, 'nod': 2, 'side': 3,'up': 4, 'down': 5, 'sway': 6, 'other' : 7}

def get_cohens_kappa(annotation_arrays):
    """
    Calculates Cohen's kappa for the given annotation arrays.
    """
    agreement_matrix = np.array(annotation_arrays)
    result = cohen_kappa_score(agreement_matrix[0], agreement_matrix[1])
    return result

def get_cohens_kappa_for_labels(annotation_arrays, dict_labels, annotators, show=True):
    """
    Calculates Cohen's kappa for each label in the annotation arrays.
    """
    labels = DICT_LABELS.values()
    results = []
    for label in labels:
        annotation_arrays_copy = np.array(annotation_arrays.copy())
        for i in range(len(annotation_arrays_copy)):
            annotation_arrays_copy[i] = np.where(annotation_arrays_copy[i] == label, 1, 0)
        try:
            ck = get_cohens_kappa(annotation_arrays_copy)
        except:
            ck = 0
        results.append(ck)

    if show:
        plt.bar(dict_labels.values(), results)
        plt.title("Inter annotator agreement,\n "+annotators[0].replace("ann", "annotator ")+" and "+annotators[1].replace("ann", "annotator "), fontsize=16)
        plt.ylabel("Cohen's kappa", fontsize=14)
        plt.xticks(rotation=45, fontsize=14)
        plt.yticks(fontsize=14)
        plt.show()

def get_krippendorff(annotation_arrays):
    """
    Calculates the Krippendorff's alpha for the given annotation arrays.
    """

    # convert to pandas series
    annotation_series = []
    for annotation_array in annotation_arrays:
        annotation_series.append(pd.Series(annotation_array))

    agreement_matrix = pd.concat(annotation_series, axis = 1).T

    result = kd.alpha(agreement_matrix, level_of_measurement='nominal')

    return result

def get_krippendorff_for_labels(annotation_arrays, dict_labels, annotators, show=True):
    """
    Calculates the Krippendorff's alpha for each label in the annotation arrays.
    """

    labels = DICT_LABELS.values()

    results = []
    for label in labels:
        annotation_arrays_copy = np.array(annotation_arrays.copy())
        for i in range(len(annotation_arrays_copy)):
            annotation_arrays_copy[i] = np.where(annotation_arrays_copy[i] == label, 1, 0)
        try:
            kd = get_krippendorff(annotation_arrays_copy)
        except:
            kd = 0
        results.append(kd)

    if show:
        plt.bar(dict_labels.values(), results)
        plt.title("Inter annotator agreement per label for "+annotators[0]+" and "+annotators[1])
        plt.xlabel('Label')
        plt.ylabel('Krippendorff\'s alpha')
        plt.xticks(rotation=90)
        plt.show()

def get_video_lenght(filename):
    """
    Returns the duration of the given video in milliseconds and the number of frames
    """
    
    import cv2
    video = cv2.VideoCapture(filename)

    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count/fps*1000
    # print("Number of frames: ", frame_count)
    # print("Duration: ", duration)
    
    return fps, int(frame_count)


def first_indexes_of_vals(array):
    """
    Returns the first indexes of 1's in the given array.
    """
    first_indexes_of_ones = []

    for i in range(len(array)):
        if array[i] == 1:  # If the element is 1
            if i == 0 or array[i - 1] == 0:  # If it's the first 1 in the sequence or it's preceded by a 0
                first_indexes_of_ones.append(i)
    return first_indexes_of_ones

def count_elements_within_difference(arr1, arr2, threshold):
    """
    Count the number of elements within difference range
    """
    count = 0
    for x in arr1:
        for y in arr2:
            if abs(x - y) <= threshold:
                count += 1
                break  # Break to avoid counting the same element in arr1 multiple times
    return count

def count_starts_within_other(indexes, binary_array, threshold):
    """
    Count the number of starts within the other array
    """
    result = 0
    # Add 5 1s on either side of every 1 in the binary array (replacement)
    binary_array_copy = binary_array.copy()
    for i in range(len(binary_array)):
        if binary_array[i] == 1:
            for j in range(i - threshold, i + threshold + 1):
                if j >= 0 and j < len(binary_array):
                    binary_array_copy[j] = 1
    binary_array = binary_array_copy

    # Iterate through each index in the given list
    for index in indexes:
        # Check if the index is aligned with a 1
        if binary_array[index] == 1:
            result += 1

    return result

def plot_precision_recall(precisions, recalls, title = 'Agreement scores by label', annotators = ['Annotator 1', 'Annotator 2']):
    # plot precision and recall
    labels = list(DICT_LABELS.keys())
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, precisions, width, label='Precision')
    rects2 = ax.bar(x + width/2, recalls, width, label='Recall')
    ax.set_ylabel('Scores')
    ax.set_title(title + ",\n" + annotators[0].replace("ann", "annotator ") + " as ground truth" + " and " + annotators[1].replace("ann", "annotator ") + " as predicted")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.tick_params(axis='x', rotation=45)
    ax.legend()
    plt.show()

def compare_annotation_starts(annotation_arrays):
    """
    Compares the starts of the annotations of the given annotation arrays.
    """
    precisions, recalls = [[] for _ in range(len(DICT_LABELS.values()))], [[] for _ in range(len(DICT_LABELS.values()))]
    overlap_count = [0 for _ in range(len(DICT_LABELS.values()))]
    positives_0, positives_1 = [0 for _ in range(len(DICT_LABELS.values()))], [0 for _ in range(len(DICT_LABELS.values()))]
    true_positives_0, true_positives_1 = [0 for _ in range(len(DICT_LABELS.values()))], [0 for _ in range(len(DICT_LABELS.values()))]
    false_negatives_0, false_negatives_1 = [0 for _ in range(len(DICT_LABELS.values()))], [0 for _ in range(len(DICT_LABELS.values()))]

    # for each annotator
    for annotator in ([0,1]):
        ann_arrays = [inner_list[annotator] for inner_list in annotation_arrays]

        for value in DICT_LABELS.values():
            
            # get all the first indexes of the values
            binary_values, first_indexes_values = [], []
            for arr in ann_arrays:
                # makes a binary array of the values, e.g. [0,0,0,1,1,1,1,0,0,1,0,0]
                binary_values.append((np.array(arr) == int(value)).astype(int))
                # get the first indexes of the binary list where it is 1, e.g. [3,9]
                first_indexes_values.append(first_indexes_of_vals(binary_values[-1]))

            # calculate precision and recall for combination of annotators
            precision, recall, count = 0, 0, 0
            precision_0, recall_0, precision_1, recall_1, count_0, count_1 = 0, 0, 0, 0, 0, 0
            # If both arrays are empty, precision and recall are 1 because they are equal
            if len(first_indexes_values[0]) == 0 and len(first_indexes_values[1]) == 0:
                precision, recall, precision_0, recall_0, precision_1, recall_1 = 1, 1, 1, 1, 1, 1
            # If one of the arrays is empty, 
            elif (len(first_indexes_values[0]) == 0 and len(first_indexes_values[1]) != 0):
                # precision, recall, precision_0, recall_0, precision_1, recall_1 = 0, 0, 0, 1, 1, 0
                positives_1[value] += len(first_indexes_values[1])
                false_negatives_0[value] += len(first_indexes_values[1])
            elif (len(first_indexes_values[0]) != 0 and len(first_indexes_values[1]) == 0):
                # precision, recall, precision_0, recall_0, precision_1, recall_1 = 0, 0, 1, 0, 0, 1
                positives_0[value] += len(first_indexes_values[0])
                false_negatives_1[value] += len(first_indexes_values[0])
            else:
                # Stricty count the number of starts that match a start in the other array with a certain range (strict starts)
                count = count_elements_within_difference(first_indexes_values[0], first_indexes_values[1], 5)
                if count > 0:
                    precision, recall = count / len(first_indexes_values[1]), count / len(first_indexes_values[0])
                # Count the number of starts that fall within the range of that label in the other array
                count_0 = count_starts_within_other(first_indexes_values[0], binary_values[1], 5)
                count_1 = count_starts_within_other(first_indexes_values[1], binary_values[0], 5)
                # if count_0 > 0:
                #     precision_0, recall_0 = count_0 / len(first_indexes_values[0]), count_0 / (count_0 + len(first_indexes_values[1]) - count_1)
                # if count_1 > 0:
                #     precision_1, recall_1 = count_1 / len(first_indexes_values[1]), count_1 / (count_1 + len(first_indexes_values[0]) - count_0)
                positives_0[value] += len(first_indexes_values[0])
                positives_1[value] += len(first_indexes_values[1])
                true_positives_0[value] += count_0
                true_positives_1[value] += count_1
                false_negatives_0[value] += len(first_indexes_values[1]) - count_1
                false_negatives_1[value] += len(first_indexes_values[0]) - count_0
            # if ground truth is the first tier
            precisions[value].append(precision)
            recalls[value].append(recall)
            overlap_count[value] += count

    return overlap_count, positives_0, positives_1, true_positives_0, true_positives_1, false_negatives_0, false_negatives_1


def count_starts_within_other_in(first_indexes, binary_array, threshold):
    """
    Count the number of starts within the other array
    """
    result = 0

    # Iterate through each index in the given list
    for first_index in first_indexes:
        # Check if the index is aligned with a 1
        if first_index+threshold < len(binary_array):
            first_index = first_index+threshold
        if binary_array[first_index] == 1:
            result += 1

    return result


def compare_annotation_start_labels(conf_matrix_A1, conf_matrix_A2, annotation_arrays):
    """
    Compares the starts of the annotations of the given annotation arrays.
    """
    # for each annotator
    for annotator in (0,1):
        ann_arrays = [inner_list[annotator] for inner_list in annotation_arrays]
        ann_array_S1 = ann_arrays[0]
        ann_array_S2 = ann_arrays[1]
        for index_1, value_1 in enumerate(DICT_LABELS.values()):
            binary_values_A1 = (np.array(ann_array_S1) == int(value_1)).astype(int)
            first_indexes_values_A1 = first_indexes_of_vals(binary_values_A1)
            for index_2, value_2 in enumerate(DICT_LABELS.values()):
                binary_values_A2 = (np.array(ann_array_S2) == int(value_2)).astype(int)
                first_indexes_values_A2 = first_indexes_of_vals(binary_values_A2)
            
                count_A1 = count_starts_within_other_in(first_indexes_values_A1, binary_values_A2, 5)
                conf_matrix_A1[index_1][index_2] += count_A1
                count_A2 = count_starts_within_other_in(first_indexes_values_A2, binary_values_A1, 5)
                conf_matrix_A2[index_1][index_2] += count_A2
    

def get_labels_from_eafs(eafs, video, annotators, conf_matrix_A1, conf_matrix_A2, show = True):
    """
    Loads a list of eaf files and adds the annotations of head movements to a list of labels.
    """
    annotation_arrays = []

    print(eafs[0])

    from dataset.labels.eaf_parser import Annotation

    fps, n_frames = get_video_lenght(video)
    print("Nr of frames in video: ", n_frames)

    # for each annotator
    for eaf in eafs:

        annotations_one_speaker = []

        eaf_path = Path(eaf)
        eaf = Eaf(eaf_path, 'pympi')

        for speaker in ["S1", "S2"]:
            annotations = []
            labels = np.zeros(n_frames).astype(int)
            
            # get annotations
            for annotation in eaf.tiers["Head movement "+speaker][0].values():
                label = annotation[2]
                start = int(eaf.timeslots[annotation[0]] * (fps*0.001))
                end = int(eaf.timeslots[annotation[1]] * (fps*0.001))

                annotations.append(Annotation(label, start, end))

            # add annotations to labels
            for annotation_i, annotation in enumerate(annotations):
                label = annotation.label.lower()
                # if the last annotation is other, remove the rest of the labels
                if annotation_i == len(annotations) - 1 and label == 'other' and annotation.end > len(labels)-35 and annotation.start < len(labels)-35:
                    labels[annotation.start:n_frames] = [-1] * (n_frames - annotation.start)
                elif label in DICT_LABELS:
                    labels[annotation.start:annotation.end] = DICT_LABELS[label]

            annotations_one_speaker.append(labels.tolist())

        annotation_arrays.append(annotations_one_speaker)
    
    # get the minimum length of the annotations
    lengths = [n_frames, n_frames]
    for annotation_array_i, annotation_array in enumerate(annotation_arrays):
        for length_i, length in enumerate(lengths):
            this_length = n_frames - 25
            if -1 in annotation_array[length_i]:
                this_length = annotation_array[length_i].index(-1)
            if this_length < lengths[length_i]:
                lengths[length_i] = this_length
            print("Nr frames found for annotator "+str(annotation_array_i)+" for speaker S"+str(length_i+1)+" is "+str(this_length-75))
    
    # remove the rest of the labels
    for annotation_array_i, annotation_array in enumerate(annotation_arrays):
        for i, length in enumerate(lengths):
            annotation_arrays[annotation_array_i][i] = annotation_array[i][75:length]
    print("Nr labels for S1: "+ str(len(annotation_arrays[0][0])))
    print("Nr labels for S2: "+ str(len(annotation_arrays[0][1])))

    overlap_count, p_0, p_1, tp_0, tp_1, fn_0, fn_1 = compare_annotation_starts(annotation_arrays)
    compare_annotation_start_labels(conf_matrix_A1, conf_matrix_A2, annotation_arrays)

    # Add up the annotations
    for annotation_array_i, annotation_array in enumerate(annotation_arrays):
        annotation_arrays[annotation_array_i] = list(itertools.chain.from_iterable(annotation_array))

    annotation_arrays = np.array(annotation_arrays)

    inverse_dict = {v: k for k, v in DICT_LABELS.items()}

    get_krippendorff_for_labels(annotation_arrays, inverse_dict, annotators, show=False)
    if show:
        print("Krippendorff total: ", get_krippendorff(annotation_arrays))

    combinations = itertools.combinations(annotation_arrays, 2)
    from validation.cross_validation import plot_confusion_matrix
    from sklearn.metrics import confusion_matrix
    if show:
        for combination in combinations:
            conf_matr = confusion_matrix(combination[0], combination[1], labels = [0,1,2,3,4,5,6,7])
            print(conf_matr)
    
    print('\n')
    return annotation_arrays, overlap_count, p_0, p_1, tp_0, tp_1, fn_0, fn_1


def plot_annotation_confusion_matrix(matrix, title="Confusion matrix", labels = ['background', 'head-shake', 'head-nod'], annotators=['Annotator 1', 'Annotator 2']):
    """
    Plot a confusion matrix with annotator labels.
    """
    disp = ConfusionMatrixDisplay(matrix, display_labels=labels)
    disp.plot(cmap="Blues", values_format='.0f', colorbar=False)
    plt.title(f"{title}", fontsize=12, y=1.1)
    plt.xlabel(annotators[1].replace("ann", "annotator "))  # Annotator 2 as x-axis label
    plt.ylabel(annotators[0].replace("ann", "annotator "))  # Annotator 1 as y-axis label
    plt.xticks(rotation=45)
    plt.show()

def plot_confusion_matrix_with_percentage(matrix, show_normalized_colors=True, labels=['background', 'head-shake', 'head-nod'], annotators=['Annotator 1', 'Annotator 2']):
    """
    Plot a confusion matrix with normalized values in percentage, along with totals on the other axis.
    
    Parameters:
        matrix (array-like): Confusion matrix.
        labels (list): List of class labels.
    """
    # Calculate row totals
    row_totals = np.sum(matrix, axis=1)
    normalized_matrix = matrix / row_totals[:, np.newaxis] * 100

    # Plot confusion matrix with normalized values in percentage
    if len(labels) > 3:
        _, ax = plt.subplots(figsize=(6, 5))
    else: 
        _, ax = plt.subplots(figsize=(4, 3))  # Use 6,5 for inter-annotator agreement

    # Plot confusion matrix with normalized values in percentage
    if not show_normalized_colors:
        ax.imshow(normalized_matrix, cmap='Blues', vmin=0, vmax=100)
    else:
        ax.imshow(normalized_matrix, cmap='Blues')
    threshold = 0.5  # Threshold for text color

    # Show ticks and labels
    ax.set_xticks(np.arange(len(labels) + 1))  # Add 1 for the total column
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels + ['Total'], rotation=45, ha='center')  # Rotate labels for better fit
    ax.set_yticklabels(labels)
    ax.set_xlabel(annotators[1].replace("ann", "annotator "))
    ax.set_ylabel(annotators[0].replace("ann", "annotator "))
    ax.set_title('Confusion Matrix (row-wise normalized)')
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    plt.gca().set_xlim([-0.5, len(labels)+0.5])    

    # Loop over data dimensions and create text annotations for percentage values
    for i in range(len(labels)):
        for j in range(len(labels)):
            color = 'white' if normalized_matrix[i, j] >= threshold * 100 else 'black'
            text = str(matrix[i, j])+"\n"+str(int(matrix[i, j] / row_totals[i] * 100))+"%" if matrix[i, j] > 0 else ""
            ax.text(j, i, f'{text}', ha="center", va="center", color=color)

    # Plot row totals on the other axis
    for i, row_total in enumerate(row_totals):
        ax.text(len(labels), i, f'{row_total}\n100%',
                    ha="center", va="center", color="black")

    # make only the right column dark and the text white
    ax.set_facecolor('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    plt.show()
    

def run_inter_annotator_agreement(eaf_files, annotators, print_confusion = True):
    """
    Run the several steps for inter annotator agreement analysis.
    """
    # set empty arrays
    
    num_labels = len(DICT_LABELS)
    conf_matrix_A1, conf_matrix_A2 = np.zeros((num_labels, num_labels), dtype=int), np.zeros((num_labels, num_labels), dtype=int)
    anns_1, anns_2 = [], []
    ps_0, ps_1, tps_0, tps_1, fns_0, fns_1 = [0 for _ in range(len(DICT_LABELS.values()))], [0 for _ in range(len(DICT_LABELS.values()))], [0 for _ in range(len(DICT_LABELS.values()))], [0 for _ in range(len(DICT_LABELS.values()))], [0 for _ in range(len(DICT_LABELS.values()))], [0 for _ in range(len(DICT_LABELS.values()))]
    a_0_starts, a_1_starts, counts = [0 for _ in range(len(DICT_LABELS.values()))], [0 for _ in range(len(DICT_LABELS.values()))], [0 for _ in range(len(DICT_LABELS.values()))]
    
    # retrieve lists of labels and nr of true positives/true negatives/etc. from eafs
    for eaf_file in eaf_files:
        (a1, a2), overlap_count, p_0, p_1, tp_0, tp_1, fn_0, fn_1 = get_labels_from_eafs([f'data/eaf_inter_annotator/{annotators[0]}/{eaf_file}.eaf', f'data/eaf_inter_annotator/{annotators[1]}/{eaf_file}.eaf'], f'data/eaf_inter_annotator/{eaf_file}.mp4', annotators, conf_matrix_A1, conf_matrix_A2, show=True)
        anns_1.extend(a1)
        anns_2.extend(a2)
        ps_0, ps_1, tps_0, tps_1, fns_0, fns_1 = [x + y for x, y in zip(ps_0, p_0)], [x + y for x, y in zip(ps_1, p_1)], [x + y for x, y in zip(tps_0, tp_0)], [x + y for x, y in zip(tps_1, tp_1)], [x + y for x, y in zip(fns_0, fn_0)], [x + y for x, y in zip(fns_1, fn_1)]
        for i in range(len(DICT_LABELS.values())):
            a_0_starts[i] += p_0[i]
            a_1_starts[i] += p_1[i]
            counts[i] += overlap_count[i]

    print(conf_matrix_A1)
    print(conf_matrix_A2)

    # plot precision recall
    precisions_0, recalls_0, precisions_1, recalls_1 = [], [], [], []
    for i in range(len(DICT_LABELS.values())):
        if ps_0[i] == 0:
            precisions_0.append(0)
        else:
            precisions_0.append(tps_0[i] / ps_0[i])
        if ps_1[i] == 0:
            precisions_1.append(0)
        else:
            precisions_1.append(tps_1[i] / ps_1[i])
        if tps_0[i] == 0 and fns_0[i] == 0:
            recalls_0.append(0)
        else:
            recalls_0.append(tps_0[i] / (tps_0[i] + fns_0[i]))
        if tps_1[i] == 0 and fns_1[i] == 0:
            recalls_1.append(0)
        else:
            recalls_1.append(tps_1[i] / (tps_1[i] + fns_1[i]))

    annotators_turned = [annotators[1], annotators[0]]
    plot_precision_recall(precisions_0, recalls_0, title= "Annotation start frame agreement", annotators = annotators_turned)
    plot_precision_recall(precisions_1, recalls_1, title= "Annotation start frame agreement", annotators = annotators)
    print(ps_0, ps_1)

    # plot_annotation_confusion_matrix(conf_matrix_A1, title="Confusion matrix between annotators on event level for "+annotators[0], labels = ['bg', 'shake', 'nod', 'side', 'up', 'down', 'sway', 'other'], annotators=annotators)
    plot_confusion_matrix_with_percentage(conf_matrix_A1, labels = ['bg', 'shake', 'nod', 'side', 'up', 'down', 'sway', 'other'], annotators=annotators)
    # plot_annotation_confusion_matrix(conf_matrix_A2.T, title="Confusion matrix between annotators on event level for "+annotators_turned[0], labels = ['bg', 'shake', 'nod', 'side', 'up', 'down', 'sway', 'other'], annotators=annotators_turned)
    plot_confusion_matrix_with_percentage(conf_matrix_A2.T, labels = ['bg', 'shake', 'nod', 'side', 'up', 'down', 'sway', 'other'], annotators=annotators_turned)
    

    # plot precision recall
    precisions, recalls = [], []
    for i in range(len(DICT_LABELS.values())):
        if a_1_starts[i] == 0:
            precisions.append(0)
        else:
            precisions.append(counts[i]/a_1_starts[i])
        if a_0_starts[i] == 0:
            recalls.append(0)
        else:
            recalls.append(counts[i]/a_0_starts[i])
    plot_precision_recall(precisions, recalls, title = "Annotation start agreement scores per label STRICT", annotators = annotators)

    annotation_arrays = [anns_1, anns_2]

    inverse_dict = {v: k for k, v in DICT_LABELS.items()}
    get_cohens_kappa_for_labels(annotation_arrays, inverse_dict, annotators)
    print("Cohens Kappa total: ", get_cohens_kappa(annotation_arrays))
    get_krippendorff_for_labels(annotation_arrays, inverse_dict, annotators)
    print("Krippendorff total: ", get_krippendorff(annotation_arrays))

    if print_confusion:
        combinations = itertools.combinations(annotation_arrays, 2)
        from validation.cross_validation import plot_confusion_matrix
        from sklearn.metrics import confusion_matrix
        for combination in combinations:
            conf_matr = confusion_matrix(combination[0], combination[1])
            # plot_annotation_confusion_matrix(conf_matr, title="Confusion matrix between annotators on frame level", labels = ['bg', 'shake', 'nod', 'side', 'up', 'down', 'sway', 'other'], annotators=annotators)
            plot_confusion_matrix_with_percentage(conf_matr, labels = ['bg', 'shake', 'nod', 'side', 'up', 'down', 'sway', 'other'], annotators=annotators)
            plot_confusion_matrix_with_percentage(conf_matr.T, labels = ['bg', 'shake', 'nod', 'side', 'up', 'down', 'sway', 'other'], annotators=annotators_turned)





def get_cohens_kappa_for_labels_eval(annotation_arrays, orig_labels, dict_labels, show=True):
    """
    Calculates Cohen's kappa for each label in the annotation arrays.
    """
    labels = orig_labels.values()
    results = []
    for label in labels:
        annotation_arrays_copy = np.array(annotation_arrays.copy())
        for i in range(len(annotation_arrays_copy)):
            annotation_arrays_copy[i] = np.where(annotation_arrays_copy[i] == label, 1, 0)
        try:
            ck = get_cohens_kappa(annotation_arrays_copy)
        except:
            ck = 0
        results.append(ck)

    print(results)
    if show:
        plt.bar(dict_labels.values(), results)
        plt.title("Frame-level agreement", fontsize=16)
        plt.ylabel("Cohen's kappa", fontsize=14)
        plt.xticks(rotation=45, fontsize=14)
        plt.yticks(fontsize=14)
        plt.show()

def plot_precision_recall_event(precisions, recalls, labels_dict, title = 'Start frame agreement scores by label'):
    # plot precision and recall
    labels = list(labels_dict.keys())
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots()
    ax.bar(x - width/2, precisions, width, label='Precision')
    ax.bar(x + width/2, recalls, width, label='Recall')
    ax.set_ylabel('Scores', fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=14)
    ax.tick_params(axis='x', rotation=45, labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.legend(fontsize=14)

    plt.show()

def compare_preds_trues_start_labels(conf_matrix_pred, predicted_list, labeled_list, labels_dict):
    """
    Compares the starts of the annotations of the given annotation arrays.
    """
    
    for index_1, value_1 in enumerate(labels_dict.values()):
        binary_values_pred = (np.array(predicted_list) == int(value_1)).astype(int)
        first_indexes_values_pred = first_indexes_of_vals(binary_values_pred)
        for index_2, value_2 in enumerate(labels_dict.values()):
            binary_values_labd = (np.array(labeled_list) == int(value_2)).astype(int)
        
            count_pred = count_starts_within_other_in(first_indexes_values_pred, binary_values_labd, 5)
            conf_matrix_pred[index_1][index_2] += count_pred

def compare_preds_trues_starts(predicted_list, labeled_list, labels_dict):
    """
    Compares the starts of the annotations of the given annotation arrays.
    """
    false_positives_pred = [0 for _ in range(len(labels_dict.values()))]
    true_positives_pred = [0 for _ in range(len(labels_dict.values()))]
    false_negatives_pred = [0 for _ in range(len(labels_dict.values()))]

    for value in labels_dict.values():
        
        # makes a binary array of the values, e.g. [0,0,0,1,1,1,1,0,0,1,0,0]
        binary_values_pred = ((np.array(predicted_list) == int(value)).astype(int))
        binary_values_labd = ((np.array(labeled_list) == int(value)).astype(int))
        # get the first indexes of the binary list where it is 1, e.g. [3,9]
        first_indexes_values_pred = first_indexes_of_vals(binary_values_pred)
        first_indexes_values_labd = first_indexes_of_vals(binary_values_labd)

        # If one of the arrays is empty, 
        if (len(first_indexes_values_pred) == 0 and len(first_indexes_values_labd) != 0):
            false_negatives_pred[value] += len(first_indexes_values_labd)
        elif (len(first_indexes_values_pred) != 0 and len(first_indexes_values_labd) == 0):
            false_positives_pred[value] += len(first_indexes_values_pred)
        elif (len(first_indexes_values_pred) != 0 and len(first_indexes_values_labd) != 0):
            # Count the number of starts that fall within the range of that label in the other array
            true_positives_pred[value] += count_starts_within_other(first_indexes_values_pred, binary_values_labd, 5)
            true_positives_labd = count_starts_within_other(first_indexes_values_labd, binary_values_pred, 5)
            false_positives_pred[value] += len(first_indexes_values_pred) - true_positives_pred[value]
            false_negatives_pred[value] += len(first_indexes_values_labd) - true_positives_labd

    return false_positives_pred, true_positives_pred, false_negatives_pred

def get_labels_from_lists(predicted_list, trues_list, conf_matrix_pred, conf_matrix_labd, labels_dict):
    """
    Loads a list of eaf files and adds the annotations of head movements to a list of labels.
    """

    fp_pred, tp_pred, fn_pred = compare_preds_trues_starts(predicted_list, trues_list, labels_dict)
    compare_preds_trues_start_labels(conf_matrix_pred, predicted_list, trues_list, labels_dict)
    compare_preds_trues_start_labels(conf_matrix_labd, trues_list, predicted_list, labels_dict)

    return fp_pred, tp_pred, fn_pred

def run_event_level_agreement(predicted_lists, trues_lists, annotators = ["predicted", "true"], labels_dict = {'background' : 0, 'shake': 1, 'nod': 2}):
    """
    Run the several steps for inter annotator agreement analysis.
    """

    inverse_dict = {v: k for k, v in labels_dict.items()}
    num_labels = len(labels_dict)
    conf_matrix_pred, conf_matrix_labd = np.zeros((num_labels, num_labels), dtype=int), np.zeros((num_labels, num_labels), dtype=int)
    predicted_scores, true_scores = [], []
    fps_pred, tps_pred, fns_pred = [0 for _ in range(num_labels)], [0 for _ in range(num_labels)], [0 for _ in range(num_labels)]
    
    # retrieve lists of labels and nr of true positives/true negatives/etc. from eafs
    for predicted_list_i, predicted_list in enumerate(predicted_lists):
        trues_list = trues_lists[predicted_list_i]
        false_positives_pred, true_positives_pred, false_negatives_pred = get_labels_from_lists(predicted_list, trues_list, conf_matrix_pred, conf_matrix_labd, labels_dict)
        predicted_scores.extend(predicted_list)
        true_scores.extend(trues_list)
        fps_pred, tps_pred, fns_pred = [x + y for x, y in zip(fps_pred, false_positives_pred)], [x + y for x, y in zip(tps_pred, true_positives_pred)], [x + y for x, y in zip(fns_pred, false_negatives_pred)], 

    print("TPS: ", tps_pred)
    print("FPS: ", fps_pred)
    print("FNS: ", fns_pred)
    # plot precision recall
    precisions_pred, recalls_pred = [], []
    for i in range(len(labels_dict.values())):
        precisions_pred.append(tps_pred[i] / (tps_pred[i] + fps_pred[i]))
        recalls_pred.append(tps_pred[i] / (tps_pred[i] + fns_pred[i]))

    print("Precisions: ", precisions_pred)
    print("Recalls: ", recalls_pred)
    plot_precision_recall_event(precisions_pred, recalls_pred, labels_dict)
    inverse_annotators = [annotators[1], annotators[0]]
    labels = [inverse_dict[i] for i in range(len(labels_dict.values()))]
    plot_confusion_matrix_with_percentage(conf_matrix_pred, show_normalized_colors=False, labels = labels, annotators=annotators)
    plot_confusion_matrix_with_percentage(conf_matrix_labd, show_normalized_colors=False, labels = labels, annotators=inverse_annotators)

    predicted_trues_arrays = [predicted_scores, true_scores]

    get_cohens_kappa_for_labels_eval(predicted_trues_arrays, labels_dict, inverse_dict)
    print("Cohens Kappa total: ", get_cohens_kappa(predicted_trues_arrays))

    conf_matr = confusion_matrix(predicted_scores, true_scores)
    plot_confusion_matrix_with_percentage(conf_matr, show_normalized_colors=False, labels = labels, annotators=annotators)
    conf_matr = confusion_matrix(true_scores, predicted_scores)
    plot_confusion_matrix_with_percentage(conf_matr, show_normalized_colors=False, labels = labels, annotators=inverse_annotators)