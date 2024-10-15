import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from utils.helper_functions import sort_dict

def count_events_in_labels(labels, event_label):
    """ Count the number of a specific event (1 for shake, 2 for nod) in the labels array. """
    previous_value = 0
    event_count = 0
    
    for current_value in labels:
        if current_value == event_label and previous_value != event_label:
            event_count += 1
        previous_value = current_value

    return event_count

def main(frames_csv):
    df_frames = pd.read_csv(frames_csv)

    background_frame_counts, shake_frame_counts, nod_frame_counts, shake_event_counts, nod_event_counts  = {}, {}, {}, {}, {}
    split_to_signers = {}

    for _, row in df_frames.iterrows():
        labels = np.load(row['labels_path'])
        shake_frame_count = np.sum(labels==1)
        nod_frame_count = np.sum(labels==2)
        background_frame_count = int(row['end_frame']) - int(row['start_frame']) - shake_frame_count - nod_frame_count
        shake_event_count = count_events_in_labels(labels, 1)
        nod_event_count = count_events_in_labels(labels, 2)

        try:
            shake_frame_counts[row['split']] += shake_frame_count
            nod_frame_counts[row['split']] += nod_frame_count
            background_frame_counts[row['split']] += background_frame_count
            shake_event_counts[row['split']] += shake_event_count
            nod_event_counts[row['split']] += nod_event_count
            split_to_signers[row['split']].add(row['video_id'].split('_')[1])
        except KeyError:
            shake_frame_counts[row['split']] = shake_frame_count
            nod_frame_counts[row['split']] = nod_frame_count
            background_frame_counts[row['split']] = background_frame_count
            shake_event_counts[row['split']] = shake_event_count
            nod_event_counts[row['split']] = nod_event_count
            split_to_signers[row['split']] = {row['video_id'].split('_')[1]}
            
    background_frame_counts = sort_dict(background_frame_counts)
    nod_frame_counts = sort_dict(nod_frame_counts)
    shake_frame_counts = sort_dict(shake_frame_counts)
    shake_event_dict, nod_event_dict = {}, {}
    shake_event_dict['train'] = shake_event_counts['fold_1']+shake_event_counts['fold_2']+shake_event_counts['fold_3']+shake_event_counts['fold_4']
    shake_event_dict['val'] = shake_event_counts['fold_5']
    shake_event_dict['test'] = shake_event_counts['test']
    nod_event_dict['train'] = nod_event_counts['fold_1']+nod_event_counts['fold_2']+nod_event_counts['fold_3']+nod_event_counts['fold_4']   
    nod_event_dict['val'] = nod_event_counts['fold_5']
    nod_event_dict['test'] = nod_event_counts['test']
    print("Background frames train/val/test: ", background_frame_counts['fold_1']+background_frame_counts['fold_2']+background_frame_counts['fold_3']+background_frame_counts['fold_4'], " / ", background_frame_counts['fold_5'], " / ", background_frame_counts['test'])
    print("Shake frames train/val/test: ", shake_frame_counts['fold_1']+shake_frame_counts['fold_2']+shake_frame_counts['fold_3']+shake_frame_counts['fold_4'], " / ", shake_frame_counts['fold_5'], " / ", shake_frame_counts['test'])
    print("Nod frames train/val/test: ", nod_frame_counts['fold_1']+nod_frame_counts['fold_2']+nod_frame_counts['fold_3']+nod_frame_counts['fold_4'], " / ", nod_frame_counts['fold_5'], " / ", nod_frame_counts['test'])
    print("Shake events train/val/test: ", shake_event_dict['train'], " / ", shake_event_dict['val'], " / ", shake_event_dict['test'], " = ", shake_event_dict['train']+shake_event_dict['val']+shake_event_dict['test'])
    print("Nod events train/val/test: ", nod_event_dict['train'], " / ", nod_event_dict['val'], " / ", nod_event_dict['test'], " = ", nod_event_dict['train']+nod_event_dict['val']+nod_event_dict['test'])
    split_to_signers = sort_dict(split_to_signers)

    sns.set_theme()
    sns.set_style('whitegrid')
    sns.set_context('paper')

    plt.title(f'Frame counts for the different splits', fontsize=15)
    plt.bar(list(shake_frame_counts.keys()),
            list(shake_frame_counts.values()),
            align='center')
    plt.bar(list(nod_frame_counts.keys()),
            list(nod_frame_counts.values()),
            align='center',
            bottom=list(shake_frame_counts.values()))
    plt.bar(list(background_frame_counts.keys()),
            list(background_frame_counts.values()),
            align='center',
            bottom=list([x + y for x, y in zip(shake_frame_counts.values(), nod_frame_counts.values())]))
    plt.legend(['Shakes', 'Nods', 'Background'])
    plt.show()

    plt.title(f'Signers per split', fontsize=15)
    plt.bar(list(split_to_signers.keys()),
            list([len(value) for value in split_to_signers.values()]),
            align='center')
    plt.show()

    return shake_event_dict, nod_event_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('frames_csv', metavar='frames-csv', type=Path)
    args = parser.parse_args()

    main(args.frames_csv)
