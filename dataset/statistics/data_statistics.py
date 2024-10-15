import argparse
import pickle
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.offsetbox import AnchoredText
from tqdm import tqdm

from pose.tier_detector import TierDetector, find_eaf_and_videos, EAF_DIR, find_speaker_id
from utils.exceptions import EAFParsingError


def find_videos(file):
    """Find the videos and the annotations for a given file."""
    ngt_id = file.stem
    print(ngt_id)
    eaf, video_left, video_right = find_eaf_and_videos(ngt_id)
    tiers_1, tiers_2 = collect_annotations(eaf)

    return eaf, video_left, video_right, tiers_1, tiers_2


def collect_annotations(eaf):
    """Detect tiers and collect annotations."""
    tier_detector = TierDetector(eaf)
    time_name, time_time = tier_detector.conv_timeslots()
    tiers_1, tiers_2 = tier_detector.conv_tiers(time_name, time_time)

    return tiers_1, tiers_2


def dump_raw_statistics(output_file):
    """Dump raw statistics to a file."""

    output_file = Path(output_file)
    
    files = list(EAF_DIR.glob('*.eaf'))
    single_speaker_ids = []
    data = []

    with ProcessPoolExecutor() as executor:
        future_list = []

        for file in files:
            future = executor.submit(find_videos, file)
            print(future)
            future_list.append(future)

        for future in tqdm(future_list):
            try:
                data.append(future.result())
            except EAFParsingError as err:
                single_speaker_ids.append(err.ngt_id)

    with open(output_file, 'wb') as output_handle:
        pickle.dump(data, output_handle)
    with open((output_file.parent / 'single_speakers').with_suffix(output_file.suffix), 'wb') as output_handle:
        pickle.dump(single_speaker_ids, output_handle)


def load_raw_statistics(labels_file, single_speakers_file=None):
    """Load raw statistics from a file and return the data and single speaker ids."""

    labels_file = Path(labels_file)

    if not single_speakers_file:
        single_speakers_file = (labels_file.parent / 'single_speakers').with_suffix(labels_file.suffix)

    with open(labels_file, 'rb') as input_handle:
        data = pickle.load(input_handle)
    with open(single_speakers_file, 'rb') as input_handle:
        single_speaker_ids = pickle.load(input_handle)

    return data, single_speaker_ids


def plot_statistics(videos):
    """Plot statistics for the given data, including the number of annotations per speaker and the number of occurrences"""

    labels_valid = ['n', 'nf', 'nx', 'nn', 'nn:l', 'nnf', 'shake']
    labels_valid.extend(['?' + label for label in labels_valid])

    speaker_to_n_annotations = {}
    n_videos, n_total_videos, n_annotations = 0, 0, 0
    seconds, seconds_valid, labels = [], [], []

    for video in videos:
        n_total_videos += 1

        if len(video[3]) > 0 or len(video[4]) > 0:
            n_videos += 1
            n_annotations += len(video[3])
            n_annotations += len(video[4])

            for annotation in video[3]:
                file = Path(video[1])
                speaker_id = find_speaker_id(file)
                increment_speaker_dict(speaker_to_n_annotations, speaker_id)

                labels.append(annotation[2])
                seconds.append((annotation[1] - annotation[0]) * 0.001)
            for annotation in video[4]:
                file = Path(video[2])
                speaker_id = find_speaker_id(file)
                increment_speaker_dict(speaker_to_n_annotations, speaker_id)

                labels.append(annotation[2])
                seconds.append((annotation[1] - annotation[0]) * 0.001)

    clean_labels = [label.lower() if label.lower() in labels_valid else 'other' for label in labels]
    counts = Counter(clean_labels)
    counts_raw = Counter(labels)

    speaker_to_n_annotations = {k: v for k, v in sorted(speaker_to_n_annotations.items(),
                                                        key=lambda item: item[1],
                                                        reverse=True)}

    sns.set_theme()
    sns.set_style('whitegrid')
    sns.set_context('paper')
    fig, ax = plt.subplots()
    plt.title('Number of annotations per speaker', fontsize=15)
    plt.bar(range(len(speaker_to_n_annotations)), list(speaker_to_n_annotations.values()), align='center')
    plt.xticks(range(len(speaker_to_n_annotations)), list(speaker_to_n_annotations.keys()),
               rotation=70, ha='center', fontsize=10)
    plt.xlim(-0.5, len(speaker_to_n_annotations) - 0.5)
    ax.xaxis.grid(False)
    plt.tight_layout()
    plt.show()

    counts = {k: v for k, v in sorted(counts.items(),
                                      key=lambda item: item[1],
                                      reverse=True)}
    counts_raw = {k: v for k, v in sorted(counts_raw.items(),
                                          key=lambda item: item[1],
                                          reverse=True)}
    occurrences = {'Sure': [counts['n'], counts['nf'], counts['other'], counts['nx'],
                            counts['nn'], counts['nn:l'], counts['nnf'], counts['shake']],
                   'Unsure': [counts['?n'], counts['?nf'], counts['?other'], counts['?nx'],
                              counts['?nn'], counts['?nn:l'], counts['?nnf'], counts['?shake']]}

    fig, ax = plt.subplots()
    plt.title('Number of occurrences per label', fontsize=15)
    sns.set_context(rc={'patch.linewidth': 0.0})
    plt.bar(range(len(occurrences['Sure'])),
            list(occurrences['Sure']),
            align='center')
    plt.bar(range(len(occurrences['Unsure'])),
            list(occurrences['Unsure']),
            align='center',
            bottom=list(occurrences['Sure']))

    sns.set_context('paper')
    plt.legend(['Sure', 'Unsure'], fontsize=15, loc='center right')
    plt.xticks(range(len(occurrences['Sure'])), ['n', 'nf', 'nx', 'nn', 'nn:l', 'nnf', 'shake'],
               rotation=70, ha='center', fontsize=10)
    plt.xlim(-0.5, len(occurrences['Sure']) - 0.5)
    ax.xaxis.grid(False)
    at = textonly(plt.gca(),
                  "N   --> head-shake negation during signing\n"
                  "Nf  --> head-shake negation without signing\n"
                  "Nx  --> head-shake negation between signing\n"
                  "Nn  --> head-shake no negation during signing\n"
                  "Nn:l  --> head-shake no negation during listening\n"
                  "Nnf --> head-shake no negation without signing\n",
                  "shake --> head-shake",
                  fontsize=10,
                  font='monospace',
                  loc=1)

    plt.tight_layout()
    plt.show()

    plt.hist(seconds, bins=100, log=True)
    plt.title('Histogram of annotation lengths', fontsize=15)
    plt.xlabel('Length (seconds)')
    plt.ylabel('Occurrences (log)')
    plt.show()

    print(f'Labels: {n_annotations}')
    print(f'Videos: {n_total_videos}')
    print(f'Annotated videos: {n_videos}')
    print(f'Average length (seconds): {np.mean(seconds)}')
    print(f'Std length (seconds): {np.std(seconds)}')


def textonly(ax, txt, fontsize=14, loc=2, font='dejavu sans', *args, **kwargs):
    at = AnchoredText(txt,
                      prop=dict(size=fontsize, font=font),
                      frameon=True,
                      loc=loc)
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax.add_artist(at)
    return at


def increment_speaker_dict(dictionary, speaker_id):
    try:
        dictionary[speaker_id] += 1
    except KeyError:
        dictionary[speaker_id] = 1


def count_shakes_nods(frames_csv, version):
    import pandas as pd
    from pympi.Elan import Eaf
    from utils.config import Config
    config = Config()

    df_frames = pd.read_csv(frames_csv)
    df_frames.head()
    files_done = []
    hm_tiers = []

    def parse_tier(tier):
        annotations = []

        for annotation in tier[0].values():
            label = annotation[2]
            annotations.append(label)

        return annotations

    for _, row in df_frames.iterrows():
        unique_id = row['video_id']
        if unique_id not in files_done:
            files_done.append(unique_id)
            eaf = Eaf(config.content["media"]["eaf"] + "/" + version + "/" + unique_id+'.eaf', 'pympi')
            hm_tiers.append(parse_tier(eaf.tiers['Head movement']))

    nod_tier_count, shake_tier_count = 0, 0
    nod_count, shake_count = 0, 0
    for hm in hm_tiers:
        nod_found = False
        shake_found = False
        for annot in hm:
            label = annot.lower().replace("?", "")
            if label in ['d', 'df', 'db', 'nod:l', 'nod', 'nod:n']:
                nod_count += 1
                nod_found = True
            if label in ['n', 'nf', 'nx', 'nn', 'nn:l', 'nnf', 'shake']:
                shake_count += 1
                shake_found = True
        if nod_found:
            nod_tier_count += 1
        if shake_found:
            shake_tier_count += 1

    print("Nr of nods: ", nod_count)
    print("Nr of tiers that contain a nod: ", nod_tier_count)
    print("Nr of shakes: ", shake_count)
    print("Nr of tiers that contain a shake: ", shake_tier_count)
    print("Nr of tiers: ", len(hm_tiers))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('labels_file', metavar='labels_file', type=str)
    parser.add_argument('--single_speakers_file', type=str, default=None)
    args = parser.parse_args()

    load_raw_statistics(args.labels_file, args.single_speakers_file)
