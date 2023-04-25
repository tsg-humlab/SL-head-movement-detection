import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.offsetbox import AnchoredText
from tqdm import tqdm

from dataset.pose.pose_face_detection_old import TierDetector, find_eaf_and_videos, EAF_DIR, find_speaker_id
from utils.exceptions import EAFParsingError

from concurrent.futures import ProcessPoolExecutor

import pickle
from collections import Counter
from pathlib import Path


def find_videos(file):
    ngt_id = file.stem

    eaf, video_left, video_right = find_eaf_and_videos(ngt_id)
    tiers_left, tiers_right = collect_annotations(eaf)

    return eaf, video_left, video_right, tiers_left, tiers_right


def collect_annotations(eaf):
    tier_detector = TierDetector(eaf)
    time_name, time_time = tier_detector.conv_timeslots()
    tiers_left, tiers_right = tier_detector.conv_tiers(time_name, time_time)

    return tiers_left, tiers_right


def dump_raw_statistics(output_file):
    output_file = Path(output_file)

    files = list(EAF_DIR.glob('*.eaf'))
    single_speaker_ids = []
    data = []

    with ProcessPoolExecutor() as executor:
        future_list = []

        for file in files:
            future = executor.submit(find_videos, file)
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
    labels_file = Path(labels_file)

    if single_speakers_file is None:
        single_speakers_file = (labels_file.parent / 'single_speakers').with_suffix(labels_file.suffix)

    with open(labels_file, 'rb') as input_handle:
        data = pickle.load(input_handle)
    with open(single_speakers_file, 'rb') as input_handle:
        single_speaker_ids = pickle.load(input_handle)

    return data, single_speaker_ids


def plot_statistics(data):
    labels_valid = ['N', 'Nf', 'Nx', 'Nn', 'Ns', 'Nsx']
    labels_valid.extend(['?' + label for label in labels_valid])

    speaker_to_n_annotations = {}
    n_videos = 0
    n_total_videos = 0
    n_annotations = 0
    seconds = []
    seconds_valid = []
    labels = []

    for video in data:
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

    clean_labels = [label if label in labels_valid else 'other' for label in labels]
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
    occurrences = {'Sure': [counts['N'], counts['Nf'], counts['other'], counts['Nx'],
                            counts['Ns'], counts['Nn'], counts['Nsx']],
                   'Unsure': [counts['?N'], counts['?Nf'], 0, counts['?Nx'],
                              counts['?Ns'], counts['?Nn'], counts['?Nsx']]}

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
    plt.xticks(range(len(occurrences['Sure'])), ['N', 'Nf', 'other', 'Nx', 'Ns', 'Nn', 'Nsx'],
               rotation=70, ha='center', fontsize=10)
    plt.xlim(-0.5, len(occurrences['Sure']) - 0.5)
    ax.xaxis.grid(False)
    at = textonly(plt.gca(),
                  "N   --> head-shake negation during sign\n"
                  "Nf  --> head-shake negation without sign\n"
                  "Nx  --> head-shake negation between signs\n"
                  "Ns  --> head-sway negation\n"
                  "Nn  --> head-shake no negation during sign\n"
                  "Nsx --> head-sway no negation",
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

    ...


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


if __name__ == '__main__':
    plot_statistics(load_raw_statistics(
        Path(r'C:\Users\casva\PycharmProjects\Thesis\data\statistics\labels.pickle'))[0]
                    )
