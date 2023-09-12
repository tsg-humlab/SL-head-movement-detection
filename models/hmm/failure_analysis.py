import argparse
from pathlib import Path
import random
import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from models.hmm.filters import majority_filter
from models.hmm.prediction_visualisation import load_predictions, flips_indices
from models.simple.detector import verify_window_size
from pose.review_conflicts import show_frame
from utils.array_manipulation import get_uninterrupted_ones
from utils.frames_csv import load_df, get_splits, load_all_labels
from utils.io import mkdir_if_not_exists


def prepare_evaluation(frames_csv, predictions_path, window=48):
    window = verify_window_size(window)
    cut = int((window - 1) / 2)

    df_frames = load_df(frames_csv)
    predictions = load_predictions(predictions_path)
    df_val = df_frames.iloc[:0].copy()

    splits = get_splits(df_frames)
    for fold in splits:
        df_val = pd.concat([df_val, df_frames[df_frames['split'] == fold]], ignore_index=True)
    labels = load_all_labels(df_val, shift=1, window=window)

    return df_val, cut, predictions, labels


def review_dataset(frames_csv, predictions_path, window=48, filter_size=None):
    df_val, cut, predictions, labels = prepare_evaluation(frames_csv, predictions_path, window)

    indices = list(df_val.index)
    random.Random(14).shuffle(indices)
    video_n = 49

    for index in indices[video_n:]:
        video_n += 1

        prediction = predictions[index]
        label = labels[index]

        if filter_size:
            prediction = majority_filter(prediction, filter_size)

        capture = cv2.VideoCapture(df_val.iloc[index]['media_path'])
        add = cut + df_val.iloc[index]['start_frame']

        prediction_indices = get_group_indices(prediction)
        truth_indices = get_group_indices(label)

        sentinel = True
        show_preds = True
        show_truths = True
        while sentinel:
            if show_truths:
                prediction_index = 0

                for pair in truth_indices:
                    prediction_index += 1

                    for i in range(*pair):
                        acc = round(accuracy_score(label[pair[0]:pair[1]], prediction[pair[0]:pair[1]]) * 100, 1)

                        frame = show_frame(capture, i + add, return_frame=True)

                        cv2.putText(frame,
                                    f'Video {video_n}/{len(indices)} ({df_val.iloc[index]["video_id"]})',
                                    (50, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1,
                                    (255, 255, 255),
                                    2)

                        cv2.putText(frame,
                                    f'Truth {prediction_index}/{len(truth_indices)}',
                                    (50, 100),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1,
                                    (255, 255, 255),
                                    2)

                        cv2.putText(frame,
                                    f'Accuracy: {acc}%',
                                    (50, 150),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1,
                                    (255, 255, 255),
                                    2)

                        cv2.imshow('Predictions', frame)
                        cv2.waitKey(1)

            if show_preds:
                prediction_index = 0

                for pair in prediction_indices:
                    prediction_index += 1

                    for i in range(*pair):
                        acc = round(accuracy_score(label[pair[0]:pair[1]], prediction[pair[0]:pair[1]]) * 100, 1)

                        if acc != 0:
                            continue

                        frame = show_frame(capture, i + add, return_frame=True)

                        cv2.putText(frame,
                                    f'Video {video_n}/{len(indices)} ({df_val.iloc[index]["video_id"]})',
                                    (50, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1,
                                    (255, 255, 255),
                                    2)

                        cv2.putText(frame,
                                    f'Prediction {prediction_index}/{len(prediction_indices)}',
                                    (50, 100),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1,
                                    (255, 255, 255),
                                    2)

                        cv2.putText(frame,
                                    f'Accuracy: {acc}%',
                                    (50, 150),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1,
                                    (255, 255, 255),
                                    2)

                        cv2.imshow('Predictions', frame)
                        cv2.waitKey(1)

            response = input('Continue? Y/n/t')
            if response.lower() != 'n':
                sentinel = False
                show_preds = True
            if response.lower() == 't':
                sentinel = True
                show_preds = False

        capture.release()
        cv2.destroyAllWindows()


def get_group_indices(array, target=0):
    return np.flatnonzero(np.diff(np.r_[target, array, target]) != 0).reshape(-1, 2) - [0, 1]


def play_case(media_file):
    media_file = Path(media_file)

    idx = 0
    capture = cv2.VideoCapture(str(media_file))

    while capture.isOpened():
        ret, frame = capture.read()

        if ret:
            cv2.imshow(media_file.stem, frame)
            cv2.waitKey(1)

            idx += 1
        else:
            break

    capture.release()
    cv2.destroyAllWindows()


def extract_prediction(frames_csv,
                       predictions_path,
                       video_id,
                       prediction_index,
                       output_dir,
                       context_seconds=2,
                       window=48,
                       filter_size=None):
    df_val, cut, predictions, labels = prepare_evaluation(frames_csv, predictions_path, window)

    video_index = df_val.index[df_val['video_id'] == video_id].to_list()[0]
    prediction = predictions[video_index]
    if filter_size:
        prediction = majority_filter(prediction, filter_size)

    prediction_indices = get_group_indices(prediction)
    group = prediction_indices[prediction_index]

    start = int(df_val.iloc[video_index]['start_frame'])
    mkdir_if_not_exists(output_dir)
    media_path = Path(df_val.iloc[video_index]['media_path'])
    write_video(str(media_path),
                str(output_dir / media_path.name),
                group[0] + start + cut,
                group[1] + start + cut,
                context_seconds * 24)


def write_video(infile, outfile, start, end, context=0):
    """Credit to Red:
    https://stackoverflow.com/questions/67349721/how-to-save-specific-parts-of-video-into-separate-videos-using-opencv-in-python
    """
    cap = cv2.VideoCapture(infile)
    ret, frame = cap.read()
    h, w, _ = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(outfile, fourcc, 20.0, (w, h))

    f = 0
    while ret:
        f += 1
        if (start - context) <= f <= (end + context):
            if f < start:
                cv2.putText(frame,
                            'PRE CONTEXT',
                            (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (255, 255, 255),
                            2)
            elif f > end:
                cv2.putText(frame,
                            'POST CONTEXT',
                            (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (255, 255, 255),
                            2)
            else:
                cv2.putText(frame,
                            'PREDICTION',
                            (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (255, 255, 255),
                            2)

            writer.write(frame)
        ret, frame = cap.read()

    writer.release()
    cap.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('frames_csv', type=Path)
    parser.add_argument('predictions_csv', type=Path)
    parser.add_argument('-f', '--filter_size', type=int)
    subparsers = parser.add_subparsers(dest='subparser')

    extract_parser = subparsers.add_parser('extract')
    extract_parser.add_argument('video_id', type=str)
    extract_parser.add_argument('prediction_index', type=int)
    extract_parser.add_argument('output_dir', type=Path)
    extract_parser.add_argument('--context_seconds', '-s', type=int, default=2)

    args = parser.parse_args()

    if args.subparser == 'extract':
        extract_prediction(args.frames_csv,
                           args.predictions_csv,
                           args.video_id,
                           args.prediction_index,
                           args.output_dir,
                           context_seconds=args.context_seconds,
                           filter_size=args.filter_size)
    else:
        review_dataset(args.frames_csv, args.predictions_csv, filter_size=args.filter_size)
