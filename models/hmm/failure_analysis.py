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
    video_n = 0

    start = 0

    for index in indices[start:]:
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('frames_csv', type=Path)
    parser.add_argument('predictions_csv', type=Path)
    parser.add_argument('-f', '--filter_size', type=int)
    args = parser.parse_args()

    review_dataset(args.frames_csv, args.predictions_csv, filter_size=args.filter_size)
