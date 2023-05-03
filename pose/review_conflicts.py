import argparse
import os
from pathlib import Path

import cv2
import numpy as np

from pose import KEYPOINTS, VIDEO, BOXES
from pose.analyze_results import is_output_dir
from utils.io import write_json


def review_all(results_dir, output_json):
    assert is_output_dir(results_dir)

    if not os.path.exists(output_json):
        write_json({}, output_json)

    files = (results_dir / KEYPOINTS).glob('*.npy')

    for file in files:
        media_path = results_dir / VIDEO / f'{file.stem}.mp4'
        boxes_path = results_dir / BOXES / f'{file.stem}.npy'
        review_case(file, boxes_path, media_path)


def review_case(keypoints_file, boxes_file, media_file):
    keypoints = np.load(keypoints_file)
    boxes = np.load(boxes_file)
    if keypoints.shape[1] == 1:
        return True

    capture = cv2.VideoCapture(str(media_file))
    empty_prediction = np.zeros((17, 3))

    for i, frame in enumerate(keypoints):
        if not np.array_equal(frame[1], empty_prediction):
            key = show_frame(capture, i, media_file.stem)

            if key == ord('q'):
                break
            elif key == ord('d'):
                print('#' * 50)
                print(f'DEBUG frame {i + 1} from {media_file.stem}')
                for j in range(len(keypoints[i])):
                    if not np.array_equal(keypoints[i, j], empty_prediction):
                        print(f'Person conf: {np.round(boxes[i, j, 4], 2)}')
                        print(f'nose       : x = {np.round(keypoints[i, j, 0, 0])} | y = {np.round(keypoints[i, j, 0, 1])} | conf = {np.round(keypoints[i, j, 0, 2], 2)}')
                        print(f'eye (left) : x = {np.round(keypoints[i, j, 0, 0])} | y = {np.round(keypoints[i, j, 0, 1])} | conf = {np.round(keypoints[i, j, 0, 2], 2)}')
                        print(f'eye (right): x = {np.round(keypoints[i, j, 0, 0])} | y = {np.round(keypoints[i, j, 0, 1])} | conf = {np.round(keypoints[i, j, 0, 2], 2)}')
                        print('-' * 50)
                print('#' * 50)
                cv2.waitKey()

    capture.release()
    cv2.destroyAllWindows()


def show_frame(capture, frame_i, title='CNGT frame'):
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_i)
    ret, frame = capture.read()

    if not ret:
        capture.release()
        cv2.destroyAllWindows()

        raise RuntimeError(f"Couldn't load frame {frame_i} from {title}")

    cv2.imshow(title, frame)
    return cv2.waitKey()


def get_frame_ratio(frame, duration, fps):
    return frame / (duration * fps)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('results_dir', metavar='results-dir', type=Path)
    parser.add_argument('output_json', metavar='output-json', type=Path)
    args = parser.parse_args()

    review_all(args.results_dir, args.output_json)
