import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from pose import KEYPOINTS, VIDEO, BOXES
from pose.analyze_results import is_output_dir
from utils.draw import draw_opaque_box


def review_all(results_dir, output_csv, overwrite=False):
    """Review all the videos in a pose prediction output directory.

    :param results_dir: Path to the pose prediction output directory
    :param output_csv: Path to the output CSV
    :param overwrite: True if you want to overwrite an existing CSV
    """
    assert is_output_dir(results_dir)

    if not os.path.exists(output_csv) or overwrite:
        df_output = pd.DataFrame(columns=['case_id', 'comment'])
    else:
        df_output = pd.read_csv(output_csv)

    files = (results_dir / KEYPOINTS).glob('*.npy')

    for file in files:
        media_path = results_dir / VIDEO / f'{file.stem}.mp4'
        boxes_path = results_dir / BOXES / f'{file.stem}.npy'

        if len(df_output[df_output['case_id'].str.contains(media_path.stem)]) > 0:
            continue

        play_case(file, boxes_path, media_path)

        sentinel = True

        while sentinel:
            result = input('Approve case? (y/n)').lower()

            if result == 'y':
                sentinel = False
            elif result == 'n':
                review_case(file, boxes_path, media_path)
            else:
                print('Invalid input')

        comment = input('Comment: ')
        df_output.loc[len(df_output)] = [media_path.stem, comment]
        df_output.to_csv(output_csv, index=False)


def play_case(keypoints_file, boxes_file, media_file):
    keypoints = np.load(keypoints_file)
    boxes = np.load(boxes_file)

    idx = 0
    capture = cv2.VideoCapture(str(media_file))

    while capture.isOpened():
        ret, frame = capture.read()

        if ret:
            target_signer = find_subject(keypoints[idx], boxes[idx])
            target_bbox = boxes[idx, target_signer]
            if target_bbox[4] > 0:
                draw_opaque_box(frame, boxes[idx, target_signer], alpha=0.8)

            cv2.imshow(media_file.stem, frame)
            cv2.waitKey(1)

            idx += 1
        else:
            break

    capture.release()
    cv2.destroyAllWindows()


def review_case(keypoints_file, boxes_file, media_file):
    """Review one video for conflicts (moments where more than one person is predicted to be in the video).

    :param keypoints_file: Path to a numpy file with predicted keypoints
    :param boxes_file: Path to a numpy file with predicted bounding boxes
    :param media_file: Path to the video file
    """
    keypoints = np.load(keypoints_file)
    boxes = np.load(boxes_file)
    if keypoints.shape[1] == 1:
        return True

    capture = cv2.VideoCapture(str(media_file))
    empty_prediction = np.zeros((17, 3))

    for i, frame in enumerate(keypoints):
        if not np.array_equal(frame[1], empty_prediction):
            target_signer = find_subject(keypoints[i], boxes[i])

            key = show_frame(capture,
                             i,
                             target_bbox=boxes[i, target_signer, :4],
                             title=media_file.stem)

            if key == ord('q'):
                break
            elif key == ord('d'):
                print('#' * 50)
                print(f'DEBUG frame {i + 1} from {media_file.stem}')
                for j in range(len(keypoints[i])):
                    if not np.array_equal(keypoints[i, j], empty_prediction):
                        print(f'Person conf: {np.round(boxes[i, j, 4], 2)}')
                        print(f'nose       : '
                              f'x = {np.round(keypoints[i, j, 0, 0])} | '
                              f'y = {np.round(keypoints[i, j, 0, 1])} | '
                              f'conf = {np.round(keypoints[i, j, 0, 2], 2)}')
                        print(f'eye (left) : '
                              f'x = {np.round(keypoints[i, j, 0, 0])} | '
                              f'y = {np.round(keypoints[i, j, 0, 1])} | '
                              f'conf = {np.round(keypoints[i, j, 0, 2], 2)}')
                        print(f'eye (right): '
                              f'x = {np.round(keypoints[i, j, 0, 0])} | '
                              f'y = {np.round(keypoints[i, j, 0, 1])} | '
                              f'conf = {np.round(keypoints[i, j, 0, 2], 2)}')
                        print('-' * 50)
                print('#' * 50)
                cv2.waitKey()

    capture.release()
    cv2.destroyAllWindows()


def show_frame(capture, frame_i, target_bbox=None, title='CNGT frame'):
    """Show a specific frame from a capture using the frame index.

    The index starts at 0 and ends at N_frames - 1.

    You can optionally draw an opaque square over a single bounding box or provide a more descriptive title.

    :param capture: OpenCV capture of a video
    :param frame_i: Index of the frame
    :param target_bbox: Bounding box of a person that should be highlighted
    :param title: Title of the window
    """
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_i)
    ret, frame = capture.read()

    if not ret:
        capture.release()
        cv2.destroyAllWindows()

        raise RuntimeError(f"Couldn't load frame {frame_i} from {title}")

    if target_bbox is not None:
        draw_opaque_box(frame, target_bbox, alpha=0.8)

    cv2.imshow(title, frame)
    return cv2.waitKey()


def find_subject(keypoints, boxes):
    """Find the subject of the video using a simple weighted average of the confidence values.

    The eyes and nose of the subject are averaged together and then combined with the bounding box confidence to obtain
    a single confidence value for every person in the frame.

    Note that this function will only work when an image doesn't contain multiple persons facing the camera, if this is
    the case then you should consider the output a random person facing the camera.

    :param keypoints: Keypoint predictions on a single frame
    :param boxes: Bounding box predictions on a single frame
    :return: Index of the subject
    """
    face_conf = np.average(keypoints[:, :3, 2], axis=1)
    bbox_conf = boxes[:, 4]
    weighted_conf = (face_conf + bbox_conf) / 2

    return np.argmax(weighted_conf)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('results_dir', metavar='results-dir', type=Path)
    parser.add_argument('output_json', metavar='output-json', type=Path)
    parser.add_argument('-o', '--overwrite', action='store_true')
    args = parser.parse_args()

    review_all(args.results_dir, args.output_json, overwrite=args.overwrite)
