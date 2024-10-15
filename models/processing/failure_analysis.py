import argparse
from pathlib import Path
import random
import cv2
import numpy as np
import pandas as pd
import math
from sklearn.metrics import accuracy_score

from models.processing.filters import majority_filter
from models.processing.facial_movement import _single_pitch_yaw_roll_internal
from models.hmm.prediction_visualisation import load_predictions
from models.simple.detector import verify_window_size
from pose.review_conflicts import show_frame
from pose import KEYPOINTS
from utils.frames_csv import load_df, get_splits, load_all_labels
from utils.io import mkdir_if_not_exists


def prepare_evaluation(frames_csv, predictions_path, window=36, test=False):
    window = verify_window_size(window)
    cut = int((window - 1) / 2)

    df_frames = load_df(frames_csv)
    predictions = load_predictions(predictions_path)
    df_val = df_frames.iloc[:0].copy()

    if not test:
        splits = get_splits(df_frames)
    else:
        splits = ['test']
    for fold in splits[4:5]:
        print("Preparing fold ", fold)
        df_val = df_frames[df_frames['split'] == fold].reset_index()
        # df_val = pd.concat([df_val, df_frames[df_frames['split'] == fold]], ignore_index=True)
    labels = load_all_labels(df_val, shift=1, window=window)
    print("Len labels, len preds", len(labels[0]), len(predictions[0]))

    return df_val, cut, predictions, labels


def get_group_indices(array, target=0, shake = True, nod = False):
    """
    Get the indices (start and end) of the groups of a target value in an array.
    """
    if shake:
        indices_shake = np.array(array, copy=True)  
        indices_shake[indices_shake != 1] = 0
        indices_shake = np.r_[target, indices_shake, target]
        indices_shake = np.where(np.diff(indices_shake) != 0)[0]
        indices_shake = indices_shake.reshape(-1, 2) - [0, 1]
        if shake and not nod:
            return indices_shake
    
    if nod:
        indices_nod = np.array(array, copy=True)  
        indices_nod[indices_nod != 2] = 0
        indices_nod[indices_nod != 0] = 1
        indices_nod = np.r_[target, indices_nod, target]
        indices_nod = np.where(np.diff(indices_nod) != 0)[0]
        indices_nod = indices_nod.reshape(-1, 2) - [0, 1] 
        if nod and not shake:
            return indices_nod
        
    if not shake and not nod:
        indices_bg = np.array(array, copy=True)
        indices_bg[indices_bg != 0] = 1
        indices_bg = np.r_[target, indices_bg, target]
        indices_bg = np.where(np.diff(indices_bg) != 0)[0]
        indices_bg = indices_bg.reshape(-1, 2) - [0, 1]
        return indices_bg
    
    return indices_shake, indices_nod


def extract_predicted_video(frames_csv, predictions_path, video_id, output_dir, window=35, filter_size=None, test=False,
                            show_preds=True, show_truths=True, show_keypoints=True, show_pose=True):
    
    if predictions_path is not None:
        """
        Extract all predictions and truths and show them on top of the video.
        """
        df_val, cut, predictions, labels = prepare_evaluation(frames_csv, predictions_path, window)
        df_val.head()
        video_index = df_val.index[df_val['video_id'] == video_id].to_list()
        print(video_index)
        if len(video_index) == 0:
            print(f'Video {video_id} not found.')
            return
        
        video_index = video_index[0]
        label = labels[video_index]
        nod_label, shake_label = np.array(label), np.array(label)
        truth_indices_nod = get_group_indices(nod_label, shake=False, nod=True)
        truth_indices_shake = get_group_indices(shake_label, shake=True, nod=False)

        prediction = predictions[video_index]

        # Filtering was already done in the validation/prediction step

        # Separate nod and shake predictions
        nod_predictions, shake_predictions = np.array(prediction), np.array(prediction)
        prediction_indices_nod = get_group_indices(nod_predictions, shake=False, nod=True)
        prediction_indices_shake = get_group_indices(shake_predictions, shake=True, nod=False)
        print("Len prediction: ", len(prediction), ", Len labels: ", len(label))

    else:
        cut = 0
        df_val = load_df(frames_csv)
        video_index = df_val.index[df_val['video_id'] == video_id].to_list()[0]

    keypoints_path = df_val.iloc[video_index]['keypoints_path']
    headboxes_path = df_val.iloc[video_index]['headboxes_path']
    headposes_path = df_val.iloc[video_index]['headposes_path']

    keypoints = np.load(keypoints_path)
    headboxes = np.load(headboxes_path)
    headposes = np.load(headposes_path)

    infile = Path(df_val.iloc[video_index]['media_path'])
    outfile = Path(output_dir) / infile.name
    # TO FIX: hacky
    outfile = Path('data/results/'+video_id+'.mp4')
    capture = cv2.VideoCapture(str(infile))
    # TO FIX: hacky
    capture = cv2.VideoCapture('data/videos/'+video_id+'.mp4')
    # capture = cv2.VideoCapture('data/test_pose/'+video_id)
    ret, frame = capture.read()
    h, w, _ = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(str(outfile), fourcc, 20.0, (w, h))

    frames_to_begin = int(df_val.iloc[video_index]['start_frame'])
    frames_to_end = int(df_val.iloc[video_index]['end_frame'])
    print("Cut: ", cut, ", Frame start: ", frames_to_begin, ", Frame end: ", frames_to_end)
    print("Nr frames start to end: ", frames_to_end - frames_to_begin)

    f = 0
    while ret:
        if f >= frames_to_begin and f < frames_to_end:
            if show_truths:
                cv2.putText(frame, 'ANNOTATED', (200, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                for pair in truth_indices_nod:
                    for i in range(pair[0], pair[1]+1):
                        if f == i + frames_to_begin + cut:
                            cv2.putText(frame, 'NOD', (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                for pair in truth_indices_shake:
                    for i in range(pair[0], pair[1]+1):
                        if f == i + frames_to_begin + cut:
                            cv2.putText(frame, 'SHAKE', (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if show_preds:
                cv2.putText(frame, 'PREDICTED', (1000,150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                for pair in prediction_indices_nod:
                    for i in range(pair[0], pair[1]+1):
                        if f == i + frames_to_begin + cut:
                            cv2.putText(frame, 'NOD', (1000, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                for pair in prediction_indices_shake:
                    for i in range(pair[0], pair[1]+1):
                        if f == i + frames_to_begin + cut:
                            cv2.putText(frame, 'SHAKE', (1000, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if show_keypoints:
                for i in range(7):
                    x = keypoints[f][0][i][0]
                    y = keypoints[f][0][i][1]
                    if (x,y) != (0,0):
                        cv2.circle(frame, (int(x), int(y)), 3, (255, 255, 255), -1)
                        cv2.putText(frame, str(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 2)

            if show_pose:
                center_face = (headboxes[f][0], headboxes[f][1])  #bbox = [center_x, center_y, width, height]

                # Add center_bb as parameter in case of using the bounding boxes
                _, _, _, _, s_angle = _single_pitch_yaw_roll_internal(keypoints[f][0], center_face)
                (z_angle,y_angle,x_angle) = headposes[f] #headpose = [r,y,p]
                z_angle = -z_angle
                y_angle = -y_angle
                x_angle = -x_angle
                
                if center_face != (0,0):
                    # show headbox
                    cv2.rectangle(frame, (int(center_face[0]-headboxes[f][2]/2), int(center_face[1]-headboxes[f][3]/2)), (int(center_face[0]+headboxes[f][2]/2), int(center_face[1]+headboxes[f][3]/2)), (255, 255, 255), 2)

                    # show center of headbox
                    cv2.circle(frame, center_face, 3, (0, 0, 0), -1)
                    
                    # Show pitch, yaw, roll feedback in video
                    cv2.putText(frame, 'Yaw', (50, 550), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.putText(frame, str(int(y_angle)), (150, 550), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.putText(frame, 'Pitch', (50, 600), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.putText(frame, str(int(x_angle)), (150, 600), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.putText(frame, 'Roll', (50, 650), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.putText(frame, str(int(z_angle)), (150, 650), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.putText(frame, 'Shoulder', (50, 700), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.putText(frame, str(int(s_angle)), (200, 700), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                    # See where the user's head tilting
                    if y_angle > 10:
                        text = "Looking Left"
                    elif y_angle < -10:
                        text = "Looking Right"
                    elif x_angle > 10:
                        text = "Looking Down"
                    elif x_angle < -10:
                        text = "Looking Up"
                    else:
                        text = "Forward"
                    cv2.putText(frame, text, (50, 500), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    
                    # Draw line to show head tilt
                    p1 = (center_face[0], center_face[1])
                    p2 = (int(center_face[0]+y_angle*5), int(center_face[1]+x_angle*5))
                    cv2.line(frame, p1, p2, (255, 255, 255), 3)

            writer.write(frame)
        ret, frame = capture.read()
        f += 1

    capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('frames_csv', type=Path)
    parser.add_argument('predictions_csv', type=Path)
    parser.add_argument('video_id', type=str)
    parser.add_argument('output_dir', type=Path)
    parser.add_argument('-f', '--filter_size', type=int)
    parser.add_argument('-t', '--test', action='store_true')
    parser.add_argument('-p', '--show_preds', action='store_true')
    parser.add_argument('-l', '--show_truths', action='store_true')
    parser.add_argument('-k', '--show_keypoints', action='store_true')
    parser.add_argument('-s', '--show_pose', action='store_true')

    args = parser.parse_args()

    extract_predicted_video(args.frames_csv, args.predictions_csv, args.video_id, args.ouput_dir, args.filter_size, \
                            args.test, args.show_preds, args.show_truths, args.show_keypoints, args.show_pose)