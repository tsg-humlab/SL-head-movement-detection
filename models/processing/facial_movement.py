import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import math
from matplotlib import pyplot as plt

from models.hmm import HMM_DECODER
from models.processing.preparation import smoothen_curve
from pose import BOXES, KEYPOINTS, HEADBOXES, HEADPOSES
from utils.array_manipulation import find_subject_video
from utils.media import get_session_from_cngt_file, get_signer_from_cngt_file
from utils.frames_csv import load_df


def create_pose_csv(frames_csv, results_csv, pose_dir):
    """
    Add the paths to the pose files (keypoints and boxes) to the frames csv
    """
    df_frames = load_df(frames_csv)
    keypoints, boxes, headboxes, headposes = [], [], [], []

    for _, row in df_frames.iterrows():
        video_id = Path(row['video_id'])

        keypoints.append(Path(pose_dir) / KEYPOINTS / f'{video_id}.npy')
        boxes.append(Path(pose_dir) / BOXES / f'{video_id}.npy')
        headboxes.append(Path(pose_dir) / HEADBOXES / f'{video_id}.npy')
        headposes.append(Path(pose_dir) / HEADPOSES / f'{video_id}.npy')

    df_frames['keypoints_path'], df_frames['boxes_path'], df_frames['headboxes_path'], df_frames['headposes_path'] = keypoints, boxes, headboxes, headposes
    df_frames.to_csv(results_csv, index=False)


def ordinal_from_csv(frames_csv, threshold):
    """
    Save the movements in the form of ordinal vectors (up/down, left/right) of pitch and yaw
    """
    df_frames = load_df(frames_csv)
    movement_list = []

    for _, row in df_frames.iterrows():
        keypoints_arr = np.load(row['keypoints_path'])
        boxes_arr = np.load(row['boxes_path'])

        pitch, yaw, _, _, _, _, _, _ = calc_pitch_yaw_roll(
            keypoints_arr[row['start_frame']:row['end_frame']],
            boxes_arr[row['start_frame']:row['end_frame']]
        )

        movement_list.append(compute_ordinal_vectors(pitch, yaw, threshold=threshold))
    return movement_list


def derivatives_from_csv(frames_csv, take_diff=False):
    """
    Save the movements in the form of pitch, yaw, roll
    """
    df_frames = load_df(frames_csv)
    movement_list = []

    for _, row in df_frames.iterrows():

        keypoints_arr, boxes_arr = np.load(row['keypoints_path']), np.load(row['boxes_path'])

        try:
            headposes_arr = np.load(row['headposes_path'])
        except:
            headposes_arr = np.array([[0,0,0]] * len(keypoints_arr))

        _, _, _, _, pitch, yaw, roll, shoulder = calc_pitch_yaw_roll(
            keypoints_arr[row['start_frame']:row['end_frame']],
            boxes_arr[row['start_frame']:row['end_frame']],
            headposes_arr[row['start_frame']:row['end_frame']],
            take_diff = take_diff
        )
        movement_list.append(np.stack([pitch, yaw, roll, shoulder], axis=1))
           
    return movement_list


def _focus_keypoints(keypoints, boxes):
    """
    only keep 1 person from the keypoints, those where indices are true
    """
    indices = find_subject_video(keypoints, boxes)
    new_keypoints = []
    for index, kp in enumerate(keypoints):
        trueindex = 0
        for i, x in enumerate(indices[index]):
            if x:
                trueindex = i
        new_keypoints.append(kp[trueindex])
    new_keypoints = np.array(new_keypoints)
    return new_keypoints


def _single_pitch_yaw_roll_internal(keypoints, center_face=None):
    """
    Calculate the pitch, yaw and roll from the keypoints
    """
    # Indexes of keypoints: 0=nose, 1,2=eyes, 3,4=ears, 5,6=shoulders
    frame_keypoints = np.array(keypoints[[0,1,2,3,4,5,6]])
    non_zero_keypoints = frame_keypoints[~np.any(frame_keypoints == 0, axis=1)]
    # if less than 3 keypoints are detected, return 0
    if len(non_zero_keypoints) < 3:
        return (0,0), 0, 0, 0, 0

    # In case center face wasn't determined by face (bounding box) detection, use the keypoints to make an estimate
    if not center_face:
        center_face = np.mean(non_zero_keypoints, axis=0)[0:2]
        center_face_x = int(center_face[0])
        center_face_y = int(center_face[1])-int((keypoints[1][0]-keypoints[2][0])/2)
        center_face = (center_face_x, center_face_y)
    if center_face == (0,0):
        print("No face found")
        return (0,0), 0, 0, 0, 0

    if keypoints[2][0] >= keypoints[1][0]:
        # print("Eyes are switched")
        return (0,0), 0, 0, 0, 0

    # pitch (nose y - center y) / (left eye x - right eye x)
    x_angle = np.arctan((keypoints[0][1] - center_face[1]) / (keypoints[1][0]-keypoints[2][0])) * (180 / math.pi)

    # yaw (midpoint eyes x - center x) / (left eye y - right eye y)
    eyes_shift_x = (keypoints[1][0]-keypoints[2][0])/2
    eyes_x = keypoints[1][0] - eyes_shift_x
    y_angle = np.arctan((eyes_x - center_face[0]) / (keypoints[1][0] - keypoints[2][0])) * (180 / math.pi)

    # roll (right eye - left eye y) / (left eye x - right eye x)
    z_angle = np.arctan((keypoints[1][1] - keypoints[2][1]) / (keypoints[1][0]-keypoints[2][0])) * (180 / math.pi)

    # shoulder roll (right shoulder - left shoulder y) / (left shoulder - right shoulder x)
    shoulder_angle = np.arctan((keypoints[5][1] - keypoints[6][1]) / abs(keypoints[5][0]-keypoints[6][0])) * (180 / math.pi)
    
    if math.isnan(x_angle):
        x_angle = 0
    if math.isnan(y_angle):
        y_angle = 0
    if math.isnan(z_angle):
        z_angle = 0
    if math.isnan(shoulder_angle):
        shoulder_angle = 0

    return center_face, x_angle, y_angle, z_angle, shoulder_angle


def _pitch_yaw_roll_internal(keypoints, headposes, take_diff = False):
    """
    Calculate the pitch, yaw and roll from the keypoints
    """
    # the original values
    pitch_orig = keypoints[:, 0, 1]                             # positive = to the bottom
    yaw_orig = keypoints[:, 0, 0]                               # positive = to the right
    roll_orig = keypoints[:, 1, 1] - keypoints[:, 2, 1]
    shoulder_orig = keypoints[:, 6, 1] - keypoints[:, 5, 1]

    shoulder = []

    # Only shoulder roll (diff in height) is calculated from the keypoints
    # pitch, yaw, roll = [], [], []
    for kp in keypoints:
        _, _, _, _, s = _single_pitch_yaw_roll_internal(kp)
        # pitch.append(p)
        # yaw.append(y)
        # roll.append(r)
        shoulder.append(s)

    # the values retrieved from headpose estimation through the bounding box + CNN
    roll = np.array(headposes[:, 0])
    yaw = np.array(headposes[:, 1])
    pitch = np.array(headposes[:, 2])

    # use the differences between frames
    if take_diff:
        pitch = np.diff(pitch)
        yaw = np.diff(yaw)
        roll = np.diff(roll)
        shoulder = np.diff(shoulder)

    # slightly smoothen the curves to lose some noise
    to_smoothen = [pitch_orig, yaw_orig, roll_orig, shoulder_orig, pitch, yaw, roll, shoulder]
    smoothened_curves = []
    for t_s in to_smoothen:
        smoothened_curves.append(smoothen_curve(t_s, 3))

    return smoothened_curves


def calc_pitch_yaw_roll(keypoints, boxes, headposes, take_diff=False):
    """
    Calculate the pitch, yaw and roll from the keypoints and boxes
    """
    keypoints = _focus_keypoints(keypoints, boxes)
    
    return _pitch_yaw_roll_internal(keypoints, headposes, take_diff=take_diff)


def compute_ordinal_vectors(pitch, yaw, threshold):
    """
    Compute the ordinal up/down vectors from the pitch and yaw vectors
    """
    result = np.argmax(
        np.absolute(
            np.vstack((pitch, yaw, np.full(pitch.shape, threshold)))
        ), axis=0)

    pitch[pitch >= 0] = 2  # down
    pitch[pitch < 0] = 1  # up
    yaw[yaw >= 0] = 4  # right
    yaw[yaw < 0] = 3  # left
    lookup_table = np.vstack((pitch, yaw, np.full(pitch.shape, 0))).astype(int)

    movement = np.zeros(pitch.shape).astype(int)

    for i in range(len(pitch)):
        movement[i] = lookup_table[result[i], i]

    return movement


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('frames_csv', type=Path)
    parser.add_argument('results_csv', type=Path)
    parser.add_argument('results_dir', type=Path)
    args = parser.parse_args()

    create_pose_csv(args.frames_csv, args.results_csv, args.results_dir)
