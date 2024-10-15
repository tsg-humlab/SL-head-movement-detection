import argparse
from pathlib import Path
import random
import cv2
import numpy as np
import pandas as pd
import math
import torch
import pickle
import os

from pose.detector import scale_bbox
from models.processing.facial_movement import calc_pitch_yaw_roll
from models.simple.detector import verify_window_size
from models.processing.filters import majority_filter
from dataset.labels.eaf_parser import Annotation
from models.hmm.test_hmm import BACKGROUND_HMM_FILENAME, SHAKE_HMM_FILENAME, NOD_HMM_FILENAME

from lightweight_hpe.network import Network, load_snapshot
from ultralytics import YOLO
from torchvision import transforms
from PIL import Image
from lightweight_hpe.camera_normalize import drawAxis, printWords
from IPython.display import clear_output
from pympi.Elan import Eaf


def process_frames(video):
    """ Process the frames of a video to extract head poses and facial keypoints. """

    # Load the models
    face_detector = YOLO('yolov8n-face.pt')
    face_keypoint_detector = YOLO('yolov8n-pose.pt')
    pose_estimator = Network(bin_train=False)
    load_snapshot(pose_estimator,'lightweight_hpe/model-b66.pkl')
    pose_estimator = pose_estimator.eval()
    
    # Define the transformations
    transform_test = transforms.Compose([transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                              std=[0.229, 0.224, 0.225])])

    # Process the frames
    capture = cv2.VideoCapture(str(video))
    headposes_list, headboxes_list, boxes_list, keypoints_list = [], [], [], []
    while capture.isOpened():
        ret, frame = capture.read()

        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_keypoint_detector(frame_rgb, verbose=False)

        # Extract the bounding boxes and keypoints
        boxes = results[0].boxes.data.cpu().numpy()
        keypoints = results[0].keypoints.data.cpu().numpy()
        n_results = boxes.shape[0]

        if n_results == 0:
            boxes = np.zeros((1, 6))
            keypoints = np.zeros((1, 17, 3))

        boxes_list.append(boxes)
        keypoints_list.append(keypoints)

        # Detect faces and headposes
        headpose = [0,0,0]
        bbox = [0,0,0,0]
        faces = face_detector(frame_rgb, verbose=False)
        boxes = faces[0].boxes
        if boxes:
            top_left_x = int(boxes[0].xyxy.tolist()[0][0])
            top_left_y = int(boxes[0].xyxy.tolist()[0][1])
            bottom_right_x = int(boxes[0].xyxy.tolist()[0][2])
            bottom_right_y = int(boxes[0].xyxy.tolist()[0][3])
            width = bottom_right_x - top_left_x
            height = bottom_right_y - top_left_y
            center_x = int((top_left_x+bottom_right_x)/2)
            center_y = int((top_left_y+bottom_right_y)/2)
            bbox = [center_x, center_y, width, height]

            # draw the bounding box and estimate the head pose
            face_images, face_tensors = [], []
            x,y, w,h = scale_bbox(bbox,1.5)
            frame_rgb = cv2.rectangle(frame_rgb,(x-int(w/2),y-int(h/2)), (x+int(w/2), y+int(h/2)),color=(0,0,255),thickness=2)
            face_img = frame_rgb[y-int(h/2):y+int(h/2),x-int(w/2):x+int(w/2)]
            face_images.append(face_img)
            
            try:
                pil_img = Image.fromarray(cv2.cvtColor(cv2.resize(face_img,(224,224)), cv2.COLOR_BGR2RGB))
                face_tensors.append(transform_test(pil_img)[None])

                if len(face_tensors)>0:
                    with torch.no_grad():
                        face_tensors = torch.cat(face_tensors,dim=0)
                        roll, yaw, pitch = pose_estimator(face_tensors)
                        for _, r,y,p in zip(face_images, roll,yaw,pitch):
                            headpose = [r,y,p]
                            drawAxis(frame_rgb, headpose, center=(center_x, center_y))
                            printWords(frame_rgb, r, y, p)
            except:
                print("Error in processing face image")

        headposes_list.append(headpose)
        headboxes_list.append(bbox)

    capture.release()
    
    headposes_list, headboxes_list, boxes_list, keypoints_list = np.array(headposes_list), np.array(headboxes_list), np.array(boxes_list), np.array(keypoints_list)
    return headposes_list, headboxes_list, boxes_list, keypoints_list

def make_predictions(vector, window_size):
    """ Make predictions for a given vector using the trained HMMs. """

    window_size = verify_window_size(window_size)

    with open(Path('data/folds/fold_5') / BACKGROUND_HMM_FILENAME, 'rb') as input_handle:
        hmm_bg = pickle.load(input_handle)
    with open(Path('data/folds/fold_5') / SHAKE_HMM_FILENAME, 'rb') as input_handle:
        hmm_shake = pickle.load(input_handle)
    with open(Path('data/folds/fold_5') / NOD_HMM_FILENAME, 'rb') as input_handle:
        hmm_nod = pickle.load(input_handle)

    pred_len = len(vector) - window_size + 1

    windows = []
    for i in range(pred_len):
        windows.append(vector[i:i + window_size])

    windows = np.stack(windows)
    pred_bg = hmm_bg.forward_backward(windows)
    pred_shake = hmm_shake.forward_backward(windows)
    pred_nod = hmm_nod.forward_backward(windows)
    log_probs = np.stack([pred_bg[4].cpu().detach().numpy(), pred_shake[4].cpu().detach().numpy(), pred_nod[4].cpu().detach().numpy()])

    predictions = np.argmax(log_probs, axis=0)
    return predictions


def main(video):
    capture = cv2.VideoCapture(str(video))
    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps != 25:
        print(f"FPS: {fps}")
        print("Converting video to 25 FPS...")
        import subprocess
        output_video = str(video).split('.')[0] + '_25fps.mp4'
        os.system(f'ffmpeg -y -i {video} -r 25 -loglevel quiet {output_video}')
        video = output_video
    capture.release()
    fps = 25
    
    print("Processing video...")
    headposes_list, headboxes_list, boxes_list, keypoints_list = process_frames(output_video)
    _, _, _, _, pitch, yaw, roll, shoulder = calc_pitch_yaw_roll(keypoints_list, boxes_list, headposes_list)
    vector = np.stack([pitch, yaw, roll, shoulder], axis=1)

    print("Making predictions...")
    window_size = 37
    predictions = make_predictions(vector, window_size)
    predictions = majority_filter(predictions, 7)

    print("Converting predictions to annotations...")
    annotations = []
    start_frame, start_label = 0, 'background'
    for pred_i, pred in enumerate(predictions):
        pred = 'shake' if pred == 1 else 'nod' if pred == 2 else 'background'
        if pred != start_label or pred_i == len(predictions) - 1:
            if start_frame != pred_i and start_label != 'background':
                start_ms = int(start_frame * (1000/fps))
                end_ms = int(pred_i * (1000/fps))
                annotations.append(Annotation(start_label, start_ms, end_ms))
            start_frame = pred_i
            start_label = pred

    print("Saving annotations...")

    new_eaf = Eaf()
    new_eaf.add_tier("Head movement")
    for annotation in annotations:
        new_eaf.add_annotation("Head movement", annotation.start, annotation.end, annotation.label)
    new_eaf.add_linked_file(str(video), str(video), 'video/mp4')
    new_eaf.to_file(str(video).split('.')[0] + '_head_movement.eaf')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('video', type=Path)
    args = parser.parse_args()
    main(args.video)