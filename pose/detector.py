import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import torch

from pose import KEYPOINTS, BOXES, KEYPOINT_VIDEO, HEADPOSES, HEADBOXES, POSE_VIDEO
from utils.array_manipulation import stack_with_padding
from utils.config import Config
from utils.media import get_metadata
from lightweight_hpe.network import Network, load_snapshot
from ultralytics import YOLO
from torchvision import transforms
from PIL import Image
from lightweight_hpe.camera_normalize import drawAxis, printWords
from IPython.display import clear_output


def process_dataset(output_dir, overview_path=None, log_file='log.txt', model_name='yolov8n-pose.pt'):
    """Make pose predictions over an entire dataset using YOLO and pose detection models.

    The overview file can either be provided explicitly or it will be read out of the config.yml file.

    :param output_dir: Directory where the predictions should be stored
    :param overview_path: Path to the overview CSV
    :param log_file: Log file with success/error statements for every video in the overview (will be overwritten!)
    :param model_name: Name of the YOLO checkpoint that should be used (default nano)
    """
    if overview_path is None:
        config = Config()
        overview_path = config.content['overview']
    df_overview = pd.read_csv(overview_path)

    create_subdirs(output_dir)

    log_handle = open(log_file, 'w')

    for rowi, row in df_overview.iterrows():
        clear_output()
        print(rowi)
        # when using frames.csv...
        unique_id = f"{row['video_id']}"
        video_path = f"data/videos/{unique_id}.mp4"

        # Check if the headbox predictions already exist
        if not os.path.exists(f"data/pose_detections/headboxes/{unique_id}.npy"):
            # noinspection PyBroadException
            try:
                process_video_headposes(unique_id, video_path, output_dir)
                log_handle.write(f'Successfully processed headposes of {unique_id}\n')
            except Exception as err:
                log_handle.write(f'Error processing headposes of {unique_id}\n')
                log_handle.write(str(err) + '\n')

        # Check if the headbox predictions already exist
        if not os.path.exists(f"data/pose_detections/boxes/{unique_id}.npy"):
            # noinspection PyBroadException
            try:
                process_video_keypoints(unique_id, row['media_path'], output_dir, model_name=model_name)
                log_handle.write(f'Successfully processed keypoints of {unique_id}\n')
            except Exception as err:
                log_handle.write(f'Error processing keypoints of {unique_id}\n')
                log_handle.write(str(err) + '\n')

    log_handle.close()


def create_subdirs(path):
    """Create (sub)directories for the prediction outputs.

    If any directories already exists, they will be ignored without raising an exception.

    :param path: Path to the output directory
    """
    path = Path(path)

    if not os.path.exists(path):
        os.mkdir(path)
    if not os.path.exists(path / KEYPOINT_VIDEO):
        os.mkdir(path / KEYPOINT_VIDEO)
    if not os.path.exists(path / BOXES):
        os.mkdir(path / BOXES)
    if not os.path.exists(path / KEYPOINTS):
        os.mkdir(path / KEYPOINTS)
    if not os.path.exists(path / POSE_VIDEO):
        os.mkdir(path / POSE_VIDEO)
    if not os.path.exists(path / HEADBOXES):
        os.mkdir(path / HEADBOXES)
    if not os.path.exists(path / HEADPOSES):
        os.mkdir(path / HEADPOSES)


def process_video_keypoints(unique_id, video_path, output_dir, model_name='yolov8n-pose.pt', custom_fps=False, save_video=False):
    """Make pose predictions on a single video.

    I recommend using a GPU for this process, although the nano model can still be run in a reasonable time window
    through only a CPU.

    :param unique_id: Unique identifier for the video (results with the same ID will be overwritten!)
    :param video_path: Path to the video file
    :param output_dir: Directory to store the results in (subdirs will be created if necessary)
    :param model_name: Name of the YOLO checkpoint that should be used (default nano)
    """
    
    print("Processing keypoints ", unique_id)

    create_subdirs(output_dir)
    model = YOLO(model_name)

    duration, fps = get_metadata(video_path)
    if custom_fps:
        fps = custom_fps
    n_frames = round(duration * fps)

    capture = cv2.VideoCapture(video_path)
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_file = output_dir / Path(KEYPOINT_VIDEO) / f'{unique_id}.mp4'
        output_video = cv2.VideoWriter(str(output_file), fourcc, fps, (int(capture.get(3)), int(capture.get(4))))

    idx = 1

    boxes_list = []
    keypoints_list = []

    while capture.isOpened():
        ret, frame = capture.read()

        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model(frame_rgb, verbose=False)

            boxes = results[0].boxes.data.cpu().numpy()
            keypoints = results[0].keypoints.data.cpu().numpy()
            n_results = boxes.shape[0]

            if n_results == 0:
                boxes_list.append(np.zeros((1, 6)))
                keypoints_list.append(np.zeros((1, 17, 3)))
            else:
                boxes_list.append(boxes)
                keypoints_list.append(keypoints)

            output_frame = results[0].plot()
            output_frame = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)
            if save_video:
                output_video.write(output_frame)

            idx += 1
        else:
            break

    capture.release()
    if save_video:
        output_video.release()

    boxes = stack_with_padding(boxes_list)
    keypoints = stack_with_padding(keypoints_list)

    np.save(output_dir / Path(BOXES) / f'{unique_id}.npy', boxes)
    np.save(output_dir / Path(KEYPOINTS) / f'{unique_id}.npy', keypoints)

    print("finished keypoints ", unique_id)

    assert n_frames == boxes.shape[0]
    assert n_frames == keypoints.shape[0]



def scale_bbox(bbox, scale):
    """
    Scale the bounding box (to a square box) by a given factor.
    """
    w = max(bbox[2], bbox[3]) * scale
    return np.asarray([bbox[0],bbox[1],w,w],np.int64)


def process_video_headposes(unique_id, video_path, output_dir, face_model='yolov8n-face.pt', headpose_model='lightweight_hpe/model-b66.pkl', custom_fps=False):
    """
    credits: https://github.com/Shaw-git/Lightweight-Head-Pose-Estimation/tree/main
    Make head pose predictions (pitch, yaw, roll) on a single video.

    :param unique_id: Unique identifier for the video (results with the same ID will be overwritten!)
    :param video_path: Path to the video file
    :param output_dir: Directory to store the results in (subdirs will be created if necessary)
    :param model_name: Name of the model that will be used for the predictions (default lightweight_hpe)
    """
    create_subdirs(output_dir)

    print("Processing poses ", unique_id)
    # Load the models
    face_detector = YOLO(face_model)
    pose_estimator = Network(bin_train=False)
    load_snapshot(pose_estimator,headpose_model)
    pose_estimator = pose_estimator.eval()
    
    transform_test = transforms.Compose([transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                              std=[0.229, 0.224, 0.225])])

    duration, fps = get_metadata(video_path)
    if custom_fps:
        fps = custom_fps
    n_frames = round(duration * fps)

    capture = cv2.VideoCapture(video_path)
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # output_file = output_dir / Path(POSE_KEYPOINT_VIDEO) / f'{unique_id}.mp4'
    # output_video = cv2.VideoWriter(str(output_file), fourcc, fps, (int(capture.get(3)), int(capture.get(4))))

    headposes_list, headboxes_list = [], []

    while capture.isOpened():
        ret, frame = capture.read()

        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        headpose = [0,0,0]
        bbox = [0,0,0,0]

        # Detect faces
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
        # if output_video is not None:
        #     output_frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        #     output_video.write(output_frame)

    capture.release()
    # output_video.release()

    np.save(output_dir / Path(HEADBOXES) / f'{unique_id}.npy', np.array(headboxes_list))
    np.save(output_dir / Path(HEADPOSES) / f'{unique_id}.npy', np.array(headposes_list))

    print("Finished poses ", unique_id)

    assert(n_frames == np.array(headposes_list).shape[0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output-dir', type=Path, required=True)
    parser.add_argument('-v', '--overview-path', type=Path)
    parser.add_argument('-l', '--log-file', type=Path, default='log.txt')
    args = parser.parse_args()

    process_dataset(output_dir=args.output_dir,
                    overview_path=args.overview_path,
                    log_file=args.log_file)
