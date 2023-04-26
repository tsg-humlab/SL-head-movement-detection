import re
import time
from glob import glob
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from pympi.Elan import Eaf

from pose.mp_detector import MediaPipePoseDetector
from utils.exceptions import EAFParsingError
from utils.config import Config

EAF_DIR = Path(r'E:\CorpusNGT\EAF_export_HeadMov')
MEDIA_DIR = Path(r'E:\CorpusNGT\CNGT 720p')


class TierDetector:
    def __init__(self, eaf, fps=25):
        self.eaf = eaf
        self.fps = fps

    def conv_timeslots(self):
        time_name, time_time = [], []
        for key in self.eaf.timeslots:
            time_name.append(key)
            time_time.append(self.eaf.timeslots[key])
        return time_name, time_time

    def conv_tiers(self, time_name, time_time):
        tiers_1, tiers_2 = [], []
        for tier in self.eaf.tiers:
            if "Head movement S1" in tier:
                tiers_1 = self.get_annotations(self.eaf.tiers[tier], self.eaf.timeslots, time_name, time_time)
            if "Head movement S2" in tier:
                tiers_2 = self.get_annotations(self.eaf.tiers[tier], self.eaf.timeslots, time_name, time_time)
        return tiers_1, tiers_2

    def get_ms(self, fr):
        ms_per_frame = 1000 / self.fps
        return fr * ms_per_frame

    def find_annotations(self, fr, tiers):
        nr_ms = int(self.get_ms(fr))
        for tier in tiers:
            if tier[0] < nr_ms < tier[1]:
                return tier[2]

    @staticmethod
    def get_annotations(tier_tiers, tier_timeslots, time_name, time_time):
        tier_list = []
        for tier in tier_tiers[0]:
            ts_begin, ts_end = tier_timeslots[tier_tiers[0][tier][0]], tier_timeslots[tier_tiers[0][tier][1]]
            for t_i, t_t in enumerate(time_time):
                if time_name[t_i] == tier_tiers[0][tier][1] and t_i < (len(time_time) - 1):
                    ts_end = time_time[t_i + 1]
            tier_list.append((ts_begin, ts_end, tier_tiers[0][tier][2]))
        return tier_list


class FaceDetector:
    def __init__(self, confidence=0.5, distance=0):

        self.confidence = confidence
        self.distance = distance

        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.confidence, self.distance)

        self.results = None

    def find_faces(self, img, draw=True, draw_pct=False):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(img_rgb)
        bbox_list = []

        if self.results.detections:
            for face_id, detection in enumerate(self.results.detections):
                bbox_c = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape

                bbox = int(bbox_c.xmin * iw), int(bbox_c.ymin * ih), \
                    int(bbox_c.width * iw), int(bbox_c.height * ih)
                bbox_list.append([bbox, detection.score])

                if draw:
                    self.fancy_draw(img, bbox)
                    if draw_pct:
                        cv2.putText(img, f'{int(detection.score[0] * 100)}%', (bbox[0], bbox[1] - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 225, 0), 2)

        return img, bbox_list

    @staticmethod
    def fancy_draw(img, bbox, rt=2):
        cv2.rectangle(img, bbox, (255, 255, 0), rt)

        return img


def resize_with_aspect_ratio(image, width=None, height=None, inter=cv2.INTER_AREA):
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)


def draw_landmarks(image, face_detector, pose_detector, tier_detector, frame, tiers):
    face_detector.find_faces(image)
    pose_detector.find_pose(image, draw=False)
    landmarks = pose_detector.find_position(image, draw=True)
    if len(landmarks) > 0:
        for i in range(0, 11):
            cv2.circle(image, (landmarks[i][1], landmarks[i][2]), 5, (255, 0, 0), cv2.FILLED)
    gloss_text = tier_detector.find_annotations(frame, tiers)
    if gloss_text is not None:
        cv2.putText(image, gloss_text, (100, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 3)


def find_eaf_and_videos(ngt_id):
    config = Config()

    eaf = Eaf(str(Path(config.content["media"]["eaf"]) / f'{ngt_id}.eaf'), 'pympi')

    sign_videos = glob(str(Path(config.content["media"]["body_720"]) / f'{ngt_id}_S{"[0-9]" * 3}*'))

    if len(sign_videos) == 1:
        raise EAFParsingError(ngt_id=ngt_id, message='This EAF is only linked to one speaker')
    assert len(sign_videos) == 2

    speaker_id = find_speaker_id(sign_videos[0])
    if (int(speaker_id[3]) % 2) == 0:
        video_right, video_left = sign_videos
    else:
        video_left, video_right = sign_videos

    return eaf, video_left, video_right


def find_speaker_id(filepath):
    matches = re.findall(r'S[0-9][0-9][0-9]', str(filepath))

    return matches[-1]


def main(ngt_id):
    eaf, video_left, video_right = find_eaf_and_videos(ngt_id)

    capture_left = cv2.VideoCapture(video_left)
    capture_right = cv2.VideoCapture(video_right)

    p_time = 0
    face_detector_left = FaceDetector(0.70, 1)
    pose_detector_left = MediaPipePoseDetector()
    face_detector_right = FaceDetector(0.70, 1)
    pose_detector_right = MediaPipePoseDetector()

    tier_detector = TierDetector(eaf)
    time_name, time_time = tier_detector.conv_timeslots()
    tiers_left, tiers_right = tier_detector.conv_tiers(time_name, time_time)
    frame = 0
    print("Tiers S1: ", tiers_left)
    print("Tiers S2: ", tiers_right)
    print("-------------------------------")

    sentinel = True

    while sentinel:
        success, image_left = capture_left.read()
        if not success:
            sentinel = False
            continue
        success, image_right = capture_right.read()
        if not success:
            sentinel = False
            continue

        draw_landmarks(image_left, face_detector_left, pose_detector_left, tier_detector, frame, tiers_left)
        draw_landmarks(image_right, face_detector_right, pose_detector_right, tier_detector, frame, tiers_right)

        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time
        cv2.putText(image_left, str(int(fps)), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 225, 0), 3)

        cv2.imshow("Image", resize_with_aspect_ratio(np.hstack([image_left, image_right]), width=1440))
        # cv2.imshow("Image", image_left)
        cv2.waitKey(1)
        frame += 1


if __name__ == '__main__':
    main('CNGT0004')
    # main('CNGT2103')
