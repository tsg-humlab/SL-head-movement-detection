import cv2
import mediapipe as mp


class MediaPipePoseDetector:
    def __init__(self, static_image_mode=False, model_complexity=0, smooth_landmarks=True,
                 enable_segmentation=False, smooth_segmentation=True, min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):

        self.static_image_mode = static_image_mode,
        self.model_complexity = model_complexity,
        self.smooth_landmarks = smooth_landmarks,
        self.enable_segmentation = enable_segmentation,
        self.smooth_segmentation = smooth_segmentation,
        self.min_detection_confidence = min_detection_confidence,
        self.min_tracking_confidence = min_tracking_confidence

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(static_image_mode, model_complexity, smooth_landmarks, enable_segmentation,
                                     smooth_segmentation, min_detection_confidence, min_tracking_confidence)

        self.results = None

    def find_pose(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(img_rgb)

        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

        return img

    def find_position(self, img, draw=True):
        lm_list = []
        if self.results.pose_landmarks:
            for landmark_id, landmark in enumerate(self.results.pose_landmarks.landmark):
                height, width, channels = img.shape
                cx, cy = int(landmark.x * width), int(landmark.y * height)
                lm_list.append([landmark_id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 3, (0, 0, 255), cv2.FILLED)
        return lm_list
