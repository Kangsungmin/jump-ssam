import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import PoseLandmarkerResult
import cv2
import numpy as np
from pathlib import Path

import config


# Tasks API landmark 인덱스 (COCO 33-point)
LANDMARK_INDEX = {
    "NOSE": 0,
    "LEFT_EYE_INNER": 1, "LEFT_EYE": 2, "LEFT_EYE_OUTER": 3,
    "RIGHT_EYE_INNER": 4, "RIGHT_EYE": 5, "RIGHT_EYE_OUTER": 6,
    "LEFT_EAR": 7, "RIGHT_EAR": 8,
    "MOUTH_LEFT": 9, "MOUTH_RIGHT": 10,
    "LEFT_SHOULDER": 11, "RIGHT_SHOULDER": 12,
    "LEFT_ELBOW": 13, "RIGHT_ELBOW": 14,
    "LEFT_WRIST": 15, "RIGHT_WRIST": 16,
    "LEFT_PINKY": 17, "RIGHT_PINKY": 18,
    "LEFT_INDEX": 19, "RIGHT_INDEX": 20,
    "LEFT_THUMB": 21, "RIGHT_THUMB": 22,
    "LEFT_HIP": 23, "RIGHT_HIP": 24,
    "LEFT_KNEE": 25, "RIGHT_KNEE": 26,
    "LEFT_ANKLE": 27, "RIGHT_ANKLE": 28,
    "LEFT_HEEL": 29, "RIGHT_HEEL": 30,
    "LEFT_FOOT_INDEX": 31, "RIGHT_FOOT_INDEX": 32,
}

POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26),
    (25, 27), (26, 28), (27, 29), (28, 30), (29, 31), (30, 32),
]

# 인원별 스켈레톤 색상 (BGR)
_PERSON_COLORS_BGR = [
    (0, 200, 100),   # 초록
    (0, 100, 255),   # 주황
    (255, 50, 50),   # 파랑
    (200, 0, 255),   # 보라
    (0, 255, 255),   # 노랑
    (255, 255, 0),   # 시안
    (100, 255, 100), # 연두
    (255, 100, 200), # 분홍
]


class PoseResults:
    def __init__(self, result: PoseLandmarkerResult):
        self._result = result
        self.pose_landmarks_list = result.pose_landmarks  # list[list[NormalizedLandmark]]
        self.pose_landmarks = self.pose_landmarks_list[0] if self.pose_landmarks_list else None

    @property
    def num_persons(self) -> int:
        return len(self.pose_landmarks_list)


class PoseAnalyzer:
    def __init__(self):
        model_path = config.MODEL_PATH
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"모델 파일이 없습니다: {model_path}\n"
                f"다음 명령으로 다운로드하세요:\n"
                f"curl -L -o {model_path} {config.MODEL_DOWNLOAD_URL}"
            )

        options = mp_vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp_vision.RunningMode.VIDEO,
            min_pose_detection_confidence=config.POSE_CONFIDENCE,
            min_tracking_confidence=config.POSE_CONFIDENCE,
            num_poses=config.MAX_TRACKED_PERSONS,
            output_segmentation_masks=False,
        )
        self._landmarker = mp_vision.PoseLandmarker.create_from_options(options)
        self._frame_ts_ms = 0

    def process(self, frame) -> PoseResults:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self._frame_ts_ms += int(1000 / config.TARGET_FPS)
        result = self._landmarker.detect_for_video(mp_image, self._frame_ts_ms)
        return PoseResults(result)

    def draw_landmarks(self, frame, results: PoseResults):
        h, w = frame.shape[:2]
        for pose_idx, lms in enumerate(results.pose_landmarks_list):
            color = _PERSON_COLORS_BGR[pose_idx % len(_PERSON_COLORS_BGR)]
            for start, end in POSE_CONNECTIONS:
                if start >= len(lms) or end >= len(lms):
                    continue
                x1, y1 = int(lms[start].x * w), int(lms[start].y * h)
                x2, y2 = int(lms[end].x * w), int(lms[end].y * h)
                cv2.line(frame, (x1, y1), (x2, y2), color, 2)
            for lm in lms:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 4, (255, 255, 255), -1)
                cv2.circle(frame, (cx, cy), 4, color, 1)

    def get_landmark(self, results: PoseResults, landmark_name: str, person_id: int = 0):
        if person_id >= len(results.pose_landmarks_list):
            return None
        lms = results.pose_landmarks_list[person_id]
        idx = LANDMARK_INDEX.get(landmark_name)
        if idx is None or idx >= len(lms):
            return None
        lm = lms[idx]
        return np.array([lm.x, lm.y, lm.z, lm.visibility if hasattr(lm, "visibility") else 1.0])

    def close(self):
        self._landmarker.close()
