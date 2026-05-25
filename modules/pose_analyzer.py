import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import PoseLandmarkerResult
import cv2
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

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
    (0, 200, 100),
    (0, 100, 255),
    (255, 50, 50),
    (200, 0, 255),
    (0, 255, 255),
    (255, 255, 0),
    (100, 255, 100),
    (255, 100, 200),
]


@dataclass
class FaceBox:
    """정규화 좌표 기반 얼굴 바운딩박스."""
    x: float       # 좌상단 x (0~1)
    y: float       # 좌상단 y (0~1)
    w: float       # 너비 (0~1)
    h: float       # 높이 (0~1)
    score: float   # 감지 신뢰도

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2

    def to_pixel(self, frame_w: int, frame_h: int) -> tuple[int, int, int, int]:
        """픽셀 좌표 (x1, y1, x2, y2) 반환."""
        x1 = int(self.x * frame_w)
        y1 = int(self.y * frame_h)
        x2 = int((self.x + self.w) * frame_w)
        y2 = int((self.y + self.h) * frame_h)
        return x1, y1, x2, y2

    def crop(self, frame) -> np.ndarray:
        """프레임에서 얼굴 영역을 크롭하여 반환."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = self.to_pixel(w, h)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        return frame[y1:y2, x1:x2]


class PoseResults:
    def __init__(
        self,
        result: PoseLandmarkerResult,
        face_boxes: Optional[list[Optional[FaceBox]]] = None,
    ):
        self._result = result
        self.pose_landmarks_list = result.pose_landmarks  # list[list[NormalizedLandmark]]
        self.pose_landmarks = self.pose_landmarks_list[0] if self.pose_landmarks_list else None
        # pose_idx별 매핑된 얼굴 box (감지 실패 시 None)
        self.face_boxes: list[Optional[FaceBox]] = face_boxes or [None] * len(self.pose_landmarks_list)

    @property
    def num_persons(self) -> int:
        return len(self.pose_landmarks_list)


class PoseAnalyzer:
    def __init__(self):
        self._init_pose()
        self._init_face()
        self._frame_ts_ms = 0

    def _init_pose(self):
        model_path = config.MODEL_PATH
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"포즈 모델 파일이 없습니다: {model_path}\n"
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

    def _init_face(self):
        face_model = config.FACE_MODEL_PATH
        if not Path(face_model).exists():
            # 얼굴 모델 없으면 face detection 비활성화 (경고만 출력)
            print(f"[경고] 얼굴 감지 모델 없음: {face_model} — 얼굴 감지 비활성화")
            self._face_detector = None
            return
        options = mp_vision.FaceDetectorOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(face_model)),
            running_mode=mp_vision.RunningMode.VIDEO,
            min_detection_confidence=config.FACE_CONFIDENCE,
        )
        self._face_detector = mp_vision.FaceDetector.create_from_options(options)

    def process(self, frame) -> PoseResults:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self._frame_ts_ms += int(1000 / config.TARGET_FPS)

        pose_result = self._landmarker.detect_for_video(mp_image, self._frame_ts_ms)

        face_boxes: list[Optional[FaceBox]] = [None] * len(pose_result.pose_landmarks)

        if self._face_detector and pose_result.pose_landmarks:
            face_result = self._face_detector.detect_for_video(mp_image, self._frame_ts_ms)
            raw_boxes = _parse_face_boxes(face_result, frame.shape[1], frame.shape[0])
            face_boxes = _map_faces_to_poses(pose_result.pose_landmarks, raw_boxes)

        return PoseResults(pose_result, face_boxes)

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

            # 얼굴 box 표시
            fb = results.face_boxes[pose_idx] if pose_idx < len(results.face_boxes) else None
            if fb is not None:
                x1, y1, x2, y2 = fb.to_pixel(w, h)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

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
        if self._face_detector:
            self._face_detector.close()


def _parse_face_boxes(face_result, frame_w: int, frame_h: int) -> list[FaceBox]:
    """FaceDetectorResult → 정규화 FaceBox 리스트 변환."""
    boxes = []
    for det in face_result.detections:
        bb = det.bounding_box
        boxes.append(FaceBox(
            x=bb.origin_x / frame_w,
            y=bb.origin_y / frame_h,
            w=bb.width / frame_w,
            h=bb.height / frame_h,
            score=det.categories[0].score if det.categories else 0.0,
        ))
    return boxes


def _map_faces_to_poses(
    pose_landmarks_list: list,
    face_boxes: list[FaceBox],
) -> list[Optional[FaceBox]]:
    """각 포즈에 가장 가까운 얼굴 box를 매핑한다. 이미 배정된 box는 재사용 금지."""
    result: list[Optional[FaceBox]] = [None] * len(pose_landmarks_list)
    used = set()

    for pose_idx, lms in enumerate(pose_landmarks_list):
        if not lms:
            continue
        # 코 또는 어깨 중점을 얼굴 기준점으로 사용
        nose = lms[0]
        ref_x, ref_y = nose.x, nose.y

        best_fb, best_dist = None, float("inf")
        for i, fb in enumerate(face_boxes):
            if i in used:
                continue
            d = float(np.hypot(fb.cx - ref_x, fb.cy - ref_y))
            if d < best_dist:
                best_dist, best_fb, best_i = d, fb, i

        if best_fb is not None and best_dist < config.FACE_POSE_MAP_THRESH:
            result[pose_idx] = best_fb
            used.add(best_i)

    return result
