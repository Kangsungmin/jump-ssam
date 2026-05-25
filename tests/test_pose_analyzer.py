import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# PoseAnalyzer는 모델 파일이 필요해 단위 테스트가 제한적
# 여기서는 LANDMARK_INDEX 정합성만 검증

from modules.pose_analyzer import LANDMARK_INDEX


def test_landmark_index_has_33_points():
    assert len(LANDMARK_INDEX) == 33, "MediaPipe Pose는 33개 키포인트를 가져야 함"


def test_landmark_index_values_are_unique():
    values = list(LANDMARK_INDEX.values())
    assert len(values) == len(set(values)), "인덱스 값에 중복이 없어야 함"


def test_required_landmarks_exist():
    required = [
        "LEFT_SHOULDER", "RIGHT_SHOULDER",
        "LEFT_HIP", "RIGHT_HIP",
        "LEFT_KNEE", "RIGHT_KNEE",
        "LEFT_ANKLE", "RIGHT_ANKLE",
        "LEFT_WRIST", "RIGHT_WRIST",
        "LEFT_ELBOW", "RIGHT_ELBOW",
    ]
    for name in required:
        assert name in LANDMARK_INDEX, f"{name} 이 LANDMARK_INDEX에 없음"
