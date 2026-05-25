from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ropemetrics.base import Landmark, LandmarkName, LandmarkProvider
from modules.pose_analyzer import LANDMARK_INDEX, PoseResults


class MediaPipeLandmarkProvider(LandmarkProvider):
    """
    PoseAnalyzer 결과를 LandmarkProvider 인터페이스로 노출하는 경량 어댑터.
    person_id로 다중 인원 중 특정 포즈를 선택한다.
    """

    def get_landmark(
        self,
        results: PoseResults,
        name: LandmarkName,
        person_id: int = 0,
    ) -> Optional[Landmark]:
        if person_id >= len(results.pose_landmarks_list):
            return None

        lms = results.pose_landmarks_list[person_id]
        idx = LANDMARK_INDEX.get(name)
        if idx is None or idx >= len(lms):
            return None

        lm = lms[idx]
        return np.array(
            [lm.x, lm.y, lm.z,
             lm.visibility if hasattr(lm, "visibility") else 1.0],
            dtype=np.float32,
        )

    def get_landmarks(
        self,
        results: PoseResults,
        names: Sequence[LandmarkName],
        person_id: int = 0,
    ) -> dict[LandmarkName, Optional[Landmark]]:
        return {name: self.get_landmark(results, name, person_id) for name in names}


class BoundLandmarkProvider(LandmarkProvider):
    """
    특정 person의 pose_idx에 바인딩된 제공자.
    pose_idx는 매 프레임 tracker 결과로 갱신된다.
    JumpCounter에 1:1로 할당되어 ropemetrics 인터페이스 변경 없이 다중 인원을 지원한다.
    """

    def __init__(self, base: MediaPipeLandmarkProvider):
        self._base = base
        self.pose_idx: int = 0

    def get_landmark(self, results: PoseResults, name: LandmarkName) -> Optional[Landmark]:
        return self._base.get_landmark(results, name, person_id=self.pose_idx)

    def get_landmarks(
        self,
        results: PoseResults,
        names: Sequence[LandmarkName],
    ) -> dict[LandmarkName, Optional[Landmark]]:
        return {name: self.get_landmark(results, name) for name in names}
