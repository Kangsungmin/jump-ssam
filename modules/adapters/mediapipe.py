from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ropemetrics.base import Landmark, LandmarkName, LandmarkProvider
from modules.pose_analyzer import LANDMARK_INDEX, PoseResults


class MediaPipeLandmarkProvider(LandmarkProvider):
    """
    PoseAnalyzer의 결과를 LandmarkProvider 인터페이스로 노출하는 경량 어댑터.

    설계 의도:
      - PoseAnalyzer는 MediaPipe 초기화·영상처리·draw 등 렌더링 책임을 유지
      - 이 어댑터는 "랜드마크 조회" 책임만 담당하며 내부 상태 없음
      - PoseResults를 공유하므로 데이터 복사 없음
      - MediaPipe Tasks API 버전 차이는 PoseAnalyzer가 흡수하고
        이 클래스는 그 결과만 읽음
    """

    def get_landmark(
        self,
        results: PoseResults,
        name: LandmarkName,
    ) -> Optional[Landmark]:
        if results.pose_landmarks is None:
            return None

        idx = LANDMARK_INDEX.get(name)
        if idx is None or idx >= len(results.pose_landmarks):
            return None

        lm = results.pose_landmarks[idx]
        return np.array(
            [lm.x, lm.y, lm.z,
             lm.visibility if hasattr(lm, "visibility") else 1.0],
            dtype=np.float32,
        )

    def get_landmarks(
        self,
        results: PoseResults,
        names: Sequence[LandmarkName],
    ) -> dict[LandmarkName, Optional[Landmark]]:
        return {name: self.get_landmark(results, name) for name in names}
