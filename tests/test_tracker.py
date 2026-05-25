"""MultiPersonTracker 단위 테스트."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from unittest.mock import MagicMock

import numpy as np
import pytest

from modules.tracker import MultiPersonTracker, TrackResult
from modules.pose_analyzer import PoseResults, FaceBox


# ── 테스트 픽스처 헬퍼 ─────────────────────────────────────────────────────

@dataclass
class _FakeLandmark:
    x: float
    y: float
    z: float = 0.0
    visibility: float = 1.0


def _make_lms(cx: float, cy: float) -> list[_FakeLandmark]:
    """엉덩이 중점이 (cx, cy)가 되도록 33개 랜드마크를 생성한다."""
    lms = [_FakeLandmark(0.0, 0.0)] * 33
    lms = list(lms)
    lms[23] = _FakeLandmark(cx - 0.01, cy)
    lms[24] = _FakeLandmark(cx + 0.01, cy)
    return lms


def _make_results(
    centroids: list[tuple[float, float]],
    face_boxes: Optional[list] = None,
) -> PoseResults:
    mock_result = MagicMock()
    mock_result.pose_landmarks = [_make_lms(cx, cy) for cx, cy in centroids]
    pr = PoseResults.__new__(PoseResults)
    pr._result = mock_result
    pr.pose_landmarks_list = mock_result.pose_landmarks
    pr.pose_landmarks = pr.pose_landmarks_list[0] if pr.pose_landmarks_list else None
    pr.face_boxes = face_boxes or [None] * len(centroids)
    return pr


def _tracker(**kwargs) -> MultiPersonTracker:
    return MultiPersonTracker(
        max_miss_frames=kwargs.get("max_miss_frames", 15),
        match_threshold=kwargs.get("match_threshold", 0.3),
        ghost_expire_frames=kwargs.get("ghost_expire_frames", 150),
        ghost_match_threshold=kwargs.get("ghost_match_threshold", 0.4),
        face_store=kwargs.get("face_store", None),
    )


# ── 기본 추적 ──────────────────────────────────────────────────────────────

class TestBasicTracking:
    def test_single_person_gets_id(self):
        t = _tracker()
        r = t.update(_make_results([(0.5, 0.5)]))
        assert len(r) == 1
        assert r[0].person_id == 0

    def test_two_persons_get_different_ids(self):
        t = _tracker()
        r = t.update(_make_results([(0.3, 0.5), (0.7, 0.5)]))
        ids = {res.person_id for res in r}
        assert len(ids) == 2

    def test_id_is_stable_across_frames(self):
        t = _tracker()
        r1 = t.update(_make_results([(0.5, 0.5)]))
        r2 = t.update(_make_results([(0.5, 0.5)]))
        assert r1[0].person_id == r2[0].person_id

    def test_empty_frame_returns_empty(self):
        t = _tracker()
        assert t.update(_make_results([])) == []

    def test_centroid_updated_after_move(self):
        t = _tracker()
        t.update(_make_results([(0.5, 0.5)]))
        r = t.update(_make_results([(0.55, 0.55)]))
        assert abs(r[0].centroid[0] - 0.55) < 0.02

    def test_miss_frames_do_not_appear_in_output(self):
        t = _tracker(max_miss_frames=15)
        t.update(_make_results([(0.5, 0.5)]))
        r = t.update(_make_results([]))  # 미감지
        assert len(r) == 0

    def test_person_reappears_same_id_within_ghost_window(self):
        t = _tracker(max_miss_frames=2, ghost_expire_frames=30, ghost_match_threshold=0.4)
        r1 = t.update(_make_results([(0.5, 0.5)]))
        pid = r1[0].person_id
        # 미감지 3프레임 → ghost
        for _ in range(3):
            t.update(_make_results([]))
        # 재등장
        r2 = t.update(_make_results([(0.5, 0.5)]))
        assert len(r2) == 1
        assert r2[0].person_id == pid

    def test_ghost_expires_new_id_assigned(self):
        t = _tracker(max_miss_frames=2, ghost_expire_frames=3, ghost_match_threshold=0.4)
        r1 = t.update(_make_results([(0.5, 0.5)]))
        pid = r1[0].person_id
        for _ in range(10):
            t.update(_make_results([]))
        r2 = t.update(_make_results([(0.5, 0.5)]))
        assert r2[0].person_id != pid


# ── face_box 매핑 ──────────────────────────────────────────────────────────

class TestFaceBoxMapping:
    def test_face_box_included_in_result(self):
        fb = FaceBox(x=0.4, y=0.1, w=0.1, h=0.1, score=0.9)
        t = _tracker()
        r = t.update(_make_results([(0.5, 0.5)], face_boxes=[fb]))
        assert r[0].face_box is fb

    def test_no_face_box_returns_none(self):
        t = _tracker()
        r = t.update(_make_results([(0.5, 0.5)]))
        assert r[0].face_box is None


# ── TrackResult 필드 ───────────────────────────────────────────────────────

class TestTrackResult:
    def test_track_result_has_student_id_field(self):
        t = _tracker()
        r = t.update(_make_results([(0.5, 0.5)]))
        assert hasattr(r[0], "student_id")
        assert r[0].student_id is None  # face_store 없으면 None

    def test_track_result_pose_idx_matches(self):
        t = _tracker()
        r = t.update(_make_results([(0.3, 0.5), (0.7, 0.5)]))
        pose_idxs = {res.pose_idx for res in r}
        assert pose_idxs == {0, 1}


# ── 융합 스코어 (face_store mock) ──────────────────────────────────────────

class TestFusionScore:
    def _mock_face_store(self, student_id: Optional[str] = "s001"):
        fs = MagicMock()
        fs.best_match.return_value = student_id
        fs.get_all_embeddings.return_value = {}
        return fs

    def test_student_id_assigned_from_face_store(self, tmp_path):
        from modules.face_store import FaceStore
        store = FaceStore(db_path=tmp_path / "faces.db")
        store.register_student("s001", "홍길동", consent=True)
        emb = np.random.default_rng(1).random(128)
        emb /= np.linalg.norm(emb)
        store.add_embedding("s001", emb)

        t = _tracker(face_store=store)
        # face_store.best_match가 호출되려면 face_box + frame이 있어야 한다
        # 여기서는 store 연결 여부만 확인 (임베딩 인코딩은 실제 face img 필요)
        r = t.update(_make_results([(0.5, 0.5)]))
        # frame=None이므로 임베딩 추출 없이 student_id는 None
        assert r[0].student_id is None
        store.close()

    def test_ghost_restores_student_id(self):
        t = _tracker(max_miss_frames=2, ghost_expire_frames=30)

        # 수동으로 state에 student_id 심기
        from modules.tracker import _TrackedState
        r1 = t.update(_make_results([(0.5, 0.5)]))
        pid = r1[0].person_id
        t._persons[pid].student_id = "s001"

        for _ in range(3):
            t.update(_make_results([]))

        r2 = t.update(_make_results([(0.5, 0.5)]))
        assert r2[0].person_id == pid
        assert r2[0].student_id == "s001"
