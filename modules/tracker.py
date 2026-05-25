from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from modules.pose_analyzer import PoseResults, FaceBox
from modules.face_store import FaceStore

import config


@dataclass
class TrackResult:
    person_id: int
    pose_idx: int
    centroid: tuple[float, float]       # 정규화 (x, y)
    face_box: Optional[FaceBox] = None  # 매핑된 얼굴 box (없으면 None)
    student_id: Optional[str] = None    # 융합 매칭된 학생 ID (없으면 None)


class _TrackedState:
    def __init__(
        self,
        centroid: tuple[float, float],
        pose_idx: int,
        student_id: Optional[str] = None,
    ):
        self.centroid = centroid
        self.pose_idx = pose_idx
        self.miss_frames = 0
        self.student_id: Optional[str] = student_id
        self.embedding: Optional[np.ndarray] = None   # 마지막 인코딩된 얼굴 임베딩
        self.embed_frame: int = 0                      # 임베딩이 갱신된 프레임 번호


class _GhostState:
    """이탈한 인물의 마지막 centroid와 임베딩을 보관한다. 재등장 시 ID 복원에 사용."""
    def __init__(
        self,
        centroid: tuple[float, float],
        student_id: Optional[str] = None,
        embedding: Optional[np.ndarray] = None,
    ):
        self.centroid = centroid
        self.ghost_frames = 0
        self.student_id = student_id
        self.embedding = embedding


class MultiPersonTracker:
    """
    프레임 간 포즈를 centroid(엉덩이 중점) 기반 nearest-neighbor로 매칭해
    일관된 person_id를 부여한다.

    face_store가 제공되면 얼굴 임베딩+centroid 융합 스코어로 ghost 복원 및
    student_id 매핑을 수행한다. face_store가 None이면 centroid 방식만 사용.
    """

    def __init__(
        self,
        max_miss_frames: int = config.TRACKER_MAX_MISS_FRAMES,
        match_threshold: float = config.TRACKER_MATCH_THRESHOLD,
        ghost_expire_frames: int = config.TRACKER_GHOST_EXPIRE_FRAMES,
        ghost_match_threshold: float = config.TRACKER_GHOST_MATCH_THRESHOLD,
        face_store: Optional[FaceStore] = None,
    ):
        self._next_id = 0
        self._persons: dict[int, _TrackedState] = {}
        self._ghosts: dict[int, _GhostState] = {}
        self._max_miss_frames = max_miss_frames
        self._match_threshold = match_threshold
        self._ghost_expire_frames = ghost_expire_frames
        self._ghost_match_threshold = ghost_match_threshold
        self._face_store = face_store
        self._frame_no = 0

    def update(self, results: PoseResults, frame=None) -> list[TrackResult]:
        self._frame_no += 1
        centroids = _compute_centroids(results)
        pose_confs = _compute_confidences(results)
        n = len(centroids)

        matched_pose: dict[int, int] = {}
        matched_det: set[int] = set()

        # ── 1. 활성 추적 대상 매칭 ───────────────────────────────
        if self._persons and n > 0:
            for pid, state in self._persons.items():
                best_idx, best_dist = None, float("inf")
                for i, c in enumerate(centroids):
                    if i in matched_det:
                        continue
                    d = float(np.hypot(c[0] - state.centroid[0], c[1] - state.centroid[1]))
                    if d < best_dist:
                        best_dist, best_idx = d, i
                if best_idx is not None and best_dist < self._match_threshold:
                    matched_pose[pid] = best_idx
                    matched_det.add(best_idx)

        # ── 2. 활성 대상 상태 갱신 ───────────────────────────────
        for pid, state in self._persons.items():
            if pid in matched_pose:
                idx = matched_pose[pid]
                state.centroid = centroids[idx]
                state.pose_idx = idx
                state.miss_frames = 0

                # 얼굴 임베딩 갱신 (FUSION_EMBED_INTERVAL 프레임마다)
                if (
                    self._face_store is not None
                    and frame is not None
                    and idx < len(results.face_boxes)
                    and results.face_boxes[idx] is not None
                    and (self._frame_no - state.embed_frame) >= config.FUSION_EMBED_INTERVAL
                ):
                    fb = results.face_boxes[idx]
                    crop = fb.crop(frame)
                    if crop.size > 0:
                        emb = FaceStore.encode_face(crop)
                        if emb is not None:
                            state.embedding = emb
                            state.embed_frame = self._frame_no

                            # 임베딩으로 student_id 업데이트
                            sid = self._face_store.best_match(emb)
                            if sid is not None:
                                state.student_id = sid
            else:
                state.miss_frames += 1

        # ── 3. 오래된 미감지 대상 → ghost 풀로 이동 ──────────────
        stale = [pid for pid, s in self._persons.items() if s.miss_frames > self._max_miss_frames]
        for pid in stale:
            s = self._persons[pid]
            self._ghosts[pid] = _GhostState(
                centroid=s.centroid,
                student_id=s.student_id,
                embedding=s.embedding,
            )
            del self._persons[pid]

        # ── 4. 미매칭 감지 포즈 → ghost 복원 또는 신규 ID 발급 ───
        for i, c in enumerate(centroids):
            if i in matched_det:
                continue

            conf = pose_confs[i] if i < len(pose_confs) else 1.0

            # 현재 프레임 얼굴 임베딩 (있을 경우)
            cur_emb: Optional[np.ndarray] = None
            if (
                self._face_store is not None
                and frame is not None
                and i < len(results.face_boxes)
                and results.face_boxes[i] is not None
            ):
                fb = results.face_boxes[i]
                crop = fb.crop(frame)
                if crop.size > 0:
                    cur_emb = FaceStore.encode_face(crop)

            # ghost 풀에서 최적 후보 탐색
            restored_pid = self._find_best_ghost(c, conf, cur_emb)

            if restored_pid is not None:
                ghost = self._ghosts[restored_pid]
                new_state = _TrackedState(
                    centroid=c,
                    pose_idx=i,
                    student_id=ghost.student_id,
                )
                if cur_emb is not None:
                    new_state.embedding = cur_emb
                    new_state.embed_frame = self._frame_no
                    # student_id 재확인
                    sid = self._face_store.best_match(cur_emb) if self._face_store else None
                    if sid:
                        new_state.student_id = sid
                self._persons[restored_pid] = new_state
                del self._ghosts[restored_pid]
            else:
                # 신규 ID 발급
                new_state = _TrackedState(centroid=c, pose_idx=i)
                if cur_emb is not None:
                    new_state.embedding = cur_emb
                    new_state.embed_frame = self._frame_no
                    if self._face_store:
                        sid = self._face_store.best_match(cur_emb)
                        new_state.student_id = sid
                self._persons[self._next_id] = new_state
                self._next_id += 1

        # ── 5. ghost 나이 증가 및 만료 처리 ──────────────────────
        for ghost in self._ghosts.values():
            ghost.ghost_frames += 1
        expired = [pid for pid, g in self._ghosts.items() if g.ghost_frames > self._ghost_expire_frames]
        for pid in expired:
            del self._ghosts[pid]

        # ── 6. 현재 프레임 감지 대상만 반환 ──────────────────────
        out = []
        for pid, s in self._persons.items():
            if s.miss_frames != 0:
                continue
            fb = None
            if s.pose_idx < len(results.face_boxes):
                fb = results.face_boxes[s.pose_idx]
            out.append(TrackResult(
                person_id=pid,
                pose_idx=s.pose_idx,
                centroid=s.centroid,
                face_box=fb,
                student_id=s.student_id,
            ))
        return out

    def _find_best_ghost(
        self,
        centroid: tuple[float, float],
        pose_conf: float,
        embedding: Optional[np.ndarray],
    ) -> Optional[int]:
        """
        융합 스코어(얼굴 0.4 + centroid 0.4 + pose_conf 0.2)로 가장 유사한
        ghost를 찾아 person_id를 반환. 임계값 미만이면 None.
        """
        best_pid, best_score = None, -1.0

        for pid, ghost in self._ghosts.items():
            centroid_dist = float(np.hypot(
                centroid[0] - ghost.centroid[0],
                centroid[1] - ghost.centroid[1],
            ))
            # centroid 거리를 [0,1] 유사도로 변환 (임계값 이상이면 0)
            centroid_sim = max(0.0, 1.0 - centroid_dist / self._ghost_match_threshold)

            face_sim = 0.0
            if embedding is not None and ghost.embedding is not None:
                face_sim = FaceStore.similarity(embedding, ghost.embedding)

            w_f = config.FUSION_WEIGHT_FACE
            w_c = config.FUSION_WEIGHT_CENTROID
            w_p = config.FUSION_WEIGHT_POSE_CONF

            if embedding is not None and ghost.embedding is not None:
                score = w_f * face_sim + w_c * centroid_sim + w_p * pose_conf
            else:
                # 임베딩 없으면 centroid + pose_conf 비율로 재배분
                total = w_c + w_p
                score = (w_c / total) * centroid_sim + (w_p / total) * pose_conf

            if score > best_score:
                best_score, best_pid = score, pid

        threshold = config.FUSION_MATCH_THRESHOLD
        if best_pid is not None and best_score >= threshold:
            return best_pid

        # 융합 임계값 미달이라도 centroid 거리가 짧으면 centroid 방식으로 복원
        if best_pid is not None:
            ghost = self._ghosts[best_pid]
            d = float(np.hypot(
                centroid[0] - ghost.centroid[0],
                centroid[1] - ghost.centroid[1],
            ))
            if d < self._ghost_match_threshold:
                return best_pid

        return None


def _compute_centroids(results: PoseResults) -> list[tuple[float, float]]:
    centroids = []
    for lms in results.pose_landmarks_list:
        if len(lms) > 24:
            cx = (lms[23].x + lms[24].x) / 2
            cy = (lms[23].y + lms[24].y) / 2
        elif len(lms) > 12:
            cx = (lms[11].x + lms[12].x) / 2
            cy = (lms[11].y + lms[12].y) / 2
        elif lms:
            cx, cy = lms[0].x, lms[0].y
        else:
            continue
        centroids.append((cx, cy))
    return centroids


def _compute_confidences(results: PoseResults) -> list[float]:
    """각 포즈의 평균 가시성(visibility)을 신뢰도로 반환."""
    confs = []
    for lms in results.pose_landmarks_list:
        if not lms:
            confs.append(0.0)
            continue
        vis_vals = [
            lm.visibility if hasattr(lm, "visibility") and lm.visibility is not None else 1.0
            for lm in lms
        ]
        confs.append(float(np.mean(vis_vals)))
    return confs
