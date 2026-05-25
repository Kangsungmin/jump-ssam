from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from modules.pose_analyzer import PoseResults


@dataclass
class TrackResult:
    person_id: int
    pose_idx: int
    centroid: tuple[float, float]  # 정규화 (x, y)


class _TrackedState:
    def __init__(self, centroid: tuple[float, float], pose_idx: int):
        self.centroid = centroid
        self.pose_idx = pose_idx
        self.miss_frames = 0


class _GhostState:
    """이탈한 인물의 마지막 centroid를 보관한다. 재등장 시 ID 복원에 사용."""
    def __init__(self, centroid: tuple[float, float]):
        self.centroid = centroid
        self.ghost_frames = 0


class MultiPersonTracker:
    """
    프레임 간 포즈를 centroid(엉덩이 중점) 기반 nearest-neighbor로 매칭해
    일관된 person_id를 부여한다.

    이탈한 인물은 ghost 풀에 일시 보관되며, 재등장 시 마지막 centroid와
    비교해 동일 인물로 판정되면 기존 person_id를 복원한다.
    """

    def __init__(
        self,
        max_miss_frames: int = 15,
        match_threshold: float = 0.3,
        ghost_expire_frames: int = 150,
        ghost_match_threshold: float = 0.4,
    ):
        self._next_id = 0
        self._persons: dict[int, _TrackedState] = {}
        self._ghosts: dict[int, _GhostState] = {}   # person_id → GhostState
        self._max_miss_frames = max_miss_frames
        self._match_threshold = match_threshold
        self._ghost_expire_frames = ghost_expire_frames
        self._ghost_match_threshold = ghost_match_threshold

    def update(self, results: PoseResults) -> list[TrackResult]:
        centroids = _compute_centroids(results)
        n = len(centroids)

        matched_pose: dict[int, int] = {}  # person_id → pose_idx
        matched_det: set[int] = set()      # 이미 배정된 pose_idx

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
            else:
                state.miss_frames += 1

        # ── 3. 오래된 미감지 대상 → ghost 풀로 이동 ──────────────
        stale = [pid for pid, s in self._persons.items() if s.miss_frames > self._max_miss_frames]
        for pid in stale:
            self._ghosts[pid] = _GhostState(centroid=self._persons[pid].centroid)
            del self._persons[pid]

        # ── 4. 미매칭 감지 포즈 → ghost 복원 또는 신규 ID 발급 ───
        for i, c in enumerate(centroids):
            if i in matched_det:
                continue

            # ghost 풀에서 가장 가까운 후보 탐색
            restored_pid, best_ghost_dist = None, float("inf")
            for pid, ghost in self._ghosts.items():
                d = float(np.hypot(c[0] - ghost.centroid[0], c[1] - ghost.centroid[1]))
                if d < best_ghost_dist:
                    best_ghost_dist, restored_pid = d, pid

            if restored_pid is not None and best_ghost_dist < self._ghost_match_threshold:
                # 기존 ID 복원
                self._persons[restored_pid] = _TrackedState(centroid=c, pose_idx=i)
                del self._ghosts[restored_pid]
            else:
                # 신규 ID 발급
                self._persons[self._next_id] = _TrackedState(centroid=c, pose_idx=i)
                self._next_id += 1

        # ── 5. ghost 나이 증가 및 만료 처리 ──────────────────────
        for ghost in self._ghosts.values():
            ghost.ghost_frames += 1
        expired = [pid for pid, g in self._ghosts.items() if g.ghost_frames > self._ghost_expire_frames]
        for pid in expired:
            del self._ghosts[pid]

        # ── 6. 현재 프레임 감지 대상만 반환 ──────────────────────
        return [
            TrackResult(person_id=pid, pose_idx=s.pose_idx, centroid=s.centroid)
            for pid, s in self._persons.items()
            if s.miss_frames == 0
        ]


def _compute_centroids(results: PoseResults) -> list[tuple[float, float]]:
    centroids = []
    for lms in results.pose_landmarks_list:
        if len(lms) > 24:  # 엉덩이 키포인트 사용
            cx = (lms[23].x + lms[24].x) / 2
            cy = (lms[23].y + lms[24].y) / 2
        elif len(lms) > 12:  # 폴백: 어깨 중점
            cx = (lms[11].x + lms[12].x) / 2
            cy = (lms[11].y + lms[12].y) / 2
        elif lms:
            cx, cy = lms[0].x, lms[0].y
        else:
            continue
        centroids.append((cx, cy))
    return centroids
