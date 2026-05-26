"""
운동 세션 기록 모듈.

- 운동 시작/종료 사이에만 인원별 점프 횟수를 누적한다.
- 비활성 상태에서는 아무것도 기록하지 않는다.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PersonStat:
    person_id: int
    jump_count: int
    duration_sec: float

    @property
    def jumps_per_min(self) -> float:
        if self.duration_sec <= 0:
            return 0.0
        return self.jump_count / self.duration_sec * 60


class SessionRecorder:
    """
    운동 세션의 시작/종료를 관리하고 인원별 통계를 집계한다.

    - start() 시점의 카운트를 스냅샷으로 저장.
    - update()를 매 프레임 호출해 현재 카운트를 추적.
    - stop() 시 (현재 카운트 - 스냅샷)으로 세션 중 점프 수 산출.
    - 세션 도중 사라진 인물도 마지막으로 확인된 카운트로 집계.
    """

    def __init__(self):
        self._active = False
        self._start_time: float = 0.0
        self._start_counts: dict[int, int] = {}
        self._peak_counts: dict[int, int] = {}
        self._last_stats: list[PersonStat] = []

    @property
    def active(self) -> bool:
        return self._active

    @property
    def elapsed_sec(self) -> float:
        if not self._active:
            return 0.0
        return time.time() - self._start_time

    @property
    def last_stats(self) -> list[PersonStat]:
        return self._last_stats

    def start(self, persons: dict):
        """운동 시작. 현재 카운트를 기준점으로 스냅샷."""
        self._active = True
        self._start_time = time.time()
        self._start_counts = {pid: p["counter"].count for pid, p in persons.items()}
        self._peak_counts = dict(self._start_counts)
        self._last_stats = []

    def update(self, persons: dict):
        """매 프레임 호출. 현재 감지 중인 인원의 카운트를 갱신."""
        if not self._active:
            return
        for pid, p in persons.items():
            self._peak_counts[pid] = p["counter"].count

    def stop(self, persons: dict) -> list[PersonStat]:
        """운동 종료. 세션 통계 반환."""
        if not self._active:
            return []

        self.update(persons)
        elapsed = time.time() - self._start_time
        self._active = False

        stats = []
        for pid, peak in self._peak_counts.items():
            start = self._start_counts.get(pid, peak)
            delta = max(0, peak - start)
            stats.append(PersonStat(
                person_id=pid,
                jump_count=delta,
                duration_sec=elapsed,
            ))

        self._last_stats = sorted(stats, key=lambda s: s.person_id)
        return self._last_stats
