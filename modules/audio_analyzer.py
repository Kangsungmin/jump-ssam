# Phase 1 stub — Phase 1 완료 후 구현 예정
# 기능: 마이크 실시간 BPM 감지, 박자 그리드 생성, 점프 타이밍 동기화
#
# 구현 계획:
#   - PyAudio로 마이크 스트림 수음
#   - librosa 또는 aubio로 BPM 감지
#   - 박자 정확도 점수 산출 (오차 허용: ±BEAT_TOLERANCE_MS)

import config  # noqa: F401 — 설정값 참조용


class AudioAnalyzer:
    def __init__(self):
        raise NotImplementedError("AudioAnalyzer는 Phase 1 완료 후 구현 예정입니다.")

    def start(self):
        pass

    def get_bpm(self) -> float:
        return 0.0

    def get_beat_accuracy(self, jump_timestamp_ms: float) -> float:
        return 0.0

    def stop(self):
        pass
