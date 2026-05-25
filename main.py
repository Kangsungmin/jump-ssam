import cv2
import time

import config
from modules.pose_analyzer import PoseAnalyzer
from modules.posture_analyzer import PostureAnalyzer
from modules.display import draw_hud
from ropemetrics import JumpCounter, JumpCounterConfig, JumpEvent
from ropemetrics.strategies import AnkleStrategy
from modules.adapters.mediapipe import MediaPipeLandmarkProvider


def main():
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError("카메라를 열 수 없습니다. 카메라 권한을 확인하세요.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          config.TARGET_FPS)

    pose_analyzer    = PoseAnalyzer()
    posture_analyzer = PostureAnalyzer()

    # ── 카운터 조립 ───────────────────────────────────────────────────────
    # 전략·프로바이더·카운터를 독립적으로 교체 가능
    # 예: AnkleStrategy → HipShoulderStrategy, JumpCounterConfig() → .fast()
    cfg      = JumpCounterConfig()
    strategy = AnkleStrategy(cfg)
    provider = MediaPipeLandmarkProvider()

    jump_flash_left = 0

    def _on_jump(event: JumpEvent) -> None:
        nonlocal jump_flash_left
        jump_flash_left = config.JUMP_FLASH_FRAMES
        # 향후 확장: db.record(event), audio_feedback(event) 등

    counter = JumpCounter(strategy=strategy, config=cfg, on_jump=_on_jump)

    prev_time = time.time()
    report    = None

    print("줄넘기 분석 시작 — 'r' 카운터 초기화 / 'q' 종료")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        results = pose_analyzer.process(frame)
        pose_analyzer.draw_landmarks(frame, results)

        counter.update(provider, results)   # on_jump 콜백이 flash 처리

        if results.pose_landmarks:
            report = posture_analyzer.analyze(results, pose_analyzer)

        flash = jump_flash_left > 0
        if jump_flash_left > 0:
            jump_flash_left -= 1

        now       = time.time()
        fps       = 1.0 / (now - prev_time + 1e-6)
        prev_time = now

        draw_hud(frame, counter.count, fps, report, flash)
        cv2.imshow("AI Jump Rope Analyzer", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            counter.reset()
            print("카운터 초기화")

    cap.release()
    cv2.destroyAllWindows()
    pose_analyzer.close()
    print(f"최종 점프 횟수: {counter.count}")


if __name__ == "__main__":
    main()
