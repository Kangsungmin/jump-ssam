import cv2
import time

import config
from modules.pose_analyzer import PoseAnalyzer
from modules.posture_analyzer import PostureAnalyzer
from modules.display import draw_hud, draw_stats_overlay, make_button_layout, ButtonLayout
from modules.tracker import MultiPersonTracker, TrackResult
from modules.session import SessionRecorder
from ropemetrics import JumpCounter, JumpCounterConfig, JumpEvent
from ropemetrics.strategies import AnkleStrategy
from modules.adapters.mediapipe import MediaPipeLandmarkProvider, BoundLandmarkProvider


# 민감도 프리셋 순환 목록
_SENS_PRESETS = [
    ("레벨 1", JumpCounterConfig.slow),
    ("레벨 2", JumpCounterConfig),
    ("레벨 3", JumpCounterConfig.fast),
]


def _make_counter(cfg: JumpCounterConfig, on_jump) -> JumpCounter:
    return JumpCounter(strategy=AnkleStrategy(cfg), config=cfg, on_jump=on_jump)


def main():
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError("카메라를 열 수 없습니다. 카메라 권한을 확인하세요.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          config.TARGET_FPS)

    pose_analyzer    = PoseAnalyzer()
    posture_analyzer = PostureAnalyzer()
    tracker = MultiPersonTracker(
        max_miss_frames=config.TRACKER_MAX_MISS_FRAMES,
        match_threshold=config.TRACKER_MATCH_THRESHOLD,
        ghost_expire_frames=config.TRACKER_GHOST_EXPIRE_FRAMES,
        ghost_match_threshold=config.TRACKER_GHOST_MATCH_THRESHOLD,
    )
    session = SessionRecorder()

    base_provider = MediaPipeLandmarkProvider()
    persons: dict = {}
    _frame_no = 0
    _last_results = None

    # UI 상태 (dict 사용으로 콜백에서 변경 가능)
    ui = {
        "quit":        False,
        "sens_idx":    1,
        "show_stats":  False,   # 통계 오버레이 표시 여부
    }
    layout: ButtonLayout = make_button_layout(config.FRAME_WIDTH, config.FRAME_HEIGHT)

    def _rebuild_counters():
        _, cfg_cls = _SENS_PRESETS[ui["sens_idx"]]
        cfg = cfg_cls() if cfg_cls is JumpCounterConfig else cfg_cls()
        for pid, p in persons.items():
            def _make_on_jump(p_id: int):
                def _on_jump(event: JumpEvent) -> None:
                    persons[p_id]["flash"] = config.JUMP_FLASH_FRAMES
                return _on_jump
            p["counter"] = _make_counter(cfg, _make_on_jump(pid))

    def _reset_all():
        for p in persons.values():
            p["counter"].reset()
        print("전체 카운터 초기화")

    def _toggle_session():
        if session.active:
            stats = session.stop(persons)
            ui["show_stats"] = True
            print("=== 운동 종료 ===")
            for s in stats:
                print(f"  #{s.person_id + 1}: {s.jump_count}회  ({s.jumps_per_min:.1f}회/분)")
        else:
            session.start(persons)
            ui["show_stats"] = False
            print("=== 운동 시작 ===")

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if ui["show_stats"]:
            ui["show_stats"] = False
            return
        if layout.hit("session", x, y):
            _toggle_session()
        elif layout.hit("reset", x, y):
            _reset_all()
        elif layout.hit("sensitivity", x, y):
            ui["sens_idx"] = (ui["sens_idx"] + 1) % len(_SENS_PRESETS)
            label, _ = _SENS_PRESETS[ui["sens_idx"]]
            print(f"민감도 변경: {label}")
            _rebuild_counters()
        elif layout.hit("quit", x, y):
            ui["quit"] = True

    win_name = "AI Jump Rope Analyzer"
    cv2.namedWindow(win_name)
    cv2.setMouseCallback(win_name, on_mouse)

    prev_time = time.time()
    _, initial_cfg_cls = _SENS_PRESETS[ui["sens_idx"]]
    current_cfg = initial_cfg_cls()

    print("줄넘기 분석 시작 — '운동 시작' 버튼으로 세션 시작 / 'r' 리셋 / 'q' 종료")
    if config.PI5_MODE:
        print(f"[Pi5 모드] pose_skip={config.PERF_POSE_SKIP}, scale={config.PERF_RESIZE_SCALE}")

    while not ui["quit"]:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        _frame_no += 1

        # 프레임 스킵: PERF_POSE_SKIP 마다 포즈 추정 실행
        if _frame_no % config.PERF_POSE_SKIP == 0:
            proc_frame = frame
            if config.PERF_RESIZE_SCALE != 1.0:
                fh, fw = frame.shape[:2]
                proc_frame = cv2.resize(
                    frame,
                    (int(fw * config.PERF_RESIZE_SCALE), int(fh * config.PERF_RESIZE_SCALE)),
                )
            _last_results = pose_analyzer.process(proc_frame)

        results = _last_results
        if results is None:
            cv2.imshow(win_name, frame)
            cv2.waitKey(1)
            continue

        pose_analyzer.draw_landmarks(frame, results)

        track_results: list[TrackResult] = tracker.update(results, frame)
        active_ids = {tr.person_id for tr in track_results}

        for pid in list(persons.keys()):
            if pid not in active_ids:
                del persons[pid]

        _, cfg_cls = _SENS_PRESETS[ui["sens_idx"]]
        current_cfg = cfg_cls()

        for tr in track_results:
            pid = tr.person_id

            if pid not in persons:
                provider = BoundLandmarkProvider(base_provider)

                def _make_on_jump(p_id: int):
                    def _on_jump(event: JumpEvent) -> None:
                        if session.active:
                            persons[p_id]["flash"] = config.JUMP_FLASH_FRAMES
                    return _on_jump

                persons[pid] = {
                    "counter":  _make_counter(current_cfg, _make_on_jump(pid)),
                    "provider": provider,
                    "report":   None,
                    "flash":    0,
                    "centroid": tr.centroid,
                }

            p = persons[pid]
            p["centroid"]          = tr.centroid
            p["provider"].pose_idx = tr.pose_idx
            p["counter"].update(p["provider"], results)
            p["report"] = posture_analyzer.analyze(results, pose_analyzer, tr.pose_idx)

            if p["flash"] > 0:
                p["flash"] -= 1

        # 세션 활성 중이면 카운트 갱신
        session.update(persons)

        now       = time.time()
        fps       = 1.0 / (now - prev_time + 1e-6)
        prev_time = now

        sens_label, _ = _SENS_PRESETS[ui["sens_idx"]]
        draw_hud(
            frame, persons, fps, layout, sens_label,
            session_active=session.active,
            elapsed_sec=session.elapsed_sec,
        )

        # 통계 오버레이 (운동 종료 직후)
        if ui["show_stats"] and session.last_stats:
            draw_stats_overlay(frame, session.last_stats, session.last_stats[0].duration_sec)

        cv2.imshow(win_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            _reset_all()
        elif key == ord("s"):
            _toggle_session()
        elif ui["show_stats"] and key != 0xFF:
            ui["show_stats"] = False

    cap.release()
    cv2.destroyAllWindows()
    pose_analyzer.close()

    print("=== 최종 결과 ===")
    for pid, p in sorted(persons.items()):
        print(f"  학생 #{pid + 1}: {p['counter'].count}회")


if __name__ == "__main__":
    main()
