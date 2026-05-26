import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image
from dataclasses import dataclass
from typing import Optional

import config
from modules.posture_analyzer import PostureReport
from modules.session import PersonStat

_WHITE_RGB  = (255, 255, 255)
_BLACK_RGB  = (0,   0,   0)
_RED_RGB    = (255, 60,  0)
_GRAY_RGB   = (180, 180, 180)
_YELLOW_RGB = (255, 210, 0)
_GREEN_RGB  = (60,  220, 80)

_BTN_SESSION_START_BG = (20,  160, 60)   # 진초록 — 운동 시작
_BTN_SESSION_STOP_BG  = (200, 50,  30)   # 진빨강 — 운동 종료
_BTN_RESET_BG         = (40,  130, 80)   # 초록
_BTN_SENS_BG          = (40,  90,  180)  # 파랑
_BTN_QUIT_BG          = (180, 40,  40)   # 빨강
_BTN_BORDER           = (210, 210, 210)

_PERSON_COLORS_RGB = [
    (0,   200, 100),
    (255, 100, 0),
    (50,  50,  255),
    (255, 0,   200),
    (0,   230, 230),
    (230, 230, 0),
    (100, 255, 100),
    (255, 100, 200),
]

_font_cache: dict = {}


def _font(size: int) -> ImageFont.FreeTypeFont:
    if size not in _font_cache:
        _font_cache[size] = ImageFont.truetype(config.FONT_PATH, size)
    return _font_cache[size]


def _put_text(draw: ImageDraw.ImageDraw, text: str, pos, color, size: int):
    x, y = pos
    draw.text((x + 1, y + 1), text, font=_font(size), fill=_BLACK_RGB)
    draw.text((x, y), text, font=_font(size), fill=color)


def _draw_button(draw: ImageDraw.ImageDraw, rect: tuple, label: str, bg: tuple, size: int = 22):
    x, y, w, h = rect
    draw.rectangle([x, y, x + w, y + h], fill=bg, outline=_BTN_BORDER, width=2)
    bbox = draw.textbbox((0, 0), label, font=_font(size))
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text((x + (w - tw) // 2, y + (h - th) // 2), label, font=_font(size), fill=_WHITE_RGB)


@dataclass
class ButtonLayout:
    session:     tuple  # (x, y, w, h) — 운동 시작/종료
    reset:       tuple
    sensitivity: tuple
    quit:        tuple

    def hit(self, name: str, mx: int, my: int) -> bool:
        bx, by, bw, bh = getattr(self, name)
        return bx <= mx <= bx + bw and by <= my <= by + bh


def make_button_layout(frame_w: int, frame_h: int) -> ButtonLayout:
    bh = 52
    by = frame_h - bh - 10
    return ButtonLayout(
        session     = (20,                 by, 190, bh),
        reset       = (230,                by, 130, bh),
        sensitivity = (frame_w // 2 - 115, by, 230, bh),
        quit        = (frame_w - 170,      by, 150, bh),
    )


def draw_hud(
    frame,
    persons: dict,
    fps: float,
    layout: ButtonLayout,
    sens_label: str,
    session_active: bool = False,
    elapsed_sec: float = 0.0,
):
    """
    persons: dict[person_id → {
        'counter': JumpCounter,
        'report': Optional[PostureReport],
        'flash': int,
        'centroid': tuple[float, float],
    }]
    """
    h, w = frame.shape[:2]

    # 점프 플래시 오버레이
    if any(p["flash"] > 0 for p in persons.values()):
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 255, 255), -1)
        cv2.addWeighted(overlay, 0.05, frame, 0.95, 0, frame)

    # 세션 활성 시 상단 빨간 녹화 표시 띠
    if session_active:
        bar = frame.copy()
        cv2.rectangle(bar, (0, 0), (w, 48), (180, 0, 0), -1)
        cv2.addWeighted(bar, 0.55, frame, 0.45, 0, frame)

    # 하단 버튼 바 배경 (반투명)
    bar_y = layout.reset[1] - 8
    bar_bg = frame.copy()
    cv2.rectangle(bar_bg, (0, bar_y), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(bar_bg, 0.6, frame, 0.4, 0, frame)

    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw    = ImageDraw.Draw(pil_img)

    # 세션 상태 표시 (상단)
    if session_active:
        mins = int(elapsed_sec) // 60
        secs = int(elapsed_sec) % 60
        _put_text(draw, f"● REC  {mins:02d}:{secs:02d}", (w // 2 - 90, 8), (255, 80, 80), size=28)

    # FPS (우상단)
    _put_text(draw, f"FPS {fps:.1f}", (w - 130, 14), _GRAY_RGB, size=26)

    # 감지 인원 수 (좌상단)
    top_y = 54 if session_active else 14
    _put_text(draw, f"감지: {len(persons)}명", (20, top_y), _WHITE_RGB, size=26)

    # 인원별 신체 위에 뱃지
    for pid, info in persons.items():
        color = _YELLOW_RGB if info["flash"] > 0 else _PERSON_COLORS_RGB[pid % len(_PERSON_COLORS_RGB)]
        cx_norm, cy_norm = info.get("centroid", (0.5, 0.5))
        cx = int(cx_norm * w)
        cy = int(cy_norm * h)

        count = info["counter"].count
        label = f"#{pid + 1}  {count}회"
        badge_y = max(cy - 130, 60)
        badge_x = max(min(cx - 40, w - 160), 10)
        _put_text(draw, label, (badge_x, badge_y), color, size=30)

        report: PostureReport | None = info.get("report")
        if report and report.warnings:
            _put_text(draw, report.warnings[0], (badge_x, badge_y + 36), _RED_RGB, size=18)

    # 우측 요약 패널
    if persons:
        panel_x = w - 200
        panel_y = 56
        _put_text(draw, "학생 현황", (panel_x, panel_y), _WHITE_RGB, size=22)
        panel_y += 30
        for pid, info in sorted(persons.items()):
            count = info["counter"].count
            color = _PERSON_COLORS_RGB[pid % len(_PERSON_COLORS_RGB)]
            status = _posture_status(info.get("report"))
            _put_text(draw, f"#{pid + 1}  {count}회  {status}", (panel_x, panel_y), color, size=20)
            panel_y += 26

    # 버튼 4개
    session_bg    = _BTN_SESSION_STOP_BG if session_active else _BTN_SESSION_START_BG
    session_label = "운동 종료" if session_active else "운동 시작"
    _draw_button(draw, layout.session,     session_label,          session_bg,    size=22)
    _draw_button(draw, layout.reset,       "리셋",                  _BTN_RESET_BG, size=22)
    _draw_button(draw, layout.sensitivity, f"민감도: {sens_label}", _BTN_SENS_BG,  size=22)
    _draw_button(draw, layout.quit,        "테스트 종료",           _BTN_QUIT_BG,  size=22)

    frame[:] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def draw_stats_overlay(frame, stats: list[PersonStat], elapsed_sec: float):
    """운동 종료 후 인원별 통계 오버레이를 그린다."""
    h, w = frame.shape[:2]

    # 반투명 배경 패널
    panel_w, panel_h = min(600, w - 80), min(80 + len(stats) * 52 + 60, h - 80)
    px = (w - panel_w) // 2
    py = (h - panel_h) // 2

    overlay = frame.copy()
    cv2.rectangle(overlay, (px, py), (px + panel_w, py + panel_h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
    cv2.rectangle(frame, (px, py), (px + panel_w, py + panel_h), (200, 200, 200), 2)

    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw    = ImageDraw.Draw(pil_img)

    mins = int(elapsed_sec) // 60
    secs = int(elapsed_sec) % 60
    title = f"운동 완료  {mins:02d}:{secs:02d}"
    _put_text(draw, title, (px + 24, py + 18), _WHITE_RGB, size=28)

    # 헤더
    hy = py + 60
    _put_text(draw, "학생",     (px + 24,          hy), _GRAY_RGB, size=20)
    _put_text(draw, "점프 횟수", (px + 120,         hy), _GRAY_RGB, size=20)
    _put_text(draw, "분당 횟수", (px + panel_w - 180, hy), _GRAY_RGB, size=20)

    # 구분선
    lx = px + 16
    draw.line([(lx, hy + 28), (lx + panel_w - 32, hy + 28)], fill=_GRAY_RGB, width=1)

    # 인원별 행
    row_y = hy + 36
    for stat in stats:
        color = _PERSON_COLORS_RGB[stat.person_id % len(_PERSON_COLORS_RGB)]
        _put_text(draw, f"#{stat.person_id + 1}",           (px + 24,          row_y), color,      size=22)
        _put_text(draw, f"{stat.jump_count}회",             (px + 120,         row_y), _WHITE_RGB, size=22)
        _put_text(draw, f"{stat.jumps_per_min:.1f} 회/분",  (px + panel_w - 180, row_y), _WHITE_RGB, size=22)
        row_y += 52

    # 하단 안내
    _put_text(draw, "아무 키나 누르면 닫힘", (px + 24, py + panel_h - 36), _GRAY_RGB, size=18)

    frame[:] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def _posture_status(report: PostureReport | None) -> str:
    if report is None:
        return "..."
    if report.warnings:
        return "⚠"
    return "✓"
