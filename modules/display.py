import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image
from dataclasses import dataclass

import config
from modules.posture_analyzer import PostureReport

_WHITE_RGB  = (255, 255, 255)
_BLACK_RGB  = (0,   0,   0)
_RED_RGB    = (255, 60,  0)
_GRAY_RGB   = (180, 180, 180)
_YELLOW_RGB = (255, 210, 0)

_BTN_RESET_BG = (40,  130, 80)   # 초록
_BTN_SENS_BG  = (40,  90,  180)  # 파랑
_BTN_QUIT_BG  = (180, 40,  40)   # 빨강
_BTN_BORDER   = (210, 210, 210)

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
    reset:       tuple  # (x, y, w, h)
    sensitivity: tuple
    quit:        tuple

    def hit(self, name: str, mx: int, my: int) -> bool:
        bx, by, bw, bh = getattr(self, name)
        return bx <= mx <= bx + bw and by <= my <= by + bh


def make_button_layout(frame_w: int, frame_h: int) -> ButtonLayout:
    bh = 52
    by = frame_h - bh - 10
    return ButtonLayout(
        reset       = (20,              by, 150, bh),
        sensitivity = (frame_w // 2 - 115, by, 230, bh),
        quit        = (frame_w - 170,   by, 150, bh),
    )


def draw_hud(frame, persons: dict, fps: float, layout: ButtonLayout, sens_label: str):
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

    # 하단 버튼 바 배경 (반투명)
    bar_y = layout.reset[1] - 8
    bar_bg = frame.copy()
    cv2.rectangle(bar_bg, (0, bar_y), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(bar_bg, 0.6, frame, 0.4, 0, frame)

    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw    = ImageDraw.Draw(pil_img)

    # FPS (우상단)
    _put_text(draw, f"FPS {fps:.1f}", (w - 130, 14), _GRAY_RGB, size=26)

    # 감지 인원 수 (좌상단)
    _put_text(draw, f"감지: {len(persons)}명", (20, 14), _WHITE_RGB, size=26)

    # 인원별 신체 위에 뱃지
    for pid, info in persons.items():
        color = _YELLOW_RGB if info["flash"] > 0 else _PERSON_COLORS_RGB[pid % len(_PERSON_COLORS_RGB)]
        cx_norm, cy_norm = info.get("centroid", (0.5, 0.5))
        cx = int(cx_norm * w)
        cy = int(cy_norm * h)

        count = info["counter"].count
        label = f"#{pid + 1}  {count}회"
        badge_y = max(cy - 130, 50)
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

    # 버튼 3개
    _draw_button(draw, layout.reset,       "카운터 리셋",          _BTN_RESET_BG, size=22)
    _draw_button(draw, layout.sensitivity, f"민감도: {sens_label}", _BTN_SENS_BG,  size=22)
    _draw_button(draw, layout.quit,        "테스트 종료",           _BTN_QUIT_BG,  size=22)

    frame[:] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def _posture_status(report: PostureReport | None) -> str:
    if report is None:
        return "..."
    if report.warnings:
        return "⚠"
    return "✓"
