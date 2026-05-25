import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image
from typing import Optional

import config
from modules.posture_analyzer import PostureReport


_GREEN_RGB  = (0, 220, 0)
_YELLOW_RGB = (255, 210, 0)
_RED_RGB    = (255, 60, 0)
_WHITE_RGB  = (255, 255, 255)
_BLACK_RGB  = (0, 0, 0)

_font_cache: dict = {}


def _font(size: int) -> ImageFont.FreeTypeFont:
    if size not in _font_cache:
        _font_cache[size] = ImageFont.truetype(config.FONT_PATH, size)
    return _font_cache[size]


def _put_text(draw: ImageDraw.ImageDraw, text: str, pos, color, size: int):
    x, y = pos
    draw.text((x + 1, y + 1), text, font=_font(size), fill=_BLACK_RGB)
    draw.text((x, y), text, font=_font(size), fill=color)


def draw_hud(frame, jump_count: int, fps: float, report: Optional[PostureReport], just_jumped: bool):
    h, w = frame.shape[:2]

    if just_jumped:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 255, 255), -1)
        cv2.addWeighted(overlay, 0.07, frame, 0.93, 0, frame)

    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw    = ImageDraw.Draw(pil_img)

    # 점프 카운트 (좌상단)
    count_color = _GREEN_RGB if not just_jumped else _YELLOW_RGB
    _put_text(draw, f"점프: {jump_count}회", (20, 10), count_color, size=52)

    # FPS (우상단)
    _put_text(draw, f"FPS {fps:.1f}", (w - 130, 14), _WHITE_RGB, size=28)

    # 자세 분석 패널 (좌하단)
    if report is None:
        _put_text(draw, "포즈 인식 중...", (20, h - 46), _YELLOW_RGB, size=30)
    else:
        items = [
            ("어깨 수평",   _kor_shoulder(report.shoulder_level)),
            ("팔 대칭",     "대칭" if report.arm_symmetry == "symmetric" else "비대칭"),
            ("무릎 굽힘",   _kor_knee(report.knee_bend)),
            ("상체 기울기", _kor_lean(report.body_lean)),
        ]
        panel_y = h - 170
        for label, value in items:
            color = _GREEN_RGB if _is_good(value) else _YELLOW_RGB
            _put_text(draw, f"{label}: {value}", (20, panel_y), color, size=26)
            panel_y += 34

        if report.warnings:
            msg  = report.warnings[0]
            bbox = draw.textbbox((0, 0), msg, font=_font(28))
            tw   = bbox[2] - bbox[0]
            _put_text(draw, msg, (w // 2 - tw // 2, h - 44), _RED_RGB, size=28)

    frame[:] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def _is_good(value: str) -> bool:
    return value in ("수평", "대칭", "좋음", "수직")

def _kor_shoulder(v):
    return {"level": "수평", "tilted_left": "왼쪽 높음", "tilted_right": "오른쪽 높음"}.get(v, v)

def _kor_knee(v):
    return {"good": "좋음", "too_straight": "너무 펴짐", "too_bent": "너무 굽힘"}.get(v, v)

def _kor_lean(v):
    return {"upright": "수직", "leaning_forward": "앞으로 기움", "leaning_back": "뒤로 기움"}.get(v, v)
