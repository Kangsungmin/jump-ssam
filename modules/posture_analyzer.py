import numpy as np
from dataclasses import dataclass, field
from typing import Optional

import config


@dataclass
class PostureReport:
    shoulder_level: str = "unknown"   # "level" | "tilted_left" | "tilted_right"
    arm_symmetry:   str = "unknown"   # "symmetric" | "asymmetric"
    knee_bend:      str = "unknown"   # "good" | "too_straight" | "too_bent"
    body_lean:      str = "unknown"   # "upright" | "leaning_forward" | "leaning_back"
    warnings: list = field(default_factory=list)


def _angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba  = a - b
    bc  = c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))


class PostureAnalyzer:
    def analyze(self, pose_results, pose_analyzer) -> Optional[PostureReport]:
        get = lambda name: pose_analyzer.get_landmark(pose_results, name)

        ls = get("LEFT_SHOULDER");  rs = get("RIGHT_SHOULDER")
        lh = get("LEFT_HIP");       rh = get("RIGHT_HIP")
        lk = get("LEFT_KNEE");      rk = get("RIGHT_KNEE")
        la = get("LEFT_ANKLE");     ra = get("RIGHT_ANKLE")
        lw = get("LEFT_WRIST");     rw = get("RIGHT_WRIST")

        if any(v is None for v in [ls, rs, lh, rh, lk, rk, la, ra]):
            return None

        report = PostureReport()

        # 어깨 수평
        shoulder_diff = abs(ls[1] - rs[1])
        if shoulder_diff < config.SHOULDER_LEVEL_THRESHOLD:
            report.shoulder_level = "level"
        elif ls[1] < rs[1]:
            report.shoulder_level = "tilted_right"
            report.warnings.append("오른쪽 어깨가 올라가 있어요")
        else:
            report.shoulder_level = "tilted_left"
            report.warnings.append("왼쪽 어깨가 올라가 있어요")

        # 팔 대칭
        if lw is not None and rw is not None:
            wrist_diff = abs(lw[1] - rw[1])
            report.arm_symmetry = "symmetric" if wrist_diff < config.ARM_SYMMETRY_THRESHOLD else "asymmetric"
            if report.arm_symmetry == "asymmetric":
                report.warnings.append("양팔 높이가 비대칭이에요")

        # 무릎 굽힘
        avg_knee = (_angle_deg(lh[:2], lk[:2], la[:2]) + _angle_deg(rh[:2], rk[:2], ra[:2])) / 2
        if avg_knee > config.KNEE_STRAIGHT_ANGLE:
            report.knee_bend = "too_straight"
            report.warnings.append("무릎을 살짝 구부려 착지 충격을 줄이세요")
        elif avg_knee < config.KNEE_BENT_ANGLE:
            report.knee_bend = "too_bent"
            report.warnings.append("무릎이 너무 많이 굽혀져 있어요")
        else:
            report.knee_bend = "good"

        # 상체 기울기
        lean = ((ls[0] + rs[0]) / 2) - ((lh[0] + rh[0]) / 2)
        if abs(lean) < config.BODY_LEAN_THRESHOLD:
            report.body_lean = "upright"
        elif lean > 0:
            report.body_lean = "leaning_forward"
            report.warnings.append("상체가 앞으로 기울어 있어요")
        else:
            report.body_lean = "leaning_back"
            report.warnings.append("상체가 뒤로 기울어 있어요")

        return report
