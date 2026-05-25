from pathlib import Path

# ── 카메라 ──────────────────────────────────────────────
CAMERA_INDEX  = 0
FRAME_WIDTH   = 1280
FRAME_HEIGHT  = 720
TARGET_FPS    = 30

# ── 포즈 추정 ────────────────────────────────────────────
MODEL_PATH            = Path("data/models/pose_landmarker.task")
POSE_CONFIDENCE       = 0.6   # detection / tracking 공통 신뢰도
MODEL_DOWNLOAD_URL    = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)

# ── 얼굴 감지 ─────────────────────────────────────────────
FACE_MODEL_PATH       = Path("data/models/face_detector.tflite")
FACE_CONFIDENCE       = 0.5   # 얼굴 감지 최소 신뢰도
FACE_POSE_MAP_THRESH  = 0.25  # 얼굴-포즈 매핑 거리 임계값 (정규화 좌표)
FACE_MODEL_DOWNLOAD_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
)

# ── 횟수 카운팅 ──────────────────────────────────────────
JUMP_THRESHOLD    = 0.025   # 발목 Y좌표 변화 임계값 (정규화 좌표)
MIN_JUMP_FRAMES   = 3       # 최소 점프 인정 프레임 수
COOLDOWN_FRAMES   = 8       # 연속 카운팅 방지 쿨다운
JUMP_HISTORY_SIZE = 10      # 발목 Y좌표 이동평균 윈도우 크기

# ── 자세 분석 임계값 ─────────────────────────────────────
SHOULDER_LEVEL_THRESHOLD = 0.03   # 어깨 수평 판정 Y차이 (정규화)
ARM_SYMMETRY_THRESHOLD   = 0.06   # 손목 대칭 판정 Y차이 (정규화)
KNEE_STRAIGHT_ANGLE      = 165    # 무릎 각도 이상이면 "너무 펴짐" (도)
KNEE_BENT_ANGLE          = 130    # 무릎 각도 이하이면 "너무 굽힘" (도)
BODY_LEAN_THRESHOLD      = 0.04   # 상체 기울기 X차이 (정규화)

# ── 박자 분석 (Phase 1 - 추후 구현) ──────────────────────
AUDIO_SAMPLE_RATE  = 44100
BPM_HOP_LENGTH     = 512
BEAT_TOLERANCE_MS  = 50     # 박자 오차 허용 범위 (ms)

# ── 다중 인원 추적 ──────────────────────────────────────
MAX_TRACKED_PERSONS        = 10   # MediaPipe 최대 동시 감지 인원
TRACKER_MAX_MISS_FRAMES    = 15   # 미감지 허용 프레임 수 (30fps 기준 ~0.5초)
TRACKER_MATCH_THRESHOLD    = 0.3  # 활성 추적 centroid 매칭 거리 임계값 (정규화 좌표)
TRACKER_GHOST_EXPIRE_FRAMES = 150 # ghost 보관 프레임 수 (30fps 기준 5초)
TRACKER_GHOST_MATCH_THRESHOLD = 0.4  # 재등장 매칭 거리 임계값 (정규화 좌표)

# ── 얼굴+신체 융합 식별 ─────────────────────────────────────
FUSION_WEIGHT_FACE      = 0.4   # 얼굴 임베딩 유사도 가중치
FUSION_WEIGHT_CENTROID  = 0.4   # 신체 centroid 거리 가중치
FUSION_WEIGHT_POSE_CONF = 0.2   # 포즈 신뢰도 가중치
FUSION_MATCH_THRESHOLD  = 0.6   # 동일 인물 판정 융합 스코어 임계값
FUSION_EMBED_INTERVAL   = 30    # 임베딩 갱신 주기 (프레임 수)

# ── 디스플레이 ───────────────────────────────────────────
FONT_PATH        = "/System/Library/Fonts/AppleSDGothicNeo.ttc"  # macOS
JUMP_FLASH_FRAMES = 4       # 점프 시 플래시 지속 프레임 수
