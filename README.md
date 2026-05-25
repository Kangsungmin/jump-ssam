# jump-ssam — AI 줄넘기 자율학습 서비스

중고등학교 체육 수업의 단체 줄넘기 / 음악줄넘기 활동에서 AI로 학생 개개인의 자세를 실시간 분석·피드백하고, 음악 박자 정확도를 측정하여 **교사의 수업 퀄리티 향상**을 지원하는 교육용 AI 시스템입니다.

---

## 주요 기능

- **실시간 포즈 추정** — MediaPipe로 신체 33개 키포인트 추출 및 자세 분석
- **점프 횟수 자동 카운팅** — 발목 Y좌표 변화 기반 사이클 감지 (목표 정확도 95%)
- **다중 인원 동시 추적** — 최대 10명 동시 감지, ghost 풀 기반 re-identification
- **자세 점수 산출** — 어깨 수평, 팔 대칭, 무릎 굽힘, 상체 기울기 실시간 피드백
- **테스트 UI** — 카운터 리셋 / 민감도 조절 / 종료 버튼 내장

---

## 개발 환경

| 항목 | 내용 |
|---|---|
| Language | Python 3.10+ |
| 포즈 추정 | MediaPipe PoseLandmarker |
| 영상 처리 | OpenCV |
| 카운팅 엔진 | [ropemetrics](https://github.com/Kangsungmin/ropemetrics) |
| 플랫폼 | macOS (프로토타입), Raspberry Pi 5 (목표) |

---

## 설치

### 1. 저장소 클론

```bash
git clone https://github.com/Kangsungmin/jump-ssam.git
cd jump-ssam
```

### 2. 가상환경 생성 및 패키지 설치

```bash
python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

# ropemetrics는 로컬 경로로 설치 (PyPI 미등록)
pip install -e /path/to/ropemetrics
```

### 3. MediaPipe 모델 다운로드

```bash
curl -L -o data/models/pose_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
```

---

## 실행

```bash
source .venv/bin/activate
python main.py
```

### 조작 방법

| 키 / 버튼 | 동작 |
|---|---|
| 화면 버튼 **카운터 리셋** | 전체 점프 횟수 초기화 |
| 화면 버튼 **민감도: 기본** | 느림 → 기본 → 빠름 순환 |
| 화면 버튼 **테스트 종료** | 프로그램 종료 |
| `r` 키 | 전체 카운터 초기화 |
| `q` 키 | 종료 |

---

## 테스트

```bash
source .venv/bin/activate
python -m pytest tests/ -v
```

---

## 프로젝트 구조

```
jump-ssam/
├── main.py                      # 메인 실행 진입점
├── config.py                    # 설정값 (임계값, 카메라 인덱스 등)
├── requirements.txt
│
├── modules/
│   ├── pose_analyzer.py         # MediaPipe 포즈 추정 + 다중 인원 지원
│   ├── posture_analyzer.py      # 자세 점수화 (어깨·팔·무릎·상체)
│   ├── tracker.py               # 다중 인원 추적 + re-identification
│   ├── display.py               # 실시간 HUD + 테스트 버튼
│   ├── audio_analyzer.py        # 마이크 BPM 감지 (Phase 2 예정)
│   ├── dashboard.py             # 교사 대시보드 (Phase 3 예정)
│   └── adapters/
│       └── mediapipe.py         # ropemetrics LandmarkProvider 어댑터
│
├── data/
│   ├── models/                  # MediaPipe 모델 파일
│   └── sessions/                # 수업 기록 SQLite DB
│
└── tests/
    ├── test_counter.py
    ├── test_pose_analyzer.py
    └── test_audio_analyzer.py
```

---

## 개발 로드맵

| Phase | 내용 | 상태 |
|---|---|---|
| Phase 1 | 단일 인원 프로토타입 (웹캠, 포즈 추정, 카운팅, 자세 분석) | ✅ 완료 |
| Phase 2 | 다중 인원 확장 (10~20명 동시 추적, 전용 카메라 연동) | 🔄 진행 중 |
| Phase 3 | 교사 대시보드 및 수업 기록 (SQLite, CSV 내보내기) | 📋 예정 |
| Phase 4 | Raspberry Pi 5 포팅 및 실제 수업 파일럿 테스트 | 📋 예정 |

---

## 라이선스

This project is licensed under the **Business Source License 1.1 (BUSL-1.1)**.

- 비상업적 이용(교육 연구, 개인 학습, 비영리 기관)은 자유롭게 허용됩니다.
- 상업적 이용은 별도 계약이 필요합니다 — 문의: beloksm@gmail.com
- 보호 기간 종료일: **2031-05-25** (이후 Apache License 2.0으로 전환)

자세한 내용은 [LICENSE](./LICENSE) 파일을 참고하세요.
