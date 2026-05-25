---
description: 개발 중 발생한 이슈를 GitHub에 생성한다. ai-jumprope 앱 이슈는 Kangsungmin/jump-ssam에, ropemetrics 라이브러리 이슈(버그·API 개선·새 백엔드)는 Kangsungmin/ropemetrics에 생성한다.
---

## 이슈 분류 기준

**Kangsungmin/jump-ssam** — 앱 레벨 이슈:
- HUD, 자세 분석, 오디오 BPM, 교사 대시보드
- MediaPipe 연동 문제 (PoseAnalyzer, PoseResults)
- Phase 기능 구현 태스크 (SPEC.md 기준)
- 카메라, OpenCV, 한글 폰트 등 앱 환경 문제

**Kangsungmin/ropemetrics** — 라이브러리 이슈:
- `JumpCounter`, `JumpStrategy`, `LandmarkProvider`, `JumpCounterConfig` 버그·개선
- `AnkleStrategy` / `HipShoulderStrategy` 오탐·미탐
- 새 포즈 백엔드(MoveNet, YOLOv8-Pose 등) 어댑터 요청
- pyproject.toml, 패키징, 배포 관련
- 라이브러리 공개 API 설계 변경

분류가 애매하면 사용자에게 어느 레포인지 확인한 후 진행한다.

## 이슈 생성 절차

1. `$ARGUMENTS`(또는 대화 맥락)에서 제목·내용·유형을 파악한다.
2. 분류 기준으로 대상 레포를 결정한다.
3. 적절한 라벨을 선택한다 (아래 라벨 가이드 참고).
4. `gh issue create` 명령으로 이슈를 생성한다.
5. 생성된 이슈 URL을 사용자에게 출력한다.

## 명령 형식

```bash
# ai-jumprope 이슈
gh issue create \
  --repo Kangsungmin/jump-ssam \
  --title "<제목>" \
  --body "<내용>" \
  --label "<라벨>"

# ropemetrics 이슈
gh issue create \
  --repo Kangsungmin/ropemetrics \
  --title "<제목>" \
  --body "<내용>" \
  --label "<라벨>"
```

## 라벨 가이드

| 라벨 | 사용 시 |
|------|---------|
| `bug` | 오류, 예외, 잘못된 동작 |
| `enhancement` | 기존 기능 개선 |
| `feature` | 신규 기능 요청 |
| `documentation` | 문서 부족 또는 오류 |
| `question` | 설계·방향 논의 |

## 이슈 본문 형식

ropemetrics 이슈는 재현 방법을 포함한다:

```
## 문제
<현상 설명>

## 재현 방법
<최소 재현 코드 또는 단계>

## 기대 동작
<정상적으로 동작해야 하는 방식>

## 환경
- ropemetrics: <버전>
- Python: <버전>
- 포즈 백엔드: MediaPipe <버전>
```

ai-jumprope 이슈는 SPEC.md Phase와 관련 모듈을 언급한다.
