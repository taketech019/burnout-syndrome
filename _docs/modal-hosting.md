# Modal 호스팅 운영 가이드 — KoAlpaca NF4 + LoRA

| 작성일 | 2026-05-24 |
| --- | --- |
| 대상 | 본 리포 운영자, 공모전 심사 기간(~2주) 동안 F3 요약 모델 호스팅 책임자 |
| 예상 비용 | $0~15 (사용량 따라, 신규 $30 Free credit 안) |
| 엔드포인트 | `https://taketech019--koalpaca-nf4-inference-summarize.modal.run` |

## 1. 왜 Modal인가

`_docs/뻘짓.md` + `_docs/path-a-results.md` 참조.

요지: 1070 Ti 8GB로는 12.8B 모델 NF4 + LoRA를 학습 환경(A100 전부 GPU)대로 호스팅 불가. CPU offload는 LoRA 효과 약화, llama.cpp는 양자화 mismatch. Modal A10G(24GB)에서 전부 GPU 적재로 학습 환경 재현 → **partial pass (2/4 헤더, raw 폴백)** 안정적 재현.

## 2. 배포 / 재배포

```bash
# 신규 또는 코드 변경 시
modal deploy ai-model/nf4_modal.py
```

요구:
- Modal 계정 + CLI (`pip install modal && modal token new`)
- `koalpaca-secret` Modal secret 존재 (KOALPACA_API_KEY 키)
  - 신규 생성: `modal secret create koalpaca-secret KOALPACA_API_KEY=<랜덤 32byte>`
- 어댑터(`ai-model/koalpaca_save/`)는 코드와 함께 image에 mount됨

**처음 1회 cold start**: ~3분 (12.8B base 24GB를 HuggingFace → Modal Volume `koalpaca-hf-cache`로 다운). 이후 cold start는 Volume cache hit으로 ~30~60초.

## 3. 비용 구조

A10G: **$0.000306/sec** ≈ $1.10/시간

| 시나리오 | 시간 | 비용 |
| --- | --- | --- |
| 데모/심사 (3시간 사용/2주) | 3h | $3.30 |
| 적극 사용 (10시간 사용) | 10h | $11.00 |
| 매일 1시간 × 14일 always-warm | 14h | $15.40 |

신규 계정 $30 free credit 안에서 모두 커버.

**scaledown_window=120s**: 마지막 요청 후 2분 idle → 컨테이너 종료 → 청구 중단. 다음 요청 시 cold start(~30~60s).

scaledown_window 조정 (`ai-model/nf4_modal.py` 내):
- 더 짧게(60s) → 비용 절감, 콜드스타트 잦음
- 더 길게(600s) → 같은 세션 안에서 warm, 비용 증가

## 4. 운영 명령어

| 작업 | 명령어 |
| --- | --- |
| 앱 상태 확인 | `modal app list` |
| 로그 보기 | `modal app logs koalpaca-nf4` |
| 즉시 중지 (idle 대기 없이) | `modal app stop koalpaca-nf4 --yes` |
| 재시작 (= 재배포) | `modal deploy ai-model/nf4_modal.py` |
| 컨테이너 강제 콜드 (캐시 워밍) | `curl https://...inference-health.modal.run` 호출 후 대기 |
| 비용 대시보드 | https://modal.com/settings/usage |

## 5. 호출 인터페이스

```http
POST https://taketech019--koalpaca-nf4-inference-summarize.modal.run
Content-Type: application/json

{
  "api_key": "<KOALPACA_API_KEY>",
  "transcript": "내담자\\t...\n상담사\\t...",
  "max_new_tokens": 1024,
  "temperature": 0.0,
  "repetition_penalty": 1.2,
  "no_repeat_ngram_size": 3
}
```

응답:
```json
{ "text": "주요 증상: ...\n위험 요인: ...\n..." }
```

`src/summarizer.py`가 이 호출을 wrap. API 키/URL은 `.env`의 `KOALPACA_ENDPOINT_URL`, `KOALPACA_API_KEY`로 주입.

## 6. 입력 요구사항 (필수)

`src/summarizer.py`가 자동 처리하지만 호출자가 알아야 할 사항:

| 항목 | 요구사항 |
| --- | --- |
| 발화자 구분 | `발화자\t텍스트` (탭 구분). `발화자 : 텍스트` 입력 시 summarizer가 자동 변환 |
| 비식별화 | `@NAME`, `@AGE`, `@PLACE` 등 마커 (학습 분포). MVP 데모 데이터는 이미 적용됨 |
| 길이 | **최소 5000자**. 3000자 이하면 4섹션 트리거 실패 (검증 완료) |
| 권장 | 5000~9000자. 9000자 초과는 모델 max_pos 2048 토큰 초과로 truncate |

## 7. 알려진 한계 / partial pass

- 4섹션 중 **주요 증상 + 위험 요인 = 안정적 출력 (2/4)**
- 개선 요인 / 개입 요인은 자주 누락 → `raw` 폴백으로 전체 텍스트 반환
- AI Hub 공식 F-1 60.94% 환경 100% 재현 불가 (xlsx 학습 데이터 입수 불가, 데이터 leakage 가능성)
- 응답 시간: warm 컨테이너 기준 60~120초 (max_tokens=1024, 9000자 입력)

## 8. 심사 종료 후 정리

```bash
modal app stop koalpaca-nf4 --yes
modal volume delete koalpaca-hf-cache  # 선택 — 24GB HF 캐시 영구 삭제
modal secret delete koalpaca-secret    # 선택
```

코드는 리포에 보존 (`ai-model/nf4_modal.py`, `koalpaca_save/`), 추후 다른 LoRA로 재활용 가능.

## 9. 트러블슈팅

| 증상 | 원인 / 대응 |
| --- | --- |
| 401 invalid api_key | `.env`의 `KOALPACA_API_KEY` ≠ Modal secret 값. 재확인 또는 secret 재생성 |
| 408 timeout | 입력이 너무 길거나 cold start. `_TIMEOUT=300` 충분, 그래도 timeout이면 Modal 로그 확인 |
| 응답에 헤더 0개 | 입력이 5000자 미만 → summarizer가 미리 에러 반환. 또는 OOD 입력. raw 폴백 활용 |
| Cold start 5분+ | Modal Volume 캐시 미스 (재배포 후 첫 호출). 정상. 두 번째 호출은 빠름 |
| modal CLI Windows에서 cp949 에러 | `PYTHONUTF8=1 modal deploy ...` 로 우회 |

## 10. 관련 파일

- `ai-model/nf4_modal.py` — Modal 앱 정의 (Inference 클래스, /health, /summarize)
- `ai-model/koalpaca_save/` — LoRA 어댑터 + tokenizer (image에 포함, 50MB)
- `src/summarizer.py` — Modal /summarize 호출 wrap
- `.env.example` — 환경변수 양식
- `_docs/뻘짓.md` — 왜 로컬 호스팅 포기했나
- `_docs/path-a-results.md` — Path A 시도/실패 상세
