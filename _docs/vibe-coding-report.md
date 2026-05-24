# Vibe Coding Report — MVP F1~F5 통합 구현 (2026-05-24)

| 작성일 | 2026-05-24 04:00 ~ 05:10 KST |
| --- | --- |
| 범위 | PRD §3 MVP 전 기능 (F1~F5) 통합 구현 + KlueBERT 진단 + RAG 인덱싱 |
| 환경 | Windows 11, Python 3.13, CPU only (GTX 1070 Ti 미사용) |

## 한 줄 요약

PRD F1~F5 MVP 전 기능을 동작 상태로 통합 완성. KlueBERT는 변별력 부족 진단 후 **Gemma 4 31B 1차 판별**로 우회. RAG 인덱스 3,316건 임베딩 완료. KoAlpaca Modal 콜드 시 **Gemma 4 31B 폴백**으로 F3 보장.

## 변경 사항 요약

| 모듈 | 변경 | 검증 |
| --- | --- | --- |
| `src/classifier.py` | Gemma 4 31B 0~3 정도값 기반 1차 판별 (KlueBERT 변별력 부족 진단). 환경변수로 로컬 KlueBERT 보조 추론 가능. | 우울/불안/중독 정상군 4종 텍스트로 변별 확인 ✓ |
| `src/gemma_client.py` (신규) | Gemma 4 31B 공통 클라이언트. 500/transient 재시도, Gemini 2.5 Flash 폴백, JSON 추출, 영어 reasoning bullet 자동 strip. | 단위 호출 OK ✓ |
| `src/factor_extractor.py` | Gemini → Gemma 4 31B 교체. 64-라벨 발화별 0/1 분류. | 15발화 데모에서 자살 사고=2, 사회적 고립=2 정확히 라벨링 ✓ |
| `src/summarizer.py` | KoAlpaca → 실패/짧은 입력 시 **Gemma 4 31B 폴백** 추가. 4섹션 헤더 파싱 강화 (bullet/heading 양식 모두). | 586자 데모 입력으로 4섹션 모두 한국어 의미있는 요약 생성 ✓ |
| `src/rag/chain.py` | Ollama Qwen2.5 → Gemma 4 31B 교체. KoSBERT 임베딩 유지. reasoning strip 적용. | RAG 응답 한국어 단일 단락 + 출처 3개 ✓ |
| `src/rag/ingest.py` | AI Hub `.zip.part0` ZIP 직접 처리, PDF + DOCX + HIRA CSV 통합 인덱싱. KoSBERT 배치 임베딩. | 3,316개 문서 인덱싱 → `chroma_db/chroma.sqlite3` 12MB ✓ |
| `src/hira.py` (신규) | 건보심사평가원 시군구·성별·연령별 CSV (cp949) 로드, 내담자 메타로 환자수 매칭. | 30대 여성 서울 우울증 → 56,274명 (전국 160,392명) ✓ |
| `src/dashboard.py` | HIRA stub → 실데이터 호출. | E2E ✓ |
| `app.py` | Streamlit 5페이지 라우팅 (실수로 HF Space 코드로 덮인 것 복구). | `streamlit run app.py` → HTTP 200 ✓ |
| `requirements.txt` | torch / transformers / chromadb / langchain / sentence-transformers / weasyprint / pypdf / python-docx / google-generativeai / kaleido 모두 명시. | 신규 venv에서 재현 가능. |

## 핵심 결정

### KlueBERT 변별력 없음 — Gemma로 대체

- **증거**: HF Space `/healthz`의 `regressor_weight_norm=0.517, mean=-0.0001, std=0.018, bias=0.0001` — 학습된 가중치는 있지만 어떤 입력이든 raw 출력 ~1.2 (binary=1)로 변별 0.
- **로컬 weights도 같음**: `ai-model/kluebert/2.AI학습모델파일/` weights 직접 로드해도 정상/우울/불안/중독 4종 텍스트 모두 ~1.2 출력.
- **원인 추정**: MSE 손실 학습 시 라벨 분포가 1 근처로 mode collapse. 재학습 없이는 복구 불가.
- **해결**: 1차 판별을 **Gemma 4 31B 0~3 정도값**으로 교체. 정상군 변별 + PRD 4분류 모두 충족. KlueBERT는 환경변수 `CLASSIFIER_INCLUDE_KLUEBERT=1` 시 보조 raw 값 첨부.

### Gemma 4 31B 특성과 후처리

- Google AI Studio `models/gemma-4-31b-it` 사용 가능 확인 (input_token_limit=262K).
- **JSON 모드 미지원** (`response_mime_type` 무시) → 프롬프트 + `extract_json()` 균형 괄호 매칭으로 추출.
- **영어 reasoning bullet 강박 출력** ("* Role:", "* Material 1:") → `strip_reasoning()`로 한국어 단락만 남김.
- **500 Internal Error 빈도** → 재시도 + Gemini 2.5 Flash 폴백.

### RAG 인덱스 구성 (3,316건)

| 소스 | 건수 |
| --- | --- |
| AI Hub 심리상담 라벨링 ZIP 발화 | 2,500 (cap) |
| PDF (종검보고서 워크샵 교안) | 31 페이지 |
| DOCX (한국상담심리학회 윤리규정 + 한국상담학회 윤리강령) | 336 |
| HIRA CSV (4개 통계 파일 헤더 + 행) | 449 |

- 임베딩: KoSBERT (`snunlp/KR-SBERT-V40K-klueNLI-augSTS`) CPU 배치 200건.
- 인덱싱 시간: ~5분 (3316건 / 200 batch × ~18 배치).
- 환경변수로 `RAG_MAX_AIHUB_DOCS` 조정 가능 (기본 3000).

### F3 KoAlpaca 폴백

- KoAlpaca Modal 콜드스타트 60~120초, 또는 5000자 미만 입력에선 4섹션 안 나옴 (학습 분포 median 22,872자).
- **Gemma 4 31B 폴백** 추가 → 짧은 시연 입력 (~600자)에서도 4섹션 한국어 요약 완성.
- KoAlpaca 살아 있고 입력 ≥ 5000자 시 우선 사용. 그 외는 Gemma.

## 데모 시드

`python scripts/seed_demo.py`
- 데모A (여성/32/서울) — 우울 + 자살 사고 + 음주, transcript 600자 14발화. F1+F3 완료 상태.
- 데모B (남성/45/부산) — 회기 없음 (정상군 시연용).

## 알려진 제약

1. **`.pdf` 다운로드**: weasyprint Windows GTK 의존성으로 비활성. `.md`/`.docx` 권장. UI에서 graceful fallback.
2. **kaleido PNG 임베드**: Windows에서 처음 호출 시 chrome 부트스트랩으로 느림 (~30s). docx 차트는 best-effort.
3. **KlueBERT raw 디버그**: `CLASSIFIER_INCLUDE_KLUEBERT=1` 시에만 보조 raw 값 첨부 (기본 OFF, Gemma 결과만).
4. **Gemma 4 31B 500 에러**: 자동 재시도 + Gemini Fallback. 그래도 실패 시 명시적 error 키 반환.
5. **HIRA CSV "정실질환"**: 원본 파일명에 오타 (정실 = 정신). 컬럼 자동 매칭은 "상별구분"으로 OK.

## 확인 명령

```bash
# (1) 환경
venv/Scripts/python -c "import torch, transformers, chromadb, langchain, sentence_transformers; print('OK')"

# (2) 데모 시드
python scripts/seed_demo.py

# (3) RAG 인덱스 (이미 빌드됨)
ls chroma_db/   # chroma.sqlite3 12MB

# (4) Streamlit 실행
streamlit run app.py
```
