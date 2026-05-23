# PR #3 변경 요청 — feature/minjung

| 작성일 | 2026-05-23 |
|--------|------------|
| 대상 | PR #3 작성자 (민정) |
| PR | https://github.com/taketech019/burnout-syndrome/pull/3 (`feature/minjung` → `main`) |
| 상태 | Request Changes — 머지 차단 4건 |

KlueBERT classifier, factor extractor, Koalpaca summarizer를 실제 백엔드와 연결한 작업 자체는 큰 진척입니다. 다만 머지 전에 반드시 잡아야 할 항목 4개가 있어 정리합니다. 본 PR을 다른 PC(`feature/yoon`)에서 worktree로 격리 체크아웃해 실측한 결과 기준입니다.

---

## 1. 분석 대시보드 즉시 크래시 (HIGH) — `app.py:1167`

분석 대시보드 페이지가 mock·실제 백엔드 무관 100% 크래시합니다. 들여쓰기 한 단계 실수로 `backend` 변수가 unreachable.

### 재현
1. worktree로 `feature/minjung` 체크아웃
2. `streamlit run app.py`
3. 상담내역 화면에서 **AI 분석 실행** → 자동으로 분석 대시보드로 이동
4. `UnboundLocalError: cannot access local variable 'backend' where it is not associated with a value`

### 원인 (app.py:1163-1167)

```python
if result is None:
    st.info("아직 분석 결과가 없습니다. ...")
    return

    backend = result.get("model_info", {}).get("backend", "unknown")  # ← 4칸 들여쓰기 잘못
```

`backend = ...`가 `if result is None:` 블록 **내부**에 들여쓰여 있습니다. `return` 뒤에 있어 어떤 경우에도 실행되지 않고, 라인 1186 `if backend == "mock":`에서 참조 실패.

### 수정 (1줄)

```python
if result is None:
    st.info("아직 분석 결과가 없습니다. ...")
    return

backend = result.get("model_info", {}).get("backend", "unknown")
```

### 추가 권고
로컬에서 모든 페이지(상담내역·**대시보드**·보고서·챗봇)를 한 번씩 클릭 테스트한 뒤 push 부탁드립니다. `ruff check app.py` 또는 `python -m pyflakes app.py`로 같은 패턴(unreachable / unbound)이 더 있는지도 확인 부탁드립니다.

---

## 2. `.python-version` 파일 삭제됨

PR diff에 `.python-version | 1 -` 로 표시됩니다. main에 있던 한 줄짜리 파일(`3.11`)이 통째로 사라집니다.

### 영향
- Streamlit Community Cloud가 디폴트 Python(현재 **3.14.5**)으로 환경을 만듬
- 우리가 쓰는 패키지들 다수가 Python 3.14용 wheel이 없어서 소스 빌드 → **uv pip install이 그대로 hang** (이번 주에 같은 증상으로 5분+ 멈춤 발생, 결국 `.python-version`을 추가해 해결한 이력)

### 요청
`.python-version` 파일을 그대로 복원해주세요. 내용은 한 줄:
```
3.11
```

CLAUDE.md §환경 항목도 Python 3.11 기준입니다.

---

## 3. ML 의존성을 루트 `requirements.txt`에 추가 — 명세 위반

`torch`, `transformers`, `safetensors` 3개가 루트에 추가되었습니다. 하지만 `_docs/nf4-hosting.md` §Step 1과 CLAUDE.md는 명시적으로 이 패키지들을 **분리**하라고 적고 있습니다.

### 분리 원칙
| 파일 | 환경 | 들어가는 것 |
|------|------|-------------|
| 루트 `requirements.txt` | Streamlit Cloud (경량) | streamlit, plotly, pandas 등 UI/보고서용 |
| `ai-model/requirements-nf4.txt` (신규) | 로컬 GPU 서버 | torch, transformers, bitsandbytes, peft 등 |

CLAUDE.md 인용:
> **Koalpaca 12.8B**: Streamlit Cloud에서 실행 불가 → 외부 추론 API(Modal 등)로 호출

즉 Streamlit Cloud 빌드에 `torch`가 있을 이유 자체가 없습니다. `src/summarizer.py`는 KoAlpaca를 HTTP(`KOALPACA_ENDPOINT_URL`)로만 호출하므로 Streamlit 측은 `requests`만 있으면 충분합니다.

### 요청

**(a)** 루트 `requirements.txt`에서 다음 3줄 제거:
```
torch
transformers
safetensors
```

**(b)** `ai-model/requirements-nf4.txt` 신규 생성 후 다음 내용 (nf4-hosting.md §Step 1과 동일):
```
bitsandbytes>=0.43.0
transformers>=4.40.0
peft>=0.10.0
accelerate>=0.30.0
torch>=2.1.0
safetensors
fastapi>=0.110.0
uvicorn[standard]>=0.27.0
pydantic>=2.0
sentencepiece
protobuf
```

버전 핀이 들어간 이유: 현재 PR은 `torch`·`transformers`·`safetensors`에 버전 명시가 없어 빌드마다 다른 버전이 깔립니다. NF4 호스팅과 호환되는 최소 버전을 명시해야 재현성이 확보됩니다.

---

## 4. main의 `dae1dda` 정리가 통째로 되돌려짐

PR base가 옛 main(`dae1dda` 이전)에서 분기되어 있어, 머지 시 다음 정리들이 모두 회귀합니다.

### 회귀하는 변경

| 부활하는 패키지 | 문제 |
|------|------|
| `langchain-openai==0.1.6`, `openai==1.30.1` | PRD는 Google AI(Gemma 4 31B) 사용 — OpenAI 미사용 |
| `matplotlib==3.9.0`, `seaborn==0.13.2`, `altair==5.3.0` | 코드 어디서도 import 안 함. altair는 Streamlit이 번들링 |
| `chromadb==0.4.24`, `sentence-transformers==2.7.0`, `langchain==0.1.20` | RAG(F4) 미구현 상태에서 미리 추가됨. sentence-transformers는 torch를 또 끌어옴 → 빌드 시간 폭증 |

### 누락되는 변경 (F3 보고서 기능 패키지)

| 누락 패키지 | 영향 |
|------|------|
| `kaleido==0.2.1` | plotly 차트를 이미지로 export — 보고서에 차트 삽입 불가 |
| `markdown==3.6` | .md 보고서 생성 |
| `python-docx==1.1.2` | .docx 보고서 생성 |

세 패키지는 PRD §F3에 명시된 기능이며, 머지하면 보고서 다운로드 기능이 깨집니다.

### 요청
PR 브랜치를 최신 `main`(현재 HEAD `dae1dda` 이후)으로 **rebase** 부탁드립니다.

```bash
git fetch origin
git checkout feature/minjung
git rebase origin/main
# 충돌 발생 시: requirements.txt는 main 기준(dae1dda 정리본)을 살린 채,
# 본인이 추가한 ML 의존성은 위 §3의 ai-model/requirements-nf4.txt로 옮기기
git push --force-with-lease
```

---

## 권장 작업 순서

1. `main` rebase (위 §4)
2. `.python-version` 복원 (위 §2)
3. `app.py:1167` 들여쓰기 수정 (위 §1)
4. ML 의존성을 `ai-model/requirements-nf4.txt`로 분리 (위 §3)
5. 로컬에서 모든 페이지 클릭 테스트 — 상담내역 입력 → AI 분석 → **분석 대시보드** → AI 보고서 → 챗봇
6. `ruff check` 또는 `pyflakes`로 lint 통과 확인
7. force-push 후 PR 갱신

## 참고 문서
- `_docs/PRD.md` §7 기술 스택
- `_docs/nf4-hosting.md` §1 환경 가정 / §Step 1 의존성 분리
- `CLAUDE.md` (안티패턴, 환경 격리)

## 문의
위 4건 중 의도가 다르거나 막히는 부분 있으면 `feature/yoon` 쪽에 코멘트 주세요. 같이 풀어봅시다.
