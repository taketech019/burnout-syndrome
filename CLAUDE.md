# CLAUDE.md

본 저장소에서 작업하는 AI 코딩 에이전트용 가이드.

## 프로젝트

- **사업명**: 한국 수험생/직장인 번아웃 조기 감지 (가칭)
- **공모전**: 제12회 보건의료빅데이터·AI 활용 창업경진대회 (마감 2026-05-29)
- **컨셉**: 정신과 의사 진료 보조 도구. HIRA 공공데이터 + 임상 가이드라인 RAG + 위험도 대시보드
- **타겟**: 정신과 의사 (1차)
- **팀**: 학생 4명. Streamlit·LangChain·Git 초급. 코드는 단순·명료 우선.

## 도메인 용어

- **HIRA** (opendata.hira.or.kr): 진료 통계 공공데이터
- **PhysioNet**: 생체신호 공개 데이터. *시연·검증용으로만* 사용 (메인 아님)
- **PHQ-9 / GAD-7 / MBI**: 우울·불안·번아웃 표준 스크리닝 도구

## 기술 스택

- Python 3.11 (3.12 비추), Streamlit, LangChain 0.1.x, ChromaDB (로컬)
- 임베딩: `sentence-transformers` (`BAAI/bge-m3` 권장)
- LLM: OpenAI API (GPT-4o)
- 시각화: plotly, matplotlib, seaborn, altair, 환경변수: python-dotenv

**금지**: Docker, 외부 SQL DB, async 프레임워크, React/Vue, 한글 식별자

## 디렉토리 구조

```
app.py                  # Streamlit 진입점. 라우팅만, 비즈니스 로직 금지
config.py               # 경로·상수 중앙화
requirements.txt
.env / .env.example     # .env는 gitignored

data/                   # ⚠️ 데이터 파일 gitignored, 폴더 구조(.gitkeep)는 Git 추적
  raw/                  # 원본, 읽기 전용
  processed/            # 전처리 결과
  references/           # RAG용 PDF

src/
  data_loader.py        # 로딩만
  analysis.py           # 통계·점수 계산
  rag/                  # 격리된 RAG 파이프라인
    ingest.py           # PDF → 청킹 → 벡터화
    retriever.py
    chain.py
  ui/                   # Streamlit 컴포넌트
    dashboard.py
    chatbot.py

pages/                  # Streamlit 멀티페이지
notebooks/              # 탐색용, 프로덕션 코드 아님
chroma_db/              # ⚠️ gitignored, 벡터 인덱스
```

## 핵심 규칙

### 데이터
- 모든 데이터는 `data/`에. `data/raw/`는 읽기 전용 (덮어쓰기 금지)
- 파일명: `{출처}_{주제}_{필터}_{버전}_{날짜}.csv`
  - 예: `hira_mental_health_seoul_20s_v1_20260513.csv`
- "latest", "final", "최신" 같은 이름 금지
- 데이터 파일 Git 커밋 금지
- `.gitignore` 패턴: `data/**` + `!data/*/` + `!data/**/.gitkeep` (폴더 구조만 추적)

### 비밀 키
- 모든 API 키는 `.env`. 코드 하드코딩 금지
- `os.getenv()` + 누락 시 명확한 에러 메시지
- LLM 호출은 try/except (한도 초과 대비)

### 환자 정보
- 개인식별정보를 LLM에 전송 금지. 익명화·마스킹된 텍스트만

### 경로
- 하드코딩 금지. `config.DATA_DIR` 사용

## 브랜치·커밋

### 브랜치
- `main`: 보호, 직접 push 금지, PR로만 머지
- 작업: `feature/{기능}`, `fix/{버그}`, `chore/{잡일}`

### 커밋 메시지

```
feat:     새 기능
fix:      버그 수정
docs:     문서
refactor: 동작 변경 없는 정리
chore:    의존성·설정
```

예: `feat: HIRA 정신질환 데이터 로더 구현`
금지: `update`, `wip`, `최종` 같은 모호한 메시지

## 코딩 컨벤션

- PEP 8, 줄 길이 100
- `snake_case` 함수/변수, `PascalCase` 클래스, `UPPER_SNAKE_CASE` 상수
- 공개 함수 타입 힌트 + docstring (한국어 OK)
- 임포트 순서: 표준 → 서드파티 → 로컬
- Streamlit: 컴포넌트는 함수로 감싸기, 무거운 연산은 `@st.cache_data`

## 명령어

```bash
# 최초 셋업
python -m venv venv
source venv/bin/activate         # 윈도우: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env

# 실행
streamlit run app.py

# RAG 인덱스 빌드 (PDF 추가/변경 시)
python -m src.rag.ingest

# 의존성 갱신
pip install <패키지>
pip freeze > requirements.txt
```

## 안티패턴 (하지 말 것)

1. 데이터 파일을 Git에 커밋
2. API 키 하드코딩
3. `main`에 직접 push
4. 노트북 코드 그대로 import (안정화 후 `src/`로 이관)
5. `from x import *`
6. 상대경로 하드코딩 (`config.DATA_DIR` 사용)
7. 환자 개인정보를 LLM에 전송
8. requirements.txt 갱신 누락
9. 같은 파일을 두 명이 동시 수정

## 작업 시작 전

- 현재 브랜치가 `main`이 아닌지
- 작업 영역(폴더) 파악
- 새 의존성은 사용자 확인 후 추가
- 데이터 출력은 `data/processed/`에 새 파일 (덮어쓰기 X)
- LLM 호출 try/except + 에러 메시지
- 변경 후 변경 파일·이유 요약