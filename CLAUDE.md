# CLAUDE.md

AI 코딩 에이전트용 가이드. 작업 전 필독.

## 프로젝트

**"CounsHelper"**: 사설 상담센터 심리상담사를 위한 **AI 상담 기록·분석·보고서 자동화 플랫폼**.
공모전 출품작, 마감 2026-05-29. 팀 4명, Streamlit·LangChain 초급. 코드는 단순·명료 우선.

## 기능 (F1~F5)

항상 _docs/PRD.md를 참고할 것.

## 기술 스택

- Python 3.11, Streamlit
- transformers + torch (KlueBERT 판별, Koalpaca 요약)
- Gemini API (28요인 분류, 챗봇)
- LangChain, ChromaDB, sentence-transformers (bge-m3)
- plotly + kaleido (차트 이미지화), python-docx, weasyprint

**금지**: Docker, 외부 SQL DB, async 프레임워크, 한글 식별자

## 디렉토리

```
app.py              # Streamlit 진입점, 라우팅만
config.py           # 경로·상수
data/               # ⚠️ gitignored (raw/ processed/ references/)
src/
  classifier.py     # F1: KlueBERT 판별 + Gemini 28요인
  summarizer.py     # F3: Koalpaca 요약 호출
  report.py         # F3: .md/.pdf/.docx 출력
  dashboard.py      # F2: plotly 차트
  hira.py           # HIRA 통계 매칭
  rag/              # F4: ingest / retriever / chain
pages/              # Streamlit 멀티페이지
chroma_db/          # ⚠️ gitignored, 벡터 인덱스
```

## 핵심 규칙

- **데이터**: 모두 `data/`, `raw/`는 읽기 전용, Git 커밋 금지
- **API 키**: `.env`만, 하드코딩 금지, LLM 호출은 try/except
- **경로**: `config.py` 상수 사용, 하드코딩 금지
- **환자 데이터**: MVP는 데모 데이터만, 실제 식별정보 입력 금지
- **Koalpaca 12.8B**: Streamlit Cloud에서 실행 불가 → 외부 추론 API(Modal 등)로 호출

## Git 전략

- `main`: 보호됨, 직접 push 금지, PR로만 머지
- 작업 브랜치: `feature/{기능}` (예: `feature/f1-classifier`)
- 커밋 접두사: `feat:` `fix:` `docs:` `refactor:` `chore:`
  - 예: `feat: KlueBERT 판별 로직 구현`
- 작업 전 항상 `git pull`, 같은 파일 동시 수정 금지

## 안티패턴 (하지 말 것)

1. 데이터 파일을 Git에 커밋
2. API 키 하드코딩
3. `main`에 직접 push
4. 상대경로 하드코딩 (`config.py` 사용)
5. requirements.txt 갱신 누락

## 명령어

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
python -m src.rag.ingest      # RAG 인덱스 빌드
```