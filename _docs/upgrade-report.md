# CounsHelper 업그레이드 비교 보고서

| 항목 | 내용 |
|------|------|
| 작성일 | 2026-05-25 |
| 원본 | `main` 브랜치 / counshel-per.streamlit.app |
| 개선 버전 | `wip/vibe-coding` 브랜치 (로컬 실행 확인) |
| 분석 방법 | 코드 직접 비교 (원본 사이트 Streamlit Cloud 인증 차단으로 WebFetch 불가, git diff + 코드 정독으로 대체) |

---

## 1. 구조 비교 요약

| 항목 | 원본 (main) | 개선 버전 (wip/vibe-coding) |
|------|------------|--------------------------|
| **페이지 수** | 4 | 6 |
| **페이지 목록** | 상담내역 기록·추가 / 분석 대시보드 / AI 보고서 / 챗봇 | 내담자 홈 / 상담내역 기록·추가 / 분석 대시보드 / 통계 / AI 보고서 / 챗봇 |
| **네비게이션** | 상단 4-버튼 가로 행 | 사이드바 세로 메뉴 + Material 아이콘 |
| **색상 테마** | Blue #2563EB | Violet #7C3AED (WeeNote) |
| **폰트** | 시스템 폰트 | Pretendard (CDN) |
| **데이터 저장** | 메모리 DataFrame (새로고침 시 소멸) | SQLite (영속적) |
| **AI 모델** | Mock (키워드 매칭 규칙 기반) | 실제 AI (Gemma 4 31B + Gemini API + KoAlpaca) |
| **사이드바** | 텍스트 검색 입력만 | 브랜드 헤더 + 사용자 정보 + selectbox 검색 + 신규 등록 폼 |
| **st.secrets 버그** | 있음 (경고 배너 반복) | 수정됨 (os.getenv() 우선) |
| **테스트** | 없음 | tests/ 5개 파일 |

---

## 2. 기능별 상세 비교

### 2-1. 내담자 홈 (신규 추가)

- **원본**: 없음. 사이드바 텍스트 검색으로 내담자를 선택하면 바로 기록 화면으로 이동.
- **개선**: 전용 홈 페이지 신설. 프로필 카드(이름·성별·연령·지역·메모), 최근 회기 요약, 분류 플래그(우울/불안/중독), HIRA 통계 매칭 한 줄 요약 표시.
- **평가**: 공모전 시연 시 내담자 맥락을 한눈에 보여주는 첫 화면으로 큰 UX 개선.

### 2-2. 상담내역 기록·추가

- **원본**: 정적 CLIENTS/SESSIONS DataFrame 기반. 새로고침 시 데이터 초기화. 상담 스크립트 미리보기(text_area, 모델 입력 형태 표시) 있음.
- **개선**: SQLite 연동. 신규 내담자를 사이드바 expander에서 등록. 회기 저장 후 재분석(`reanalyze_session()`) 가능. 스크립트 미리보기는 유지.
- **평가**: 데이터 영속성 확보가 핵심 개선. 공모전 발표 중 실시간 데이터 입력·저장·재조회 시연 가능.

### 2-3. 분석 대시보드

- **원본**: Mock 분류(우울/불안/중독 1/0) + 28요인 키워드 매칭 + 회기별 추이 예시 차트. AI 없이 규칙 기반.
- **개선**: Gemma 4 31B 실제 판별(0~3 정도값) + Gemini API 64-라벨 발화별 분류. AI 도우미 패널(RAG 연결) 오른쪽에 토글. src/insight.py 통해 AI 서술형 통찰 생성.
- **평가**: 실제 AI 결과를 보여줄 수 있어 공모전 임팩트 크게 향상.

### 2-4. 통계 (신규 추가)

- **원본**: 없음.
- **개선**: HIRA(건강보험심사평가원) 시군구·성별·연령별 CSV 연동. 내담자 메타 기준으로 국가 통계와 매칭(예: 30대 여성 서울 우울증 → 전국 160,392명 중 56,274명). 전체 케이스 집계(aggregate_global_stats), 분류 분포, 요인 top-N 차트.
- **평가**: 실제 공공데이터와의 연동은 공모전 차별성 포인트. 단, HIRA CSV가 data/ 폴더에 있어야 동작.

### 2-5. AI 보고서

- **원본**: Mock 요약(템플릿 문자열 조합). `.md` 다운로드만.
- **개선**: KoAlpaca Modal API 호출 → 실패/짧은 입력 시 Gemma 4 31B 폴백. 4섹션 파싱 강화. `.md` + `.docx` 다운로드. PDF는 Windows GTK 의존성으로 graceful fallback(비활성 안내).
- **평가**: 실제 LLM 기반 보고서 생성. KoAlpaca 콜드스타트(60~120s) 대기 시간이 시연 리스크.

### 2-6. 챗봇 (RAG)

- **원본**: Mock 응답(고정 텍스트). ChromaDB 미연결.
- **개선**: ChromaDB 3,316건 인덱싱 완료(AI Hub 발화 2,500 + PDF 31p + DOCX 336 + HIRA CSV 449). KoSBERT 임베딩. Gemma 4 31B 답변 생성. 출처 3개 표시. AI 도우미 패널로도 접근 가능(회기 컨텍스트 첨부).
- **평가**: RAG 실연결이 F4 기능의 핵심 구현. 단, `chroma_db/` 인덱스가 로컬에만 있고 Streamlit Cloud 배포 시 재인덱싱 필요.

---

## 3. 기술 스택 변화

| 모듈 | 원본 | 개선 |
|------|------|------|
| **1차 판별** | KeywordMatcher (Mock) | Gemma 4 31B (0~3 정도값) |
| **28요인 추출** | KeywordMatcher (Mock) | Gemini API (64-라벨 발화별 0/1) |
| **요약** | 템플릿 문자열 | KoAlpaca Modal API + Gemma fallback |
| **RAG** | Mock 응답 | ChromaDB + KoSBERT + Gemma 4 31B |
| **DB** | 메모리 DataFrame | SQLite (`src/db.py`) |
| **통계** | 없음 | HIRA CSV + src/stats.py + src/insight.py |
| **보고서 출력** | `.md` 만 | `.md` + `.docx` (PDF graceful fallback) |
| **KlueBERT** | Mock placeholder | **mode collapse 진단 후 Gemma로 대체** (raw 값 0.517±0.018, 어떤 입력이든 binary=1 출력) |

---

## 4. 디자인·UX 비교

| 항목 | 원본 | 개선 |
|------|------|------|
| **색상** | #2563EB (Blue) | #7C3AED (Violet, WeeNote) |
| **폰트** | 시스템 기본 | Pretendard (한국어 가독성 우수) |
| **네비게이션** | 상단 가로 4버튼 | 사이드바 세로 6메뉴 + Material 아이콘 |
| **버튼 스타일** | 라운드 pill (border-radius: 999px) | 라운드 rect (border-radius: 0.7rem) |
| **사이드바** | 검색 입력 단독 | 브랜드 + 사용자 카드 + 검색 + 신규등록 expander + 메뉴 |
| **카드 디자인** | 파란 카드 (CARD_BLUE) | 흰 카드 + 보라 테두리 (insight-card) |
| **AI 도우미** | 없음 | 우측 토글 패널 (회기 컨텍스트 첨부) |
| **위험 경고** | 초록/빨강 st.success/st.error | alert-card CSS 커스텀 카드 |

---

## 5. 개선 점수 평가

| 분류 | 평가 |
|------|------|
| 기능 완성도 | ★★★★★ — F1~F5 전 기능 실제 동작 |
| AI 품질 | ★★★★☆ — 실 모델 연동, KlueBERT 한계 정직하게 대체 |
| 데이터 영속성 | ★★★★★ — SQLite로 완전 해결 |
| UX/디자인 | ★★★★☆ — Pretendard + Material 아이콘으로 향상, 사이드바 nav는 호불호 있음 |
| 공모전 시연 적합성 | ★★★☆☆ — KoAlpaca 콜드스타트, chroma_db 로컬 의존성 리스크 존재 |
| 코드 품질 | ★★★★☆ — 테스트 추가, src/ 모듈화 강화 |

---

## 6. 원본(main)에서 차용해야 할 점

### 6-1. 색상 테마 재검토
원본의 **파란색 (#2563EB)** 계열은 의료·임상 맥락에서 신뢰감·전문성이 높음. 보라색은 세련되지만 심리상담 플랫폼의 공식성과 거리감이 있을 수 있음. 공모전 심사 기준(실용성)을 고려하면 파란색 유지 또는 절충안(보라+파란 혼용) 검토 권장.

### 6-2. 상단 탑-네비게이션
원본의 **상단 4버튼 가로 네비게이션**은 처음 보는 사람이 전체 기능을 즉시 파악하기 쉬움. 사이드바 nav는 메뉴를 열어야 보여서 발표/시연 시 어디서 무엇을 클릭하는지 설명이 필요. 공모전 발표에서는 탑-nav가 더 직관적일 수 있음.

### 6-3. 상담 스크립트 미리보기 text_area
원본 `render_record_editor()`의 **"모델 입력 형태" 미리보기**는 AI 모델에 어떤 텍스트가 들어가는지 심사위원에게 투명하게 보여주는 장치. 개선 버전에서 이 UI 요소가 축소된 경우 복원 고려.

### 6-4. 법적 면책 문구
원본 사이드바 하단의 `"본 시스템은 상담사의 임상적 판단을 대체하지 않습니다"` 문구. 의료/심리 관련 도구에서 필수적인 면책 요소로, 개선 버전에서도 눈에 띄는 위치에 유지해야 함.

### 6-5. st.secrets 경고 수정 (이미 stash에 보관)
`main` 브랜치에서 발생하는 **st.secrets 경고 배너** 문제는 `os.getenv()` 우선 패턴으로 수정된 내용이 현재 git stash에 보관 중. 배포 전 반드시 적용 필요.

---

## 7. 배포 전 체크리스트

개선 버전을 counshel-per.streamlit.app에 배포하기 위한 전제 조건:

| 항목 | 상태 | 비고 |
|------|------|------|
| `chroma_db/` 인덱스 | ⚠️ 로컬 전용 | Streamlit Cloud에 재인덱싱 필요 또는 원격 벡터 DB 전환 |
| `data/` HIRA CSV | ⚠️ gitignore | 통계 기능을 위해 Cloud 환경에 별도 제공 필요 |
| `.env` → Streamlit Secrets | ⚠️ 미설정 | GEMINI_API_KEY, KOALPACA_ENDPOINT_URL 등 Cloud secrets에 등록 필요 |
| KoAlpaca 콜드스타트 | ⚠️ 60~120s | 시연 전 미리 warm-up 호출 권장 |
| PDF 출력 | ❌ Windows 전용 제약 | weasyprint GTK 의존성 → Cloud에서도 비활성 유지 |
| SQLite `counseling.db` | ✅ 자동 생성 | `db.init_db()` 호출로 처리됨 |
| st.secrets 버그 수정 | ⚠️ stash 대기 중 | `git stash pop` 후 PR 머지 필요 |

---

## 8. 권장 액션

1. **즉시**: `main`에서 `git stash pop` 후 st.secrets 수정분 커밋 → 원본 배포본 경고 배너 제거
2. **단기**: wip/vibe-coding의 `통계` 및 `내담자 홈` 페이지를 main에 병합
3. **공모전 전**: 색상 테마(파란/보라) 팀 합의 → 탑-nav vs 사이드바-nav 최종 결정
4. **배포 시**: Streamlit Cloud secrets 등록, chroma_db 재인덱싱, KoAlpaca warm-up 절차 정립
