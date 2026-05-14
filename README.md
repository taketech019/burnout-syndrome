# 한국 수험생/직장인 번아웃 조기 감지 (가칭)

> 제12회 보건의료빅데이터·AI 활용 창업경진대회 출품작

정신과 의사의 진료를 돕는 보조 도구. 환자 증상을 챗봇에 입력하면 HIRA 공공데이터 기반 통계 비교, 임상 가이드라인 기반 RAG 답변, 위험도 시각화 대시보드를 동시에 제공합니다.

---

## 📌 첫 방문이라면 — 5분 안에 셋업하기

순서대로 따라하시면 됩니다. 막히면 디스코드에 물어보세요.

### 사전 준비

- Python 3.11 설치 (3.10 이상 가능, 3.12는 비추천)
- Git 설치 (또는 [GitHub Desktop](https://desktop.github.com) — GUI 추천)
- VS Code 또는 익숙한 에디터
- Gemini API 키 (팀 슬랙/디스코드 DM으로 받기)

### 1️⃣ 저장소 받기

```bash
git clone https://github.com/<팀명>/<레포명>.git
cd <레포명>
```

GitHub Desktop을 쓰면 "File → Clone repository"로 클릭만 하면 됩니다.

### 2️⃣ 가상환경 만들기

가상환경은 *이 프로젝트만의 독립된 파이썬 작업 공간*입니다. 다른 프로젝트와 패키지가 섞이지 않게 해줍니다.

```bash
python -m venv venv
```

폴더 안에 `venv/` 가 생기면 성공.

### 3️⃣ 가상환경 활성화

**맥/리눅스:**
```bash
source venv/bin/activate
```

**윈도우 (PowerShell):**
```powershell
venv\Scripts\Activate.ps1
```

**윈도우 (cmd):**
```cmd
venv\Scripts\activate.bat
```

성공하면 터미널 줄 앞에 `(venv)` 가 붙습니다:
```
(venv) C:\Users\hong\project>
```

⚠️ **새 터미널 창을 열 때마다 다시 활성화해야 합니다.** 자동으로 안 됩니다.

### 4️⃣ 패키지 설치

```bash
pip install -r requirements.txt
```

5~10분 정도 걸립니다. 의료/AI 라이브러리가 크고 많습니다.

### 5️⃣ 환경변수 파일 만들기

`.env.example` 파일을 복사해서 `.env` 로 이름 변경:

**맥/리눅스:**
```bash
cp .env.example .env
```

**윈도우:**
```powershell
copy .env.example .env
```

그 다음 `.env` 파일을 에디터로 열어 본인 API 키를 채워넣으세요:

```
GEMINI_API_KEY=여기에_본인_키_붙여넣기
```

⚠️ `.env` 파일은 **절대 Git에 올라가지 않습니다** (`.gitignore`에 포함). 안심하고 키를 적으세요.

### 6️⃣ 데이터 받기

데이터는 용량 문제로 Git에 없습니다. 구글 드라이브 공유 폴더에서 받아주세요:

> 📁 [팀 공유 드라이브 링크 — PM이 첨부]
> └── 01_raw_data/ → 본인 컴퓨터의 `data/raw/`로 복사
> └── 03_레퍼런스/ → 본인 컴퓨터의 `data/references/`로 복사

### 7️⃣ 실행

```bash
streamlit run app.py
```

브라우저가 자동으로 열리며 http://localhost:8501 에서 앱이 뜹니다.

🎉 끝났습니다. 안 되면 아래 트러블슈팅 보세요.

---

## 📂 폴더 구조 한눈에

```
project-root/
├── app.py                  # 여기서 실행 (streamlit run app.py)
├── config.py               # 경로·상수
├── requirements.txt        # 필요한 패키지 목록
├── .env                    # API 키 (본인이 만들어야 함, gitignored)
├── .env.example            # .env 만들 때 참고할 템플릿
│
├── data/                   # 데이터 (gitignored, 직접 받아야 함)
│   ├── raw/                # 원본 — 수정 금지!
│   ├── processed/          # 전처리 결과
│   └── references/         # RAG용 임상 가이드라인 PDF
│
├── src/                    # 코드 본체
│   ├── data_loader.py      # 데이터 로딩
│   ├── analysis.py         # 통계, 위험도 계산
│   ├── rag/                # RAG 챗봇 관련
│   └── ui/                 # Streamlit 컴포넌트
│
├── pages/                  # Streamlit 멀티페이지
├── notebooks/              # EDA·실험용 (프로덕션 코드 아님)
│
├── README.md               # 이 파일
└── CLAUDE.md               # AI 코딩 에이전트 참고용 (Claude Code 등)
```

### 작업 영역 분담

- **풀스택**: `src/ui/`, `pages/`, `app.py`
- **AI/RAG**: `src/rag/`
- **데이터**: `src/data_loader.py`, `src/analysis.py`, `notebooks/`
- **PM**: 코드 최소 수정. 문서·기획서 위주.

같은 파일을 둘이 동시에 만지지 마세요. 충돌 사고의 80%가 여기서 납니다.

---

## 🌳 Git 작업 흐름

### 큰 원칙

- `main` 브랜치에는 **직접 push 금지**. PR로만 머지.
- 작업은 **본인 브랜치**에서.
- 충돌은 무서운 거 아닙니다. 도움 요청하세요.

### GitHub Desktop으로 (권장 — 클릭만)

1. **Current Branch** 드롭다운 → "New Branch" → `feature/내기능` 이름
2. 코드 수정
3. 변경사항이 자동으로 보임 → 메시지 적고 "Commit to feature/내기능" 클릭
4. "Push origin" 버튼 클릭
5. 우측 상단 "Create Pull Request" 버튼 → 브라우저에서 PR 생성

끝.

### CLI로 (한 번 익히면 빠름)

```bash
# 새 브랜치 만들기
git checkout main
git pull origin main
git checkout -b feature/내기능

# 작업 후
git add .
git commit -m "feat: 기능 설명"
git push origin feature/내기능

# 그 다음 GitHub 웹에서 PR 만들기
```

### 작업 중 main이 업데이트됐다면

```bash
git checkout main
git pull origin main
git checkout feature/내기능
git merge main      # 또는 GitHub Desktop의 "Update from main"
```

### 브랜치 이름 규칙

- 새 기능: `feature/dashboard-layout`, `feature/rag-chain`
- 버그 수정: `fix/chroma-windows-bug`
- 문서·잡일: `chore/update-readme`

---

## 💬 커밋 메시지 컨벤션

### 형식

```
타입: 메시지

(선택) 상세 설명
```

### 자주 쓰는 타입

| 타입 | 용도 | 예시 |
|---|---|---|
| `feat` | 새 기능 | `feat: HIRA 정신질환 데이터 로더 구현` |
| `fix` | 버그 수정 | `fix: ChromaDB 경로 윈도우 호환성` |
| `docs` | 문서 | `docs: README 셋업 가이드 보완` |
| `refactor` | 동작은 같고 정리만 | `refactor: data_loader 함수 분리` |
| `chore` | 잡일 | `chore: pypdf 패키지 추가` |

### 좋은 메시지 / 나쁜 메시지

✅ `feat: 챗봇에 출처 인용 기능 추가`
✅ `fix: PDF 빈 페이지 처리 시 크래시`

❌ `update`
❌ `wip`
❌ `버그 고침`
❌ `최종`

---

## 🔄 일상 워크플로우

### 작업 시작 시

```bash
# 1. 본인 브랜치로
git checkout feature/내기능

# 2. main 최신 가져오기
git checkout main && git pull origin main && git checkout feature/내기능 && git merge main

# 3. 가상환경 활성화
source venv/bin/activate   # 또는 venv\Scripts\activate

# 4. 새 패키지가 추가됐을 수 있으니
pip install -r requirements.txt
```

### 작업 끝나고

```bash
# 1. 변경사항 확인
git status

# 2. 의도한 것만 커밋
git add .
git commit -m "feat: ..."

# 3. push
git push origin feature/내기능
```

### 새 패키지 설치했다면

```bash
pip install 새패키지
pip freeze > requirements.txt   # ← 이거 잊지 말기!
git add requirements.txt
git commit -m "chore: 새패키지 추가 (이유)"
```

다른 팀원이 받으면 `pip install -r requirements.txt` 한 번만 돌리면 됩니다.

---

## 🚀 Streamlit Cloud 배포

시제품 증빙용 URL이 공모전 제출에 필요합니다.

1. https://share.streamlit.io 로그인 (GitHub 계정)
2. "New app" → 본 레포 선택, branch `main`, main file `app.py`
3. "Secrets" 메뉴에서 `.env` 내용을 그대로 입력:
   ```
   GEMINI_API_KEY = "..."
   ```
4. Deploy 클릭

`main`에 push될 때마다 자동 재배포됩니다.

⚠️ **마감 일주일 전에는 배포 한 번 테스트해두세요.** 패키지 호환성 문제로 처음 배포 시 반나절씩 날아갑니다.

---

## 🔐 비밀 키 관리

### 절대 하지 말 것

- ❌ API 키를 코드에 직접 적기
- ❌ `.env` 파일을 Git에 올리기 (`.gitignore`에 포함되어 있지만 확인 필수)
- ❌ 디스코드 단톡방에 키 공유 (DM으로만)

### 키가 노출됐다면

1. 즉시 해당 API 콘솔에서 키 *재발급* (구 키 폐기)
2. PM에게 알림
3. Git 히스토리에서 제거 (PM이 처리)

API 콘솔에서 **월 사용 한도(spending limit)** 를 1만원 정도로 설정해두세요. 사고 발생해도 피해 제한.

---

## ❓ 트러블슈팅

### "ModuleNotFoundError: No module named 'streamlit'"
가상환경이 활성화 안 되어 있습니다. `(venv)` 표시가 보이는지 확인 후 `pip install -r requirements.txt` 다시.

### `venv\Scripts\Activate.ps1` 실행 정책 에러 (윈도우)
PowerShell을 관리자 모드로 열고 한 번만:
```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

### "GEMINI_API_KEY가 .env에 없습니다"
`.env` 파일이 프로젝트 루트(README.md 옆)에 있는지 확인. 파일명에 오타 없는지 확인 (`env.txt` 같은 거 아님).

### `git push` 거부 ("rejected, fetch first")
원격에 내가 모르는 새 커밋이 있습니다.
```bash
git pull origin <브랜치명>
```
충돌이 나면 PM에게 화면공유 요청.

### Streamlit 앱이 안 뜨고 8501 포트 사용 중 에러
다른 Streamlit이 이미 실행 중입니다. 종료하거나:
```bash
streamlit run app.py --server.port 8502
```

### ChromaDB가 윈도우에서 에러
SQLite 호환성 이슈일 수 있습니다. AI 담당에게 문의.

### `pip install`이 끝없이 느림
M1/M2 맥에서 sentence-transformers 컴파일 중일 수 있습니다. 15분 정도 기다려보세요.

### 그 외 모든 에러
1. 에러 메시지 전문을 복사
2. 본인이 한 마지막 작업이 뭐였는지
3. 디스코드에 올리기

혼자 30분 이상 잡고 있지 마세요. 더 빠른 사람이 있을 수 있습니다.

---

## 👥 팀

| 역할 | 담당자 | 주 영역 |
|---|---|---|
| PM/기획 | TBD | 문서, BM, 발표 |
| 데이터 분석 | TBD | HIRA EDA, 통계 |
| 풀스택 | TBD | Streamlit, UI |
| AI/RAG | TBD | RAG 파이프라인, LLM |

## 📅 주요 일정

- **2026-04-06 ~ 05-29**: 공모 접수 기간
- **2026-05-29 17:00**: 제출 마감 ⚠️
- **2026-06-26**: 1차 서류 평가 결과
- **2026-07-06 ~ 07**: 2차 인터뷰 평가
- **2026-07-22**: 3차 최종 발표 평가

## 📚 참고 자료

- 공모전 페이지: https://opendata.hira.or.kr (창업경진대회)
- HIRA 보건의료빅데이터: https://opendata.hira.or.kr
- 공공데이터 포털: https://www.data.go.kr
- LangChain 한국어 자료: https://wikidocs.net/231154
- Streamlit 한국어 자료: https://docs.streamlit.io