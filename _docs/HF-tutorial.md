# 튜토리얼 — KlueBERT Hugging Face Spaces 호스팅 (Yun 전용 가이드)

| 작성일 | 2026-05-23 |
| --- | --- |
| 대상 독자 | Yun (HF 계정 `Yun-choice`, HF Hub/CLI 사용 경험 중급) |
| 선행 문서 | `_docs/PRD.md` §F1, 호스팅 플랜 (`C:\Users\choic\.claude\plans\model-details-model-summary-lexical-hammock.md`) |
| 예상 총 소요 | **90~120분** (실제 업로드 60~90분이 대부분) |
| 마감 | 2026-05-29 (D-6) |
| 목적 | 학습 완료된 KlueBERT 3개 모델을 HF Hub + Spaces에 올려, Streamlit Cloud에서 `src/classifier.classify()`가 실제로 동작하도록 만든다 |

---

## 사전 준비 체크리스트

본 문서 시작 전, 다음이 모두 충족되어야 한다.

- [ ] HF 계정 `Yun-choice` 로그인 가능
- [ ] HF 계정에서 **write** 권한 토큰을 새로 발급할 수 있음
- [ ] 로컬에 다음 폴더 3개 존재 (각 ~423MB):
  - `ai-model/kluebert/2.AI학습모델파일/trained_model_kluebert_anxiety/`
  - `ai-model/kluebert/2.AI학습모델파일/trained_model_kluebert_depression/`
  - `ai-model/kluebert/2.AI학습모델파일/trained_model_kluebert_addiction/`
- [ ] 로컬에 `hf-space/` 디렉토리와 그 안의 4개 파일 존재:
  - `app.py`, `modeling_kluebert.py`, `requirements.txt`, `README.md`
- [ ] 가정용 인터넷 회선으로 **60~90분 연속 업로드** 가능한 시간대 (1.27 GB)
- [ ] PowerShell이 프로젝트 루트(`C:\Users\choic\Desktop\Projects_Dev\PR_BOAZ_2nd`)에서 열려 있음

> 모든 PowerShell 명령은 **프로젝트 루트**에서 실행한다고 가정한다. `cd`는 명시된 곳에서만 수행한다.

---

## Phase 0 — 환경 준비 (5분)

> **CLI 마이그레이션 안내**: `huggingface_hub` v0.34부터 CLI 진입점이 `huggingface-cli` → **`hf`** 로 전환됐다. 구 명령은 일부 플래그가 silently 깨지므로 본 가이드의 `hf` 명령을 그대로 사용. 만약 이전에 구 명령으로 일부 단계를 마쳤더라도 만들어진 repo/Space는 그대로 재사용 가능 — 누락된 부분만 신 CLI로 다시 실행.

### Step 0-1. `huggingface_hub` 설치 + 신 CLI 확인

```powershell
pip install -U huggingface_hub
hf --version
```

`hf --version`이 명령 없음 에러가 나면 설치가 환경에 반영 안 된 것 — PowerShell 새 창을 열거나 `python -m pip install -U huggingface_hub`로 강제 설치.

### Step 0-2. write 권한 토큰 발급

1. 브라우저로 https://huggingface.co/settings/tokens 접속
2. **New token** 클릭
3. Name: `counshelper-deploy` (자유), Type/Role: **Write** 선택
4. **Create token** → 화면에 표시된 토큰 문자열을 클립보드에 복사

### Step 0-3. CLI 로그인

```powershell
hf auth login
```

프롬프트에 토큰 paste → 엔터. `Add token as git credential? (Y/n)`은 **Y** 권장 (Phase 4 git push 시 필요).

이미 다른 토큰으로 로그인된 상태에서 다시 로그인하려면:
```powershell
hf auth login --force
```

### 성공 조건

```powershell
hf auth whoami
```
출력에 `Yun-choice` 표시되면 통과.

---

## Phase 1 — 한글 경로 ASCII 스테이징 (5분)

> **왜 필요한가**: Windows + 한글 폴더명(`2.AI학습모델파일`) + `hf upload` 조합은 인코딩 버그가 자주 발생한다. 한 번만 ASCII 경로로 복사해 두면 이후 모든 명령이 안정적으로 동작한다. `hf-upload/`는 이미 `.gitignore`되어 있어 안전.

### Step 1-1. 스테이징 디렉토리 생성 + 모델 복사

```powershell
New-Item -ItemType Directory -Force -Path .\hf-upload\anxiety, .\hf-upload\depression, .\hf-upload\addiction | Out-Null

Copy-Item -Recurse "ai-model\kluebert\2.AI학습모델파일\trained_model_kluebert_anxiety\*"    .\hf-upload\anxiety\
Copy-Item -Recurse "ai-model\kluebert\2.AI학습모델파일\trained_model_kluebert_depression\*" .\hf-upload\depression\
Copy-Item -Recurse "ai-model\kluebert\2.AI학습모델파일\trained_model_kluebert_addiction\*"  .\hf-upload\addiction\
```

### 성공 조건

```powershell
Get-ChildItem .\hf-upload\anxiety | Select-Object Name, Length
```

다음 6개 파일이 보여야 함:
- `config.json`
- `model.safetensors` (≈ 443,000,000 bytes)
- `special_tokens_map.json`
- `tokenizer_config.json`
- `training_args.bin`
- `vocab.txt`

3개 폴더 모두 같은 구성인지 빠르게 확인:
```powershell
foreach ($d in 'anxiety','depression','addiction') {
  "$d : $((Get-ChildItem .\hf-upload\$d | Measure-Object).Count) files"
}
```

---

## Phase 2 — 모델 repo 3개 생성 + 업로드 (60~90분)

> **이 Phase가 가장 긴 단계**. 업로드 시간 대부분이 회선 속도에 묶여 있다. 백그라운드에서 진행하면서 다른 작업 가능.

### Step 2-1. 모델 repo 3개 생성 (각 public, 기본값)

```powershell
hf repos create Yun-choice/kluebert-anxiety    --exist-ok
hf repos create Yun-choice/kluebert-depression --exist-ok
hf repos create Yun-choice/kluebert-addiction  --exist-ok
```

- 모델 repo는 `--repo-type` 생략(기본값). Space만 명시 필요.
- 기본값 public이며 private 원하면 `--private` 추가 — 본 가이드는 public.
- `--exist-ok`: 이미 존재해도 에러 없이 통과 (재실행 안전).

### Step 2-2. 모델 파일 업로드 (3개 모두)

**순차 업로드 (가장 안전)** — 한 PowerShell 창에서 차례로:

```powershell
hf upload Yun-choice/kluebert-anxiety    .\hf-upload\anxiety\    .
hf upload Yun-choice/kluebert-depression .\hf-upload\depression\ .
hf upload Yun-choice/kluebert-addiction  .\hf-upload\addiction\  .
```

`hf upload <repo_id> <local_path> <path_in_repo>` 순서. 마지막 `.`은 "repo 루트에 업로드" 의미.

**병렬 업로드 (3배 빠름, 회선이 받쳐줄 때만)** — PowerShell 창 3개 띄워 각각 한 줄씩 실행. 업로드 대역폭이 충분하지 않으면 오히려 느려지므로 회선 속도 확인 후 결정.

업로드 중간에 네트워크가 끊겨도 같은 명령을 다시 실행하면 이어서 업로드한다 (`huggingface_hub`는 LFS hash 기반 멱등).

### 성공 조건

각 repo URL을 브라우저로 열어 파일 6개가 모두 보이는지 확인:
- https://huggingface.co/Yun-choice/kluebert-anxiety/tree/main
- https://huggingface.co/Yun-choice/kluebert-depression/tree/main
- https://huggingface.co/Yun-choice/kluebert-addiction/tree/main

각 repo의 `model.safetensors` 옆에 `LFS` 뱃지와 `≈ 423 MB` 표시되어야 정상.

---

## Phase 3 — 모델 repo 라이선스 메타데이터 (5분)

### Step 3-1. 각 repo README 편집 (웹 UI)

3개 repo 각각에서:

1. repo 페이지 우상단 **⋯ 메뉴** → **Edit model card** 클릭 (또는 README.md 파일 클릭 → `Edit` 아이콘)
2. **상단에 YAML 헤더 삽입** (이미 있으면 덮어쓰기):

```yaml
---
license: cc-by-sa-4.0
language: ko
base_model: klue/bert-base
tags:
  - bert
  - korean
  - mental-health
  - regression
---
```

3. 본문 한 줄 (label 자리에 `anxiety`/`depression`/`addiction` 채움):

```markdown
# kluebert-<label>

한국어 심리상담 채록 1,609건으로 fine-tune된 `klue/bert-base` 회귀 모델 (CounsHelper F1 1단계).
회귀 출력을 `[0, 3]` 정수에 클리핑한 뒤 `0 → 0, 그 외 → 1`로 binary 변환하여 사용한다.
추론 사용 예시는 [Space](https://huggingface.co/spaces/Yun-choice/kluebert-counshelper) 참조.
```

4. **Commit changes to main** → 커밋 메시지는 기본값 OK.

### 성공 조건

repo 메인 페이지 상단에 `license: cc-by-sa-4.0`, `Korean`, `BERT` 뱃지가 표시된다.

---

## Phase 4 — Space 생성 + 코드 푸시 (10분)

> **이전에 구 `huggingface-cli` 명령으로 시도해서 실패했다면** 4-1부터 다시 신 CLI로 실행. 4-1이 멱등이라 같은 Space가 이미 있어도 OK.

### Step 4-1. Space repo 생성 (Gradio SDK)

```powershell
hf repos create Yun-choice/kluebert-counshelper --repo-type space --space-sdk gradio --exist-ok
```

> ⚠️ 플래그명 주의: 구 명령의 `--space_sdk`(언더스코어)가 아닌 **`--space-sdk`(하이픈)**. 구 명령은 silently 무시되어 SDK 없는 Space가 만들어지거나 생성 자체가 실패한다.

**성공 조건**: 출력에 `https://huggingface.co/spaces/Yun-choice/kluebert-counshelper` URL이 표시됨. 브라우저로 한 번 열어서 Space 페이지가 보이는지(아직 비어 있어도 OK) 확인한 뒤 4-2로.

### Step 4-2. 로그인 상태 + Space repo 클론

> ⚠️ 프로젝트 안에 이미 `hf-space/`가 있으므로, 클론 디렉토리는 **`hf-space-remote/`**로 다른 이름 사용.

```powershell
hf auth whoami
git clone https://huggingface.co/spaces/Yun-choice/kluebert-counshelper hf-space-remote
```

- `hf auth whoami` 출력이 `Yun-choice`여야 함 (다르면 `hf auth login --force`로 재로그인)
- `git clone`에서 자격증명 프롬프트가 뜨면 Username `Yun-choice` + Password 자리에 Phase 0의 토큰 paste

### Step 4-2.5. 디렉토리 존재 검증 (Step 4-3 진입 게이트)

```powershell
Test-Path .\hf-space-remote
Get-ChildItem .\hf-space-remote
```

- `Test-Path` 결과가 `True`여야 함
- `Get-ChildItem` 결과에 최소 `.gitattributes`, `README.md` 등이 보여야 함

`False`거나 비어 있으면 4-1/4-2 재실행 (clone 실패 원인은 95% 자격증명 또는 Space 미생성).

### Step 4-3. 4개 파일 복사 (단일 명령)

```powershell
Copy-Item .\hf-space\* .\hf-space-remote\ -Recurse -Force
```

이전 가이드의 4줄 분할 명령은 PowerShell 백틱 줄바꿈/paste 사고가 잦아 단일 와일드카드 명령으로 단순화. 결과는 동일 (`app.py`, `modeling_kluebert.py`, `requirements.txt`, `README.md` 4개 복사 + 기존 `README.md`는 덮어쓰기).

**검증**:
```powershell
Get-ChildItem .\hf-space-remote | Select-Object Name
```
4개 파일이 모두 나오면 OK.

### Step 4-4. 커밋 + 푸시

```powershell
cd hf-space-remote
git add app.py modeling_kluebert.py requirements.txt README.md
git commit -m "feat: KlueBERT inference API on free CPU Space"
git push
cd ..
```

푸시 시 자격증명 프롬프트가 다시 뜨면 4-2와 동일하게 paste.

### 성공 조건

https://huggingface.co/spaces/Yun-choice/kluebert-counshelper 접속 시:
- 우상단 "Building" 또는 "Running" 상태 뱃지 확인
- 파일 목록에 `app.py`, `modeling_kluebert.py`, `requirements.txt`, `README.md` 4개 보임

빌드는 자동 시작되지만 Phase 5의 Secret이 아직 없어서 401만 반환할 것이므로 다음 Phase로.

---

## Phase 5 — Space Secret 등록 (5분)

### Step 5-1. 랜덤 토큰 생성

```powershell
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

출력된 문자열(예: `xK7Q...44자`)을 **클립보드에 복사 + 별도 메모장에 저장** (Phase 7에서 같은 값을 `.env`에 적어야 함).

### Step 5-2. Space Secret 등록 (웹 UI)

1. https://huggingface.co/spaces/Yun-choice/kluebert-counshelper/settings 접속
2. 페이지 중간 **Variables and secrets** 섹션 → **New secret** 클릭
3. Name: `HF_SPACE_API_KEY` (정확히 이 이름)
4. Value: Step 5-1에서 복사한 문자열
5. **Save** 클릭

Secret 저장 시 Space가 자동 재시작된다.

### 성공 조건

같은 settings 페이지 Variables and secrets 섹션에 `HF_SPACE_API_KEY` 항목이 (값은 마스킹된 채) 표시된다.

---

## Phase 6 — Space 빌드 모니터링 (10~20분)

### Step 6-1. Logs 탭 관찰

https://huggingface.co/spaces/Yun-choice/kluebert-counshelper?logs=build

빌드 로그에서 다음 신호를 차례로 확인:

1. `Installing dependencies` → pip install 진행
2. `Build complete` → 컨테이너 시작
3. Container logs에 다음 줄들이 차례로 보임:
   - `INFO 모델 로드 시작 (CPU, fp32, 3 모델)`
   - `INFO 로드 중: Yun-choice/kluebert-anxiety`
   - (3개 모델 로드)
   - `INFO 모델 로드 완료: ['anxiety', 'depression', 'addiction']`
4. Uvicorn이 0.0.0.0:7860 listening

3개 모델 다운로드 + 로드는 첫 부팅 시 60~120초 걸린다.

### Step 6-2. 헬스 체크

빌드 완료 후:

```powershell
curl https://yun-choice-kluebert-counshelper.hf.space/healthz
```

### 성공 조건

```json
{"status":"ok","models":["anxiety","depression","addiction"]}
```

이 응답이 나오면 Space가 정상 가동 중.

---

## Phase 7 — 프로젝트 `.env` 작성 (1분)

프로젝트 루트의 `.env` 파일을 (없으면 생성, gitignore되어 있음) 열어 다음 두 줄을 추가/갱신:

```
KLUEBERT_ENDPOINT_URL=https://yun-choice-kluebert-counshelper.hf.space
KLUEBERT_API_KEY=<Phase 5-1에서 생성·저장해둔 값>
```

> `KLUEBERT_API_KEY` 값은 Phase 5에서 Space Secret `HF_SPACE_API_KEY`에 넣은 값과 **반드시 동일**해야 한다. 다르면 모든 호출이 401로 거부된다.

기존 `OPENAI_API_KEY`, `KOALPACA_*` 항목은 건드리지 말 것.

### 성공 조건

```powershell
Get-Content .env | Select-String "KLUEBERT"
```
두 줄(`KLUEBERT_ENDPOINT_URL`, `KLUEBERT_API_KEY`)이 출력된다.

---

## Phase 8 — 스모크 테스트 (5분)

### Step 8-1. PowerShell 환경변수 로드

```powershell
$env:KLUEBERT_ENDPOINT_URL = (Get-Content .env | Select-String '^KLUEBERT_ENDPOINT_URL=').Line.Split('=',2)[1].Trim()
$env:KLUEBERT_API_KEY      = (Get-Content .env | Select-String '^KLUEBERT_API_KEY=').Line.Split('=',2)[1].Trim()
```

> `.Trim()` 필수: `.env` 라인 끝의 trailing whitespace/`\r`(CRLF)이 헤더 값에 그대로 들어가면 서버 Secret과 1바이트 차이로 비교 실패 → 401. 자세한 진단은 트러블슈팅 #9.

### Step 8-2. 정상 호출 (카논 테스트 문장)

> ⚠️ Windows PowerShell에서 한국어 JSON을 `curl.exe`로 보내면 콘솔 코드페이지(CP949)와 UTF-8 충돌로 body가 깨진다 (자세한 내용은 트러블슈팅 #9). 대신 PowerShell 네이티브 `Invoke-RestMethod` + UTF-8 bytes 변환 사용:

```powershell
$payload = @{ text = "내가 가는 이 길이 어디로 가는지 어디로 날 데려가는지 그곳은 어딘지 알 수 없지만 오늘도 난 걸어가고 있네." } | ConvertTo-Json -Compress
$bodyBytes = [System.Text.Encoding]::UTF8.GetBytes($payload)
Invoke-RestMethod -Uri "$env:KLUEBERT_ENDPOINT_URL/predict" `
  -Method Post `
  -Headers @{ "X-API-Key" = $env:KLUEBERT_API_KEY } `
  -ContentType "application/json" `
  -Body $bodyBytes
```

기대 응답 (PowerShell이 JSON을 객체로 자동 파싱):
```
anxiety depression addiction
------- ---------- ---------
      0          0         0
```
(값은 모델 판별 결과에 따라 0/1 조합)

### Step 8-3. 401 음성 테스트

```powershell
try {
  Invoke-RestMethod -Uri "$env:KLUEBERT_ENDPOINT_URL/predict" `
    -Method Post `
    -ContentType "application/json" `
    -Body ([System.Text.Encoding]::UTF8.GetBytes('{"text":"hi"}'))
} catch {
  $_.Exception.Response.StatusCode   # 기대 출력: Unauthorized
}
```

### Step 8-4. 디버그 UI

브라우저로 https://yun-choice-kluebert-counshelper.hf.space/ui 접속 → 텍스트 입력 후 **판별** 클릭 → "이진 결과"와 "원시 회귀값" JSON 두 개가 모두 표시되어야 함.

### 성공 조건

- Step 8-2: HTTP 200 + `{"anxiety":<0|1>,"depression":<0|1>,"addiction":<0|1>}` 정확히 3-키 응답
- Step 8-3: HTTP 401 + `{"detail":"unauthorized"}`
- Step 8-4: UI에서 두 JSON 모두 표시

---

## Phase 9 — Streamlit 통합 검증 (5분)

### Step 9-1. Streamlit 실행

```powershell
streamlit run app.py
```

브라우저가 자동으로 열림. 일단 그냥 두고 다음 단계.

### Step 9-2. Python REPL에서 클라이언트 직접 호출

별도 PowerShell 창을 열어 프로젝트 루트에서:

```powershell
python
```

REPL에서:

```python
from src.classifier import classify
r = classify("내가 가는 이 길이 어디로 가는지 어디로 날 데려가는지 그곳은 어딘지 알 수 없지만 오늘도 난 걸어가고 있네.")
print(r)

# 검증 assert
assert set(r) >= {"anxiety", "depression", "addiction", "is_normal"}
assert all(r[k] in (0, 1) for k in ("anxiety", "depression", "addiction"))
assert r["is_normal"] == (r["anxiety"] == 0 and r["depression"] == 0 and r["addiction"] == 0)
assert "error" not in r
print("OK")
```

### 성공 조건

`OK` 출력 + 에러 없음. 응답 dict가 4개 키를 모두 가지고 있고 `is_normal`이 binary 합과 일치.

---

## 데모 당일 워밍업

Space는 48시간 미사용 시 sleep 상태가 된다. 시연/발표 **5분 전**에 다음 한 줄로 깨워둔다:

```powershell
$env:KLUEBERT_ENDPOINT_URL = (Get-Content .env | Select-String '^KLUEBERT_ENDPOINT_URL=').Line.Split('=',2)[1]
$env:KLUEBERT_API_KEY      = (Get-Content .env | Select-String '^KLUEBERT_API_KEY=').Line.Split('=',2)[1]
curl.exe "$env:KLUEBERT_ENDPOINT_URL/healthz"
curl.exe -X POST "$env:KLUEBERT_ENDPOINT_URL/predict" -H "X-API-Key: $env:KLUEBERT_API_KEY" -H "Content-Type: application/json" -d '{"text":"워밍업 호출"}'
```

- `healthz`가 즉시 응답하지 않고 60~120초 걸린다면 cold-start 중 (정상)
- 두 번째 호출까지 정상 응답하면 ~6시간 동안 활성 유지

---

## 트러블슈팅 (자주 나오는 케이스)

### 0. 구 `huggingface-cli` 명령으로 일부 단계를 이미 진행했다
- 만들어진 repo/Space는 그대로 재사용 가능. 누락된 부분만 신 `hf` 명령으로 다시 수행.
- 어디까지 됐는지 확인:
  ```powershell
  hf auth whoami   # Yun-choice 표시?
  ```
- 모델 repo 존재 확인: 브라우저로 `https://huggingface.co/Yun-choice/kluebert-anxiety` (depression, addiction 동일) — 페이지가 열리고 파일이 있으면 Phase 2 완료, 없으면 Phase 2 재실행
- Space 존재 확인: `https://huggingface.co/spaces/Yun-choice/kluebert-counshelper` — 페이지가 열리면 Phase 4-1 완료. 비어 있어도 OK, Phase 4-2부터 진행

### 1. `hf auth login` 토큰 거부 / `whoami` 인증 실패
- 토큰이 만료됐거나 권한이 read였을 가능성
- 조치: https://huggingface.co/settings/tokens 에서 **write 권한**으로 새 토큰 발급 → `hf auth login --force`
- 기존 토큰 캐시 정리: `Remove-Item $env:USERPROFILE\.cache\huggingface\token -ErrorAction SilentlyContinue`

### 2. 한글 경로 업로드 깨짐 (`UnicodeEncodeError`, `Path not found`)
- Phase 1 ASCII 스테이징을 건너뛰었을 가능성
- 조치: Phase 1을 다시 수행해 `hf-upload/<label>/`로 복사 후, `hf upload`의 로컬 경로 인자에 한글이 한 글자도 없는지 확인

### 3. 업로드 중 네트워크 끊김 / `RemoteDisconnected`
- LFS hash 기반 멱등이므로 같은 `hf upload ...` 명령을 재실행하면 이어서 업로드
- 너무 자주 끊긴다면 회선이 불안정한 시간대를 피하거나, 3개 모델을 순차 업로드로 전환

### 4. Step 4-3 `Copy-Item : ... 경로의 일부를 찾을 수 없습니다 (DirectoryNotFoundException)`
- `hf-space-remote/` 디렉토리가 만들어지지 않은 상태에서 Copy-Item을 실행한 것
- 근본 원인: Step 4-1의 Space 생성이 (구 CLI 문법 `--space_sdk`로) silently 실패했거나, Step 4-2의 `git clone`이 자격증명 부족으로 실패
- 조치:
  1. `hf repos create Yun-choice/kluebert-counshelper --repo-type space --space-sdk gradio --exist-ok` 재실행 (신 문법, 하이픈)
  2. `hf auth whoami`로 로그인 확인
  3. `git clone https://huggingface.co/spaces/Yun-choice/kluebert-counshelper hf-space-remote` 재실행
  4. **Step 4-2.5의 `Test-Path .\hf-space-remote`가 `True`인지 확인 후** Step 4-3 실행

### 5. Space 빌드 실패: `transformers` import 에러 / 버전 충돌
- `hf-space/requirements.txt`의 `transformers==4.46.3` 핀이 풀려 있을 가능성 (편집된 적 있는지 확인)
- 조치: 핀이 맞다면 Space settings → **Factory rebuild** 클릭 (캐시 무시 재빌드)
- 그래도 실패하면 빌드 로그의 첫 번째 에러 줄을 그대로 검색

### 5-2. 부팅 로그는 정상인데 곧바로 Runtime error로 빠지고 healthz 응답이 "Your space is in error"
- 증상: Logs에 `모델 로드 완료: ['anxiety', 'depression', 'addiction']`이 1~2회 찍히고 그 후 침묵 → Space 우상단 "Runtime error" 뱃지
- 원인: HF Spaces Gradio SDK의 자동 launcher가 `gr.mount_gradio_app(api, _demo, ...)` + FastAPI `app` 변수 패턴을 안정적으로 띄우지 못하는 경우 발생. 컨테이너가 7860 포트에 정상 listening하지 못해 startup probe 실패 → kill → 반복 → "Runtime error".
- 조치: **Docker SDK로 전환** (uvicorn을 명시적으로 시작해 모든 ambiguity 제거):
  1. `hf-space/Dockerfile` 신규 생성:
     ```dockerfile
     FROM python:3.11-slim
     WORKDIR /app
     RUN useradd -m -u 1000 user
     USER user
     ENV HOME=/home/user
     ENV PATH=/home/user/.local/bin:$PATH
     COPY --chown=user requirements.txt requirements.txt
     RUN pip install --no-cache-dir --upgrade -r requirements.txt
     COPY --chown=user . /app
     CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
     ```
  2. `hf-space/README.md`의 YAML 헤더를 Docker SDK 형식으로 교체:
     ```yaml
     ---
     title: KlueBERT CounsHelper
     emoji: 🧠
     colorFrom: blue
     colorTo: green
     sdk: docker
     app_port: 7860
     pinned: false
     license: cc-by-sa-4.0
     ---
     ```
     (Gradio용 `sdk_version`, `app_file`, `python_version` 필드는 모두 제거)
  3. Phase 4-3~4-4 절차로 재푸시 (`Copy-Item .\hf-space\* .\hf-space-remote\ -Recurse -Force`로 Dockerfile까지 한 번에 복사).

### 5-1. Space 빌드 로그에 `ModuleNotFoundError: No module named 'audioop'` (또는 `pyaudioop`)
- 원인: HF Space의 기본 Python 런타임이 3.13으로 잡혔는데, Python 3.13에서 표준 라이브러리 `audioop`이 **PEP 594에 따라 제거**됨. gradio → pydub → audioop 의존 체인이 깨져 부팅 불가.
- 조치: `hf-space/README.md` YAML 헤더의 `app_file: app.py` 줄 뒤에 다음 한 줄 추가:
  ```yaml
  python_version: "3.11"
  ```
  변경 후 Phase 4-3~4-4 절차로 `hf-space-remote/`에 복사 → commit + push:
  ```powershell
  Copy-Item .\hf-space\README.md .\hf-space-remote\README.md -Force
  cd hf-space-remote
  git add README.md
  git commit -m "fix: pin python_version to 3.11 (audioop removed in 3.13)"
  git push
  cd ..
  ```
  Space가 자동 재빌드되어 Python 3.11 컨테이너로 부팅 → gradio 정상 import.

### 6. Space 부팅 시 401 / `RepositoryNotFoundError` / `from_pretrained` 실패
- 증상: Space 로그에 `401 Client Error: Unauthorized for url: https://huggingface.co/Yun-choice/kluebert-<label>/resolve/main/tokenizer_config.json` + `RepositoryNotFoundError`
- 주의: HF는 **미존재 repo와 private repo를 모두 401로 응답**(정보 노출 방지)하므로 메시지만으로는 두 원인을 구별할 수 없음. 진단부터.

**진단 — 어느 쪽인지 확인**

브라우저 본인 계정 로그인 상태로 `https://huggingface.co/Yun-choice` 페이지 열어 실제 모델 repo 이름을 확인:
- 이름이 **정확히 `kluebert-anxiety`/`kluebert-depression`/`kluebert-addiction`이 아니면** → 원인 6-A (오타)
- 이름은 정확한데 시크릿 창에서 404가 뜨면 → 원인 6-B (private)

**원인 6-A. repo 이름 오타** (예: `klubert-anxiety` ← `e` 누락)

`hf-space/app.py`는 정확한 이름 `kluebert-*`(klue + bert)를 찾고 있음. repo를 정확한 이름으로 rename하면 코드 변경 불필요:

```powershell
hf repos move Yun-choice/<오타이름>-anxiety    Yun-choice/kluebert-anxiety
hf repos move Yun-choice/<오타이름>-depression Yun-choice/kluebert-depression
hf repos move Yun-choice/<오타이름>-addiction  Yun-choice/kluebert-addiction
```

대안 (web UI): 각 repo Settings → "Rename or transfer this model" → 새 이름 입력. rename은 파일 보존(재업로드 불필요).

**원인 6-B. private repo**

```powershell
hf repos settings Yun-choice/kluebert-anxiety    --private false
hf repos settings Yun-choice/kluebert-depression --private false
hf repos settings Yun-choice/kluebert-addiction  --private false
```

또는 private 유지가 필요하면 Space settings에 `HF_TOKEN` Secret 추가 + `hf-space/app.py` 상단에 `from huggingface_hub import login; login(token=os.environ["HF_TOKEN"])` 추가 후 재푸시.

**공통 마무리** (6-A/6-B 모두)

Space는 자동으로 재시도하지 않으므로:
- `https://huggingface.co/spaces/Yun-choice/kluebert-counshelper/settings` → **Factory rebuild** 클릭
- Logs 탭에서 `INFO 모델 로드 완료: ['anxiety', 'depression', 'addiction']` 확인

### 7. `POST /predict`가 항상 401
- Space Secret 이름이 `HF_SPACE_API_KEY`와 **대소문자까지 일치**하는지 확인
- 프로젝트 `.env`의 `KLUEBERT_API_KEY` 값이 Space Secret 값과 **공백 없이 정확히 일치**하는지 확인 (앞뒤 공백, 줄바꿈 주의)
- Phase 8-1로 환경변수를 다시 로드한 뒤 재시도

### 8. `requests.exceptions.Timeout` (Streamlit 호출 시) / Space cold-start
- Space가 sleep 상태였을 가능성
- 조치: 위 **데모 당일 워밍업** 스크립트로 깨우기 → 다시 호출
- `_TIMEOUT = 90`(`src/classifier.py`)이 부족한 경우는 거의 없으나, 회선이 매우 느린 경우 일시적으로 120으로 올렸다가 원복 가능

### 9. PowerShell `curl.exe`로 한국어 body POST 시 `Could not resolve host: 媛?...` + 서버는 `{"input":{}}` 422
- 증상: curl 출력에 한국어로 보이는 깨진 hostname 에러가 여러 줄 + 서버 응답 `"input":{}` (받은 body가 빈 객체)
- 원인: Windows PowerShell 콘솔 코드페이지(CP949)와 UTF-8 한국어 JSON의 충돌. `ConvertTo-Json`이 만든 raw UTF-8 한국어가 PowerShell→`curl.exe` 인자 전달 과정에서 CP949로 잘못 인코딩되고, 백틱 멀티라인 paste 시 일부 라인이 분리되면서 한국어가 별도 인자로 들어가 curl이 hostname으로 해석. body에는 빈 `{}`만 남음.
- 조치: Step 8-2/8-3의 새 명령처럼 `curl.exe` 대신 **`Invoke-RestMethod` + `[System.Text.Encoding]::UTF8.GetBytes()`** 사용. PowerShell 네이티브이므로 한국어가 안전하게 UTF-8로 전달됨.
- 영어/숫자만 POST하는 경우는 curl.exe로도 문제없음 — 한국어 멀티바이트 문자가 들어갈 때만 발생.

### 11. 모든 입력에 anxiety=depression=addiction=1로 동일하게 떨어짐 (변별력 부족)
- 증상: 우울/불안/중독/정상 어떤 한국어 상담 텍스트든 동일하게 세 라벨 모두 1, raw 값도 입력 내용과 무관하게 늘 비슷한 패턴 (depression≈1.30 > anxiety≈1.27 > addiction≈1.23 고정)
- 진단: 학습 노트북 `kluebert_train.ipynb` 셀 출력에서 학습된 모델 자체가 가사 한 줄("비도 오고 그래서 네 생각이 났어...") 입력에 `Predicted numbers: 1` 출력. 즉 **학습은 정상이지만 모델이 거의 모든 입력에 1 예측하도록 수렴**. 우리 Space 추론은 노트북과 완전 일치 (raw 1.2791, closest_integer→1).
- 원인: AI Hub 데이터셋 자체의 라벨 분포 편향(정상 15% minority) + 작은 학습 샘플(질환별 528건) → 모델이 majority baseline(~73%)에서 수렴. model card 정확도 66~74%는 majority 항상 1 예측의 baseline 정확도.
- 조치 (단기): `src/classifier.py`의 `KLUEBERT_BYPASS=True`로 1차 우회 → 2차 Gemini 28요인이 실제 분류 수행. 모델이 결과적으로 거의 모든 입력에 1이므로 우회와 동일 동작. PRD §F1의 "정상이면 종료" 분기는 사실상 dead code.
- 조치 (근본): AI Hub 전체 데이터셋(465K건)으로 재학습. 정상 라벨 비율을 인위적으로 늘리거나 class weight 조정. 또는 다른 한국어 정신건강 분류 모델로 교체. 마감 D-5 이후 본선/시상식 전 작업 가능.

### 10. Space는 가동 중이고 Secret도 정확한데 `Invoke-RestMethod`가 계속 401 unauthorized
- 증상: 부팅 로그에 401 unauthorized + Space settings에 `HF_SPACE_API_KEY` 정상 등록 + `.env`의 `KLUEBERT_API_KEY` 값도 본인이 등록한 그대로
- 원인: `.env` 라인 끝의 **trailing whitespace 또는 `\r`(CRLF)**이 `Split('=',2)[1]`에 그대로 따라와서 헤더 값에 포함됨. 서버 Secret에는 그 1바이트가 없으니 비교 실패 → 401. 육안으론 같아 보임.
- 진단: 클라이언트 값의 길이와 마지막 문자 코드 확인
  ```powershell
  $key = (Get-Content .env | Select-String '^KLUEBERT_API_KEY=').Line.Split('=',2)[1]
  Write-Output "Length: $($key.Length)"
  Write-Output "Last char code: $([int]$key[-1])"
  ```
  `secrets.token_urlsafe(32)` 출력은 항상 43자. 길이가 44+이거나 last char code가 13(`\r`)/32(공백)/10(`\n`)이면 확정.
- 조치: Step 8-1 명령에 `.Trim()` 추가:
  ```powershell
  $env:KLUEBERT_API_KEY = (Get-Content .env | Select-String '^KLUEBERT_API_KEY=').Line.Split('=',2)[1].Trim()
  ```
  endpoint URL 라인도 같은 패턴 적용 (Step 8-1 본문 참조).

---

## 핵심 참조 파일

- `hf-space/app.py` — Space 서버 본체 (Phase 4에서 복사)
- `hf-space/modeling_kluebert.py` — `CustomBertForSequenceRegression` 정의
- `hf-space/README.md` — Space YAML 메타데이터
- `src/classifier.py` — Streamlit 측 HTTP 클라이언트
- `config.py` (25~27행) — `KLUEBERT_ENDPOINT_URL`, `KLUEBERT_API_KEY`
- `.env.example` (8~11행) — `.env` 작성 양식
- `ai-model/kluebert/1.모델소스코드/py 소스코드/kluebert_run.py` — 원본 추론 코드 (회귀 → 0~3 정수 변환 로직)

---

## 완료 후 정리 (선택)

업로드 검증이 끝나면 `hf-upload/`는 더 이상 필요 없다 (1.27 GB 차지). 디스크 여유가 필요할 때 삭제:

```powershell
Remove-Item -Recurse -Force .\hf-upload\
```

`hf-space/`와 `hf-space-remote/`는 향후 Space 코드 업데이트 시 필요하므로 유지.
