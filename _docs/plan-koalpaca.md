# KoAlpaca llama.cpp 로컬 호스팅 계획

## 개요

PRD F3(요약 보고서) 기능은 KoAlpaca 12.8B 모델을 사용하지만, Streamlit Cloud에서는 8GB VRAM 부족으로 실행 불가. 개인 PC(GTX 1070 Ti, 32GB DDR4)에 llama.cpp로 양자화·서빙하고 Cloudflare Tunnel로 외부 노출 → Streamlit Cloud에서 HTTP 호출하는 구조.

## 실행 위치 분리

| 작업 | 실행 위치 |
|---|---|
| Phase 0~3: 다운로드 + 머지 + 변환 + 양자화 | 1070 Ti PC (32GB RAM + 60GB 디스크 필요) |
| Phase 4~5: llama-server + Cloudflare Tunnel | 1070 Ti PC (GPU 필수) |
| Phase 6~7: 코드 작성 (summarizer.py 등) | 개발 머신 (현재 기기) |

## 최종 아키텍처

```
[HF Hub] EleutherAI/polyglot-ko-12.8b (~25GB)
       + ai-model/koalpaca_save/ (LoRA 어댑터 25MB)
         ↓ merge_and_convert.py
[ai-model/koalpaca_merged/] (HF fp16)
         ↓ convert_hf_to_gguf.py (llama.cpp repo)
[ai-model/models/koalpaca-f16.gguf] (~25GB, 임시)
         ↓ llama-quantize Q4_K_M
[ai-model/models/koalpaca-q4km.gguf] (~7.5GB) ← 최종
         ↓ serve.bat (llama-server, ngl=30~35)
localhost:8080/v1/completions (OpenAI 호환)
         ↓ cloudflared tunnel
https://xxxx.trycloudflare.com
         ↓ src/summarizer.py
Streamlit Cloud 앱
```

---

## Phase 0: 사전 확인 (1070 Ti PC)

```powershell
# CUDA 버전 확인 (12.x 필요)
nvidia-smi

# 여유 디스크 확인 (최소 60GB 필요)
Get-PSDrive C

# Python 환경 준비
pip install torch transformers peft accelerate huggingface_hub gguf sentencepiece protobuf
```

> **주의**: 머지 단계에서 fp16 모델이 RAM ~26GB 점유. 실행 전 다른 앱 모두 종료.

---

## Phase 1: llama.cpp 설치 (1070 Ti PC, ~45분)

### 1-1. 사전 빌드 CUDA 바이너리

- URL: https://github.com/ggerganov/llama.cpp/releases
- 파일: `llama-<버전>-bin-win-cuda-cu12.x.x-x64.zip` 다운로드
- 압축 해제 위치: `C:\llama.cpp-bin\`
- 동작 확인: `C:\llama.cpp-bin\llama-cli.exe --version`

### 1-2. 변환 스크립트용 repo 클론 (빌드 불필요)

```powershell
git clone https://github.com/ggerganov/llama.cpp.git C:\llama.cpp-repo
```

---

## Phase 2: 모델 다운로드 + 머지 — `ai-model/merge_and_convert.py` (NEW)

> **핵심**: koalpaca_save/의 어댑터는 QLoRA 훈련 결과. 머지 시 베이스 모델을 반드시 **fp16으로** 로드 (4bit 아님).

```python
"""ai-model/merge_and_convert.py — 1회성 빌드 스크립트. ai-model/ 에서 실행."""
import torch
from pathlib import Path
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL_ID = "EleutherAI/polyglot-ko-12.8b"
ADAPTER_DIR   = Path("koalpaca_save")
MERGED_DIR    = Path("koalpaca_merged")

# 1. 베이스 모델 다운로드 (~25GB, 최초 1회)
snapshot_download(
    repo_id=BASE_MODEL_ID,
    ignore_patterns=["*.msgpack", "flax_model*", "tf_model*"],
)

# 2. fp16 로드 (CPU, 4bit 절대 사용하지 말 것)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="cpu",
    offload_folder="offload",   # OOM 안전장치
)
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)

# 3. LoRA 어댑터 머지
model = PeftModel.from_pretrained(model, str(ADAPTER_DIR))
model = model.merge_and_unload()

# 4. 저장
model.save_pretrained(str(MERGED_DIR), safe_serialization=True)
tokenizer.save_pretrained(str(MERGED_DIR))
print(f"완료: {MERGED_DIR}")
```

**RAM 부족 시 대안**: `device_map="auto"` 사용 (일부 레이어를 디스크로 오프로드, 속도 느림)

---

## Phase 3: GGUF 변환 + 양자화 (1070 Ti PC)

```powershell
cd C:\...\PR_BOAZ_2nd\ai-model

# 1. fp16 GGUF 변환 (~20분)
python C:\llama.cpp-repo\convert_hf_to_gguf.py `
    .\koalpaca_merged `
    --outfile .\models\koalpaca-f16.gguf `
    --outtype f16

# 2. Q4_K_M 양자화 (~10분, 결과 ~7.5GB)
C:\llama.cpp-bin\llama-quantize.exe `
    .\models\koalpaca-f16.gguf `
    .\models\koalpaca-q4km.gguf `
    Q4_K_M

# 3. 중간 파일 정리 (Phase 4 smoke test 통과 후)
Remove-Item -Recurse -Force .\koalpaca_merged\
Remove-Item -Force .\models\koalpaca-f16.gguf
```

**디스크 최대 사용량**: 변환 중 ~57GB. 정리 후 ~7.5GB.

---

## Phase 4: `ai-model/serve.bat` (NEW, 1070 Ti PC에서 실행)

> **ngl 튜닝**: ngl=20 시작 → 5씩 증가 → OOM 직전에서 -2. GTX 1070 Ti 8GB 기준 ngl=30 권장 시작.

```bat
@echo off
setlocal

set MODEL=C:\...\PR_BOAZ_2nd\ai-model\models\koalpaca-q4km.gguf
set PORT=8080
set NGL=%1
if "%NGL%"=="" set NGL=30
if "%KOALPACA_API_KEY%"=="" set KOALPACA_API_KEY=changeme-set-this

echo llama-server 시작 (ngl=%NGL%)...
C:\llama.cpp-bin\llama-server.exe ^
    -m "%MODEL%" ^
    -c 2048 ^
    -ngl %NGL% ^
    --host 0.0.0.0 ^
    --port %PORT% ^
    --api-key %KOALPACA_API_KEY% ^
    -t 8 ^
    --batch-size 256 ^
    --log-disable

endlocal
```

사용법: `serve.bat` (ngl=30 기본) / `serve.bat 35` (ngl 오버라이드)

**Smoke test**:
```powershell
$body = @{ prompt="### 명령어: 테스트`n`n###맥락: 테스트`n`n### 답변:"; max_tokens=50; stop=@("<|endoftext|>") } | ConvertTo-Json
Invoke-RestMethod -Method POST -Uri "http://localhost:8080/v1/completions" `
    -Headers @{ Authorization="Bearer changeme-set-this"; "Content-Type"="application/json" } `
    -Body $body
```

---

## Phase 5: Cloudflare Tunnel (1070 Ti PC에서 실행)

```powershell
# 설치: https://github.com/cloudflare/cloudflared/releases
# cloudflared-windows-amd64.exe → C:\cloudflared\cloudflared.exe

# serve.bat 실행 중인 상태에서:
C:\cloudflared\cloudflared.exe tunnel --url http://localhost:8080
# → "https://xxxx.trycloudflare.com" 출력
```

> **URL 관리**: 터널 재시작 시 URL 변경. Streamlit Cloud Secrets에서 `KOALPACA_ENDPOINT_URL` 업데이트 필요.  
> 안정적 URL 필요 시: Cloudflare 계정 + Named Tunnel 설정 (무료).

**세션 시작 체크리스트** (1070 Ti PC):
1. `serve.bat 30` 실행
2. `cloudflared tunnel --url http://localhost:8080` 실행
3. Streamlit Cloud Secrets에서 `KOALPACA_ENDPOINT_URL` 업데이트

---

## Phase 6: `src/summarizer.py` (NEW, 개발 머신)

> **프롬프트 형식 주의**: 학습(`koalpaca4bit_train.ipynb`)에서 `###맥락:` (공백 없음) 사용.  
> 추론 노트북의 `### 맥락:` (공백 있음)은 **불일치** — 학습 형식을 따름.  
> 컨텍스트 한계: 2048 토큰 → 입력 텍스트 1900자 트런케이션.

```python
"""src/summarizer.py — F3: KoAlpaca 상담 요약."""
import re
import requests
from config import KOALPACA_ENDPOINT_URL, KOALPACA_API_KEY

_INSTRUCTION = "다음과 같은 상담기록을 보고 요약서를 작성해주세요."
_MAX_INPUT_CHARS = 1900
_MAX_NEW_TOKENS = 1024
_TIMEOUT = 120
_STOP = ["<|endoftext|>", "<|sep|>", "### 명령어:"]
_SECTIONS = {
    "symptoms":             r"주요\s*증상",
    "risk_factors":         r"위험\s*요인",
    "improvement_factors":  r"개선\s*요인",
    "intervention_factors": r"개입\s*요인",
}


def _build_prompt(text: str) -> str:
    return (
        f"### 명령어: {_INSTRUCTION}\n\n"
        f"###맥락: {text[:_MAX_INPUT_CHARS]}\n\n"
        f"### 답변:"
    )


def _parse_sections(raw: str) -> dict:
    for stop in ["<|endoftext|>", "<|sep|>"]:
        raw = raw.split(stop)[0]
    raw = raw.strip()

    positions = {k: m.start() for k, p in _SECTIONS.items() if (m := re.search(p, raw))}
    if not positions:
        return {k: "" for k in _SECTIONS} | {"raw": raw}

    ordered = sorted(positions.items(), key=lambda x: x[1])
    result = {}
    for i, (key, start) in enumerate(ordered):
        end = ordered[i + 1][1] if i + 1 < len(ordered) else len(raw)
        result[key] = raw[start:end].strip()
    return {k: result.get(k, "") for k in _SECTIONS}


def summarize(text: str) -> dict:
    """F3: KoAlpaca 요약 호출. 반환: 4개 섹션 dict. 실패 시 'error' 키 포함."""
    if not KOALPACA_ENDPOINT_URL:
        return {k: "" for k in _SECTIONS} | {"error": "KOALPACA_ENDPOINT_URL 미설정"}

    url = KOALPACA_ENDPOINT_URL.rstrip("/") + "/v1/completions"
    try:
        resp = requests.post(
            url,
            json={"prompt": _build_prompt(text), "max_tokens": _MAX_NEW_TOKENS,
                  "temperature": 0.7, "stop": _STOP, "stream": False},
            headers={"Authorization": f"Bearer {KOALPACA_API_KEY}",
                     "Content-Type": "application/json"},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        return _parse_sections(resp.json()["choices"][0]["text"])
    except requests.exceptions.ConnectionError:
        return {k: "" for k in _SECTIONS} | {"error": "KoAlpaca 서버에 연결할 수 없습니다."}
    except requests.exceptions.Timeout:
        return {k: "" for k in _SECTIONS} | {"error": f"응답 시간 초과 ({_TIMEOUT}초)"}
    except Exception as e:
        return {k: "" for k in _SECTIONS} | {"error": str(e)}
```

---

## Phase 7: `config.py` / `.env.example` 수정 (개발 머신)

**`config.py` 추가** (기존 코드 아래):
```python
import os

KOALPACA_ENDPOINT_URL = os.getenv("KOALPACA_ENDPOINT_URL", "")
KOALPACA_API_KEY      = os.getenv("KOALPACA_API_KEY", "changeme-set-this")
```

**`.env.example` 추가**:
```
# KoAlpaca (llama-server + Cloudflare Tunnel) — 터널 재시작 시 URL 갱신 필요
KOALPACA_ENDPOINT_URL=https://xxxx.trycloudflare.com
KOALPACA_API_KEY=your-random-api-key-here
```

**Streamlit Cloud Secrets (TOML)**:
```toml
KOALPACA_ENDPOINT_URL = "https://xxxx.trycloudflare.com"
KOALPACA_API_KEY = "your-random-api-key-here"
```

---

## 수정 대상 파일 요약

| 파일 | 작업 | 실행 위치 |
|---|---|---|
| `ai-model/merge_and_convert.py` | NEW | 1070 Ti PC |
| `ai-model/serve.bat` | NEW | 1070 Ti PC |
| `src/summarizer.py` | NEW | 개발 머신 |
| `config.py` | UPDATE (os import + 2개 상수) | 개발 머신 |
| `.env.example` | UPDATE (KOALPACA 변수 추가) | 개발 머신 |

---

## 전체 일정 추정

| 단계 | 소요 | 위치 |
|---|---|---|
| Phase 0 사전 확인 | 30분 | 1070 Ti PC |
| Phase 1 llama.cpp 설치 | 45분 | 1070 Ti PC |
| Phase 2 다운로드 + 머지 | 2~4시간 | 1070 Ti PC |
| Phase 3 변환 + 양자화 | 30분 | 1070 Ti PC |
| Phase 4~5 서버 + 터널 | 45분 | 1070 Ti PC |
| Phase 6~7 코드 작성 | 1시간 | 개발 머신 |
| **합계** | **~6~7시간** | |
