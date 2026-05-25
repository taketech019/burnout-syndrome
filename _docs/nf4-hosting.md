# 명세서 — KoAlpaca-Polyglot-12.8B NF4 + LoRA Attach 호스팅 (길 A)

| 작성일 | 2026-05-23 |
| --- | --- |
| 대상 | Claude Code CLI |
| 선행 문서 | `_docs/PRD.md` §F3, `_docs/issue-koalpaca.md` (양자화 mismatch 분석) |
| 마감 | 2026-05-29 (D-6) |
| 목적 | 학습 시점과 동일한 양자화 환경(NF4 + bf16 compute)에서 LoRA 어댑터를 **머지하지 않고 attach**하여 4섹션 요약 instruction-following 복원 |

---

## 0. 핵심 원칙 — 시작 전 반드시 숙지

1. **LoRA 어댑터를 base에 머지하지 않는다.** `peft.PeftModel.from_pretrained(base, adapter_path)`로 attach만 한다. 이슈 문서 §1~3에서 머지 자체가 손실의 출발점임이 확인됨.
2. **양자화는 NF4 한 종류만 쓴다.** llama.cpp의 GGUF/K-quants는 사용 금지. 기존 `merge_and_convert.py`·`serve.bat`·`.gguf` 파일은 **건드리지 말고 그대로 둔다** (롤백 안전망).
3. **8GB VRAM 한계는 `device_map="auto"` + `max_memory`로 CPU offload 처리.** 모든 layer를 GPU에 올리려 하지 말 것.
4. **Preflight check가 실패하면 즉시 중단하고 사용자에게 보고.** 길 B(클라우드) 전환 결정을 사용자에게 위임. 클로드 코드가 임의로 우회·머지하지 않는다.

---

## 1. 환경 가정

| 항목 | 값 |
| --- | --- |
| OS | Windows |
| GPU | NVIDIA GTX 1070 Ti (8GB VRAM, Pascal, **CC 6.1**) |
| RAM | 32GB |
| Python | 3.11 (PRD 기준) |
| 작업 디렉토리 | 리포 루트 (`_docs/`, `src/`, `ai-model/`이 있는 곳) |
| LoRA 어댑터 | 로컬에 `adapter_config.json` + `adapter_model.bin` 또는 `.safetensors` 존재 — **정확한 경로는 작업 초기에 탐색해서 확정** |
| Base 모델 | `beomi/KoAlpaca-Polyglot-12.8B` (HuggingFace Hub에서 다운로드, ~24GB) |

---

## 2. ⚠️ Preflight Check — 본 작업 진입 전 반드시 통과

1070 Ti는 Pascal 아키텍처이며, bitsandbytes 4bit 양자화 및 bf16 compute의 하드웨어 지원이 의심된다. 본 작업의 성패를 가르는 게이트이므로 **가장 먼저** 검증한다.

### Step 0-1. 환경 정보 수집

```bash
nvidia-smi
python -c "import torch; print('cuda:', torch.cuda.is_available(), 'cc:', torch.cuda.get_device_capability(0), 'name:', torch.cuda.get_device_name(0))"
```

기대: `cuda: True`, `cc: (6, 1)`, `name: NVIDIA GeForce GTX 1070 Ti`

### Step 0-2. bitsandbytes 설치 + Windows 호환 확인

```bash
pip install "bitsandbytes>=0.43.0" "transformers>=4.40.0" "peft>=0.10.0" "accelerate>=0.30.0"
python -m bitsandbytes
```

`python -m bitsandbytes`는 self-diagnostic을 출력한다. **"compiled with CUDA support" 및 "CUDA SETUP: Successful"** 메시지가 나와야 통과. 실패 시 Windows wheel 설치 이슈 — `pip install bitsandbytes --upgrade` 또는 `bitsandbytes-windows` 별도 패키지 시도. 그래도 안 되면 **Preflight FAIL — 사용자에게 보고**.

### Step 0-3. NF4 4bit 양자화 하드웨어 지원 검증 (가장 결정적)

```python
# preflight_nf4.py — 최소 모델로 NF4 로드 가능 여부 확인
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# 작은 모델로 먼저 시도 (12.8B 다운받기 전에 호환성만 확인)
try:
    model = AutoModelForCausalLM.from_pretrained(
        "EleutherAI/polyglot-ko-1.3b",  # 1.3B로 빠르게 검증
        quantization_config=bnb_config,
        device_map="auto",
    )
    # 실제 forward 한 번 돌려서 런타임 에러 없는지 확인
    x = torch.tensor([[1, 2, 3]]).to(model.device)
    with torch.no_grad():
        out = model(x)
    print("PREFLIGHT_NF4_BF16: PASS")
except Exception as e:
    print(f"PREFLIGHT_NF4_BF16: FAIL — {type(e).__name__}: {e}")
    # bf16이 Pascal에서 안 된다면 fp16으로 폴백 시도
    bnb_config.bnb_4bit_compute_dtype = torch.float16
    try:
        model = AutoModelForCausalLM.from_pretrained(
            "EleutherAI/polyglot-ko-1.3b",
            quantization_config=bnb_config,
            device_map="auto",
        )
        x = torch.tensor([[1, 2, 3]]).to(model.device)
        with torch.no_grad():
            out = model(x)
        print("PREFLIGHT_NF4_FP16: PASS (bf16 미지원, fp16으로 대체 가능)")
    except Exception as e2:
        print(f"PREFLIGHT_NF4_FP16: FAIL — {type(e2).__name__}: {e2}")
        print("=> 1070 Ti에서 bitsandbytes NF4 자체가 동작하지 않음. 길 A 불가능. 길 B(클라우드) 전환 필요.")
```

**결과 해석**:

| 결과 | 의미 | 다음 행동 |
| --- | --- | --- |
| `PREFLIGHT_NF4_BF16: PASS` | 학습 환경 100% 재현 가능 | Step 1 진입, `compute_dtype=bfloat16` 사용 |
| `PREFLIGHT_NF4_FP16: PASS` | bf16은 안 되지만 fp16으로 NF4는 가능. **학습 forward와 미세하게 다름** — 4섹션 추출 부분 회복 가능, 완전 보장 X | Step 1 진입하되 사용자에게 "compute_dtype 변경으로 인한 잔여 위험" 보고. `compute_dtype=float16` 사용 |
| 둘 다 FAIL | Pascal CC 6.1에서 bitsandbytes 4bit 미지원 | **본 명세서 작업 중단**, 사용자에게 보고하고 길 B(클라우드) 명세서 요청 |

### Step 0-4. LoRA 어댑터 위치 확정

```bash
# Windows PowerShell 또는 git bash
find . -name "adapter_config.json" 2>nul
# 또는
where /r . adapter_config.json
```

발견된 경로를 `ADAPTER_PATH` 변수로 기록. 같은 디렉토리에 `adapter_model.bin` 또는 `adapter_model.safetensors`가 있어야 함. 둘 다 없으면 Preflight FAIL.

`adapter_config.json` 파싱해서 `base_model_name_or_path` 필드 확인 — `beomi/KoAlpaca-Polyglot-12.8B`인지, 아니면 `EleutherAI/polyglot-ko-12.8b`인지 식별. **이 값이 base 모델 로드 시 사용할 식별자.**

---

## 3. 아키텍처

```
┌──────────────────────────────────────────────────────────────┐
│  Streamlit UI (src/, 기존)                                     │
│      ↓ HTTP POST /summarize                                   │
│  FastAPI 서버 (신규, ai-model/nf4_server.py)                   │
│      ↓                                                        │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  KoAlpaca Base (NF4, beomi/KoAlpaca-Polyglot-12.8B)     │ │
│  │   ├ device_map="auto" — GPU 7GB + CPU 28GB 분할         │ │
│  │   └ compute_dtype = bfloat16 (또는 fp16 — preflight 결과)│ │
│  │  + PeftModel.from_pretrained(LoRA 어댑터)               │ │
│  │     ※ merge 금지, attach만                              │ │
│  └─────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

**기존 자산과의 관계**:
- `ai-model/serve.bat`, `*.gguf` 파일: **건드리지 말 것**. 새 서버가 검증 실패 시 롤백 경로.
- `src/summarizer.py`: 백엔드 URL만 환경변수로 분리. 기본값을 새 서버 포트로.

---

## 4. 구현 단계

### Step 1. 의존성 설치 + 신규 디렉토리 구조

```
ai-model/
├── nf4_server.py          # 신규 — FastAPI 추론 서버
├── nf4_embedded.py        # 신규 — Streamlit 직접 임베드용 모듈 (옵션 B)
├── preflight_nf4.py       # 신규 — Step 0-3 스크립트
├── serve_nf4.bat          # 신규 — Windows 기동 스크립트
├── requirements-nf4.txt   # 신규 — 본 작업용 의존성
└── serve.bat              # 기존, 건드리지 말 것
```

`requirements-nf4.txt`:
```
bitsandbytes>=0.43.0
transformers>=4.40.0
peft>=0.10.0
accelerate>=0.30.0
torch>=2.1.0
fastapi>=0.110.0
uvicorn[standard]>=0.27.0
pydantic>=2.0
sentencepiece
protobuf
```

### Step 2. 모델 로더 모듈 (공통)

`ai-model/nf4_loader.py` 신규 — FastAPI와 Streamlit 임베드가 공용:

```python
"""KoAlpaca NF4 + LoRA 로더. 추론 서버와 Streamlit 임베드가 공용."""
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# 환경변수로 경로·dtype 제어
BASE_MODEL_ID = os.getenv("KOALPACA_BASE", "beomi/KoAlpaca-Polyglot-12.8B")
ADAPTER_PATH = os.getenv("KOALPACA_ADAPTER", "")  # Step 0-4에서 확정한 경로
COMPUTE_DTYPE = os.getenv("KOALPACA_COMPUTE_DTYPE", "bfloat16")  # preflight에 따라 "bfloat16" 또는 "float16"
GPU_MEM = os.getenv("KOALPACA_GPU_MEM", "7GiB")
CPU_MEM = os.getenv("KOALPACA_CPU_MEM", "28GiB")

_MODEL = None
_TOKENIZER = None

def load():
    """Lazy load. 프로세스당 1회. 12.8B NF4 로드 ~3~5분, CPU offload 포함."""
    global _MODEL, _TOKENIZER
    if _MODEL is not None:
        return _MODEL, _TOKENIZER

    if not ADAPTER_PATH or not os.path.exists(os.path.join(ADAPTER_PATH, "adapter_config.json")):
        raise RuntimeError(f"LoRA adapter not found at KOALPACA_ADAPTER={ADAPTER_PATH}")

    dtype = torch.bfloat16 if COMPUTE_DTYPE == "bfloat16" else torch.float16

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=True,
    )

    print(f"[loader] Loading base {BASE_MODEL_ID} with NF4 (compute_dtype={COMPUTE_DTYPE})...")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        max_memory={0: GPU_MEM, "cpu": CPU_MEM},
        low_cpu_mem_usage=True,
    )

    print(f"[loader] Attaching LoRA adapter from {ADAPTER_PATH}...")
    model = PeftModel.from_pretrained(base, ADAPTER_PATH)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    _MODEL, _TOKENIZER = model, tokenizer
    print("[loader] Ready.")
    return _MODEL, _TOKENIZER


def generate(prompt: str, max_new_tokens: int = 500, temperature: float = 0.0) -> str:
    """학습 노트북(koalpaca_run.ipynb cell 11)의 gen() 형식 재현."""
    model, tokenizer = load()
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0),
            temperature=temperature if temperature > 0 else 1.0,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # 입력 프롬프트 길이를 잘라낸 뒤 답변만 디코드
    answer = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return answer


def build_prompt(transcript: str) -> str:
    """학습 데이터 변환 형식과 추론 노트북 gen() 형식 통일.
    
    이슈 문서 §시도한 진단/Prompt 형식 변경 참고:
    학습은 '###맥락:' (공백 없음), 추론 노트북은 '### 맥락:' (공백 있음).
    학습 형식이 우선 — 공백 없는 쪽을 기본으로 한다.
    """
    return f"###맥락:{transcript}\n\n### 답변:"
```

### Step 3. FastAPI 서버 (옵션 A)

`ai-model/nf4_server.py` 신규:

```python
"""KoAlpaca NF4 + LoRA attach 추론 서버. src/summarizer.py가 HTTP로 호출."""
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from nf4_loader import load, generate, build_prompt

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 서버 기동 시 모델 미리 로드 (warm start)
    print("[server] Warming up model...")
    load()
    print("[server] Ready to serve.")
    yield

app = FastAPI(lifespan=lifespan)

class SummarizeRequest(BaseModel):
    transcript: str
    max_new_tokens: int = 500
    temperature: float = 0.0

class SummarizeResponse(BaseModel):
    text: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/summarize", response_model=SummarizeResponse)
def summarize(req: SummarizeRequest):
    if not req.transcript.strip():
        raise HTTPException(400, "empty transcript")
    prompt = build_prompt(req.transcript)
    try:
        text = generate(prompt, max_new_tokens=req.max_new_tokens, temperature=req.temperature)
    except Exception as e:
        raise HTTPException(500, f"generation failed: {type(e).__name__}: {e}")
    return SummarizeResponse(text=text)
```

`ai-model/serve_nf4.bat` 신규 (Windows 기동):
```bat
@echo off
set KOALPACA_BASE=beomi/KoAlpaca-Polyglot-12.8B
set KOALPACA_ADAPTER=.\path\to\adapter_dir
set KOALPACA_COMPUTE_DTYPE=bfloat16
set KOALPACA_GPU_MEM=7GiB
set KOALPACA_CPU_MEM=28GiB

cd /d %~dp0
uvicorn nf4_server:app --host 127.0.0.1 --port 8765 --workers 1
```

**`workers=1` 필수** — 12.8B 모델 메모리 사용량 때문에 다중 워커 불가.

### Step 4. Streamlit 직접 임베드 (옵션 B)

`ai-model/nf4_embedded.py` 신규 — Streamlit `@st.cache_resource` 활용:

```python
"""Streamlit 직접 임베드. 같은 프로세스에서 모델 로드, HTTP 왕복 없음."""
import streamlit as st
from nf4_loader import load, generate, build_prompt

@st.cache_resource(show_spinner="KoAlpaca NF4 모델 로딩 중 (초기 3~5분 소요)...")
def _get_model():
    return load()

def summarize_local(transcript: str, max_new_tokens: int = 500) -> str:
    _get_model()  # cache 워밍
    prompt = build_prompt(transcript)
    return generate(prompt, max_new_tokens=max_new_tokens, temperature=0.0)
```

**옵션 A vs B 선택 가이드**:
- **옵션 A (FastAPI 권장)**: Streamlit 재시작 시 모델 재로드 안 됨, 디버깅 용이, 배포 분리 가능. **기본 권장.**
- **옵션 B (임베드)**: 인프라 단순, 그러나 Streamlit hot reload마다 RAM 점유. 데모 1회용으로만.

### Step 5. summarizer.py 연동 + 폴백 유지

`src/summarizer.py` 수정 — 백엔드 URL을 환경변수로 분리:

```python
import os
import requests

KOALPACA_BACKEND_URL = os.getenv("KOALPACA_BACKEND_URL", "http://127.0.0.1:8765/summarize")
_TIMEOUT = 600  # NF4 + CPU offload는 느림. 7~10분 응답 가능성 감안.

def summarize(transcript: str) -> dict:
    try:
        r = requests.post(
            KOALPACA_BACKEND_URL,
            json={"transcript": transcript, "max_new_tokens": 500},
            timeout=_TIMEOUT,
        )
        r.raise_for_status()
        raw = r.json()["text"]
    except Exception as e:
        return {k: "" for k in _SECTIONS} | {"raw": "", "error": str(e)}
    
    parsed = _parse_sections(raw)
    return parsed  # 기존 정규식 매칭 로직 그대로, 실패 시 raw 폴백
```

**`raw` 폴백 로직은 절대 제거하지 말 것** — NF4 attach가 4섹션을 회복할 가능성이 높지만, 보장은 아니다.

---

## 5. 검증 (Acceptance Criteria)

### V1. 모델 로드 성공

```bash
cd ai-model
serve_nf4.bat
# 다른 터미널에서:
curl http://127.0.0.1:8765/health
# {"status": "ok"} 응답
```

기준: 서버 기동 시 OOM 없이 모델 로드 완료. nvidia-smi에서 VRAM 사용량 7GB 이하, 프로세스 RAM 사용량 12~16GB.

### V2. 4섹션 추출 — 본 작업의 핵심 검증

```bash
curl -X POST http://127.0.0.1:8765/summarize \
  -H "Content-Type: application/json" \
  -d @test_request.json
```

`test_request.json`은 `data/raw/test.txt` 내용을 transcript 필드로 감싼 JSON.

**Pass 기준**:
- 응답 텍스트에 `주요 증상`, `위험 요인`, `개선요인`, `개입요인` (또는 `상담사 개입 요인`) 4개 헤더가 **모두** 등장
- 각 섹션이 빈 줄이 아니라 실제 내용을 포함
- 이슈 문서의 실패 출력처럼 "내담자 :", "상담사 :" 발화를 새로 생성하지 **않음**

**Partial pass**: 헤더 일부(2~3개)만 등장 — 모델 한계로 인정하고 `summarizer.py`의 `raw` 폴백 유지, 사용자가 직접 편집.

**Fail**: 헤더가 하나도 안 나오고 대화 이어쓰기 → preflight는 통과했지만 양자화 mismatch 외 요인(짧은 학습 등)이 주된 원인. 길 A 효과 없음. 사용자에게 보고 후 길 B 또는 옵션 C(외부 API)로 전환.

### V3. 응답 시간

- 첫 요청 (cold start, 모델 로드 포함): 5~10분 허용
- 이후 요청 (warm): max_new_tokens=500 기준 3~10분 허용. CPU offload로 인해 느림 — 이는 길 A의 알려진 한계.

PRD §8의 "텍스트 분석 ≤ 90초" 기준은 본 작업에선 미달이 정상. PM과 협의해 **데모 시 사전 캐싱 시연** 또는 **짧은 입력(500자, max_tokens=300)** 으로 제약하는 방향 권장. 명세서에 따라 작업하는 클로드 코드는 이 사실만 README에 명시.

### V4. 기존 자산 무손상

`ai-model/serve.bat`, `ai-model/*.gguf`, `ai-model/merge_and_convert.py` 파일이 본 작업 시작 전과 동일한 상태로 남아있는지 확인. 새 파일만 추가됐고 기존 파일은 수정 0건이어야 함 (`src/summarizer.py` 1개 예외).

---

## 6. 산출물

신규 생성 파일:
- `ai-model/preflight_nf4.py`
- `ai-model/nf4_loader.py`
- `ai-model/nf4_server.py`
- `ai-model/nf4_embedded.py` (옵션 B 사용 시)
- `ai-model/serve_nf4.bat`
- `ai-model/requirements-nf4.txt`
- `_docs/path-a-results.md` — 작업 결과 보고서 (preflight 결과, V1~V4 검증 결과, 응답 시간 실측, 4섹션 추출 샘플 출력)

수정 파일:
- `src/summarizer.py` — 백엔드 URL 환경변수화, timeout 600초로 상향, `raw` 폴백 유지

**절대 수정/삭제 금지**:
- `ai-model/serve.bat`, `ai-model/*.gguf`, `ai-model/merge_and_convert.py`, `ai-model/koalpaca_run.ipynb`

---

## 7. 위험 & 폴백

| 위험 | 발생 시 행동 |
| --- | --- |
| Preflight Step 0-3에서 NF4 자체 실패 (Pascal CC 6.1 미지원) | 작업 중단. 사용자에게 보고하고 길 B(클라우드) 명세서 요청. 본 명세서의 어떤 단계도 진행하지 말 것 |
| bf16 미지원, fp16으로 폴백 | 진행하되 `_docs/path-a-results.md`에 "compute_dtype 변경" 명시. V2에서 부분 통과만 해도 수용 |
| 모델 로드 중 OOM (RAM 32GB 초과) | `max_memory={0: "6GiB", "cpu": "24GiB"}`로 축소 재시도. 그래도 OOM이면 길 A 불가 보고 |
| V2에서 4섹션 추출 여전히 실패 | 양자화 mismatch가 주원인이 아니었다는 증거. **재학습(옵션 A) 또는 외부 API(옵션 C)** 외에는 길 없음을 보고 |
| 응답 시간이 너무 길어 데모 불가 (>15분) | max_new_tokens=300으로 축소, 입력도 1000자 이내로 제약. README에 데모 제약 명시 |
| bitsandbytes Windows wheel 설치 실패 | 작업 중단. WSL 사용 가능 여부 사용자 확인 |

---

## 8. 작업 순서 요약

1. Preflight Step 0-1~0-4 모두 통과 확인. **하나라도 FAIL이면 즉시 중단·보고.**
2. `requirements-nf4.txt` 작성·설치.
3. `nf4_loader.py` 작성·import 테스트 (실제 generate 호출 없이 모듈 로드만).
4. `nf4_server.py` + `serve_nf4.bat` 작성, 서버 기동, V1 검증.
5. `test.txt`로 V2 검증. 결과를 `_docs/path-a-results.md`에 기록.
6. `src/summarizer.py` 수정.
7. (V2 통과 시) Streamlit UI에서 end-to-end 시연 가능한지 확인.
8. 최종 보고서 `_docs/path-a-results.md` 작성 — preflight 결과, V1~V4 결과, 응답 시간 실측, 4섹션 추출 샘플, 데모 제약 권장사항.

---

## 9. 사용자에게 보고해야 할 시점

다음 중 하나라도 발생 시 작업을 멈추고 사용자에게 보고:

1. Preflight 어느 단계든 FAIL
2. 모델 로드 후 V2(4섹션 추출)가 명백히 실패 (헤더 0개)
3. bitsandbytes Windows 설치 자체가 막힘
4. 응답 시간이 15분을 초과해 실용성 의문
5. 명세서에 명시되지 않은 결정이 필요한 상황

판단이 애매한 경우 진행하지 말고 보고할 것. D-6 일정상 무모한 우회 시도는 시간 낭비 위험이 더 큼.