# AI 모델 호스팅 개념 설명서

이 문서는 `plan-koalpaca.md`에 등장하는 기술 개념을 처음 접하는 팀원을 위해 작성했습니다.

---

## 1. 왜 KoAlpaca를 직접 호스팅해야 하나?

KoAlpaca 12.8B 모델은 **12억 8천만 개의 파라미터(숫자)**로 이루어진 거대한 수식입니다. 이 모든 숫자를 메모리에 올려야 추론(답변 생성)이 가능합니다.

| 환경 | GPU 메모리 | 가능 여부 |
|---|---|---|
| Streamlit Cloud (무료) | ~0GB (CPU만) | ❌ 불가 |
| Google Colab 무료 | 15GB (T4 GPU) | 조건부 가능 |
| GTX 1070 Ti (개인 PC) | 8GB | ✅ Q4 양자화 시 가능 |
| A100 (Modal 유료) | 80GB | ✅ 완전 가능 |

---

## 2. 모델 파일 형식

### safetensors
HuggingFace의 표준 모델 저장 형식. Python/PyTorch 생태계에서 사용.
- 크기: 12.8B 모델 기준 fp16(반정밀도)으로 약 25GB
- `ai-model/koalpaca_save/adapter_model.safetensors` = LoRA 어댑터만 (25MB)

### GGUF (GPT-Generated Unified Format)
llama.cpp에서 사용하는 파일 형식. 하나의 파일에 모델 구조 + 가중치 + 메타데이터가 모두 담김.
- PyTorch 불필요, llama.cpp 바이너리만으로 실행 가능
- 양자화된 형태로 저장 가능 → 파일 크기 대폭 감소
- `.gguf` 확장자

**왜 GGUF로 변환하나?**
- safetensors는 Python + PyTorch + transformers 라이브러리 전체가 필요
- GGUF는 C++로 작성된 llama.cpp 실행 파일 하나로 동작 → 더 빠르고, 메모리 효율적

---

## 3. 양자화 (Quantization)

### 개념
모델의 숫자(가중치)는 원래 **float32** (32비트 소수점)로 표현됩니다. 양자화는 이를 더 적은 비트로 압축하는 기술입니다.

```
float32: 3.14159265358979...  (32비트, 4바이트)
float16: 3.14159...           (16비트, 2바이트) — 크기 절반
int8:    3                    (8비트, 1바이트)  — 크기 1/4
int4:    3                    (4비트, 0.5바이트) — 크기 1/8
```

### Q4_K_M 이란?
`Q4_K_M` = "4비트, K-quant 방식, Medium 품질"

- **4비트**: 각 파라미터를 4비트로 표현 → float16 대비 파일 크기 1/4
- **K-quant**: 단순 4비트가 아니라, 블록 단위로 스케일 팩터를 저장해 정밀도 손실 최소화
- **M (Medium)**: K-quant 내에서 중간 품질 (S < M < L)

| 양자화 타입 | 12.8B 모델 크기 | 품질 손실 |
|---|---|---|
| fp16 (원본) | ~25GB | 없음 |
| Q8_0 | ~13GB | 거의 없음 |
| **Q4_K_M** | **~7.5GB** | **낮음 (권장)** |
| Q4_0 | ~6.5GB | 보통 |
| Q3_K_M | ~5.5GB | 높음 |

**Q4_K_M을 선택하는 이유**: 품질 손실이 적으면서 1070 Ti 8GB VRAM에 들어가는 최적점.

---

## 4. LoRA와 QLoRA

### LoRA (Low-Rank Adaptation)
대형 모델 전체를 재학습하지 않고, **작은 추가 가중치(어댑터)**만 학습하는 기법.

```
원본 모델 (frozen, 변경 안 함)
  + LoRA 어댑터 (학습됨, 원본의 0.1% 크기)
= KoAlpaca (상담 데이터에 최적화된 모델)
```

- `koalpaca_save/adapter_model.safetensors` (25MB) = 이 어댑터 파일

### QLoRA
4비트 양자화된 베이스 모델 위에 LoRA를 학습하는 기법. 일반 LoRA보다 훨씬 적은 GPU 메모리로 학습 가능.

### 머지 (Merge)
LoRA 어댑터를 베이스 모델에 **수학적으로 합산**하는 과정.

```
merge_and_unload() 실행 전:
  베이스 모델 (25GB) + 어댑터 (25MB) → 두 파일 필요

merge_and_unload() 실행 후:
  merged_model (25GB) → 파일 하나, 어댑터 없이 동작
```

**왜 머지가 필요한가?**
GGUF 변환 도구는 단일 모델 파일을 입력으로 받습니다. 어댑터가 별도로 존재하면 변환이 불가능합니다.

**주의사항**: 머지 시 모델을 **fp16(float16)으로 로드**해야 합니다. QLoRA 학습 시 4비트로 로드했다가 그 상태로 저장하면, 4비트 압축 정보가 가중치에 섞여서 제대로 된 머지가 불가능합니다.

---

## 5. llama.cpp

### 개념
C++로 작성된 LLM 추론 엔진. CPU/GPU 모두 지원. PyTorch 없이 GGUF 파일만으로 모델을 실행합니다.

- GitHub: `github.com/ggerganov/llama.cpp`
- 오리지널 LLaMA(Meta) 모델용으로 시작했지만, 지금은 GPTNeoX, Mistral, Gemma 등 대부분의 아키텍처 지원

### 주요 실행 파일

| 파일 | 역할 |
|---|---|
| `llama-cli.exe` | 터미널에서 모델과 직접 대화 (디버깅용) |
| `llama-server.exe` | HTTP 서버 실행 (앱에서 API로 호출) |
| `llama-quantize.exe` | GGUF 파일을 양자화 |
| `convert_hf_to_gguf.py` | HuggingFace 모델 → GGUF 변환 (Python 스크립트) |

### llama-server의 API
OpenAI API와 호환되는 HTTP 엔드포인트를 제공합니다.

```
POST http://localhost:8080/v1/completions     ← KoAlpaca 사용 (instruction-tuned)
POST http://localhost:8080/v1/chat/completions ← ChatGPT 스타일 (chat-tuned 모델용)
GET  http://localhost:8080/health             ← 서버 상태 확인
```

---

## 6. ngl (n-gpu-layers)

`-ngl 30` = "모델의 레이어 중 30개를 GPU VRAM에 올려라"

### 레이어란?
LLM은 여러 개의 **Transformer 레이어**를 쌓아서 만들어집니다. KoAlpaca(polyglot-ko-12.8b)는 40개의 레이어를 가집니다.

```
레이어 40 ← 출력
레이어 39
...
레이어 2
레이어 1 ← 입력
```

### GPU vs CPU 처리
- GPU VRAM에 올린 레이어: 빠른 병렬 처리
- 나머지 레이어: CPU + RAM에서 처리 (느림)

```
ngl=0  → 전부 CPU (1070 Ti 전혀 안 씀)
ngl=30 → 30개 GPU, 10개 CPU
ngl=40 → 전부 GPU (8GB VRAM 초과 시 OOM 오류)
```

### 1070 Ti 8GB 기준 Q4_K_M
Q4_K_M 모델의 총 레이어 크기 ~7.5GB이므로:
- `ngl=30` 시 → ~5.6GB VRAM (안전)
- `ngl=35` 시 → ~6.6GB VRAM (KV 캐시 포함 시 한계)
- `ngl=40` 시 → ~7.5GB VRAM + 오버헤드 = OOM 가능성

**튜닝 방법**: `serve.bat 20` → `serve.bat 25` → ... 5씩 올리면서 `nvidia-smi`로 VRAM 확인.

---

## 7. KV 캐시 (Key-Value Cache)

Transformer 모델이 이전에 처리한 텍스트를 재계산하지 않기 위해 저장해 두는 중간 계산 결과.

- `-c 2048` = 컨텍스트 크기 2048 토큰
- KV 캐시 메모리 = 레이어 수 × 헤드 수 × 시퀀스 길이 × 데이터 타입 크기
- 대략 2048 토큰 컨텍스트 기준 ~640MB ~ 1GB

ngl과 KV 캐시 합산이 8GB를 넘으면 VRAM 부족 오류 발생.

---

## 8. Cloudflare Tunnel

### 문제
- llama-server는 `localhost:8080`에서만 접근 가능
- Streamlit Cloud는 외부에서 이 PC로 직접 HTTP 요청을 보낼 수 없음 (방화벽, 공유기 NAT)

### 해결책: Cloudflare Tunnel
Cloudflare의 서버가 중간 다리 역할을 합니다.

```
[Streamlit Cloud]
        ↓ HTTPS 요청
[Cloudflare 서버] ← cloudflared가 연결 유지
        ↓
[내 PC의 localhost:8080]
```

### Quick Tunnel vs Named Tunnel

| 방식 | 설정 | URL | 비용 |
|---|---|---|---|
| Quick Tunnel | 명령어 1줄 | 재시작 시 변경 | 무료 |
| Named Tunnel | Cloudflare 계정 필요 | 고정 서브도메인 | 무료 |

**Quick Tunnel (계획에서 사용)**:
```powershell
cloudflared tunnel --url http://localhost:8080
# → https://random-words.trycloudflare.com 출력
```

URL이 바뀔 때마다 Streamlit Cloud Secrets 업데이트 필요.

---

## 9. OpenAI 호환 API

### completions vs chat/completions

| 엔드포인트 | 입력 | 용도 |
|---|---|---|
| `/v1/completions` | `prompt` (문자열) | 텍스트를 이어서 생성 |
| `/v1/chat/completions` | `messages` (배열) | 멀티턴 대화 |

KoAlpaca는 **instruction-tuned 모델**입니다. 프롬프트를 직접 완성하는 방식으로 동작하므로 `/v1/completions`를 사용합니다.

```python
# completions 방식 (KoAlpaca)
{
  "prompt": "### 명령어: ...\n\n###맥락: ...\n\n### 답변:",
  "max_tokens": 1024
}

# chat/completions 방식 (ChatGPT 스타일)
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."}
  ]
}
```

---

## 10. 토큰 (Token)과 컨텍스트 윈도우

### 토큰
LLM이 처리하는 텍스트의 기본 단위. 단어보다 작을 수도 있고, 여러 단어일 수도 있습니다.

- 영어: 1단어 ≈ 1~2토큰
- 한국어: 1글자 ≈ 1~2토큰 (한글은 영어보다 토큰 효율이 낮음)

예시: "내담자는 수면 장애를 호소함" → 약 15~20토큰

### 컨텍스트 윈도우
모델이 한 번에 처리할 수 있는 최대 토큰 수. polyglot-ko-12.8b = **2048 토큰**.

```
[프롬프트 (입력)] + [생성 텍스트 (출력)] ≤ 2048 토큰

프롬프트 = 명령어(~20) + 맥락(상담 텍스트) + 템플릿(~15)
출력 = max_tokens=1024

→ 상담 텍스트: 2048 - 20 - 15 - 1024 = 989토큰
→ 한국어 기준 약 1900자
```

이것이 `summarizer.py`에서 입력을 1900자로 자르는 이유입니다.

---

## 참고 링크

- llama.cpp GitHub: https://github.com/ggerganov/llama.cpp
- llama.cpp Releases: https://github.com/ggerganov/llama.cpp/releases
- cloudflared 다운로드: https://github.com/cloudflare/cloudflared/releases
- GGUF 형식 설명: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
- Q4_K_M 양자화 비교: https://github.com/ggerganov/llama.cpp/discussions/406
