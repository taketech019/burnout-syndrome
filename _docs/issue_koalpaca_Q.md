# Issue — KoAlpaca 12.8B 호스팅에서 instruction-following 실패

| 작성일 | 2026-05-22 (개정 2026-05-23: 진단 정정) |
| --- | --- |
| 상태 | Open — 양자화 스킴 mismatch가 핵심 원인 확인, NF4 호환 호스팅 또는 외부 API 필요 |
| 영향 | F3 (요약 보고서) 4섹션 추출 불가능. `summarize()`는 `raw` 필드로 폴백 동작 |

## 증상

`src/summarizer.py`가 `data/raw/test.txt` (1700자, 발화자 마커 포함 raw 상담 다이얼로그)를 입력으로 호출하면, 모델이 **4섹션 요약 instruction을 무시하고 상담 대화를 이어 생성**한다. `### 답변:` 다음 토큰부터 새로운 "내담자 :", "상담사 :" 발화를 만들어내며, `주요 증상 / 위험 요인 / 개선요인 / 개입요인` 키워드가 한 번도 등장하지 않음.

기대 출력 형식 (추론 노트북 `ai-model/koalpaca_run.ipynb` cell 11 output 기준):
```
주요 증상: …
위험 요인: …
개선요인: …
개입요인: …
```

실제 출력 (test.txt 입력 시):
```
상담치료가 필요하다는 생각이 들 때 @NAME 씨께서는 어떻게 하십니까?
내담자 : 저 같으면 그냥 제가 먼저 연락해서 만나자고 할 것 같아요.
상담사 : 그래요, 그래요. …
```

## 시도한 진단

### 양자화 등급 escalation
| 양자화 | BPW | 파일 크기 | test.txt 결과 |
| --- | --- | --- | --- |
| Q4_K_M | 5.12 | 7.7 GB | 4섹션 미출력, 대화 이어쓰기 |
| Q5_K_M | 5.82 | 8.8 GB | 동일 |
| Q8_0 | 8.50 | 13 GB | 동일 |

Q4_K_M → Q5_K_M → Q8_0으로 정밀도를 올려도 결과 동일. 처음에는 "양자화 비트수는 충분하므로 양자화는 누명"이라 결론냈으나 — 이 결론은 **부분적으로만 옳다**. 자세한 재분석은 §근본 원인 분석 1번 참고.

### Prompt 형식 변경
- `### 맥락:` (공백, 추론 노트북 `gen()` 형식) vs `###맥락:` (공백 없음, 학습 노트북 데이터 변환 형식) 둘 다 시도 → 동일 실패
- `### 답변:` 끝에 `주요 증상:` 헤더 prepend → 첫 줄만 "우울, 불안" 출력 후 대화 모드 이탈

### 디코딩 파라미터 변경
- Greedy (`temperature=0.0, top_k=1`) — 노트북 `do_sample=False` 매칭
- Weak sampling (`temperature=0.3, top_k=40, top_p=0.9`) — 양자화 후 회복 시도
- `repeat_penalty=1.2, repeat_last_n=256` — 노트북 `repetition_penalty=1.2`, `no_repeat_ngram_size=3` 근사

모두 결과 변화 없음.

## 근본 원인 분석

### 1. **양자화 스킴 mismatch — NF4 (학습) ≠ K-quants (우리 호스팅)**

KoAlpaca 모델 카드는 "4bit 양자화 모델"이라 명시되지만, **4bit 알고리즘의 종류가 우리 호스팅과 다르다**. 같은 비트수도 표현/근사 방식이 다르면 dequant된 가중치 값이 수치적으로 달라진다.

**KoAlpaca 학습/추론**(`ai-model/koalpaca_run.ipynb` cell 4):
```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",                 # NF4 = NormalFloat-4
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
```
→ **bitsandbytes의 NF4**. 가중치는 4bit로 저장, 행렬곱 시 dequant → **bf16에서 계산**. NormalFloat은 정규분포 가중치에 최적화된 4bit 데이터타입. compute_dtype이 bf16이라 forward pass의 수치 표현이 매우 정밀.

**우리 호스팅 (Phase 3)**:
```
llama-quantize.exe ... Q4_K_M
```
→ **llama.cpp의 GGML K-quants**. super-block(256 weight) + per-block scale + min을 쓰는 양자화. NF4와 전혀 다른 알고리즘.

**핵심**: KoAlpaca LoRA 어댑터는 `bf16(NF4-dequant(W_base)) + bf16(LoRA_A @ LoRA_B)` 이라는 정확한 forward 결과 위에서 학습됨. 다른 환경(GGML K-quants dequant 값)에서는 base weight 값이 미세하게 달라져, **LoRA delta가 학습 시점에 의도한 효과를 못 냄**.

### 2. 우리 변환 경로의 손실 누적
1. `ai-model/merge_and_convert.py` (shard 단위 머지): NF4 환경을 우회하고 raw bf16 base weight를 fp16으로 LoRA 머지 (`t + (B@A) * scaling`). 이미 학습 forward와 다른 가중치 분포.
2. `convert_hf_to_gguf.py`: fp16 → GGUF fp16 (큰 손실 없음).
3. `llama-quantize`: fp16 → K-quants. 학습 시 NF4가 본 분포와 다른 노이즈 패턴으로 재양자화.

### 3. 왜 Q8_0도 안 됐는가 — **정밀도가 아닌 호환성 문제**
Q8_0의 BPW=8.50은 fp16(16.00 BPW)의 53%로 일반적으로 instruction-following을 거의 보존할 정밀도. 그럼에도 실패한 이유:
- Q8_0도 GGML 계열 양자화라 **dequant 결과 분포가 NF4와 여전히 다름**
- 가중치 표현이 학습 시점과 미세하게 다르면, fragile fine-tune은 그 차이에 무너짐
- 즉 비트수를 올려도 같은 양자화 *패밀리*(GGML)에 머물면 호환 불일치는 그대로

### 4. 짧은 학습이 mismatch에 대한 robustness를 키울 기회를 막음
모델 카드의 학습 파라미터:
```
r=8, lora_alpha=32, lora_dropout=0.05
per_device_train_batch_size=1
gradient_accumulation_steps=1
max_steps=500              ← 학습 데이터 1278~1609 건의 31~39%만 학습
learning_rate=2e-4
fp16=True
optim="paged_adamw_8bit"
```

`max_steps=500 × batch_size=1` = 500 sample만 학습. QLoRA rank `r=8`로 매우 작은 어댑터. → **NF4 dequant 값에 sharply tuned된 fragile minima**. 가중치 분포가 살짝만 흔들려도 instruction-following이 무너지는 상태.

학습이 길고/rank가 컸다면 dequant 알고리즘 변화에 어느 정도 robust할 수 있었음. 짧은 학습 + 양자화 스킴 mismatch의 **결합**이 본 이슈의 실제 원인.

### 5. 추론 노트북에서 4섹션이 잘 나왔던 이유
`ai-model/koalpaca_run.ipynb` cell 11의 예시 출력은 두 가지 조건이 동시에 만족돼서 잘 나왔음:
1. **양자화 환경 일치**: 추론도 bitsandbytes NF4 + bf16 compute — 학습 forward와 동일
2. **입력 분포 일치**: `df['filename']` 루프로 학습 데이터셋 자체 또는 매우 유사한 형식의 데이터를 추론 input으로 사용 (in-distribution)

우리 호스팅에서는 둘 다 깨짐:
1. NF4 → K-quants 변환
2. test.txt는 새 raw dialogue (학습 input 분포와 다를 수 있음)

### 6. 모델 카드의 input/output 명세
> Input: 상담자와 내담자의 발화자 구분 표시가 되어 있는 상담내용 스크립트  
> Output: 요약보고서 텍스트

test.txt는 형식상 명세에 부합. 명세만 보면 OOD가 아님. 그러나 학습 데이터 1278건 평균 길이·도메인 분포는 확인 불가 — 약간의 분포 차이도 짧은 학습 모델에는 영향. 다만 §1~3의 **양자화 스킴 mismatch가 1차 원인**이고, 입력 분포 차이는 2차 요인으로 보임.

## 현재 워크어라운드

`src/summarizer.py`는 4섹션 정규식 매칭에 실패하면 `raw` 키에 전체 응답을 담아 반환:
```python
def _parse_sections(raw: str) -> dict:
    ...
    if not positions:
        return {k: "" for k in _SECTIONS} | {"raw": raw}
```

UI 단에서는 `result.get("raw")`를 사용자가 직접 편집할 수 있도록 표시하면 기능적으로 동작 가능. 단 4섹션 자동 분류·차트화는 불가.

## 운영 관련 추가 이슈

### Cloudflare Quick Tunnel 100s idle timeout
`stream=False` + `max_tokens=1024` + 그리디(약 6.8 tok/s) → 응답 시간 ~150초 → Cloudflare 524 (서버가 100초 안에 응답 시작 못함). 현재 회피책: `max_tokens=500`, `_TIMEOUT=300`. 근본 해결: `stream=True`로 SSE 처리.

### ngl 튜닝 결과
- Q4_K_M (7.7GB) + ngl=30 → 모델 전체 + KV 캐시 VRAM에 들어감, ~6.8 tok/s
- Q5_K_M (8.8GB) + ngl=20 → 절반만 GPU, ~4 tok/s
- Q8_0 (13GB) + ngl=10 → 1/3만 GPU, ~2.5 tok/s

8GB VRAM 한도라 Q5_K_M 이상은 production 부담.

## 해결 방안 후보

### A. KoAlpaca 어댑터 재학습 (강도 ↑ + GGML 호환 환경에서 학습)
- `max_steps=500` → 3000~5000으로 확장
- `r=8` → 16 또는 32로 확장
- 가능하면 GGML K-quants dequant 시뮬레이션 환경에서 fine-tune (또는 NF4 학습 후 fp16 머지·재양자화에 robust하도록 rank·step을 늘려 둔감화)
- 소요: GPU 1대로 4~8시간

### C. 외부 API로 F3 전환 (공모전 마감 임박 시 권장)
- Gemini API, Solar API 등으로 요약 호출 (PRD F1·F4와도 일관됨)
- 호스팅 인프라 자체 불필요 — Phase 0~5 작업물은 archive
- 공모전 마감 2026-05-29 임박, 가장 안전한 단기 선택

### D. Prompt-engineering으로 base 모델 강제 (보조)
- few-shot 예시 2~3개를 prompt에 박아 in-context 학습 효과
- KoAlpaca 어댑터 효과가 약해도 base polyglot-ko가 ICL은 가능
- 단 prompt 길이 증가로 응답 시간 추가

### E. **bitsandbytes NF4로 직접 추론 — 학습 환경 재현** (신규)
- 방식: `transformers + PeftModel.from_pretrained(koalpaca_save) + BitsAndBytesConfig(nf4)` 그대로 추론 서버화. vLLM·text-generation-inference·자체 FastAPI 등.
- 학습 forward와 동일한 NF4 dequant + bf16 compute 환경 → LoRA 효과 정상 작동 기대.
- **하드웨어 요구**: 12.8B NF4 ≈ VRAM 8~10GB. 1070 Ti 8GB는 빠듯/부족. **RTX 3060 12GB 이상** 또는 클라우드(Colab T4 16GB, Lightning AI, RunPod 등) 필요.
- llama.cpp 우회. 본 호스팅 머신은 NF4 추론 부적합 — 다른 머신/클라우드 필요.

### ~~B. fp16 GGUF 호스팅~~ (제거)
- 25GB GGUF를 ngl=0 (전부 CPU)로 서빙 → ~1 tok/s, 사용 불가능. 옵션에서 제외.

## 권장 다음 단계

마감 2026-05-29 (D-6) 기준으로:

1. **단기 (지금 즉시)**: `src/summarizer.py`는 현재 상태(`raw` 폴백 유지)로 머지. 4섹션 추출은 모델 한계 명시. 옵션 **C(외부 API)** 를 PM과 협의 후 결정 — 가장 안전한 마감 대응.
2. **중기 (마감 전 여유 시)**: 옵션 **E(NF4 직접 호스팅)** 검토. NF4 호환 머신(12GB+ VRAM)이 확보되면 1~2시간 안에 가동 가능. KoAlpaca 어댑터를 학습 환경 그대로 살릴 수 있음.
3. **장기 (마감 이후)**: 옵션 **A(재학습)** — 양자화 스킴에 둔감한 더 강한 fine-tune. 본 이슈의 근본 해소.
4. **공통**: 옵션 **D(few-shot prompt)** 는 다른 옵션과 병행 가능한 보조책.

## 관련 파일·문서

- `src/summarizer.py` — 호출 로직, `raw` 폴백 구현돼 있음
- `ai-model/merge_and_convert.py` — shard 단위 LoRA 머지 (RAM 32GB 머신용)
- `ai-model/serve.bat` — llama-server 기동 스크립트 (현재 Q4_K_M / ngl=30)
- `ai-model/koalpaca_run.ipynb` cell 4 — NF4 학습 환경 설정 참고
- `_docs/PRD.md` §F3 — 요약 보고서 명세
- `_docs/plan-koalpaca.md` — 호스팅 구성 7단계 계획
- 모델 카드 출처: (사용자 제공) https://huggingface.co/beomi/KoAlpaca-Polyglot-12.8B 또는 동등 파생
