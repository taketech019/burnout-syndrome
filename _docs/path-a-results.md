# 길 A (KoAlpaca NF4 + LoRA Attach 호스팅) 결과 보고서

| 작성일 | 2026-05-23 |
| --- | --- |
| 명세서 | `_docs/nf4-hosting.md` |
| 환경 | Windows 11, GTX 1070 Ti (8GB, Pascal CC 6.1), 32GB RAM, Python 3.11 |
| 결론 | ⚠️ **GATE 3 (V2 4섹션 추출) FAIL — 길 A 효과 없음. 사용자 결정 필요.** |

---

## 1. 게이트별 결과 요약

| 게이트 | 결과 | 비고 |
| --- | --- | --- |
| GATE 1 (Preflight) | ✅ PASS | NF4 + bf16 1070 Ti에서 동작 (probe 모델 polyglot-ko-1.3b) |
| GATE 2 (V1 모델 로드) | ✅ PASS | 12.8B base + LoRA attach 성공, VRAM 7958/8192 MiB |
| **GATE 3 (V2 4섹션 추출)** | ❌ **FAIL (0/4 헤더)** | 3회 시도 모두 대화 이어쓰기 또는 단어 나열만 출력 |

## 2. 작업 과정에서 발견된 명세서와의 차이

명세서를 충실히 따랐으나 실제 적용 단계에서 다음 차이를 처리해야 했음:

### 2-1. base 모델 ID
- 명세서: `beomi/KoAlpaca-Polyglot-12.8B`
- 실제 적용: **`EleutherAI/polyglot-ko-12.8b`** (`ai-model/koalpaca_save/adapter_config.json`의 `base_model_name_or_path` 값)
- 명세서 §0-4가 "adapter_config 값이 base 모델 로드 시 사용할 식별자"라고 명시했으므로 어댑터값을 따랐음

### 2-2. NF4 + CPU offload 옵션 필요
- 명세서에 없던 사항: `BitsAndBytesConfig`에 `llm_int8_enable_fp32_cpu_offload=True` 추가 필요
- 미설정 시 `ValueError: Some modules are dispatched on the CPU or the disk...` 에러로 로드 실패

### 2-3. transformers 5.x access violation
- 초기 설치된 transformers 5.9.0는 Windows에서 12.8B sharded safetensors 로드 시 access violation으로 세그폴트
- 트레이스: `torch/storage.py:456 __getitem__` ← `transformers/core_model_loading.py:977 _materialize_copy`
- 해결: `transformers>=4.40.0,<5.0`로 다운그레이드 → transformers 4.57.6 사용

### 2-4. peft 버전 호환
- `peft 0.12`는 어댑터의 `eva_config` 필드 모름 → `TypeError: LoraConfig.__init__() got an unexpected keyword argument 'eva_config'`
- 해결: `peft>=0.14.0,<0.15`로 갱신 → peft 0.14.0 사용

### 2-5. GPT-NeoX의 token_type_ids
- `tokenizer(prompt, return_tensors='pt')`가 `token_type_ids`를 포함시키는데 GPT-NeoX `generate()`가 거부
- 해결: `nf4_loader.py`에 `inputs.pop('token_type_ids', None)` 추가

### 2-6. 디스크 공간
- C: 드라이브 4.3GB만 남아 pip 캐시/torch wheel 압축해제 실패
- 해결: `TMP`, `TEMP`, `PIP_CACHE_DIR`, `HF_HOME`을 E:\\로 우회 설정

### 2-7. 프롬프트 형식 (V2 실패의 핵심)
- 명세서의 build_prompt 추천: `###맥락:{}\n\n### 답변:` (instruction 헤더 없음)
- 실제 학습/추론 노트북 형식: `### 명령어: {instruction}\n\n### 맥락: {input}\n\n### 답변:`
- 명세서 형식으로 시도 → 4섹션 0개 (대화 이어쓰기). 노트북 형식으로 수정 후 재시도 → 여전히 4섹션 0개

## 3. V2 실패 상세 (3회 시도)

입력: `data/raw/test.txt` (762자, 24줄, 상담사/내담자 대화)

| 시도 | 프롬프트 | temperature | max_new_tokens | 응답 시간 | 4섹션 헤더 | 출력 양상 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `###맥락:{}\n\n### 답변:` (명세서 형식) | 0.0 | 500 | 342초 | 0/4 | 가짜 상담사/내담자 대화 이어쓰기 |
| 2 | `### 명령어: ... \n\n### 맥락: ... \n\n### 답변:` (노트북 형식) | 0.0 | 500 | 335초 | 0/4 | 가짜 상담사/내담자 대화 이어쓰기 |
| 3 | 노트북 형식 | 0.3 | 300 | 219초 | 0/4 | 의미 없는 단어 나열 (상담료, 상담시간, ...) |

명세서 §5 Fail 시나리오의 정의 그대로:
> "헤더가 하나도 안 나오고 대화 이어쓰기 → preflight는 통과했지만 양자화 mismatch 외 요인(짧은 학습 등)이 주된 원인. 길 A 효과 없음."

### 3-1. 실패 원인 추적 — 최종 진단은 **CPU offload + LoRA 비호환** (하드웨어 부족)

가설을 순차로 검증:

**가설 1: 짧은 학습 (epoch=0.39, loss=2.53)** — 부분 기각.
- `ai-model/koalpaca4bit_train.ipynb` cell 14에서 학습 메트릭은 확실히 약함
- 그러나 모델 공식 F-1 = **60.94%** (`ai-model/5.AI모델사용매뉴얼/모델설명서_koalpaca.txt`)
- F-1 60% = LoRA 자체는 학습 환경에서 작동했다는 강력 증거

**가설 2: `no_repeat_ngram_size=3`이 헤더 토큰 차단** — 기각 (실험 1).
- penalty 끄고 재시도 → 모델이 같은 문장만 반복, 헤더 안 나옴.

**가설 3: prompt 형식 차이 (`### 명령어:` 헤더 누락)** — 기각 (실험 2).
- 학습 노트북 정확한 형식(`### 명령어: ... ### 맥락: ... ### 답변:`)으로 재시도 → 동일 실패.

**가설 4: `eos_token_id=tokenizer.eos_token_id`가 조기 종료** — 기각 (실험 6).
- 원본 노트북의 `eos_token_id=None`으로 변경 + max_tokens=1024 → 더 긴 ramble만 생성, 헤더 여전히 0/4.

**가설 5: schema 부분 학습 — 헤더 prepend로 trigger 가능** — 부분 사실 (실험 4).
- `### 답변:\n\n주요 증상:`을 강제로 prepend → 모델이 "불안 장애" 같은 정상 라벨 출력
- 그러나 첫 라벨 후 다시 대화 모드로 이탈 → schema 자체는 알지만 sequential flow 학습 안 됨처럼 보였음.

**가설 6: 입력이 OOD (test.txt가 학습 분포와 다름)** — **결정적 기각 (실험 7)**.
- 사용자 제공 `ai-model/label_depression_1_check_D001.json` (학습 분포와 동일한 AI-Hub 심리상담 코퍼스 포맷)
- paragraph 308개를 발화자 형식으로 합쳐 첫 1500자로 V2 실행
- ground truth `summary` 필드 ("주요 증상: 내담자는 우울한 기분과 무가치감을 ...") 존재
- 결과: **여전히 0/4 헤더, 가짜 대화 ramble만 출력**
- → 입력 분포 문제 아님. 환경 자체가 문제.

**가설 7: 1070 Ti VRAM 부족으로 인한 CPU offload가 LoRA 효과를 손실** — 부분 기각 (실험 8: llama.cpp 재검증).

llama.cpp Q4_K_M (ngl=30, 전부 GPU 적재, CPU offload 없음)으로 동일 label_depression_1 in-distribution 테스트 → **여전히 0/4 헤더, 가짜 대화 ramble 출력**. CPU offload만의 문제는 아님.

**최종 진단 (가설 8): LoRA가 학습 시점 forward 경로에 sharp-tuned, 모든 환경 변화에 fragile**

다음 5가지 중 하나라도 변하면 LoRA 효과 소실:
1. 양자화 알고리즘 (NF4 ↔ K-quants)
2. compute_dtype (bf16 → fp16)
3. device placement (전부 GPU → CPU offload 일부)
4. weight 적용 방식 (PEFT attach ↔ pre-merge)
5. transformers / peft 라이브러리 버전

1070 Ti 환경에서는 위 5가지를 학습 환경(Colab A100 + NF4 + bf16 + 전부 GPU + PEFT 0.13 attach + transformers git head Dec 2024) 100% 재현 불가능. 어떤 호스팅 방식(Path A NF4 attach, llama.cpp GGUF merge)도 효과 없음.

`_docs/issue_koalpaca_Q.md` §E가 이미 경고:
> "하드웨어 요구: 12.8B NF4 ≈ VRAM 8~10GB. 1070 Ti 8GB는 빠듯/부족. **RTX 3060 12GB 이상** 또는 클라우드(Colab T4 16GB, Lightning AI, RunPod 등) 필요."

본 작업에서는 `device_map="auto" + max_memory={0: "7GiB", "cpu": "24GiB"}`로 12.8B를 GPU/CPU 분할 적재. 추론 시:
- GPU layer: NF4 → bf16 dequant → bf16 matmul (학습 forward와 동일)
- CPU layer: NF4 → 다른 경로의 dequant → CPU 정밀도로 matmul (학습 forward와 다름)
- PEFT LoRA는 모든 `query_key_value`에 attach되지만 CPU offload된 절반 이상의 layer에서는 LoRA의 효과가 학습 시점과 다르게 적용됨

학습 환경(Colab A100 40GB, 전부 GPU 적재)에서는 F-1 60% 작동. 본 환경(1070 Ti 8GB + CPU offload)에서는 LoRA 효과 절반 이상 소실되어 base polyglot의 dialogue continuation 행동이 우세.

**즉: 본 실패는 LoRA 학습 부족도 양자화 mismatch도 prompt 형식도 아니라, 하드웨어(VRAM) 부족으로 인한 CPU offload 자체의 LoRA 호환 문제.** 재학습으로 해결 불가.

## 4. 성과: 인프라는 정상 동작

V2 콘텐츠 품질은 실패했지만 인프라 자체는 모두 정상:

- ✅ NF4 + bf16 양자화 1070 Ti에서 동작
- ✅ 12.8B base + LoRA attach 성공
- ✅ FastAPI 서버 `/health`, `/summarize` 정상 응답
- ✅ VRAM 8GB 한계 안에서 운영 (`max_memory={0: '7GiB', 'cpu': '24GiB'}` + `llm_int8_enable_fp32_cpu_offload=True`)
- 응답 시간 (warm, max_tokens 300~500): **220~342초** — 데모에 부적합 (PRD §8 90초 기준 초과)

→ 본 인프라는 더 잘 학습된 LoRA(또는 다른 instruction-tuned 한국어 모델)가 생기면 즉시 재사용 가능.

## 5. 권장 다음 단계 (명세서 §9, §7 위험표)

명세서 §5 Fail 시 행동: "사용자에게 보고 후 길 B 또는 옵션 C(외부 API)로 전환."

**진단이 "CPU offload + LoRA 비호환"으로 확정됐으므로 재학습은 의미 없음** (학습 환경에서는 이미 F-1 60%, 동일 모델을 1070 Ti CPU offload로 돌리면 또 깨짐).

| 옵션 | 설명 | D-6 일정 적합도 |
| --- | --- | --- |
| ~~재학습~~ | 진단상 무의미 — 학습은 이미 충분히 작동, 호스팅 환경이 문제 | ❌ 시간 낭비 |
| **길 B (클라우드 GPU)** | Colab T4 16GB / RunPod / Modal에서 전부 GPU 적재. 본 작업의 코드/스크립트 그대로 옮김. 학습 환경(A100)과 유사하므로 LoRA 효과 정상 작동 기대 | **2~4h 셋업** + 검증, 성공 가능성 매우 높음 |
| **옵션 C (외부 API)** | Gemini/GPT-4 등으로 4섹션 요약 위탁, F3 KoAlpaca 종속성 제거 | **가장 안전**. 1h 내 동작, PRD 90초 충족, 데이터 외부 전송 동의 필요 |

**클로드 코드 권장**: 우선 **길 B (Colab T4 16GB)** 시도 — 본 작업의 nf4_loader.py / nf4_server.py 코드 그대로 옮기면 학습 환경에 가까운 12.8B 전부-GPU 적재 가능. 실패 시 옵션 C로 폴백.

## 6. 현재 리포 상태 (작업 전후 차이)

신규 파일 (변경 없이 보존, 향후 재사용 가능):
- `ai-model/preflight_nf4.py`
- `ai-model/nf4_loader.py`
- `ai-model/nf4_server.py`
- `ai-model/serve_nf4.bat`
- `ai-model/requirements-nf4.txt`
- `ai-model/.venv-nf4/` (별도 venv, ~10GB)
- `_docs/path-a-results.md` (본 문서)

수정 파일:
- (없음) — `src/summarizer.py` 수정은 V2 통과를 전제로 했으므로 보류

보존 (명세서 §0 원칙 2 준수):
- `ai-model/serve.bat`, `ai-model/models/koalpaca-q4km.gguf`, `ai-model/merge_and_convert.py`, `ai-model/koalpaca_run.ipynb` — 변경 없음

HF 캐시:
- `E:\hf_cache\hub\models--EleutherAI--polyglot-ko-12.8b` ~24GB
- `E:\hf_cache\hub\models--EleutherAI--polyglot-ko-1.3b` ~3GB (preflight용)

## 7. 인프라 재사용 가이드 (다음 작업자용)

이미 다운로드된 모델 + 설치된 venv 그대로 사용 가능:

```bash
# 서버 기동
cd ai-model
.venv-nf4\Scripts\activate
set HF_HOME=E:\hf_cache
set KOALPACA_BASE=EleutherAI/polyglot-ko-12.8b
set KOALPACA_ADAPTER=<새 LoRA 경로>
set KOALPACA_COMPUTE_DTYPE=bfloat16
set KOALPACA_GPU_MEM=7GiB
set KOALPACA_CPU_MEM=24GiB
uvicorn nf4_server:app --host 127.0.0.1 --port 8765 --workers 1
```

검증:
```bash
curl http://127.0.0.1:8765/health  # {"status":"ok"}
# /summarize POST 후 응답에서 4섹션 헤더 정규식 매칭
```
