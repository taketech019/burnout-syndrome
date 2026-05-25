"""KoAlpaca NF4 + LoRA attach 공용 로더. nf4_server.py와 임베드 경로가 공유."""
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

BASE_MODEL_ID = os.getenv("KOALPACA_BASE", "EleutherAI/polyglot-ko-12.8b")
ADAPTER_PATH = os.getenv("KOALPACA_ADAPTER", "")
COMPUTE_DTYPE = os.getenv("KOALPACA_COMPUTE_DTYPE", "bfloat16")
GPU_MEM = os.getenv("KOALPACA_GPU_MEM", "7GiB")
CPU_MEM = os.getenv("KOALPACA_CPU_MEM", "28GiB")

_MODEL = None
_TOKENIZER = None


def load():
    """Lazy load. 프로세스당 1회. 12.8B NF4 + CPU offload 콜드스타트 3~5분."""
    global _MODEL, _TOKENIZER
    if _MODEL is not None:
        return _MODEL, _TOKENIZER

    if not ADAPTER_PATH or not os.path.exists(os.path.join(ADAPTER_PATH, "adapter_config.json")):
        raise RuntimeError(f"LoRA adapter not found at KOALPACA_ADAPTER={ADAPTER_PATH!r}")

    dtype = torch.bfloat16 if COMPUTE_DTYPE == "bfloat16" else torch.float16

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=True,
        llm_int8_enable_fp32_cpu_offload=True,
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


def generate(
    prompt: str,
    max_new_tokens: int = 1024,
    temperature: float = 0.0,
    repetition_penalty: float = 1.2,
    no_repeat_ngram_size: int = 3,
) -> str:
    """추론 노트북(koalpaca_run.ipynb) gen() 정확 재현.

    핵심: eos_token_id=None 으로 모델이 EOS를 못 만들게 강제 → 4섹션 전부 끝까지 생성.
    내부 EOS 예측해도 무시하고 max_new_tokens까지 출력. 학습 시 sections 사이에 EOS
    예측 학습이 됐을 가능성 있어서 이걸 None으로 둬야 첫 섹션에서 안 멈춤.
    """
    model, tokenizer = load()
    inputs = tokenizer(prompt, return_tensors="pt", return_token_type_ids=False).to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0),
            temperature=temperature if temperature > 0 else 1.0,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            early_stopping=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=None,  # ⚠️ 원본 노트북과 일치 — 조기 종료 방지
        )

    # 원본은 skip_special_tokens 안 함. <|sep|>, <|endoftext|>를 stop marker로 사용.
    full = tokenizer.decode(out[0])
    # build_prompt 끝의 "### 답변:" 이후를 잘라 답변만 반환
    if "답변: " in full:
        answer = full.split("답변: ", 1)[1]
    elif "답변:" in full:
        answer = full.split("답변:", 1)[1]
    else:
        answer = full[len(prompt):]
    # 원본 extract_content와 동일하게 stop marker 처리
    for marker in ["<|sep|>", "<|endoftext|>", "꿋"]:
        if marker in answer:
            answer = answer.split(marker, 1)[0]
    return answer.strip()


_INSTRUCTION = "다음과 같은 상담기록을 보고 요약서를 작성해주세요."


def build_prompt(transcript: str) -> str:
    """학습 노트북(koalpaca4bit_train.ipynb cell 11) + 추론 노트북(koalpaca_run.ipynb gen()) 형식.

    학습 텍스트: '### 명령어: {instruction}\\n\\n###맥락: {input}\\n\\n### 답변: {output}<|endoftext|>'
    추론 노트북: '### 명령어: {instruction}\\n\\n### 맥락: {input}\\n\\n### 답변:'
    추론 노트북 형식을 따른다 — instruction 헤더가 빠지면 모델은 대화 이어쓰기로 떨어진다(V2 fail).
    """
    return f"### 명령어: {_INSTRUCTION}\n\n### 맥락: {transcript}\n\n### 답변:"
