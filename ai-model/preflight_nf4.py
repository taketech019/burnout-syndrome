"""preflight_nf4.py — 1070 Ti (Pascal CC 6.1)에서 NF4 + bf16/fp16 동작 검증.

_docs/nf4-hosting.md §0-3 그대로. 본 작업(NF4 + LoRA attach) 진입 전 게이트.
"""
import sys
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

PROBE_MODEL = "EleutherAI/polyglot-ko-1.3b"  # 작은 모델로 빠르게 호환성만 확인


def _try_load(compute_dtype, label):
    print(f"\n--- {label} (compute_dtype={compute_dtype}) ---")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )
    try:
        model = AutoModelForCausalLM.from_pretrained(
            PROBE_MODEL,
            quantization_config=bnb_config,
            device_map="auto",
        )
        x = torch.tensor([[1, 2, 3]]).to(model.device)
        with torch.no_grad():
            _ = model(x)
        print(f"PREFLIGHT_{label}: PASS")
        return True
    except Exception as e:
        print(f"PREFLIGHT_{label}: FAIL — {type(e).__name__}: {e}")
        return False


def main():
    print("=== Step 0-1: Environment ===")
    print(f"CUDA available : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device name    : {torch.cuda.get_device_name(0)}")
        print(f"Compute cap.   : {torch.cuda.get_device_capability(0)}")
    else:
        print("CUDA not available — preflight cannot proceed.")
        sys.exit(2)

    print("\n=== Step 0-3: NF4 load test (probe model: polyglot-ko-1.3b) ===")
    if _try_load(torch.bfloat16, "NF4_BF16"):
        print("\n=> 학습 환경(bf16) 100% 재현 가능. nf4_loader.py에서 compute_dtype=bfloat16 사용.")
        sys.exit(0)

    if _try_load(torch.float16, "NF4_FP16"):
        print("\n=> bf16 미지원, fp16으로 NF4 가능. compute_dtype=float16 사용.")
        print("   주의: 학습 forward와 미세하게 다를 수 있어 V2에서 부분 통과 가능성.")
        sys.exit(1)

    print("\n=> NF4 자체가 1070 Ti에서 동작하지 않음. 길 A 불가능.")
    print("   사용자에게 보고하고 길 B(클라우드) 전환 필요.")
    sys.exit(3)


if __name__ == "__main__":
    main()
