"""KoAlpaca NF4 + LoRA attach Modal 배포.

학습 환경(A100 + NF4 + bf16 + 전부 GPU + PEFT attach) 재현.
1070 Ti 로컬에서 실패한 환경 fragility를 A10G 24GB로 해결.

배포:
    modal deploy ai-model/nf4_modal.py

엔드포인트:
    POST https://<workspace>--koalpaca-nf4-Inference-summarize.modal.run
    헤더: X-API-Key: <KOALPACA_API_KEY>
    본문: {"transcript": "...", "max_new_tokens": 1024, "temperature": 0.0}

비용:
    A10G $0.000306/sec ≈ $1.1/시간. container_idle_timeout=120s.
    심사 데모 3시간/2주 ≈ $3.3 (Free $30 크레딧 안).

Secret:
    `modal secret create koalpaca-secret KOALPACA_API_KEY=<랜덤 키>`
"""
from pathlib import Path

import modal

ADAPTER_LOCAL = Path(__file__).parent / "koalpaca_save"

# 컨테이너 이미지: PyTorch CUDA 12.1 + NF4 의존성 + LoRA 어댑터 포함
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "transformers>=4.40.0,<5.0",
        "peft>=0.14.0,<0.15",
        "bitsandbytes>=0.43.0",
        "accelerate>=0.30.0",
        "fastapi>=0.110.0",
        "pydantic>=2.0",
        "sentencepiece",
        "protobuf",
    )
    .add_local_dir(str(ADAPTER_LOCAL), remote_path="/adapter")
)

# HuggingFace 캐시 영구화 — 첫 cold start만 base 모델(24GB) 다운, 이후 재활용
hf_cache = modal.Volume.from_name("koalpaca-hf-cache", create_if_missing=True)

app = modal.App("koalpaca-nf4")


@app.cls(
    image=image,
    gpu="A10G",
    volumes={"/cache": hf_cache},
    timeout=1800,
    scaledown_window=120,
    secrets=[modal.Secret.from_name("koalpaca-secret")],
)
class Inference:
    @modal.enter()
    def load(self):
        """컨테이너 첫 진입 시 1회 모델 로드 (~30~60초)."""
        import os
        os.environ["HF_HOME"] = "/cache/hf"
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        print("[modal] Loading base EleutherAI/polyglot-ko-12.8b (NF4 + bf16, all GPU)...")
        base = AutoModelForCausalLM.from_pretrained(
            "EleutherAI/polyglot-ko-12.8b",
            quantization_config=bnb_config,
            device_map={"": 0},  # 전부 GPU — A10G 24GB라 가능
            low_cpu_mem_usage=True,
        )

        print("[modal] Attaching LoRA adapter from /adapter...")
        model = PeftModel.from_pretrained(base, "/adapter")
        model.eval()

        # 🔑 tokenizer는 저장된 adapter 폴더에서 로드 (베이스 모델과 vocab/특수토큰 다를 수 있음).
        # 원본 추론 노트북도 AutoTokenizer.from_pretrained(output_dir)로 어댑터 폴더 사용.
        # 어댑터 tokenizer에 <|sep|>, <|acc|>, <|tel|>, <|rrn|> 같은 special token이 있음.
        tokenizer = AutoTokenizer.from_pretrained("/adapter")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        self.model = model
        self.tokenizer = tokenizer
        print("[modal] Ready.")

    def _generate(
        self,
        transcript: str,
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
        repetition_penalty: float = 1.2,
        no_repeat_ngram_size: int = 3,
    ) -> str:
        import torch

        instruction = "다음과 같은 상담기록을 보고 요약서를 작성해주세요."
        prompt = f"### 명령어: {instruction}\n\n### 맥락: {transcript}\n\n### 답변:"

        inputs = self.tokenizer(
            prompt, return_tensors="pt", return_token_type_ids=False
        ).to(self.model.device)

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=(temperature > 0),
                temperature=temperature if temperature > 0 else 1.0,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                early_stopping=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=None,  # 원본 노트북 일치 — 조기 종료 방지
            )

        full = self.tokenizer.decode(out[0])
        if "답변: " in full:
            answer = full.split("답변: ", 1)[1]
        elif "답변:" in full:
            answer = full.split("답변:", 1)[1]
        else:
            answer = full[len(prompt):]
        for marker in ["<|sep|>", "<|endoftext|>", "꿋"]:
            if marker in answer:
                answer = answer.split(marker, 1)[0]
        return answer.strip()

    @modal.fastapi_endpoint(method="GET")
    def health(self):
        return {"status": "ok"}

    @modal.fastapi_endpoint(method="POST")
    def summarize(self, request_data: dict):
        """body 안에 api_key 포함해서 호출.

        예: {"api_key": "...", "transcript": "...", "max_new_tokens": 1024}

        헤더 기반 인증(Authorization 등)은 Modal fastapi_endpoint에서 self와 함께
        Header() Depends를 쓰면 deploy-time 시그니처 직렬화 이슈가 있어 body 방식 사용.
        """
        import os
        from fastapi import HTTPException

        expected = os.environ.get("KOALPACA_API_KEY", "")
        client_key = (request_data.get("api_key") or "").strip()
        if not expected or client_key != expected:
            raise HTTPException(401, "invalid or missing api_key in body")

        transcript = (request_data.get("transcript") or "").strip()
        if not transcript:
            raise HTTPException(400, "empty transcript")

        try:
            text = self._generate(
                transcript=transcript,
                max_new_tokens=int(request_data.get("max_new_tokens", 1024)),
                temperature=float(request_data.get("temperature", 0.0)),
                repetition_penalty=float(request_data.get("repetition_penalty", 1.2)),
                no_repeat_ngram_size=int(request_data.get("no_repeat_ngram_size", 3)),
            )
        except Exception as e:
            raise HTTPException(500, f"generation failed: {type(e).__name__}: {e}")

        return {"text": text}
