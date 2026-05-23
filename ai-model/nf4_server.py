"""KoAlpaca NF4 + LoRA attach 추론 서버. src/summarizer.py가 HTTP로 호출."""
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from nf4_loader import build_prompt, generate, load


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[server] Warming up model...")
    load()
    print("[server] Ready to serve.")
    yield


app = FastAPI(lifespan=lifespan)


class SummarizeRequest(BaseModel):
    transcript: str
    max_new_tokens: int = 500
    temperature: float = 0.0
    repetition_penalty: float = 1.2
    no_repeat_ngram_size: int = 3


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
        text = generate(
            prompt,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            repetition_penalty=req.repetition_penalty,
            no_repeat_ngram_size=req.no_repeat_ngram_size,
        )
    except Exception as e:
        raise HTTPException(500, f"generation failed: {type(e).__name__}: {e}")
    return SummarizeResponse(text=text)


class RawGenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 500
    temperature: float = 0.0
    repetition_penalty: float = 1.2
    no_repeat_ngram_size: int = 3


@app.post("/raw_generate", response_model=SummarizeResponse)
def raw_generate(req: RawGenerateRequest):
    """디버깅용: build_prompt 우회하고 임의 prompt 직접 입력."""
    if not req.prompt.strip():
        raise HTTPException(400, "empty prompt")
    try:
        text = generate(
            req.prompt,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            repetition_penalty=req.repetition_penalty,
            no_repeat_ngram_size=req.no_repeat_ngram_size,
        )
    except Exception as e:
        raise HTTPException(500, f"generation failed: {type(e).__name__}: {e}")
    return SummarizeResponse(text=text)
