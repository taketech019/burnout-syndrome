"""CounsHelper KlueBERT 추론 Space — FastAPI /predict + Gradio /ui 디버그.

PII 보호: 입력 텍스트는 절대 stdout/stderr/로그에 기록하지 않음. 길이와 결과 binary만 기록.
"""
import logging
import os

import gradio as gr
import torch
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from transformers import BertTokenizer

from modeling_kluebert import CustomBertForSequenceRegression, closest_integer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("kluebert-space")

HF_USER = "Yun-choice"
LABELS = ("anxiety", "depression", "addiction")
REPO_IDS = {label: f"{HF_USER}/kluebert-{label}" for label in LABELS}

SPACE_SECRET = os.environ.get("HF_SPACE_API_KEY", "")
if not SPACE_SECRET:
    log.warning("HF_SPACE_API_KEY Space Secret 미설정 — /predict는 401만 반환")

MODELS: dict[str, CustomBertForSequenceRegression] = {}
TOKENIZERS: dict[str, BertTokenizer] = {}


def _load_all() -> None:
    log.info("모델 로드 시작 (CPU, fp32, 3 모델)")
    for label, repo_id in REPO_IDS.items():
        log.info("로드 중: %s", repo_id)
        tokenizer = BertTokenizer.from_pretrained(repo_id)
        model = CustomBertForSequenceRegression.from_pretrained(repo_id)
        model.eval()
        TOKENIZERS[label] = tokenizer
        MODELS[label] = model
    log.info("모델 로드 완료: %s", list(MODELS.keys()))


_load_all()


def _predict_one(text: str, label: str) -> tuple[int, float]:
    tokenizer = TOKENIZERS[label]
    model = MODELS[label]
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )
    with torch.no_grad():
        logits = model(**inputs)
    raw = float(logits.squeeze().item())
    clipped = closest_integer(raw)
    binary = 0 if clipped == 0 else 1
    return binary, raw


def _predict_all(text: str) -> dict[str, int]:
    return {label: _predict_one(text, label)[0] for label in LABELS}


def _predict_all_debug(text: str) -> tuple[dict[str, int], dict[str, float]]:
    binary: dict[str, int] = {}
    raw: dict[str, float] = {}
    for label in LABELS:
        b, r = _predict_one(text, label)
        binary[label] = b
        raw[label] = r
    return binary, raw


api = FastAPI(title="KlueBERT CounsHelper")


class PredictIn(BaseModel):
    text: str


@api.post("/predict")
def predict(body: PredictIn, x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> dict[str, int]:
    if not SPACE_SECRET or x_api_key != SPACE_SECRET:
        raise HTTPException(status_code=401, detail="unauthorized")
    result = _predict_all(body.text)
    log.info("predict ok len=%d result=%s", len(body.text), result)
    return result


@api.get("/healthz")
def healthz() -> dict[str, str | list[str]]:
    return {"status": "ok", "models": list(MODELS.keys())}


with gr.Blocks(title="KlueBERT CounsHelper (debug)") as demo:
    gr.Markdown(
        "## KlueBERT CounsHelper — 디버그 UI\n"
        "FastAPI 엔드포인트: `POST /predict` (X-API-Key 헤더 필요).\n"
        "이 UI는 인증 없이 로컬 추론만 수행 — **실제 환자 식별정보 입력 금지**."
    )
    inp = gr.Textbox(label="상담 텍스트", lines=8, placeholder="상담자: ...\n내담자: ...")
    btn = gr.Button("판별")
    out_bin = gr.JSON(label="이진 결과 (API 응답과 동일)")
    out_raw = gr.JSON(label="원시 회귀값 (디버그용)")
    btn.click(fn=_predict_all_debug, inputs=inp, outputs=[out_bin, out_raw])

app = gr.mount_gradio_app(api, demo, path="/ui")
