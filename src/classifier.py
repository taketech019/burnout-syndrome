"""src/classifier.py — F1 1단계: 우울/불안/중독 3-binary 판별.

⚠️ 로컬 KlueBERT + HF Space 양쪽 모두 regressor 변별력 0 진단됨 (vibe-coding-report.md).
기본 백엔드는 **Gemma 4 31B 0~3 정도값**. KlueBERT는 환경변수로 보조.

신규 시그니처: classify_text(script: str) -> dict
"""
import logging
import os
from typing import Optional

import requests

from config import (
    KLUEBERT_API_KEY,
    KLUEBERT_ENDPOINT_URL,
    KLUEBERT_LOCAL_DIR,
)

log = logging.getLogger(__name__)

_LABELS = ("depression", "anxiety", "addiction")


def _empty_result(message: str, ok: bool = False, backend: str = "none") -> dict:
    return {
        "ok": ok,
        "status": "error" if not ok else "success",
        "message": message,
        "backend": backend,
        "classification": {k: 0 for k in _LABELS},
        "scores": {k: 0.0 for k in _LABELS},
        "raw_scores": {k: 0.0 for k in _LABELS},
    }


# ── 로컬 KlueBERT (선택, 기본 OFF) ─────────────────────────────────────────────

_LOCAL_MODELS: dict = {}
_LOCAL_TOKENIZERS: dict = {}
_LOCAL_AVAILABLE: Optional[bool] = None


def _try_init_local() -> bool:
    global _LOCAL_AVAILABLE
    if _LOCAL_AVAILABLE is not None:
        return _LOCAL_AVAILABLE
    try:
        import torch  # noqa: F401
        import torch.nn as nn
        from transformers import BertForSequenceClassification, BertTokenizer

        class CustomBertForSequenceRegression(BertForSequenceClassification):  # type: ignore[misc]
            def __init__(self, config):
                super().__init__(config)
                self.num_labels = 1
                self.regressor = nn.Linear(config.hidden_size, self.num_labels)

            def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, **kwargs):
                outputs = self.bert(input_ids, attention_mask=attention_mask,
                                    token_type_ids=token_type_ids)
                pooled = outputs[0][:, 0, :]
                return self.regressor(pooled)

        for label in _LABELS:
            d = KLUEBERT_LOCAL_DIR / f"trained_model_kluebert_{label}"
            if not d.exists():
                _LOCAL_AVAILABLE = False
                return False
            _LOCAL_TOKENIZERS[label] = BertTokenizer.from_pretrained(str(d))
            m = CustomBertForSequenceRegression.from_pretrained(str(d))
            m.eval()
            _LOCAL_MODELS[label] = m
        _LOCAL_AVAILABLE = True
        return True
    except Exception as e:
        log.warning("로컬 KlueBERT skip: %s", e)
        _LOCAL_AVAILABLE = False
        return False


def _predict_local(script: str) -> dict:
    import torch
    raw, scores, classification = {}, {}, {}
    for label in _LABELS:
        tok = _LOCAL_TOKENIZERS[label]
        m = _LOCAL_MODELS[label]
        enc = tok(script, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            logits = m(input_ids=enc["input_ids"],
                       attention_mask=enc["attention_mask"],
                       token_type_ids=enc.get("token_type_ids"))
        v = float(logits.squeeze().item())
        clipped = min(max(round(v), 0), 3)
        raw[label] = round(v, 3)
        scores[label] = float(clipped)
        classification[label] = 0 if clipped == 0 else 1
    return {
        "ok": True,
        "status": "success",
        "message": "로컬 KlueBERT 추론 (regressor 변별력 부족 가능)",
        "backend": "kluebert_local",
        "classification": classification,
        "scores": scores,
        "raw_scores": raw,
    }


# ── HF Space ───────────────────────────────────────────────────────────────────


def _call_kluebert_space(script: str) -> dict:
    if not KLUEBERT_ENDPOINT_URL:
        return _empty_result("KLUEBERT_ENDPOINT_URL 미설정", backend="kluebert_hf")
    url = KLUEBERT_ENDPOINT_URL.rstrip("/") + "/predict"
    try:
        resp = requests.post(
            url, json={"text": script},
            headers={"X-API-Key": KLUEBERT_API_KEY, "Content-Type": "application/json"},
            timeout=90,
        )
        resp.raise_for_status()
        data = resp.json()
        classification = {k: int(data.get(k, 0)) for k in _LABELS}
        return {
            "ok": True,
            "status": "success",
            "message": "HF Space /predict 응답",
            "backend": "kluebert_hf",
            "classification": classification,
            "scores": {k: float(classification[k]) for k in _LABELS},
            "raw_scores": {k: float(data.get(k, 0)) for k in _LABELS},
        }
    except Exception as e:
        return _empty_result(f"HF Space 호출 실패: {e}", backend="kluebert_hf")


# ── Gemma 1차 (기본) ───────────────────────────────────────────────────────────

_GEMMA_PROMPT = """당신은 임상심리 전문가입니다. 다음 상담 텍스트를 보고 우울/불안/중독 세 가지 척도로 **0~3 정수** 정도값을 매깁니다.

**점수 가이드**
- 0: 해당 증상 없음 (정상군)
- 1: 약함 (간헐적 언급)
- 2: 중간 (반복 언급)
- 3: 심함 (자살 사고·반복 폭음 등 핵심 증상)

**출력 형식**: 단일 JSON 객체로만 답하세요. 설명·코드 블록 금지.
{"depression": 0~3, "anxiety": 0~3, "addiction": 0~3}

**상담 텍스트**
{text}
"""


def _predict_gemma(script: str) -> dict:
    from src.gemma_client import generate_json
    try:
        data = generate_json(
            _GEMMA_PROMPT.replace("{text}", script[:8000]),
            temperature=0.0, max_output_tokens=512,
        )
    except Exception as e:
        return _empty_result(f"Gemma 1차 호출 실패: {e}", backend="gemma_fallback")
    if not isinstance(data, dict):
        return _empty_result("Gemma 1차 JSON 파싱 실패", backend="gemma_fallback")
    scores, classification = {}, {}
    for k in _LABELS:
        try:
            v = int(data.get(k, 0))
        except (TypeError, ValueError):
            v = 0
        v = max(0, min(3, v))
        scores[k] = float(v)
        classification[k] = 1 if v >= 1 else 0
    return {
        "ok": True,
        "status": "success",
        "message": "Gemma 4 31B 0~3 정도값",
        "backend": "gemma_fallback",
        "classification": classification,
        "scores": scores,
        "raw_scores": dict(scores),
    }


# ── 공개 진입점 ────────────────────────────────────────────────────────────────


def classify_text(script: str) -> dict:
    """F1 1단계 판별 — 통합 진입점.

    백엔드 선택 (CLASSIFIER_BACKEND 환경변수):
      - 'gemma' (기본): Gemma 4 31B 1차
      - 'kluebert_local': 로컬 weights
      - 'kluebert_hf': HF Space
    실패 시 Gemma 폴백.

    transcript 끝의 척도 평가(PHQ-9 등) 영역은 분리되어 모델 입력에서 제외.
    """
    if not script or not script.strip():
        return _empty_result("입력 텍스트 비어 있음")

    from src.transcript_utils import split_transcript_and_scale
    script, _scale = split_transcript_and_scale(script)
    if not script.strip():
        return _empty_result("입력 텍스트 비어 있음")

    backend = os.getenv("CLASSIFIER_BACKEND", "gemma").lower()

    if backend == "kluebert_local" and _try_init_local():
        try:
            return _predict_local(script)
        except Exception as e:
            log.warning("로컬 추론 실패 → Gemma 폴백: %s", e)

    if backend == "kluebert_hf":
        r = _call_kluebert_space(script)
        if r["ok"]:
            return r

    return _predict_gemma(script)


# ── 기존 호출 호환 (구 src.classifier.classify) ───────────────────────────────


def classify(script: str) -> dict:
    """기존 호출 호환 — classify_text 결과를 옛 시그니처에 맞춰 변환."""
    r = classify_text(script)
    cls = r["classification"]
    out = {
        "anxiety": cls.get("anxiety", 0),
        "depression": cls.get("depression", 0),
        "addiction": cls.get("addiction", 0),
        "is_normal": all(v == 0 for v in cls.values()),
        "_source": r["backend"],
        "level": {k: int(r["scores"].get(k, 0)) for k in _LABELS},
        "raw": r["raw_scores"],
    }
    if not r["ok"]:
        out["error"] = r["message"]
    return out
