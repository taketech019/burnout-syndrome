"""src/classifier.py — F1 1단계: 우울/불안/중독 3-binary 판별.

⚠️ HF Space에 배포된 KlueBERT는 입력에 거의 무관하게 동일한 출력을 반환 (regressor
weight norm ≈0.5, std ≈0.018 — 학습 분포에서 거의 평탄). 2026-05-23~24 검증.

이 모듈은 3-tier 폴백:
  1) **로컬 KlueBERT** (`ai-model/kluebert/2.AI학습모델파일/trained_model_kluebert_*`)
     — torch + transformers 가 설치돼 있으면 로컬 추론. 학습된 weights 그대로 사용.
  2) **Gemma 1차 라벨** — torch 없거나 로컬 폴더 부재 시 Gemma 4 31B로 0~3 정도값
     생성 후 binary 변환. 변별력 OK.
  3) **HF Space `/predict`** — 변별력은 부족하지만 마지막 백업. `_call_kluebert_space()`.

PRD §F1 기준 (정상군 분류 + 우울/불안/중독 binary)을 충족하는 최소 변경 셋.
"""
import json
import logging
from typing import Optional

import requests

from config import (
    KLUEBERT_ENDPOINT_URL,
    KLUEBERT_API_KEY,
    KLUEBERT_LOCAL_DIR,
)

log = logging.getLogger(__name__)

_LABELS = ("anxiety", "depression", "addiction")
_TIMEOUT = 90

_LOCAL_MODELS: dict = {}
_LOCAL_TOKENIZERS: dict = {}
_LOCAL_AVAILABLE: Optional[bool] = None


def _empty(extra: Optional[dict] = None) -> dict:
    out = {k: 0 for k in _LABELS} | {"is_normal": True}
    if extra:
        out.update(extra)
    return out


# ── tier 1: 로컬 KlueBERT ─────────────────────────────────────────────────────


def _try_init_local() -> bool:
    """ai-model/kluebert/2.AI학습모델파일/trained_model_kluebert_* 로드.

    torch / transformers 미설치이거나 폴더 부재 시 False — Gemma 폴백으로.
    한 번 성공하면 모듈 전역 캐시에 유지.
    """
    global _LOCAL_AVAILABLE
    if _LOCAL_AVAILABLE is not None:
        return _LOCAL_AVAILABLE

    try:
        import torch  # noqa: F401
        from transformers import BertTokenizer

        # CustomBertForSequenceRegression 정의 — HF Space 와 동일 (modeling_kluebert.py).
        import torch.nn as nn
        from transformers import BertForSequenceClassification

        class CustomBertForSequenceRegression(BertForSequenceClassification):  # type: ignore
            def __init__(self, config):
                super().__init__(config)
                self.num_labels = 1
                self.regressor = nn.Linear(config.hidden_size, self.num_labels)

            def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, **kwargs):
                outputs = self.bert(
                    input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                )
                pooled = outputs[0][:, 0, :]
                return self.regressor(pooled)

        for label in _LABELS:
            model_dir = KLUEBERT_LOCAL_DIR / f"trained_model_kluebert_{label}"
            if not model_dir.exists():
                log.warning("로컬 KlueBERT 폴더 부재: %s", model_dir)
                _LOCAL_AVAILABLE = False
                return False
            tok = BertTokenizer.from_pretrained(str(model_dir))
            model = CustomBertForSequenceRegression.from_pretrained(str(model_dir))
            model.eval()
            _LOCAL_TOKENIZERS[label] = tok
            _LOCAL_MODELS[label] = model
        log.info("로컬 KlueBERT 로드 완료: %s", list(_LOCAL_MODELS.keys()))
        _LOCAL_AVAILABLE = True
        return True
    except ImportError as e:
        log.warning("로컬 KlueBERT skip — torch/transformers 미설치: %s", e)
        _LOCAL_AVAILABLE = False
        return False
    except Exception as e:
        log.warning("로컬 KlueBERT 로드 실패: %s", e)
        _LOCAL_AVAILABLE = False
        return False


def _predict_local(text: str) -> dict:
    """로컬 weights 직접 추론. 각 모델의 raw float + binary + level."""
    import torch

    binary, raw, level = {}, {}, {}
    for label in _LABELS:
        tok = _LOCAL_TOKENIZERS[label]
        m = _LOCAL_MODELS[label]
        enc = tok(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        # CustomBertForSequenceRegression.forward는 token_type_ids를 받지만
        # 일부 토크나이저는 반환 — 안전하게 추출.
        with torch.no_grad():
            logits = m(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                token_type_ids=enc.get("token_type_ids"),
            )
        v = float(logits.squeeze().item())
        clipped = min(max(round(v), 0), 3)
        binary[label] = 0 if clipped == 0 else 1
        raw[label] = round(v, 3)
        level[label] = clipped
    return {
        **binary,
        "is_normal": all(v == 0 for v in binary.values()),
        "raw": raw,
        "level": level,
        "_source": "local_kluebert",
    }


# ── tier 2: Gemma 1차 폴백 ────────────────────────────────────────────────────

_GEMMA_PROMPT = """당신은 임상심리 전문가입니다. 다음 상담 텍스트를 보고 우울/불안/중독 세 가지 척도로 **0~3 정수** 정도값을 매깁니다.

**점수 가이드**
- 0: 해당 증상 없음 (정상군)
- 1: 약함 (간헐적 언급, 일상 기능 정상)
- 2: 중간 (반복 언급, 일부 기능 저하)
- 3: 심함 (자살 사고·반복 폭음 등 핵심 증상 명확)

**출력 형식**: 반드시 아래 형식의 단일 JSON 객체로만 답하세요. 설명·코드블록 금지.
{"depression": 0~3, "anxiety": 0~3, "addiction": 0~3}

**상담 텍스트**
{text}
"""


def _predict_gemma(text: str) -> dict:
    """Gemma 4 31B로 0~3 정도값 생성 후 binary 변환."""
    from src.gemma_client import generate_json

    prompt = _GEMMA_PROMPT.replace("{text}", text[:8000])
    try:
        data = generate_json(prompt, temperature=0.0, max_output_tokens=512)
    except Exception as e:
        return _empty({"error": f"Gemma 1차 호출 실패: {e}"})
    if not isinstance(data, dict):
        return _empty({"error": "Gemma 1차 JSON 파싱 실패", "raw": str(data)[:300]})

    level, binary = {}, {}
    for k in _LABELS:
        try:
            v = int(data.get(k, 0))
        except (TypeError, ValueError):
            v = 0
        v = max(0, min(3, v))
        level[k] = v
        binary[k] = 1 if v >= 1 else 0
    return {
        **binary,
        "is_normal": all(v == 0 for v in binary.values()),
        "level": level,
        "_source": "gemma_1차",
        "_note": "KlueBERT 변별력 부족으로 Gemma 4 31B 0~3 정도값 산출",
    }


# ── tier 3: HF Space 폴백 ─────────────────────────────────────────────────────


def _call_kluebert_space(text: str) -> dict:
    """HF Space /predict — 변별력 부족하지만 마지막 백업."""
    if not KLUEBERT_ENDPOINT_URL:
        return _empty({"error": "KLUEBERT_ENDPOINT_URL 미설정"})
    url = KLUEBERT_ENDPOINT_URL.rstrip("/") + "/predict"
    try:
        resp = requests.post(
            url,
            json={"text": text},
            headers={"X-API-Key": KLUEBERT_API_KEY, "Content-Type": "application/json"},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        result = {k: int(data.get(k, 0)) for k in _LABELS}
        result["is_normal"] = all(v == 0 for v in result.values())
        result["_source"] = "hf_space_kluebert"
        return result
    except Exception as e:
        return _empty({"error": f"HF Space 호출 실패: {e}", "_source": "hf_space_kluebert"})


# ── 공개 진입점 ──────────────────────────────────────────────────────────────


def classify(text: str) -> dict:
    """F1 1단계 판별.

    실측 결과 KlueBERT(로컬·HF Space 모두) regressor가 입력에 거의 무관하게 ~1.2를 반환
    → **변별력 없음**. 따라서 기본은 **Gemma 4 31B 0~3 정도값** 사용.

    `CLASSIFIER_INCLUDE_KLUEBERT=1` 환경변수 켜면 로컬 KlueBERT raw 값도 함께 반환
    (참고·디버깅용). Gemma 실패 시 HF Space 폴백.

    반환: {anxiety, depression, addiction, is_normal, level?, raw?, _source, _note?, error?,
          _kluebert_local?: {raw}}
    """
    import os

    if not text or not text.strip():
        return _empty({"error": "입력 텍스트 비어 있음"})

    gemma_result = _predict_gemma(text)

    if os.getenv("CLASSIFIER_INCLUDE_KLUEBERT") == "1" and _try_init_local():
        try:
            local = _predict_local(text)
            gemma_result["_kluebert_local"] = {"raw": local.get("raw"), "level": local.get("level")}
        except Exception as e:
            log.warning("로컬 KlueBERT 보조 추론 실패: %s", e)

    if "error" not in gemma_result:
        return gemma_result

    log.warning("Gemma 1차 실패 → HF Space 폴백")
    space_result = _call_kluebert_space(text)
    space_result.setdefault(
        "_note", "Gemma 폴백 실패 후 HF Space 사용 (변별력 부족 가능, 항상 1 반환 경향)"
    )
    return space_result
