"""src/classifier.py

KlueBERT 기반 우울/불안/중독 분류 모듈.

현재 모델 구조:
- depression, anxiety, addiction 모델이 각각 따로 존재한다.
- 각 모델은 0~3 회귀 점수를 출력한다.
- 서비스에서도 원본 모델과 동일하게 0~3 점수를 유지한다.

Hugging Face model repo 예:
- dkslanjrkehlsmsrjdi/trained_model_kluebert_depression
- dkslanjrkehlsmsrjdi/trained_model_kluebert_anxiety
- dkslanjrkehlsmsrjdi/trained_model_kluebert_addiction
"""

from __future__ import annotations

import os
import gc
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification


class CustomBertForSequenceRegression(BertForSequenceClassification):
    """
    kluebert_train.ipynb / kluebert_run.ipynb에서 사용한 커스텀 회귀 모델 구조.

    주의:
    - 일반 BertForSequenceClassification의 classifier head를 쓰지 않고,
      regressor head를 새로 사용한다.
    - 따라서 저장된 model.safetensors를 이 클래스 구조로 불러와야 한다.
    """

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = 1
        self.regressor = nn.Linear(config.hidden_size, self.num_labels)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        pooled_output = sequence_output[:, 0, :]
        logits = self.regressor(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.MSELoss()
            loss = loss_fct(logits, labels)

        return (loss, logits) if loss is not None else logits


def _get_secret(key: str, default: str = "") -> str:
    """
    Streamlit Cloud에서는 st.secrets를 우선 사용하고,
    로컬 실행에서는 환경변수를 사용한다.
    """
    try:
        import streamlit as st

        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass

    return os.getenv(key, default)


def get_classifier_config() -> Dict[str, str]:
    """
    KlueBERT 모델 repo/path 설정을 반환한다.
    """
    return {
        "hf_token": _get_secret("HF_TOKEN", "").strip(),
        "depression_model": _get_secret("KLUEBERT_DEPRESSION_MODEL", "").strip(),
        "anxiety_model": _get_secret("KLUEBERT_ANXIETY_MODEL", "").strip(),
        "addiction_model": _get_secret("KLUEBERT_ADDICTION_MODEL", "").strip(),
    }


def _clip_round_score(value: float) -> int:
    """
    회귀 출력값을 0~3 정수로 변환한다.
    """
    rounded = int(round(float(value)))

    if rounded < 0:
        return 0
    if rounded > 3:
        return 3

    return rounded


def _score_to_binary(score: int) -> int:
    """
    기존 호환용 함수.
    더 이상 0/1 판별로 사용하지 않고, 0~3 점수를 그대로 반환한다.
    """
    return score


def _load_model_and_tokenizer(
    model_id: str,
    hf_token: Optional[str],
) -> Tuple[CustomBertForSequenceRegression, BertTokenizer]:
    """
    Hugging Face repo 또는 로컬 경로에서 모델과 tokenizer를 불러온다.
    """
    kwargs = {}

    if hf_token:
        kwargs["token"] = hf_token

    tokenizer = BertTokenizer.from_pretrained(model_id, **kwargs)
    model = CustomBertForSequenceRegression.from_pretrained(model_id, **kwargs)
    model.eval()

    return model, tokenizer


def _predict_single_score(
    text: str,
    model_id: str,
    hf_token: Optional[str],
    max_length: int = 512,
) -> Dict[str, Any]:
    """
    단일 질환 모델에 대해 회귀 원점수와 0~3 점수를 반환한다.

    메모리 절약을 위해 모델을 사용한 뒤 즉시 삭제한다.
    """
    if not model_id:
        return {
            "ok": False,
            "status": "not_configured",
            "message": "모델 ID가 설정되지 않았습니다.",
            "raw_score": None,
            "score": 0,
            "label": 0,
        }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = None
    tokenizer = None

    try:
        model, tokenizer = _load_model_and_tokenizer(model_id, hf_token)
        model.to(device)

        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            output = model(**inputs)

        if isinstance(output, tuple):
            logits = output[-1]
        else:
            logits = output

        raw_score = float(logits.squeeze().detach().cpu().item())
        score = _clip_round_score(raw_score)
        label = _score_to_binary(score)

        return {
            "ok": True,
            "status": "success",
            "message": "KlueBERT 예측 성공",
            "raw_score": raw_score,
            "score": score,
            "label": label,
        }

    except Exception as error:
        return {
            "ok": False,
            "status": "error",
            "message": f"KlueBERT 예측 중 오류가 발생했습니다: {error}",
            "raw_score": None,
            "score": 0,
            "label": 0,
        }

    finally:
        try:
            del model
            del tokenizer
        except Exception:
            pass

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def classify_text(text: str) -> Dict[str, Any]:
    """
    depression/anxiety/addiction 모델 3개를 순차 실행한다.

    반환 형식:
    {
        "ok": bool,
        "status": str,
        "message": str,
        "classification": {
            "depression": 0~3,
            "anxiety": 0~3,
            "addiction": 0~3
        },
        "scores": {
            "depression": 0~3,
            "anxiety": 0~3,
            "addiction": 0~3
        },
        "raw_scores": {...},
        "details": {...},
        "backend": "kluebert_hf"
    }
    """
    config = get_classifier_config()

    hf_token = config["hf_token"]

    model_map = {
        "depression": config["depression_model"],
        "anxiety": config["anxiety_model"],
        "addiction": config["addiction_model"],
    }

    details: Dict[str, Any] = {}
    classification: Dict[str, int] = {}
    scores: Dict[str, int] = {}
    raw_scores: Dict[str, Any] = {}

    for key, model_id in model_map.items():
        result = _predict_single_score(
            text=text,
            model_id=model_id,
            hf_token=hf_token,
        )

        details[key] = result
        classification[key] = int(result.get("score", 0))
        scores[key] = int(result.get("score", 0))
        raw_scores[key] = result.get("raw_score")

    all_success = all(item.get("ok") for item in details.values())

    if all_success:
        status = "success"
        message = "KlueBERT 3개 모델 예측 성공"
    else:
        status = "partial_error"
        message = "KlueBERT 모델 중 일부 예측에 실패했습니다."

    return {
        "ok": all_success,
        "status": status,
        "message": message,
        "classification": classification,
        "scores": scores,
        "raw_scores": raw_scores,
        "details": details,
        "backend": "kluebert_hf",
    }
"""src/classifier.py — F1 1단계: KlueBERT 3-binary 판별 (HF Spaces /predict 호출).

⚠️ 현재 상태: KlueBERT 우회 모드 (BYPASS)
배포된 KlueBERT 모델이 입력과 무관하게 동일한 회귀값을 출력 (변별력 0,
2026-05-23 검증). 모델 재학습 전까지 1차 판별을 우회하고 모든 입력을 양성으로
간주하여 2차 Gemini 28요인 분석이 실제 분류 수행.

원래 Space 호출 코드는 아래 _call_kluebert_space()로 보존 — 모델 재학습 후
classify() 본문을 거기로 교체하면 1차 판별 복원.
"""
import requests

from config import KLUEBERT_ENDPOINT_URL, KLUEBERT_API_KEY

_LABELS = ("anxiety", "depression", "addiction")
_TIMEOUT = 90

# F1 1차 우회 여부. False로 바꾸면 다시 KlueBERT Space 호출.
KLUEBERT_BYPASS = True


def _empty(extra: dict | None = None) -> dict:
    out = {k: 0 for k in _LABELS} | {"is_normal": True}
    if extra:
        out.update(extra)
    return out


def classify(text: str) -> dict:
    """F1 1단계: 정신질환 3-binary 판별. 반환: {anxiety, depression, addiction, is_normal}.

    BYPASS 모드: 모든 입력 양성 (is_normal=False) → 2차 Gemini 28요인이 실제 판별.
    NORMAL 모드(KLUEBERT_BYPASS=False): KlueBERT Space 호출 결과 사용.
    """
    if KLUEBERT_BYPASS:
        return {
            "anxiety": 1,
            "depression": 1,
            "addiction": 1,
            "is_normal": False,
            "_note": "F1 stage 1 bypassed — KlueBERT 모델 변별력 부족, 2차 Gemini가 실제 분류",
        }
    return _call_kluebert_space(text)


def _call_kluebert_space(text: str) -> dict:
    """KlueBERT HF Space /predict 호출 (모델 재학습 후 사용)."""
    if not KLUEBERT_ENDPOINT_URL:
        return _empty({"error": "KLUEBERT_ENDPOINT_URL 미설정"})

    url = KLUEBERT_ENDPOINT_URL.rstrip("/") + "/predict"
    try:
        resp = requests.post(
            url,
            json={"text": text},
            headers={
                "X-API-Key": KLUEBERT_API_KEY,
                "Content-Type": "application/json",
            },
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        result = {k: int(data.get(k, 0)) for k in _LABELS}
        result["is_normal"] = all(v == 0 for v in result.values())
        return result
    except requests.exceptions.ConnectionError:
        return _empty({"error": "KlueBERT Space에 연결할 수 없습니다."})
    except requests.exceptions.Timeout:
        return _empty({"error": f"응답 시간 초과 ({_TIMEOUT}초). Space cold-start 가능성 — 잠시 후 재시도."})
    except Exception as e:
        return _empty({"error": str(e)})
