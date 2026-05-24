"""src/classifier.py

KlueBERT API 기반 우울/불안/중독 분류 모듈.

현재 구조:
- KlueBERT 모델은 앱 내부에서 직접 로딩하지 않는다.
- 외부 API 서버/Hugging Face Space의 /predict endpoint를 호출한다.
- API 응답은 0/1 binary 결과를 기준으로 사용한다.

필요한 설정:
- KLUEBERT_ENDPOINT_URL
- KLUEBERT_API_KEY

실제 값은 GitHub 코드에 쓰지 않고,
로컬에서는 .streamlit/secrets.toml,
배포에서는 Streamlit Secrets에 넣는다.
"""

from __future__ import annotations

import os
from typing import Any, Dict

import requests


_LABELS = ("depression", "anxiety", "addiction")
_TIMEOUT = 90


def _get_secret(key: str, default: str = "") -> str:
    """
    Streamlit Secrets를 우선 읽고, 없으면 환경변수를 읽는다.
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
    KlueBERT API 설정을 반환한다.
    """
    return {
        "endpoint_url": _get_secret("KLUEBERT_ENDPOINT_URL", "").strip(),
        "api_key": _get_secret("KLUEBERT_API_KEY", "").strip(),
    }


def _empty_result(
    status: str,
    message: str,
    ok: bool = False,
) -> Dict[str, Any]:
    """
    실패 또는 미설정 상태에서 앱이 깨지지 않도록 기본 반환값을 만든다.
    """
    classification = {
        "depression": 0,
        "anxiety": 0,
        "addiction": 0,
    }

    return {
        "ok": ok,
        "status": status,
        "message": message,
        "classification": classification,
        "scores": classification.copy(),
        "raw_scores": {},
        "details": {},
        "backend": "kluebert_api",
    }


def _normalize_api_response(data: Dict[str, Any]) -> Dict[str, int]:
    """
    API 응답을 앱에서 쓰는 depression/anxiety/addiction 구조로 정규화한다.

    허용 예:
    {
        "depression": 1,
        "anxiety": 0,
        "addiction": 1
    }

    또는:
    {
        "result": {
            "depression": 1,
            "anxiety": 0,
            "addiction": 1
        }
    }
    """
    if "result" in data and isinstance(data["result"], dict):
        data = data["result"]

    classification = {}

    for key in _LABELS:
        try:
            classification[key] = int(data.get(key, 0))
        except Exception:
            classification[key] = 0

        if classification[key] < 0:
            classification[key] = 0
        if classification[key] > 1:
            classification[key] = 1

    return classification


def classify_text(text: str) -> Dict[str, Any]:
    """
    KlueBERT API를 호출해 우울/불안/중독 0/1 분류 결과를 반환한다.

    app.py에서 기대하는 반환 형식:
    {
        "ok": bool,
        "status": str,
        "message": str,
        "classification": {
            "depression": 0/1,
            "anxiety": 0/1,
            "addiction": 0/1
        },
        "scores": {...},
        "raw_scores": {...},
        "details": {...},
        "backend": "kluebert_api"
    }
    """
    config = get_classifier_config()
    endpoint_url = config["endpoint_url"].rstrip("/")
    api_key = config["api_key"]

    if not endpoint_url:
        return _empty_result(
            status="not_configured",
            message="KLUEBERT_ENDPOINT_URL이 설정되지 않았습니다.",
        )

    if not api_key:
        return _empty_result(
            status="not_configured",
            message="KLUEBERT_API_KEY가 설정되지 않았습니다.",
        )

    url = endpoint_url + "/predict"

    try:
        response = requests.post(
            url,
            json={"text": text},
            headers={
                "X-API-Key": api_key,
                "Content-Type": "application/json",
            },
            timeout=_TIMEOUT,
        )

        response.raise_for_status()
        data = response.json()

        classification = _normalize_api_response(data)

        return {
            "ok": True,
            "status": "success",
            "message": "KlueBERT API 예측 성공",
            "classification": classification,
            "scores": classification.copy(),
            "raw_scores": data.get("raw_scores", {}),
            "details": data,
            "backend": "kluebert_api",
        }

    except requests.exceptions.ConnectionError:
        return _empty_result(
            status="connection_error",
            message="KlueBERT API 서버에 연결할 수 없습니다.",
        )

    except requests.exceptions.Timeout:
        return _empty_result(
            status="timeout",
            message=f"KlueBERT API 응답 시간이 {_TIMEOUT}초를 초과했습니다.",
        )

    except Exception as error:
        return _empty_result(
            status="error",
            message=f"KlueBERT API 호출 중 오류가 발생했습니다: {error}",
        )


def classify(text: str) -> Dict[str, Any]:
    """
    기존 main 코드 호환용 함수.
    필요한 경우 classify_text()의 classification만 반환한다.
    """
    result = classify_text(text)
    classification = result.get(
        "classification",
        {
            "depression": 0,
            "anxiety": 0,
            "addiction": 0,
        },
    )

    classification["is_normal"] = all(value == 0 for value in classification.values())

    if not result.get("ok"):
        classification["error"] = result.get("message", "")

    return classification