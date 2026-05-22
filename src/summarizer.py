"""src/summarizer.py

KoAlpaca API 연결 전용 모듈.

현재 목적:
- KoAlpaca 호스팅이 완성되기 전까지 API 연결 자리를 만들어둔다.
- API URL/KEY가 없으면 앱이 깨지지 않고 '미설정' 상태를 반환한다.
- API 호출이 성공하면 보고서 텍스트를 반환한다.
- API 호출이 실패하면 오류 상태를 반환한다.

주의:
- 실제 API 키는 GitHub 코드에 쓰지 않는다.
- Streamlit Cloud에서는 st.secrets 또는 환경변수에서 값을 읽는다.
"""

from __future__ import annotations

import os
import re
from typing import Dict, Any

import requests


_INSTRUCTION = "다음과 같은 상담기록을 보고 요약서를 작성해주세요."
_MAX_INPUT_CHARS = 1900
_MAX_NEW_TOKENS = 1024
_TIMEOUT = 120

_STOP = ["<|endoftext|>", "<|sep|>", "###명령어:", "### 명령어:"]

_SECTIONS = {
    "symptoms": r"주요\s*증상",
    "risk_factors": r"위험\s*요인",
    "improvement_factors": r"개선\s*요인",
    "intervention_factors": r"개입\s*요인",
}


def _get_secret(key: str, default: str = "") -> str:
    """
    Streamlit Cloud에서는 st.secrets를 우선 사용하고,
    로컬 실행에서는 환경변수를 사용한다.

    st.secrets를 직접 import하지 않고 함수 내부에서만 가져오는 이유:
    - Streamlit이 아닌 환경에서 이 파일을 import해도 깨지지 않게 하기 위함.
    """
    try:
        import streamlit as st

        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass

    return os.getenv(key, default)


def get_koalpaca_config() -> Dict[str, str]:
    """
    KoAlpaca API 연결 설정을 반환한다.
    """
    return {
        "endpoint_url": _get_secret("KOALPACA_ENDPOINT_URL", "").strip(),
        "api_key": _get_secret("KOALPACA_API_KEY", "").strip(),
    }


def is_koalpaca_configured() -> bool:
    """
    KoAlpaca API URL이 설정되어 있는지 확인한다.
    API Key는 서버 설정에 따라 없을 수도 있으므로 URL만 필수로 본다.
    """
    config = get_koalpaca_config()
    return bool(config["endpoint_url"])


def _build_prompt(text: str) -> str:
    """
    KoAlpaca 입력 프롬프트를 만든다.

    현재 프로젝트 문서 기준으로 '###맥락:' 형식을 우선 사용한다.
    """
    safe_text = text[:_MAX_INPUT_CHARS]

    return (
        f"###명령어: {_INSTRUCTION}\n\n"
        f"###맥락: {safe_text}\n\n"
        f"###답변:"
    )


def _parse_sections(raw: str) -> Dict[str, str]:
    """
    KoAlpaca 응답 텍스트에서 4개 섹션을 분리한다.
    섹션 분리에 실패하면 raw에 원문을 그대로 넣는다.
    """
    for stop in ["<|endoftext|>", "<|sep|>"]:
        raw = raw.split(stop)[0]

    raw = raw.strip()

    positions = {
        key: match.start()
        for key, pattern in _SECTIONS.items()
        if (match := re.search(pattern, raw))
    }

    if not positions:
        return {
            "symptoms": "",
            "risk_factors": "",
            "improvement_factors": "",
            "intervention_factors": "",
            "raw": raw,
        }

    ordered = sorted(positions.items(), key=lambda item: item[1])
    result: Dict[str, str] = {}

    for index, (key, start) in enumerate(ordered):
        end = ordered[index + 1][1] if index + 1 < len(ordered) else len(raw)
        result[key] = raw[start:end].strip()

    return {
        "symptoms": result.get("symptoms", ""),
        "risk_factors": result.get("risk_factors", ""),
        "improvement_factors": result.get("improvement_factors", ""),
        "intervention_factors": result.get("intervention_factors", ""),
        "raw": raw,
    }


def summarize(text: str) -> Dict[str, Any]:
    """
    KoAlpaca API로 상담 요약 보고서를 생성한다.

    반환 형식:
    {
        "ok": bool,
        "status": "not_configured" | "success" | "connection_error" | "timeout" | "error",
        "message": str,
        "text": str,
        "sections": dict,
        "backend": "koalpaca_api"
    }
    """
    config = get_koalpaca_config()
    endpoint_url = config["endpoint_url"]
    api_key = config["api_key"]

    if not endpoint_url:
        return {
            "ok": False,
            "status": "not_configured",
            "message": "KoAlpaca API URL이 아직 설정되지 않았습니다.",
            "text": "",
            "sections": {},
            "backend": "koalpaca_api",
        }

    url = endpoint_url.rstrip("/") + "/v1/completions"

    headers = {
        "Content-Type": "application/json",
    }

    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "prompt": _build_prompt(text),
        "max_tokens": _MAX_NEW_TOKENS,
        "temperature": 0.0,
        "top_k": 1,
        "repeat_penalty": 1.2,
        "repeat_last_n": 256,
        "stop": _STOP,
        "stream": False,
    }

    try:
        response = requests.post(
            url,
            json=payload,
            headers=headers,
            timeout=_TIMEOUT,
        )
        response.raise_for_status()

        data = response.json()

        generated_text = ""

        if isinstance(data, dict):
            choices = data.get("choices", [])
            if choices and isinstance(choices, list):
                generated_text = choices[0].get("text", "")

        if not generated_text:
            return {
                "ok": False,
                "status": "error",
                "message": "KoAlpaca API 응답에서 생성 텍스트를 찾지 못했습니다.",
                "text": "",
                "sections": {},
                "backend": "koalpaca_api",
            }

        sections = _parse_sections(generated_text)

        return {
            "ok": True,
            "status": "success",
            "message": "KoAlpaca API 호출 성공",
            "text": generated_text.strip(),
            "sections": sections,
            "backend": "koalpaca_api",
        }

    except requests.exceptions.ConnectionError:
        return {
            "ok": False,
            "status": "connection_error",
            "message": "KoAlpaca 서버에 연결할 수 없습니다.",
            "text": "",
            "sections": {},
            "backend": "koalpaca_api",
        }

    except requests.exceptions.Timeout:
        return {
            "ok": False,
            "status": "timeout",
            "message": f"KoAlpaca API 응답 시간이 {_TIMEOUT}초를 초과했습니다.",
            "text": "",
            "sections": {},
            "backend": "koalpaca_api",
        }

    except Exception as error:
        return {
            "ok": False,
            "status": "error",
            "message": f"KoAlpaca API 호출 중 오류가 발생했습니다: {error}",
            "text": "",
            "sections": {},
            "backend": "koalpaca_api",
        }
