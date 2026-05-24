"""src/summarizer.py — F3: KoAlpaca Modal endpoint 호출.

Modal 호스팅(NF4 + LoRA attach, A10G 24GB)을 통해 4섹션 요약을 생성한다.
설정값은 Streamlit Secrets 또는 환경변수에서 읽는다.

주의:
- 실제 API 키는 GitHub 코드에 쓰지 않는다.
- 로컬에서는 .streamlit/secrets.toml, 배포에서는 Streamlit Secrets를 사용한다.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict

import requests


# Modal 호스팅 검증으로 확정된 입력 요구사항:
# - 학습 분포 median 22,872자. 3000자에서는 4섹션 트리거 실패, 5000자부터 partial pass.
# - max_position_embeddings=2048 토큰. 한국어 ~3.5 char/token. 안전선 9000자.
_MIN_INPUT_CHARS = 5000
_MAX_INPUT_CHARS = 9000
_MAX_NEW_TOKENS = 1024
_TIMEOUT = 300

_SECTIONS = {
    "symptoms": r"주요\s*증상",
    "risk_factors": r"위험\s*요인",
    "improvement_factors": r"개선\s*요인",
    "intervention_factors": r"개입\s*요인",
}

# "내담자 : 텍스트" → "내담자\t텍스트" (학습 데이터의 발화자 구분자가 탭)
_SPEAKER_COLON = re.compile(r"^([가-힣]+)\s*:\s*", re.MULTILINE)


def _get_secret(key: str, default: str = "") -> str:
    """Streamlit Secrets를 우선 읽고, 없으면 환경변수를 읽는다."""
    try:
        import streamlit as st

        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass

    return os.getenv(key, default)


def get_koalpaca_config() -> Dict[str, str]:
    return {
        "endpoint_url": _get_secret("KOALPACA_ENDPOINT_URL", "").strip(),
        "api_key": _get_secret("KOALPACA_API_KEY", "").strip(),
    }


def _empty_sections() -> Dict[str, str]:
    return {key: "" for key in _SECTIONS}


def _normalize_transcript(text: str) -> str:
    """학습 포맷(발화자\\t텍스트)으로 변환. 이미 탭이면 그대로 둔다."""
    if "\t" in text:
        return text
    return _SPEAKER_COLON.sub(r"\1\t", text)


def _parse_sections(raw: str) -> Dict[str, str]:
    """모델 출력에서 4섹션을 정규식으로 분리. 헤더 미검출 시 raw에 전체 담음."""
    raw = raw.strip()
    positions = {key: match.start() for key, pattern in _SECTIONS.items() if (match := re.search(pattern, raw))}

    if not positions:
        return _empty_sections() | {"raw": raw}

    ordered = sorted(positions.items(), key=lambda item: item[1])
    result: Dict[str, str] = {}

    for index, (key, start) in enumerate(ordered):
        end = ordered[index + 1][1] if index + 1 < len(ordered) else len(raw)
        result[key] = raw[start:end].strip()

    return {key: result.get(key, "") for key in _SECTIONS} | {"raw": raw}


def _as_app_result(
    ok: bool,
    status: str,
    message: str,
    text: str = "",
    sections: Dict[str, str] | None = None,
) -> Dict[str, Any]:
    """app.py의 KoalpacaAPISummarizer가 기대하는 공통 반환 형식."""
    return {
        "ok": ok,
        "status": status,
        "message": message,
        "text": text,
        "sections": sections or _empty_sections(),
        "backend": "koalpaca_api",
    }


def summarize(text: str) -> Dict[str, Any]:
    """F3: KoAlpaca Modal /summarize 호출. 성공 시 app.py가 표시할 보고서 text를 반환."""
    config = get_koalpaca_config()
    endpoint_url = config["endpoint_url"].rstrip("/")
    api_key = config["api_key"]

    if not endpoint_url:
        return _as_app_result(False, "not_configured", "KOALPACA_ENDPOINT_URL이 설정되지 않았습니다.")
    if not api_key:
        return _as_app_result(False, "not_configured", "KOALPACA_API_KEY가 설정되지 않았습니다.")

    transcript = _normalize_transcript(text)

    if len(transcript) < _MIN_INPUT_CHARS:
        return _as_app_result(
            False,
            "too_short",
            f"입력이 너무 짧습니다({len(transcript)}자). 4섹션 추출에는 최소 {_MIN_INPUT_CHARS}자 이상이 필요합니다.",
        )

    transcript = transcript[:_MAX_INPUT_CHARS]

    try:
        resp = requests.post(
            endpoint_url,
            json={
                "api_key": api_key,
                "transcript": transcript,
                "max_new_tokens": _MAX_NEW_TOKENS,
                "temperature": 0.0,
                "repetition_penalty": 1.2,
                "no_repeat_ngram_size": 3,
            },
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()

        raw_text = str(data.get("text", "")).strip()
        sections = _parse_sections(raw_text)

        return _as_app_result(
            True,
            "success",
            "KoAlpaca API 보고서 생성 성공",
            text=raw_text,
            sections=sections,
        )

    except requests.exceptions.ConnectionError:
        return _as_app_result(False, "connection_error", "KoAlpaca 서버에 연결할 수 없습니다.")
    except requests.exceptions.Timeout:
        return _as_app_result(False, "timeout", f"KoAlpaca API 응답 시간이 {_TIMEOUT}초를 초과했습니다.")
    except Exception as error:
        return _as_app_result(False, "error", f"KoAlpaca API 호출 중 오류가 발생했습니다: {error}")
