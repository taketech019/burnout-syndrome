"""src/summarizer.py — F3: KoAlpaca 상담 요약 (llama-server /v1/completions 호출)."""
import re
import requests

from config import KOALPACA_ENDPOINT_URL, KOALPACA_API_KEY

_INSTRUCTION = "다음과 같은 상담기록을 보고 요약서를 작성해주세요."
_MAX_INPUT_CHARS = 1900
_MAX_NEW_TOKENS = 500
_TIMEOUT = 300
_STOP = ["<|endoftext|>", "<|sep|>", "### 명령어:"]
_SECTIONS = {
    "symptoms":             r"주요\s*증상",
    "risk_factors":         r"위험\s*요인",
    "improvement_factors":  r"개선\s*요인",
    "intervention_factors": r"개입\s*요인",
}


def _build_prompt(text: str) -> str:
    return (
        f"### 명령어: {_INSTRUCTION}\n\n"
        f"### 맥락: {text[:_MAX_INPUT_CHARS]}\n\n"
        f"### 답변:"
    )


def _parse_sections(raw: str) -> dict:
    for stop in ["<|endoftext|>", "<|sep|>"]:
        raw = raw.split(stop)[0]
    raw = raw.strip()

    positions = {k: m.start() for k, p in _SECTIONS.items() if (m := re.search(p, raw))}
    if not positions:
        return {k: "" for k in _SECTIONS} | {"raw": raw}

    ordered = sorted(positions.items(), key=lambda x: x[1])
    result = {}
    for i, (key, start) in enumerate(ordered):
        end = ordered[i + 1][1] if i + 1 < len(ordered) else len(raw)
        result[key] = raw[start:end].strip()
    return {k: result.get(k, "") for k in _SECTIONS}


def summarize(text: str) -> dict:
    """F3: KoAlpaca 요약 호출. 반환: 4개 섹션 dict. 실패 시 'error' 키 포함."""
    if not KOALPACA_ENDPOINT_URL:
        return {k: "" for k in _SECTIONS} | {"error": "KOALPACA_ENDPOINT_URL 미설정"}

    url = KOALPACA_ENDPOINT_URL.rstrip("/") + "/v1/completions"
    try:
        resp = requests.post(
            url,
            json={
                "prompt": _build_prompt(text),
                "max_tokens": _MAX_NEW_TOKENS,
                "temperature": 0.3,        # 양자화 후 그리디는 instruction-following 무너짐 → 약한 sampling
                "top_k": 40,
                "top_p": 0.9,
                "repeat_penalty": 1.2,     # 노트북의 repetition_penalty=1.2
                "repeat_last_n": 256,      # no_repeat_ngram_size=3 효과 근사
                "stop": _STOP,
                "stream": False,
            },
            headers={
                "Authorization": f"Bearer {KOALPACA_API_KEY}",
                "Content-Type": "application/json",
            },
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        return _parse_sections(resp.json()["choices"][0]["text"])
    except requests.exceptions.ConnectionError:
        return {k: "" for k in _SECTIONS} | {"error": "KoAlpaca 서버에 연결할 수 없습니다."}
    except requests.exceptions.Timeout:
        return {k: "" for k in _SECTIONS} | {"error": f"응답 시간 초과 ({_TIMEOUT}초)"}
    except Exception as e:
        return {k: "" for k in _SECTIONS} | {"error": str(e)}
