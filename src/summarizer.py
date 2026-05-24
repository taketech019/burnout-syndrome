"""src/summarizer.py — F3: KoAlpaca Modal endpoint 호출.

Modal 호스팅 (NF4 + LoRA attach, A10G 24GB)을 통해 4섹션 요약 생성.
ai-model/nf4_modal.py 배포 후 KOALPACA_ENDPOINT_URL / KOALPACA_API_KEY 설정 필요.
"""
import re
import requests

from config import KOALPACA_ENDPOINT_URL, KOALPACA_API_KEY

# Modal 호스팅 검증으로 확정된 입력 요구사항:
# - 학습 분포 median 22,872자. 3000자에서는 4섹션 트리거 실패, 5000자부터 partial pass.
# - max_position_embeddings=2048 토큰. 한국어 ~3.5char/token. 안전선 9000자.
_MIN_INPUT_CHARS = 5000
_MAX_INPUT_CHARS = 9000
_MAX_NEW_TOKENS = 1024
_TIMEOUT = 300

_SECTIONS = {
    "symptoms":             r"주요\s*증상",
    "risk_factors":         r"위험\s*요인",
    "improvement_factors":  r"개선\s*요인",
    "intervention_factors": r"개입\s*요인",
}

# "내담자 : 텍스트" → "내담자\t텍스트" (학습 데이터의 발화자 구분자가 탭)
_SPEAKER_COLON = re.compile(r"^([가-힣]+)\s*:\s*", re.MULTILINE)


def _normalize_transcript(text: str) -> str:
    """학습 포맷(발화자\\t텍스트)으로 변환. 이미 탭이면 그대로."""
    if "\t" in text:
        return text
    return _SPEAKER_COLON.sub(r"\1\t", text)


def _parse_sections(raw: str) -> dict:
    """모델 출력에서 4섹션을 정규식으로 분리. 헤더 미검출 시 raw에 전체 담음."""
    raw = raw.strip()
    positions = {k: m.start() for k, p in _SECTIONS.items() if (m := re.search(p, raw))}
    if not positions:
        return {k: "" for k in _SECTIONS} | {"raw": raw}

    ordered = sorted(positions.items(), key=lambda x: x[1])
    result = {}
    for i, (key, start) in enumerate(ordered):
        end = ordered[i + 1][1] if i + 1 < len(ordered) else len(raw)
        result[key] = raw[start:end].strip()
    return {k: result.get(k, "") for k in _SECTIONS} | {"raw": raw}


def summarize(text: str) -> dict:
    """F3: KoAlpaca Modal /summarize 호출. 반환: 4섹션 dict + raw. 실패 시 error 키 포함."""
    if not KOALPACA_ENDPOINT_URL:
        return {k: "" for k in _SECTIONS} | {"error": "KOALPACA_ENDPOINT_URL 미설정"}
    if not KOALPACA_API_KEY:
        return {k: "" for k in _SECTIONS} | {"error": "KOALPACA_API_KEY 미설정"}

    transcript = _normalize_transcript(text)
    if len(transcript) < _MIN_INPUT_CHARS:
        return {k: "" for k in _SECTIONS} | {
            "error": f"입력이 너무 짧습니다 ({len(transcript)}자). 4섹션 추출에 최소 {_MIN_INPUT_CHARS}자 필요."
        }
    transcript = transcript[:_MAX_INPUT_CHARS]

    try:
        resp = requests.post(
            KOALPACA_ENDPOINT_URL,
            json={
                "api_key": KOALPACA_API_KEY,
                "transcript": transcript,
                "max_new_tokens": _MAX_NEW_TOKENS,
                "temperature": 0.0,
                "repetition_penalty": 1.2,
                "no_repeat_ngram_size": 3,
            },
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        return _parse_sections(resp.json()["text"])
    except requests.exceptions.ConnectionError:
        return {k: "" for k in _SECTIONS} | {"error": "KoAlpaca 서버에 연결할 수 없습니다."}
    except requests.exceptions.Timeout:
        return {k: "" for k in _SECTIONS} | {"error": f"응답 시간 초과 ({_TIMEOUT}초)"}
    except Exception as e:
        return {k: "" for k in _SECTIONS} | {"error": str(e)}
