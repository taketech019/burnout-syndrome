"""src/summarizer.py — F3: KoAlpaca Modal endpoint 호출 + Gemma 폴백.

기본은 KoAlpaca (Modal NF4 + LoRA, A10G 24GB) — AI Hub 학습 분포에 맞춰 partial pass 2/4.
Modal 콜드스타트(60~120s) 또는 다운 시 **Gemma 4 31B 폴백**으로 4섹션 요약 생성.
"""
import re
import requests

from config import KOALPACA_ENDPOINT_URL, KOALPACA_API_KEY
from src.gemma_client import generate, strip_reasoning

# Modal 호스팅 검증으로 확정된 입력 요구사항:
# - 학습 분포 median 22,872자. 3000자에서는 4섹션 트리거 실패, 5000자부터 partial pass.
# - max_position_embeddings=2048 토큰. 한국어 ~3.5char/token. 안전선 9000자.
_MIN_INPUT_CHARS = 5000
_MAX_INPUT_CHARS = 9000
_MAX_NEW_TOKENS = 1024
_TIMEOUT = 300

_SECTIONS = {
    "symptoms":             r"(?:^|[\s\*#>•\-])\**주요\s*증상\**\s*[:：.]?",
    "risk_factors":         r"(?:^|[\s\*#>•\-])\**위험\s*요인\**\s*[:：.]?",
    "improvement_factors":  r"(?:^|[\s\*#>•\-])\**개선\s*요인\**\s*[:：.]?",
    "intervention_factors": r"(?:^|[\s\*#>•\-])\**개입\s*요인\**\s*[:：.]?",
}

# "내담자 : 텍스트" → "내담자\t텍스트" (학습 데이터의 발화자 구분자가 탭)
_SPEAKER_COLON = re.compile(r"^([가-힣]+)\s*:\s*", re.MULTILINE)


def _normalize_transcript(text: str) -> str:
    """학습 포맷(발화자\\t텍스트)으로 변환. 이미 탭이면 그대로."""
    if "\t" in text:
        return text
    return _SPEAKER_COLON.sub(r"\1\t", text)


_HEADER_PREFIX = re.compile(r"^[\s\*#>•\-]*\**(주요\s*증상|위험\s*요인|개선\s*요인|개입\s*요인)\**\s*[:：.]?\s*")


def _find_section_positions(raw: str) -> dict[str, int]:
    """각 섹션의 시작 위치를 찾되, 모든 매치 중 '실제 내용이 뒤따르는' 것 우선.

    Gemma는 종종 헤더 1개를 도입부에 한 번, 본문 섹션을 bullet로 다시 등장시키므로
    각 섹션별로 모든 매치를 모은 뒤, 다음 섹션 매치까지 거리가 가장 긴(=실제 내용 포함)
    것을 선택.
    """
    matches: dict[str, list[int]] = {}
    for key, pattern in _SECTIONS.items():
        matches[key] = [m.start() for m in re.finditer(pattern, raw)]
    chosen: dict[str, int] = {}
    for key, starts in matches.items():
        if not starts:
            continue
        # 같은 섹션 헤더가 여러 번 나오면, 가장 뒤(=본문이 따라오는 위치) 우선
        chosen[key] = starts[-1] if len(starts) > 1 else starts[0]
    return chosen


def _parse_sections(raw: str) -> dict:
    """모델 출력에서 4섹션을 정규식으로 분리. 헤더 미검출 시 raw에 전체 담음."""
    raw = raw.strip()
    positions = _find_section_positions(raw)
    if not positions:
        return {k: "" for k in _SECTIONS} | {"raw": raw}

    ordered = sorted(positions.items(), key=lambda x: x[1])
    result = {}
    for i, (key, start) in enumerate(ordered):
        end = ordered[i + 1][1] if i + 1 < len(ordered) else len(raw)
        chunk = raw[start:end].strip()
        chunk = _HEADER_PREFIX.sub("", chunk, count=1).strip()
        chunk = re.sub(r"\n\s*\*\s*$", "", chunk).strip()
        result[key] = chunk
    return {k: result.get(k, "") for k in _SECTIONS} | {"raw": raw}


_GEMMA_SUMMARY_PROMPT = """당신은 임상심리 전문가입니다. 다음 상담 회기 텍스트를 보고 **한국어로만** 4섹션 요약을 작성합니다.

**규칙**
- 영어, 메타 설명, "Role:" 같은 라벨 금지. 한국어 답변만.
- 각 섹션의 내용은 자연스러운 한국어 문단으로 작성 (불릿·번호 금지).
- 각 섹션 1~3문장.

**출력 형식** (네 개 헤더를 정확히 그대로 사용)

주요 증상: <한 문단>

위험 요인: <한 문단>

개선 요인: <한 문단>

개입 요인: <한 문단>

---

상담 텍스트:
{text}
"""


def _try_gemma_fallback(transcript: str) -> dict:
    """Gemma 4 31B 또는 Gemini Fallback으로 4섹션 요약 생성."""
    prompt = _GEMMA_SUMMARY_PROMPT.replace("{text}", transcript)
    try:
        text = generate(prompt, temperature=0.2, max_output_tokens=2048)
        cleaned = strip_reasoning(text)
        parsed = _parse_sections(cleaned)
        parsed["_source"] = "gemma_fallback"
        # raw 키에 cleaned 보관 (디버그)
        parsed["raw"] = cleaned[:1000]
        return parsed
    except Exception as e:
        return {k: "" for k in _SECTIONS} | {"error": f"Gemma 폴백도 실패: {e}"}


def summarize(text: str, use_gemma_fallback: bool = True) -> dict:
    """F3 4섹션 요약. KoAlpaca → 실패 시 Gemma 4 31B 폴백.

    KoAlpaca는 학습 분포 median 22,872자에 맞춰져 < 5000자 입력에선 4섹션이 안 나옴.
    이 경우에도 Gemma는 짧은 입력으로 잘 동작 → 입력 짧으면 곧바로 Gemma.
    """
    transcript_norm = _normalize_transcript(text)
    transcript = transcript_norm[:_MAX_INPUT_CHARS]

    too_short = len(transcript_norm) < _MIN_INPUT_CHARS
    no_endpoint = not KOALPACA_ENDPOINT_URL or not KOALPACA_API_KEY

    if not (too_short or no_endpoint):
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
            parsed = _parse_sections(resp.json()["text"])
            parsed["_source"] = "koalpaca"
            return parsed
        except Exception as e:
            koalpaca_err = str(e)
            if not use_gemma_fallback:
                return {k: "" for k in _SECTIONS} | {"error": f"KoAlpaca 실패: {koalpaca_err}"}
    elif use_gemma_fallback:
        koalpaca_err = (
            f"입력 너무 짧음 ({len(transcript_norm)}자 < {_MIN_INPUT_CHARS}자)"
            if too_short else "KoAlpaca endpoint 미설정"
        )
    else:
        return {k: "" for k in _SECTIONS} | {
            "error": (
                f"입력이 너무 짧습니다 ({len(transcript_norm)}자). 최소 {_MIN_INPUT_CHARS}자 필요."
            ) if too_short else "KoAlpaca endpoint 미설정"
        }

    fallback = _try_gemma_fallback(transcript_norm)
    fallback["_koalpaca_skip_reason"] = koalpaca_err
    return fallback
