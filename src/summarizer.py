"""src/summarizer.py — F3 4섹션 요약. KoAlpaca Modal 전용.

신규 반환 시그니처: {ok, status, message, text, sections, source}
- text: 보고서 본문 (Markdown)
- sections: {"symptoms", "risk_factors", "improvement_factors", "intervention_factors"} dict
  → 보고서 빌더(.docx/.pdf)는 sections 사용해서 재파싱 회피
"""
import logging
import re
from typing import Optional

import requests

from config import KOALPACA_API_KEY, KOALPACA_ENDPOINT_URL
from src.gemma_client import generate, strip_reasoning

log = logging.getLogger(__name__)

_MIN_INPUT_CHARS = 5000
_MAX_INPUT_CHARS = 9000
_MAX_NEW_TOKENS = 1024
_TIMEOUT = 300

_EMPTY_SECTIONS = {
    "symptoms": "",
    "risk_factors": "",
    "improvement_factors": "",
    "intervention_factors": "",
}

_SECTION_PATTERNS = {
    "symptoms":             r"(?:^|[\s\*#>•\-])\**주요\s*증상\**\s*[:：.]?",
    "risk_factors":         r"(?:^|[\s\*#>•\-])\**위험\s*요인\**\s*[:：.]?",
    "improvement_factors":  r"(?:^|[\s\*#>•\-])\**개선\s*요인\**\s*[:：.]?",
    "intervention_factors": r"(?:^|[\s\*#>•\-])\**개입\s*요인\**\s*[:：.]?",
}

_HEADER_PREFIX = re.compile(
    r"^[\s\*#>•\-]*\**(주요\s*증상|위험\s*요인|개선\s*요인|개입\s*요인)\**\s*[:：.]?\s*"
)

_SPEAKER_COLON = re.compile(r"^([가-힣]+)\s*:\s*", re.MULTILINE)


def _normalize_transcript(text: str) -> str:
    if "\t" in text:
        return text
    return _SPEAKER_COLON.sub(r"\1\t", text)


def _empty_result(message: str, ok: bool = False) -> dict:
    return {
        "ok": ok,
        "status": "error" if not ok else "success",
        "message": message,
        "text": "",
        "brief": "",
        "sections": dict(_EMPTY_SECTIONS),
        "source": "none",
        "koalpaca_attempted": False,
        "koalpaca_sections_filled": 0,
        "gemma_sections_filled": 0,
    }


def _find_section_positions(raw: str) -> dict[str, int]:
    matches: dict[str, list[int]] = {}
    for key, p in _SECTION_PATTERNS.items():
        matches[key] = [m.start() for m in re.finditer(p, raw)]
    chosen: dict[str, int] = {}
    for key, starts in matches.items():
        if starts:
            chosen[key] = starts[-1] if len(starts) > 1 else starts[0]
    return chosen


def _parse_sections(raw: str) -> dict:
    raw = raw.strip()
    pos = _find_section_positions(raw)
    if not pos:
        return dict(_EMPTY_SECTIONS)
    ordered = sorted(pos.items(), key=lambda x: x[1])
    out = {}
    for i, (key, start) in enumerate(ordered):
        end = ordered[i + 1][1] if i + 1 < len(ordered) else len(raw)
        chunk = raw[start:end].strip()
        chunk = _HEADER_PREFIX.sub("", chunk, count=1).strip()
        chunk = re.sub(r"\n\s*\*\s*$", "", chunk).strip()
        out[key] = chunk
    return {k: out.get(k, "") for k in _EMPTY_SECTIONS}


def _sections_to_text(sections: dict) -> str:
    """sections dict → 5섹션 Markdown 본문 (다음 회기 계획 placeholder 포함)."""
    parts = [
        "## 1. 주요 증상",
        sections.get("symptoms", "") or "(추출되지 않음)",
        "",
        "## 2. 위험 요인",
        sections.get("risk_factors", "") or "(추출되지 않음)",
        "",
        "## 3. 개선 요인",
        sections.get("improvement_factors", "") or "(추출되지 않음)",
        "",
        "## 4. 상담사 개입 요인",
        sections.get("intervention_factors", "") or "(추출되지 않음)",
        "",
        "## 5. 다음 회기 계획",
        "(상담사 수동 작성 또는 LLM 보조)",
    ]
    return "\n".join(parts)


_SECTION_NAMES = {
    "symptoms":             "주요 증상",
    "risk_factors":         "위험 요인",
    "improvement_factors":  "개선 요인",
    "intervention_factors": "상담사 개입 요인",
}

_GEMMA_SECTION_PROMPT = """다음 상담 텍스트에서 **{section_name}** 섹션만 한국어로 작성하세요.
- 불릿 포인트(•) 또는 짧은 서술문으로 2~5줄
- 영어·코드블록·헤더 금지. 본문만 출력.

상담 텍스트:
{text}
"""

_GEMMA_BRIEF_PROMPT = """다음 상담 회기를 한국어 **한 문단(2~4문장)** 으로 요약합니다.
영어·메타 라벨·코드블록·헤더·불릿 금지. 자연스러운 한국어 본문만 출력.

**중요**:
- transcript 끝 척도 응답(PHQ-9 등)은 이미 제거됨 — 본문 발화 내용만 요약하세요.
- 긍정적 회기에서도 brief는 회기의 주요 흐름(주제·발견·내담자 변화·상담사 개입)을 자연스럽게 요약.
- 척도 점수나 숫자 응답은 절대 언급하지 않음.

상담 텍스트:
{text}
"""


def _gen_section(key: str, transcript: str) -> str:
    """단일 섹션 Gemini 생성. 실패 시 빈 문자열."""
    try:
        text = generate(
            _GEMMA_SECTION_PROMPT
                .replace("{section_name}", _SECTION_NAMES[key])
                .replace("{text}", transcript[:_MAX_INPUT_CHARS]),
            temperature=0.3, max_output_tokens=512,
        )
        content = strip_reasoning(text).strip()
        return f"[gemini로 생성됨]\n{content}" if content else ""
    except Exception as e:
        log.warning("섹션 %s Gemini 생성 실패: %s", key, e)
        return ""


def _gen_brief(transcript: str) -> str:
    try:
        text = generate(
            _GEMMA_BRIEF_PROMPT.replace("{text}", transcript[:_MAX_INPUT_CHARS]),
            temperature=0.2, max_output_tokens=2048,
        )
        return strip_reasoning(text).strip()
    except Exception as e:
        log.warning("brief 생성 실패: %s", e)
        return ""


# ── KoAlpaca (Modal) ───────────────────────────────────────────────────────────


def _try_koalpaca(transcript: str) -> Optional[dict]:
    if not KOALPACA_ENDPOINT_URL or not KOALPACA_API_KEY:
        return None
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
        raw = resp.json().get("text", "")
        sections = _parse_sections(raw)
        return {
            "ok": True,
            "status": "success",
            "message": "KoAlpaca Modal 응답",
            "text": _sections_to_text(sections),
            "brief": "",
            "sections": sections,
            "source": "koalpaca",
        }
    except Exception as e:
        log.warning("KoAlpaca 실패: %s", e)
        return None




# ── 머지 헬퍼 ─────────────────────────────────────────────────────────────────


def _count_filled(sections: dict) -> int:
    return sum(1 for k in _EMPTY_SECTIONS if (sections.get(k) or "").strip())


def _build_result(sections: dict, brief: str) -> dict:
    """완성된 4섹션을 result dict로 포장."""
    gemma_filled = sum(1 for v in sections.values() if (v or "").startswith("[gemini로 생성됨]"))
    ka_filled = sum(1 for v in sections.values() if v and not (v or "").startswith("[gemini로 생성됨]"))
    return {
        "ok": any(sections.values()),
        "status": "success" if any(sections.values()) else "error",
        "message": f"koalpaca {ka_filled}/4 + gemini {gemma_filled}/4",
        "text": _sections_to_text(sections),
        "brief": brief,
        "sections": sections,
        "source": "koalpaca" if gemma_filled == 0 else "koalpaca+gemini",
        "koalpaca_attempted": True,
        "koalpaca_sections_filled": ka_filled,
        "gemma_sections_filled": gemma_filled,
    }


# ── 공개 진입점 ────────────────────────────────────────────────────────────────


def summarize(script: str) -> dict:
    """F3 요약. KoAlpaca Modal 전용 — 실패 시 에러 반환 (Gemini 폴백 없음).

    transcript 끝 척도(PHQ-9 등) 영역은 분리되어 모델 입력에서 제외.
    """
    if not script or not script.strip():
        return _empty_result("입력 텍스트 비어 있음")

    from src.transcript_utils import split_transcript_and_scale
    script, _scale = split_transcript_and_scale(script)
    if not script.strip():
        return _empty_result("입력 텍스트 비어 있음")

    transcript = _normalize_transcript(script)
    capped = transcript[:_MAX_INPUT_CHARS]

    ka_result = _try_koalpaca(capped)
    sections = ka_result.get("sections", dict(_EMPTY_SECTIONS)) if ka_result else dict(_EMPTY_SECTIONS)

    for key in _EMPTY_SECTIONS:
        if not sections.get(key, "").strip():
            sections[key] = _gen_section(key, transcript)

    brief = _gen_brief(transcript)
    return _build_result(sections, brief=brief)
