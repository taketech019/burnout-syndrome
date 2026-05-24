"""src/summarizer.py — F3 4섹션 요약. KoAlpaca Modal → Gemma 폴백.

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
        brief = _gen_brief(transcript)
        return {
            "ok": True,
            "status": "success",
            "message": "KoAlpaca Modal 응답",
            "text": _sections_to_text(sections),
            "brief": brief,
            "sections": sections,
            "source": "koalpaca",
        }
    except Exception as e:
        log.warning("KoAlpaca 실패: %s", e)
        return None


# ── Gemma 폴백 ─────────────────────────────────────────────────────────────────

_GEMMA_PROMPT = """당신은 임상심리 전문가입니다. 다음 상담 회기 텍스트를 보고 **한국어로만** 4섹션 요약을 작성합니다.

**규칙**
- 영어, 메타 설명, "Role:" 같은 라벨 절대 금지. 한국어 답변만.
- 각 섹션 내용은 자연스러운 한국어 문단 (불릿·번호 금지).
- 각 섹션 1~3문장.

**중요 규칙**
- transcript 끝 척도 평가(PHQ-9 등)는 이미 입력에서 제거됨. 본문 상담 내용만 보고 작성하세요.
- **긍정적 회기**(증상 호소 약하고 자기 수용·통찰·강점 인식 강함)에서도 4섹션을 모두 작성합니다:
  - **주요 증상**: 명백한 호소가 없으면 "이번 회기에서는 두드러진 부정 증상 호소가 관찰되지 않으며, 내담자가 자신을 긍정적으로 평가하고 있습니다." 같이 한 문장.
  - **위험 요인**: 위험 신호가 없으면 "관찰되는 위험 요인 없음 — 자살/자해 사고, 약물 남용 등 부정 신호 부재." 같이 명시.
  - **개선 요인**: 자기 통찰, 자기 수용, 변화 동기, 강점 인식, 욕구 만족, 상담 참여 의지 — 본 회기에 명확히 보이면 **구체적 발화 근거와 함께** 한 문단.
  - **개입 요인**: 상담사가 사용한 기법 (예: 욕구 탐색, 공감/지지, 강점 발견, 통찰 작업, 명료화, 재구성, 직면, 행동 활성화, 다음 회기 과제 부여) — 본 회기 상담사 발화에서 명시적으로 추출. 모르겠으면 "공감·반영 중심 개입" 정도라도 한 문장.
- 네 섹션 모두 "(없음)" 또는 빈 문자열로 끝나지 않도록 합니다. 최소 한 문장씩 작성.

**출력 형식** (네 개 헤더를 정확히 그대로 사용)

주요 증상: <한 문단>

위험 요인: <한 문단>

개선 요인: <한 문단>

개입 요인: <한 문단>

---

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


def _gen_brief(transcript: str) -> str:
    """Gemma 1단락 요약 — 보고서 페이지 "요약본" textarea용.

    Gemini 2.5 Flash thinking 모드는 reasoning에 토큰 소비 — 512에서는 잘림.
    2048로 충분한 여유 확보.
    """
    try:
        text = generate(
            _GEMMA_BRIEF_PROMPT.replace("{text}", transcript[:_MAX_INPUT_CHARS]),
            temperature=0.2, max_output_tokens=2048,
        )
        return strip_reasoning(text).strip()
    except Exception as e:
        log.warning("brief 생성 실패: %s", e)
        return ""


def _try_gemma_fallback(transcript: str) -> dict:
    try:
        capped = transcript[:_MAX_INPUT_CHARS]
        # max_output_tokens 8192 — Gemini 2.5 Flash는 thinking 모드라 reasoning에 토큰 소비.
        # 2048에서는 4섹션 중 위험 요인쯤에서 잘림. 8192로 여유 확보.
        text = generate(_GEMMA_PROMPT.replace("{text}", capped),
                        temperature=0.2, max_output_tokens=8192)
        cleaned = strip_reasoning(text)
        sections = _parse_sections(cleaned)
        brief = _gen_brief(transcript)
        return {
            "ok": True,
            "status": "fallback",
            "message": "Gemma 4 31B",
            "text": _sections_to_text(sections),
            "brief": brief,
            "sections": sections,
            "source": "gemma_fallback",
        }
    except Exception as e:
        return _empty_result(f"Gemma 폴백 실패: {e}")


# ── 머지 헬퍼 ─────────────────────────────────────────────────────────────────


def _count_filled(sections: dict) -> int:
    return sum(1 for k in _EMPTY_SECTIONS if (sections.get(k) or "").strip())


def _build_result(
    sections: dict,
    brief: str,
    source: str,
    ka_attempted: bool,
    ka_filled: int,
    gemma_filled: int,
) -> dict:
    """완성된 4섹션을 result dict로 포장."""
    if source == "koalpaca":
        msg = "섹션 채움 — koalpaca 4/4"
    elif ka_attempted:
        msg = f"섹션 채움 — gemma {gemma_filled}/4 (koalpaca {ka_filled}/4 부분 응답은 무시)"
    else:
        msg = f"섹션 채움 — gemma {gemma_filled}/4 (koalpaca 미시도)"

    return {
        "ok": any(sections.values()),
        "status": "success" if any(sections.values()) else "error",
        "message": msg,
        "text": _sections_to_text(sections),
        "brief": brief,
        "sections": sections,
        "source": source,
        "koalpaca_attempted": ka_attempted,
        "koalpaca_sections_filled": ka_filled,
        "gemma_sections_filled": gemma_filled,
    }


# ── 공개 진입점 ────────────────────────────────────────────────────────────────


def summarize(script: str) -> dict:
    """F3 요약. KoAlpaca 항상 1차 시도하되 **4/4 응답 아니면 Gemma 결과 사용**.

    KoAlpaca가 학습 분포 mismatch로 partial 응답(1~3섹션) 시 PHQ-9 카탈로그 같은
    일반 출력일 가능성이 높아 신뢰 불가 → 무시하고 Gemma 결과 전적 사용.

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

    # 1) KoAlpaca 항상 1차 시도 (길이 무관)
    ka_result = _try_koalpaca(capped)
    ka_attempted = ka_result is not None
    ka_sections = (ka_result or {}).get("sections", dict(_EMPTY_SECTIONS))
    ka_filled = _count_filled(ka_sections)

    # 2) KoAlpaca 4/4 완전 응답이면 그대로 사용 (Gemma 호출 skip)
    if ka_filled == 4:
        brief = (ka_result.get("brief") or _gen_brief(transcript)) if ka_result else ""
        return _build_result(
            ka_sections, brief=brief, source="koalpaca",
            ka_attempted=True, ka_filled=4, gemma_filled=0,
        )

    # 3) KoAlpaca 부분 응답 또는 None — 신뢰 못 함, Gemma 전체 사용
    gemma_result = _try_gemma_fallback(transcript)
    gemma_sections = (gemma_result or {}).get("sections", dict(_EMPTY_SECTIONS))
    gemma_filled = _count_filled(gemma_sections)

    if gemma_filled == 0:
        return _empty_result("KoAlpaca·Gemma 모두 4섹션 추출 실패")

    return _build_result(
        gemma_sections,
        brief=(gemma_result or {}).get("brief", ""),
        source=("gemma" if ka_attempted else "gemma_only"),
        ka_attempted=ka_attempted,
        ka_filled=ka_filled,
        gemma_filled=gemma_filled,
    )
