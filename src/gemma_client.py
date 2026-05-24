"""src/gemma_client.py — Google AI Studio Gemma 4 31B 공통 클라이언트.

Gemma는 Gemini와 달리 `response_mime_type: application/json`을 지원하지 않고
체인-오브-쏘트로 reasoning 후 답을 내놓는 경향이 강하다. 따라서:
  1) 강제 JSON 프롬프트 (코드 블록 금지·설명 금지) + 재시도
  2) 응답에서 첫 번째 균형 잡힌 `{...}` 또는 `[...]` 블록을 정규식·괄호 매칭으로 추출
  3) 500/transient 에러는 짧은 backoff 재시도

Gemini 모델로 폴백 가능 (GEMINI_FALLBACK_MODEL). 기본 OFF.
"""
import json
import logging
import re
import time
from typing import Any, Optional

import google.generativeai as genai

from config import GEMINI_API_KEY, GEMINI_MODEL, GEMINI_FALLBACK_MODEL

log = logging.getLogger(__name__)

_INITIALIZED = False


def _init() -> None:
    global _INITIALIZED
    if not _INITIALIZED and GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        _INITIALIZED = True


def _strip_code_fence(text: str) -> str:
    """```json ... ``` 또는 ``` ... ``` 코드 블록 제거."""
    text = text.strip()
    m = re.match(r"^```(?:json|JSON)?\s*\n?(.*?)\n?```$", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text


def _extract_balanced(text: str, open_ch: str, close_ch: str) -> Optional[str]:
    """첫 번째 균형 잡힌 {...} 또는 [...] 블록 추출."""
    start = text.find(open_ch)
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == open_ch:
            depth += 1
        elif c == close_ch:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def extract_json(text: str) -> Optional[Any]:
    """Gemma 응답 본문에서 JSON 추출. 실패 시 None."""
    cleaned = _strip_code_fence(text)
    for raw in (cleaned, text):
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            pass
        for opener, closer in (("{", "}"), ("[", "]")):
            block = _extract_balanced(raw, opener, closer)
            if block:
                try:
                    return json.loads(block)
                except json.JSONDecodeError:
                    continue
    return None


def generate(
    prompt: str,
    *,
    temperature: float = 0.0,
    max_output_tokens: int = 2048,
    model_name: Optional[str] = None,
    max_retries: int = 3,
) -> str:
    """Gemma 4 31B 생성. 5xx/transient 에러 재시도 후, 폴백 모델 1회 시도."""
    _init()
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY 미설정")

    last_err: Optional[Exception] = None
    for attempt_model in (model_name or GEMINI_MODEL, GEMINI_FALLBACK_MODEL):
        if not attempt_model:
            continue
        for attempt in range(max_retries):
            try:
                m = genai.GenerativeModel(attempt_model)
                r = m.generate_content(
                    prompt,
                    generation_config={
                        "temperature": temperature,
                        "max_output_tokens": max_output_tokens,
                    },
                )
                if not r.candidates or not r.candidates[0].content.parts:
                    raise RuntimeError("Gemma 응답이 비어 있음 (필터/차단 가능성)")
                return r.text
            except Exception as e:
                msg = str(e)
                last_err = e
                if "429" in msg or "RESOURCE_EXHAUSTED" in msg or "quota" in msg.lower():
                    raise
                log.warning("Gemma 시도 실패 (%s, attempt %d): %s", attempt_model, attempt, msg[:120])
                time.sleep(1 + attempt * 2)
    raise RuntimeError(f"Gemma 호출 최종 실패: {last_err}")


def generate_json(prompt: str, **kwargs) -> Optional[Any]:
    """JSON 모드. 추출 실패 시 None — 호출자가 에러 처리."""
    raw = generate(prompt, **kwargs)
    return extract_json(raw)


_REASONING_BULLET = re.compile(r"^\s*\*\s+[A-Za-z][A-Za-z\s]+\d*\s*:.*$", re.MULTILINE)
_CONSTRAINT_LINE = re.compile(
    r"^\s*(\*|\d+\.)\s+(Role|Constraint|Goal|Input|Output|Task|Step|Note|Plan|Reasoning|Format|Draft|Material|Question|Answer|Context|Source|Reference|Analysis|Approach|Strategy)[\s\d]*[:.].*$",
    re.MULTILINE,
)


def _looks_korean(s: str, threshold: float = 0.3) -> bool:
    """한글 비중 threshold 이상이면 한국어로 간주."""
    if not s:
        return False
    ko = sum(1 for c in s if "가" <= c <= "힣")
    return ko / max(len(s), 1) > threshold


def strip_reasoning(text: str) -> str:
    """Gemma 4의 영어 reasoning bullet을 제거하고 한국어 답변만 남김.

    Gemma 4 31B는 프롬프트로 억제해도 `*   Role:`, `*   [Material 1]:`,
    `*   The question asks...` 등 영어 메타 추론을 bullet로 출력. 본 함수는:
      1. 영어 bullet 라인(`* `, `- `로 시작하면서 한글 비율 < 30%) 제거
      2. 빈 라인 정리
      3. 결과가 비어 있으면 원본 마지막 한국어 단락 fallback
    """
    text = text.strip()
    lines = text.splitlines()
    keep = []
    for raw in lines:
        s = raw.rstrip()
        stripped = s.lstrip()
        # 모든 bullet 라인 (`* `, `- `, `• `, `1. `) 검사
        if stripped.startswith(("* ", "*\t", "- ", "•", "1.", "2.", "3.", "4.", "5.")):
            if not _looks_korean(stripped):
                continue
            # 한국어 bullet은 유지하되 bullet 마커 제거 (가독성)
            keep.append(stripped)
            continue
        # 영어 단락도 제거 (한국어 비율 매우 낮으면)
        if stripped and not _looks_korean(stripped, threshold=0.15) and len(stripped) > 20:
            continue
        keep.append(s)

    # 연속 빈 줄 합치기
    out_lines = []
    prev_empty = False
    for ln in keep:
        is_empty = not ln.strip()
        if is_empty and prev_empty:
            continue
        out_lines.append(ln)
        prev_empty = is_empty

    result = "\n".join(out_lines).strip()
    if result:
        return result

    # Fallback: 원본에서 한국어 단락만 추출
    korean_paragraphs = [p for p in re.split(r"\n\s*\n", text) if _looks_korean(p)]
    return "\n\n".join(korean_paragraphs) if korean_paragraphs else text
