"""src/factor_extractor.py — F1 2단계: 28요인 0~3 점수 추출 (Gemma 4 31B).

사용자 제공 디자인의 SYMPTOM_SCORES 체계 (한국어 키, 5 카테고리).
- 우울 9 / 우울·위험 1 / 불안 8 / 중독 7 / 중독·기능 3 = 28
"""
import logging
from typing import Optional

from src.gemma_client import generate_json

log = logging.getLogger(__name__)


FACTOR_KEYS = [
    # 우울 9
    "우울한 기분", "무가치감", "죄책감", "사고력 저하", "흥미감소",
    "정신운동변화", "체중/식욕변화", "수면문제", "피로감",
    # 우울/위험 1
    "자살생각",
    # 불안 8
    "불안감", "비현실감", "통제력상실감", "불안조절곤란", "집중력저하",
    "사회적상황회피", "신체증상", "과민성",
    # 중독 7
    "조절실패", "갈망", "거짓말", "내성", "금단", "현저성", "자원투자",
    # 중독/기능 3
    "자기관리", "사회적문제발생", "부정적 결과",
]
assert len(FACTOR_KEYS) == 28

FACTOR_CATEGORIES = {
    **{k: "우울" for k in FACTOR_KEYS[:9]},
    "자살생각": "우울/위험",
    **{k: "불안" for k in FACTOR_KEYS[10:18]},
    **{k: "중독" for k in FACTOR_KEYS[18:25]},
    **{k: "중독/기능" for k in FACTOR_KEYS[25:28]},
}

# FACTOR_LABELS: 한국어 키 그대로 (외부 코드 호환 alias)
FACTOR_LABELS = {k: k for k in FACTOR_KEYS}


def _zero_factors() -> dict:
    return {k: 0 for k in FACTOR_KEYS}


def _empty_result(message: str, backend: str, ok: bool = False) -> dict:
    return {
        "ok": ok,
        "status": "error" if not ok else "success",
        "message": message,
        "backend": backend,
        "factors": _zero_factors(),
    }


_PROMPT = """당신은 임상심리 전문가입니다. 다음 상담 텍스트를 보고 28개 요인 각각에 대해 **0~3 정수** 점수를 매깁니다.

**점수 가이드**
- 0: 해당 요인 없음
- 1: 약함 / 간헐적 언급
- 2: 중간 / 반복 언급
- 3: 강함 / 명확한 핵심 증상

**분류 원칙**
1. 본 28요인은 모두 부정적 증상요인입니다. 긍정적 회기(자기 수용·통찰·욕구 탐색)에서는 대부분 0이 정상입니다 — 무리하게 점수 부여 금지.
2. 단, 본 회기에 **실제 증상 호소가 명시되면** (예: "잠을 못 자요", "우울해요", "술을 매일 마셔요") 1~3 점수 부여.
3. transcript 끝 척도 평가 응답 부분은 이미 입력에서 제거되었으므로 신경쓸 필요 없음 — 본문 발화 내용만 보세요.
4. 점수는 내담자 발화 + 상담사가 짚어준 증상 호소에서 추출. 단순히 키워드 등장만으로 점수 매기지 말 것 (회기에서 어떻게 사용되었는지 맥락 판단).

**28 요인** (괄호 안은 카테고리)
{labels}

**상담 텍스트**
{text}

**출력 형식**: 단일 JSON 객체로만. 설명·코드 블록·메타 설명 금지.
키는 위 한국어 라벨 그대로 사용. 28개 키가 정확히 모두 포함되어야 합니다.
{{"우울한 기분": 0~3, "무가치감": 0~3, ..., "부정적 결과": 0~3}}
"""


def _build_prompt(script: str) -> str:
    lines = "\n".join(f"- {k} ({FACTOR_CATEGORIES[k]})" for k in FACTOR_KEYS)
    return _PROMPT.replace("{labels}", lines).replace("{text}", script[:8000])


def extract_factors(
    script: str,
    classification: Optional[dict] = None,
    backend: str = "gemini_api",
) -> dict:
    """28요인 0~3 점수 추출. transcript 끝 척도 영역은 모델 입력 전 분리."""
    if not script or not script.strip():
        return _empty_result("입력 텍스트 비어 있음", backend=backend)

    from src.transcript_utils import split_transcript_and_scale
    script, _scale = split_transcript_and_scale(script)
    if not script.strip():
        return _empty_result("입력 텍스트 비어 있음", backend=backend)

    try:
        data = generate_json(_build_prompt(script), temperature=0.0, max_output_tokens=2048)
    except Exception as e:
        return _empty_result(f"Gemma 28요인 호출 실패: {e}", backend=backend)
    if not isinstance(data, dict):
        return _empty_result("Gemma 28요인 JSON 파싱 실패", backend=backend)
    factors = _zero_factors()
    for k in FACTOR_KEYS:
        try:
            v = int(data.get(k, 0))
        except (TypeError, ValueError):
            v = 0
        factors[k] = max(0, min(3, v))
    return {
        "ok": True,
        "status": "success",
        "message": "Gemma 4 31B 28요인 0~3 점수",
        "backend": backend,
        "factors": factors,
    }
