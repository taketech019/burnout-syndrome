"""src/factor_extractor.py — F1 2단계: 28요인 0~3 점수 추출 (Gemma 4 31B).

새 UI 정의 28키 (영문) + 한국어 표시명. Gemma 단일 호출로 점수 산출.
신규 시그니처: extract_factors(script, classification, backend) -> dict
"""
import logging
from typing import Optional

from src.gemma_client import generate_json

log = logging.getLogger(__name__)


FACTOR_KEYS = [
    # 우울 10
    "depressive_mood", "worthlessness", "guilt", "impaired_cognition",
    "suicidal", "anhedonia", "psychomotor_changes", "weight_appetite",
    "sleep_disturbance", "fatigue",
    # 불안 4
    "anxiety", "loss_of_control", "social_avoidance", "physical_symptom",
    # 중독 4
    "craving", "withdrawal", "tolerance", "social_problem",
    # 상담사 개입 9
    "sympathy_support", "clarification_reflection", "cognitive_restructuring",
    "information_provision", "goal_setting", "task_assignment",
    "behavioral_intervention", "coping_skill_training", "structuring",
    # 변화 1
    "motivation_for_change",
]
assert len(FACTOR_KEYS) == 28

FACTOR_LABELS = {
    "depressive_mood": "우울한 기분",
    "worthlessness": "무가치감",
    "guilt": "죄책감",
    "impaired_cognition": "사고력/집중력 저하",
    "suicidal": "자살 관련 사고",
    "anhedonia": "흥미 감소",
    "psychomotor_changes": "정신운동 변화",
    "weight_appetite": "체중/식욕 변화",
    "sleep_disturbance": "수면 문제",
    "fatigue": "피로감",
    "anxiety": "불안감",
    "loss_of_control": "통제감 상실",
    "social_avoidance": "사회적 회피",
    "physical_symptom": "신체 증상",
    "craving": "갈망",
    "withdrawal": "금단",
    "tolerance": "내성",
    "social_problem": "사회적 문제",
    "sympathy_support": "공감 및 지지",
    "clarification_reflection": "명료화 및 반영",
    "cognitive_restructuring": "인지 재구성",
    "information_provision": "정보 제공",
    "goal_setting": "목표 설정",
    "task_assignment": "과제 부여",
    "behavioral_intervention": "행동 개입",
    "coping_skill_training": "대처기술 훈련",
    "structuring": "구조화",
    "motivation_for_change": "변화 동기",
}


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

**28 요인** (영문 키 = 한국어 의미)
{labels}

**상담 텍스트**
{text}

**출력 형식**: 단일 JSON 객체로만. 설명·코드 블록·메타 설명 금지.
{{"depressive_mood": 0~3, "worthlessness": 0~3, ..., "motivation_for_change": 0~3}}
28개 키가 정확히 모두 포함되어야 합니다.
"""


def _build_prompt(script: str) -> str:
    lines = "\n".join(f"- {k}: {FACTOR_LABELS[k]}" for k in FACTOR_KEYS)
    return _PROMPT.replace("{labels}", lines).replace("{text}", script[:8000])


def extract_factors(
    script: str,
    classification: Optional[dict] = None,
    backend: str = "gemini_api",
) -> dict:
    """28요인 0~3 점수 추출. backend는 정보용 (현재는 항상 Gemma)."""
    if not script or not script.strip():
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
