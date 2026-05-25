"""src/factor_extractor.py

28요인 추출 전용 모듈.

현재 목적:
- 기존 app.py 내부 MockFactorExtractor를 나중에 이 파일로 옮기기 위한 준비.
- Gemini API 기반 28요인 추출 자리를 만들어둔다.
- GEMINI_API_KEY가 없으면 앱이 깨지지 않고 not_configured 상태를 반환한다.
- Gemini 응답은 반드시 28요인 JSON으로 파싱하는 구조를 목표로 한다.

주의:
- 실제 Gemini API 키는 GitHub 코드에 쓰지 않는다.
- Streamlit Cloud에서는 st.secrets 또는 환경변수에서 값을 읽는다.
"""

from __future__ import annotations

import json
import os
import re
from typing import Dict, Any

import requests


FACTOR_KEYS = [
    "depressive_mood",
    "worthlessness",
    "guilt",
    "impaired_cognition",
    "suicidal",
    "anhedonia",
    "psychomotor_changes",
    "weight_appetite",
    "sleep_disturbance",
    "fatigue",
    "anxiety",
    "loss_of_control",
    "social_avoidance",
    "physical_symptom",
    "craving",
    "withdrawal",
    "tolerance",
    "social_problem",
    "sympathy_support",
    "clarification_reflection",
    "cognitive_restructuring",
    "information_provision",
    "goal_setting",
    "task_assignment",
    "behavioral_intervention",
    "coping_skill_training",
    "structuring",
    "motivation_for_change",
]


def _get_secret(key: str, default: str = "") -> str:
    """
    Streamlit Cloud에서는 st.secrets를 우선 사용하고,
    로컬 실행에서는 환경변수를 사용한다.
    """
    try:
        import streamlit as st

        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass

    return os.getenv(key, default)


def get_gemini_config() -> Dict[str, str]:
    """
    Gemini API 연결 설정을 반환한다.
    """
    return {
        "api_key": _get_secret("GEMINI_API_KEY", "").strip(),
        "model": _get_secret("GEMINI_MODEL", "gemini-2.5-flash").strip(),
    }


def is_gemini_configured() -> bool:
    """
    Gemini API Key가 설정되어 있는지 확인한다.
    """
    return bool(get_gemini_config()["api_key"])


def _empty_factor_scores() -> Dict[str, int]:
    """
    모든 28요인을 0으로 초기화한다.
    """
    return {key: 0 for key in FACTOR_KEYS}


def extract_factors_mock(script: str, classification: Dict[str, int] | None = None) -> Dict[str, Any]:
    """
    기존 app.py의 MockFactorExtractor와 같은 역할을 하는 mock 함수.

    반환 형식은 Gemini 결과와 동일하게 맞춘다.
    """
    text = script.lower()

    factors = _empty_factor_scores()

    factors.update(
        {
            "depressive_mood": 2 if "우울" in text or "아무것도 하기 싫" in text else 0,
            "worthlessness": 2 if "일을 잘 못" in text or "내가 문제" in text else 0,
            "guilt": 0,
            "impaired_cognition": 2 if "집중" in text else 0,
            "suicidal": 1 if "죽고" in text or "자살" in text or "사라지고" in text else 0,
            "anhedonia": 2 if "아무것도 하기 싫" in text or "흥미" in text else 0,
            "psychomotor_changes": 0,
            "weight_appetite": 0,
            "sleep_disturbance": 3 if "잠" in text or "수면" in text else 0,
            "fatigue": 3 if "피곤" in text or "힘들" in text else 0,
            "anxiety": 3 if "불안" in text or "가슴이 답답" in text else 0,
            "loss_of_control": 1 if "통제" in text else 0,
            "social_avoidance": 2 if "피하게" in text or "만나는 것도" in text else 0,
            "physical_symptom": 2 if "가슴이 답답" in text or "두근" in text else 0,
            "craving": 2 if "술 생각" in text or "하고 싶어" in text else 0,
            "withdrawal": 0,
            "tolerance": 0,
            "social_problem": 1 if "회사" in text or "사람" in text else 0,
            "sympathy_support": 2,
            "clarification_reflection": 1,
            "cognitive_restructuring": 1,
            "information_provision": 0,
            "goal_setting": 1,
            "task_assignment": 0,
            "behavioral_intervention": 1,
            "coping_skill_training": 1,
            "structuring": 1,
            "motivation_for_change": 1,
        }
    )

    return {
        "ok": True,
        "status": "success",
        "message": "mock 28요인 추출 성공",
        "factors": factors,
        "backend": "mock",
    }


def _build_gemini_prompt(script: str, classification: Dict[str, int] | None = None) -> str:
    """
    Gemini few-shot 28요인 추출용 프롬프트를 만든다.

    핵심 원칙:
    - 진단하지 않는다.
    - 상담 발화에 근거해서만 점수화한다.
    - 각 요인은 0, 1, 2, 3 중 하나로만 출력한다.
    - JSON 외 텍스트를 출력하지 않게 지시한다.
    """
    classification = classification or {}

    return f"""
너는 심리상담 기록을 분석하는 보조 AI이다.
다음 상담 텍스트를 읽고 28개 요인을 0~3점으로 평가하라.

중요 원칙:
- 임상 진단을 확정하지 마라.
- 상담 텍스트에 직접 근거가 있는 내용만 반영하라.
- 점수는 반드시 0, 1, 2, 3 중 하나여야 한다.
- 출력은 JSON 객체만 반환하라.
- 설명 문장, 마크다운, 코드블록은 출력하지 마라.

점수 기준:
0 = 해당 근거 없음
1 = 약하게 언급됨
2 = 비교적 뚜렷하게 나타남
3 = 강하게 또는 반복적으로 나타남

분류 참고값:
{json.dumps(classification, ensure_ascii=False)}

반드시 아래 key를 모두 포함하라:
{json.dumps(FACTOR_KEYS, ensure_ascii=False)}

예시 1:
입력: "요즘 잠을 잘 못 자고 계속 피곤해요."
출력:
{{
  "depressive_mood": 0,
  "worthlessness": 0,
  "guilt": 0,
  "impaired_cognition": 0,
  "suicidal": 0,
  "anhedonia": 0,
  "psychomotor_changes": 0,
  "weight_appetite": 0,
  "sleep_disturbance": 3,
  "fatigue": 2,
  "anxiety": 0,
  "loss_of_control": 0,
  "social_avoidance": 0,
  "physical_symptom": 0,
  "craving": 0,
  "withdrawal": 0,
  "tolerance": 0,
  "social_problem": 0,
  "sympathy_support": 0,
  "clarification_reflection": 0,
  "cognitive_restructuring": 0,
  "information_provision": 0,
  "goal_setting": 0,
  "task_assignment": 0,
  "behavioral_intervention": 0,
  "coping_skill_training": 0,
  "structuring": 0,
  "motivation_for_change": 0
}}

예시 2:
입력: "상담사가 내담자의 감정을 반영하고 다음 주 목표를 함께 정했다."
출력:
{{
  "depressive_mood": 0,
  "worthlessness": 0,
  "guilt": 0,
  "impaired_cognition": 0,
  "suicidal": 0,
  "anhedonia": 0,
  "psychomotor_changes": 0,
  "weight_appetite": 0,
  "sleep_disturbance": 0,
  "fatigue": 0,
  "anxiety": 0,
  "loss_of_control": 0,
  "social_avoidance": 0,
  "physical_symptom": 0,
  "craving": 0,
  "withdrawal": 0,
  "tolerance": 0,
  "social_problem": 0,
  "sympathy_support": 2,
  "clarification_reflection": 2,
  "cognitive_restructuring": 0,
  "information_provision": 0,
  "goal_setting": 2,
  "task_assignment": 0,
  "behavioral_intervention": 0,
  "coping_skill_training": 0,
  "structuring": 1,
  "motivation_for_change": 1
}}

분석할 상담 텍스트:
{script}

JSON 출력:
""".strip()


def _extract_json_object(text: str) -> Dict[str, Any]:
    """
    Gemini 응답에서 JSON 객체만 추출한다.
    모델이 실수로 ```json 코드블록을 붙여도 최대한 복구한다.
    """
    cleaned = text.strip()

    cleaned = cleaned.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        raise ValueError("Gemini 응답에서 JSON 객체를 찾지 못했습니다.")

    return json.loads(match.group(0))


def _normalize_factor_scores(raw_scores: Dict[str, Any]) -> Dict[str, int]:
    """
    Gemini 응답을 28개 key, 0~3 정수 점수로 정규화한다.
    """
    normalized = _empty_factor_scores()

    for key in FACTOR_KEYS:
        value = raw_scores.get(key, 0)

        try:
            value = int(value)
        except Exception:
            value = 0

        if value < 0:
            value = 0
        if value > 3:
            value = 3

        normalized[key] = value

    return normalized


def extract_factors_gemini(script: str, classification: Dict[str, int] | None = None) -> Dict[str, Any]:
    """
    Gemini API를 사용해 28요인을 추출한다.

    Gemini REST API generateContent 엔드포인트를 사용한다.
    """
    config = get_gemini_config()
    api_key = config["api_key"]
    model = config["model"]

    if not api_key:
        return {
            "ok": False,
            "status": "not_configured",
            "message": "Gemini API Key가 아직 설정되지 않았습니다.",
            "factors": _empty_factor_scores(),
            "backend": "gemini_api",
        }

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key,
    }

    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": _build_gemini_prompt(script, classification)
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.0,
            "responseMimeType": "application/json",
        },
    }

    try:
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=90,
        )
        response.raise_for_status()

        data = response.json()

        candidates = data.get("candidates", [])
        if not candidates:
            return {
                "ok": False,
                "status": "error",
                "message": "Gemini API 응답에 candidates가 없습니다.",
                "factors": _empty_factor_scores(),
                "backend": "gemini_api",
            }

        parts = candidates[0].get("content", {}).get("parts", [])
        if not parts:
            return {
                "ok": False,
                "status": "error",
                "message": "Gemini API 응답에 content.parts가 없습니다.",
                "factors": _empty_factor_scores(),
                "backend": "gemini_api",
            }

        raw_text = parts[0].get("text", "")
        raw_json = _extract_json_object(raw_text)
        factors = _normalize_factor_scores(raw_json)

        return {
            "ok": True,
            "status": "success",
            "message": "Gemini API 28요인 추출 성공",
            "factors": factors,
            "backend": "gemini_api",
        }

    except requests.exceptions.Timeout:
        return {
            "ok": False,
            "status": "timeout",
            "message": "Gemini API 응답 시간이 초과되었습니다.",
            "factors": _empty_factor_scores(),
            "backend": "gemini_api",
        }

    except requests.exceptions.ConnectionError:
        return {
            "ok": False,
            "status": "connection_error",
            "message": "Gemini API에 연결할 수 없습니다.",
            "factors": _empty_factor_scores(),
            "backend": "gemini_api",
        }

    except Exception as error:
        return {
            "ok": False,
            "status": "error",
            "message": f"Gemini API 28요인 추출 중 오류가 발생했습니다: {error}",
            "factors": _empty_factor_scores(),
            "backend": "gemini_api",
        }


def extract_factors(
    script: str,
    classification: Dict[str, int] | None = None,
    backend: str = "mock",
) -> Dict[str, Any]:
    """
    28요인 추출 통합 진입점.

    backend:
    - mock
    - gemini_api
    """
    if backend == "gemini_api":
        return extract_factors_gemini(script, classification)

    return extract_factors_mock(script, classification)
