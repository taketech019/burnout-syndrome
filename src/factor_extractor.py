"""src/factor_extractor.py — F1 2단계: 발화별 64-라벨 0/1 분류 + 회기 빈도 집계.

PRD §F1 stage 2: 양성 판별된 회기에 대해 발화 단위로 4범주 라벨링.
- 증상요인 28 (우울 10 / 불안 8 / 중독 10) — primary_disease 따라 1개 세트만 사용
- 위험요인 20 (공통)
- 개선요인 5 (공통)
- 개입요인 11 (공통)

PRD가 명시한 모델은 "Gemma 4 31B"이지만 Google AI Studio가 실제로 서빙하는 모델은 Gemini 계열이라 `gemini-2.0-flash`를 기본값으로 사용. 결정 사유는 vibe-coding-report.md 참조.

라벨 이름은 AI Hub 데이터셋 명세서가 로컬에 없어 임상적으로 합리적인 구성으로 정의함. 명세서 확보 후 LABELS dict만 교체하면 됨.
"""
import json
import re
from typing import Optional

import google.generativeai as genai

from config import GEMINI_API_KEY, GEMINI_MODEL

# ── 라벨 정의 ──────────────────────────────────────────────────────────────────

SYMPTOM_LABELS = {
    "depression": [
        "우울감", "불면 또는 과수면", "식욕 또는 체중 변화",
        "무가치감 또는 죄책감", "자살 사고", "집중력 저하",
        "흥미 상실", "피로감", "정신운동 지체 또는 초조",
        "절망감",
    ],
    "anxiety": [
        "안절부절못함", "과도한 걱정", "공황 발작",
        "회피 행동", "신체 증상(심계항진/발한)",
        "수면 장애", "집중 곤란", "근육 긴장",
    ],
    "addiction": [
        "갈망", "내성 증가", "금단 증상",
        "통제력 상실", "사회 기능 손상", "위험 사용",
        "부정 및 합리화", "재발", "동반 우울/불안",
        "가족 또는 직장 갈등",
    ],
}

RISK_LABELS = [
    "가족력 정신질환", "과거 정신과 입원력", "자해/자살 시도력",
    "알코올/약물 남용", "외상 또는 학대 경험", "사회적 고립",
    "경제적 어려움", "직장 또는 학업 문제", "만성 신체질환",
    "최근 상실 경험", "양육 스트레스", "가족 갈등",
    "만성 통증", "수면 박탈", "만성 스트레스 노출",
    "약물 부작용 또는 의존", "폭력 노출", "신체적 학대 경험",
    "자존감 저하", "부정적 사고 패턴",
]

IMPROVEMENT_LABELS = [
    "사회적 지지 활용", "자기 인식 향상", "대처 기술 습득",
    "약물 순응 또는 치료 지속", "일상 활동 회복",
]

INTERVENTION_LABELS = [
    "인지 재구조화", "행동 활성화", "노출 치료",
    "이완 훈련", "마음챙김", "문제 해결 훈련",
    "자살 예방 계약", "가족 상담 권유", "약물 치료 의뢰",
    "위기 자원 안내", "다음 회기 과제 부여",
]


def get_all_labels(primary_disease: str) -> dict[str, list[str]]:
    """primary_disease에 맞는 증상요인 + 공통 라벨 세트 반환."""
    if primary_disease not in SYMPTOM_LABELS:
        primary_disease = "depression"
    return {
        "symptom_factor": SYMPTOM_LABELS[primary_disease],
        "risk_factor": RISK_LABELS,
        "improvement_factor": IMPROVEMENT_LABELS,
        "intervention_factor": INTERVENTION_LABELS,
    }


# ── 발화 분리 ──────────────────────────────────────────────────────────────────

# "상담사: ..." 또는 "내담자: ..." 형식의 한 라인 = 한 발화
_UTTERANCE_LINE = re.compile(r"^\s*([가-힣]+)\s*:\s*(.+?)\s*$", re.MULTILINE)


def parse_utterances(text: str) -> list[dict]:
    """발화자/텍스트 한 줄씩 추출. 발화 없으면 빈 리스트."""
    out = []
    for m in _UTTERANCE_LINE.finditer(text):
        speaker, content = m.group(1), m.group(2)
        if content.strip():
            out.append({"speaker": speaker, "text": content.strip()})
    return out


# ── Gemini 호출 ────────────────────────────────────────────────────────────────

_PROMPT_TEMPLATE = """당신은 임상심리 전문가입니다. 다음 상담 회기의 각 발화에 대해 4범주 라벨을 0/1로 분류해 주세요.

**라벨 정의 (총 {n_total}개)**
- 증상요인 ({n_symptom}개, {disease}): {symptom_labels}
- 위험요인 ({n_risk}개): {risk_labels}
- 개선요인 ({n_improvement}개): {improvement_labels}
- 개입요인 ({n_intervention}개): {intervention_labels}

**분류 원칙**
- 각 발화에 해당 라벨이 명확히 드러나면 1, 아니면 0.
- 발화 자체 내용 + 직전·직후 1~2개 발화의 맥락으로 판단.
- 상담사 발화에서 "개입요인"이, 내담자 발화에서 "증상/위험/개선요인"이 주로 등장.
- 불확실하면 0.

**출력 형식** (JSON only, 발화 순서 유지):
{{
  "utterances": [
    {{
      "idx": 0,
      "speaker": "...",
      "text": "...",
      "symptom_factor": [0, 1, 0, ...],
      "risk_factor": [0, 0, 1, ...],
      "improvement_factor": [0, 0, ...],
      "intervention_factor": [0, 0, ...]
    }},
    ...
  ]
}}

**분석할 발화 목록** ({n_utterances}개):
{utterances_json}
"""


def _build_prompt(utterances: list[dict], primary_disease: str) -> str:
    labels = get_all_labels(primary_disease)
    return _PROMPT_TEMPLATE.format(
        n_total=sum(len(v) for v in labels.values()),
        n_symptom=len(labels["symptom_factor"]),
        disease=primary_disease,
        symptom_labels=labels["symptom_factor"],
        n_risk=len(labels["risk_factor"]),
        risk_labels=labels["risk_factor"],
        n_improvement=len(labels["improvement_factor"]),
        improvement_labels=labels["improvement_factor"],
        n_intervention=len(labels["intervention_factor"]),
        intervention_labels=labels["intervention_factor"],
        n_utterances=len(utterances),
        utterances_json=json.dumps(
            [{"idx": i, **u} for i, u in enumerate(utterances)],
            ensure_ascii=False,
            indent=2,
        ),
    )


# ── 빈도 집계 ──────────────────────────────────────────────────────────────────

def _aggregate(utterance_labels: list[dict], primary_disease: str) -> dict:
    """발화 단위 0/1 → 회기 단위 빈도 (등장 횟수 + 비율)."""
    labels = get_all_labels(primary_disease)
    n_utt = max(len(utterance_labels), 1)

    freq = {}
    for category, names in labels.items():
        cat_freq = []
        for j, name in enumerate(names):
            count = sum(1 for u in utterance_labels if u.get(category, [0] * len(names))[j] == 1)
            cat_freq.append({
                "label": name,
                "count": count,
                "ratio": round(count / n_utt, 3),
            })
        freq[category] = cat_freq
    return freq


# ── 공개 진입점 ────────────────────────────────────────────────────────────────

def extract_factors(transcript: str, primary_disease: str = "depression") -> dict:
    """발화 단위 28+20+5+11 분류 + 회기 빈도 집계.

    반환:
    {
      "primary_disease": "depression",
      "utterance_count": N,
      "utterances": [...],          # 발화별 라벨 결과 (0/1 배열 포함)
      "frequency": {                # 라벨별 등장 횟수 + 비율
        "symptom_factor": [{"label":..., "count":..., "ratio":...}, ...],
        ...
      },
      "raw": "...",                  # 모델 응답 원문 (실패 시 진단용)
      "error": "..." (실패 시)
    }
    """
    if not GEMINI_API_KEY:
        return {"error": "GEMINI_API_KEY 미설정", "frequency": _aggregate([], primary_disease)}

    utterances = parse_utterances(transcript)
    if not utterances:
        return {
            "error": "발화 추출 실패. '상담사: ... / 내담자: ...' 형식 필요.",
            "frequency": _aggregate([], primary_disease),
        }

    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(
            _build_prompt(utterances, primary_disease),
            generation_config={
                "response_mime_type": "application/json",
                "temperature": 0.1,
            },
        )
        raw = response.text
        data = json.loads(raw)
        utt_results = data.get("utterances", [])
        return {
            "primary_disease": primary_disease,
            "utterance_count": len(utt_results),
            "utterances": utt_results,
            "frequency": _aggregate(utt_results, primary_disease),
            "raw": raw[:500],   # 진단용 일부만
        }
    except json.JSONDecodeError as e:
        return {
            "error": f"Gemini JSON 파싱 실패: {e}",
            "raw": raw if "raw" in locals() else "",
            "frequency": _aggregate([], primary_disease),
        }
    except Exception as e:
        return {"error": str(e), "frequency": _aggregate([], primary_disease)}
