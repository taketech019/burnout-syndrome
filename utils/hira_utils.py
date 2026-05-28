from pathlib import Path
import pandas as pd


# =========================
# 1. 경로 설정
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HIRA_PATH = PROJECT_ROOT / "data" / "processed" / "hira" / "hira_model_context.csv"


# =========================
# 2. 나이 → 연령구분 변환 함수
# =========================

def age_to_age_group(age: int) -> str:
    """
    상담 데이터의 숫자 나이를 HIRA 연령구분으로 변환한다.
    예: 32 → 30~39세
    """
    if age < 0:
        return "unknown"

    if age >= 100:
        return "100세이상"

    start = (age // 10) * 10
    end = start + 9

    return f"{start}~{end}세"


# =========================
# 3. 질환 키 → HIRA 질환명 변환
# =========================

CONTEXT_TO_DISEASE = {
    "depression": "우울증",
    "anxiety": "불안장애",
    "sleep": "불면증",
    "adhd": "ADHD",
    "bipolar": "조울증",
    "schizophrenia": "조현병",
}


# =========================
# 4. HIRA 매칭 함수
# =========================

from pathlib import Path
import pandas as pd


def _load_hira_df():
    project_root = Path(__file__).resolve().parents[1]
    path = project_root / "data" / "processed" / "hira" / "hira_model_context.csv"
    df = pd.read_csv(path, encoding="utf-8-sig")

    for col in ["disease", "sido", "sigungu", "gender", "age_group", "context_key"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    if "context_key" not in df.columns:
        disease_to_context_key = {
            "우울증": "depression",
            "불안장애": "anxiety",
            "불면증": "sleep",
            "ADHD": "adhd",
            "조울증": "bipolar",
            "조현병": "schizophrenia",
        }
        df["context_key"] = df["disease"].map(disease_to_context_key).fillna("etc")

    df["gender"] = df["gender"].replace(
        {
            "남자": "남",
            "남성": "남",
            "여자": "여",
            "여성": "여",
        }
    )

    return df


def infer_hira_context_keys(result):
    classification = result.get("classification", {})
    factors = result.get("factors", {})

    keys = []

    if int(classification.get("depression", 0)) == 1:
        keys.append("depression")

    if int(classification.get("anxiety", 0)) == 1:
        keys.append("anxiety")

    # 수면문제는 우울/불안의 보조 설명 항목으로만 추가
    if int(factors.get("sleep_disturbance", 0) or 0) > 0:
        keys.append("sleep")

    # 중독은 HIRA 주요 정신질환 ZIP에 직접 대응되는 질환명이 없으므로 여기서는 제외
    return list(dict.fromkeys(keys))

def get_hira_context(
    gender,
    sido,
    sigungu,
    age_group,
    context_keys,
):
    df = _load_hira_df()

    results = []

    for key in context_keys:
        base = df[df["context_key"] == key].copy()

        exact = base[
            (base["gender"] == gender)
            & (base["sido"] == sido)
            & (base["sigungu"] == sigungu)
            & (base["age_group"] == age_group)
        ]

        if exact.empty:
            # 시군구가 안 맞으면 시도 단위로 fallback
            exact = base[
                (base["gender"] == gender)
                & (base["sido"] == sido)
                & (base["age_group"] == age_group)
            ]

            if not exact.empty:
                exact = (
                    exact.groupby(["year", "disease", "sido", "gender", "age_group", "context_key"], as_index=False)
                    .agg(
                        {
                            "patients": "sum",
                            "visit_days": "sum",
                            "cost": "sum",
                        }
                    )
                )
                exact["sigungu"] = "시도 전체"

        if exact.empty:
            # 그래도 없으면 전국 단위 fallback
            exact = base[
                (base["gender"] == gender)
                & (base["age_group"] == age_group)
            ]

            if not exact.empty:
                exact = (
                    exact.groupby(["year", "disease", "gender", "age_group", "context_key"], as_index=False)
                    .agg(
                        {
                            "patients": "sum",
                            "visit_days": "sum",
                            "cost": "sum",
                        }
                    )
                )
                exact["sido"] = "전국"
                exact["sigungu"] = "전국"

        if exact.empty:
            results.append(
                {
                    "matched": False,
                    "context_key": key,
                    "message": f"{key} 조건에 맞는 HIRA 통계를 찾지 못했습니다.",
                }
            )
            continue

        row = exact.iloc[0]

        patients = int(row.get("patients", 0))
        visit_days = int(row.get("visit_days", 0))
        cost = int(row.get("cost", 0))

        results.append(
            {
                "matched": True,
                "context_key": key,
                "year": int(row.get("year", 2024)),
                "disease": row.get("disease", ""),
                "sido": row.get("sido", ""),
                "sigungu": row.get("sigungu", ""),
                "gender": row.get("gender", ""),
                "age_group": row.get("age_group", ""),
                "patients": patients,
                "visit_days": visit_days,
                "cost": cost,
                "visit_days_per_patient": visit_days / patients if patients > 0 else None,
                "cost_per_patient": cost / patients if patients > 0 else None,
            }
        )

    return results

# =========================
# 5. 보고서용 문장 생성 함수
# =========================

def format_hira_report_text(hira_results: list[dict]) -> str:
    """
    HIRA 매칭 결과를 보고서에 넣기 좋은 문장으로 변환한다.
    """

    lines = []

    for item in hira_results:
        if not item.get("matched"):
            lines.append(
                f"- {item['disease']}: 일치하는 HIRA 통계를 찾지 못했습니다."
            )
            continue

        patients = f"{item['patients']:,}"
        visit_days = f"{item['visit_days']:,}"
        cost = f"{item['cost']:,}"

        visit_days_per_patient = item.get("visit_days_per_patient")
        cost_per_patient = item.get("cost_per_patient")

        if visit_days_per_patient is not None:
            visit_days_per_patient = f"{visit_days_per_patient:.2f}"
        else:
            visit_days_per_patient = "계산 불가"

        if cost_per_patient is not None:
            cost_per_patient = f"{cost_per_patient:,.0f}"
        else:
            cost_per_patient = "계산 불가"

        line = (
            f"- {item['year']}년 HIRA 정신질환 진료 통계 기준, "
            f"{item['sido']} {item['sigungu']} 소재 요양기관에서 "
            f"{item['age_group']} {item['gender']}의 {item['disease']} 진료 환자수는 "
            f"{patients}명, 입내원일수는 {visit_days}일, "
            f"요양급여비용은 {cost}원으로 집계되었습니다. "
            f"환자 1인당 평균 입내원일수는 약 {visit_days_per_patient}일, "
            f"환자 1인당 평균 요양급여비용은 약 {cost_per_patient}원입니다."
        )

        if item.get("is_suppressed_or_zero"):
            line += (
                " 단, 해당 값은 0으로 표시되어 있으며, 실제 환자 없음이 아니라 "
                "통계 비식별 처리 또는 집계 기준상 0으로 기재된 값일 수 있습니다."
            )

        lines.append(line)

    caution = (
        "\n\n※ 주의: 위 HIRA 통계는 요양기관 소재지 기준의 공공 진료 통계이며, "
        "개별 내담자의 진단, 중증도, 위험도 판단 근거로 사용하지 않습니다. "
        "상담 보고서에서는 지역·성별·연령대별 참고 맥락으로만 활용해야 합니다."
    )

    return "\n".join(lines) + caution

def infer_hira_context_keys(analysis_result: dict) -> list[str]:
    """
    상담 분석 결과를 바탕으로 HIRA 통계 조회에 사용할 context_key를 자동 추출한다.

    현재 app.py의 run_analysis() 결과 구조:
    {
        "classification": {
            "depression": 0/1,
            "anxiety": 0/1,
            "addiction": 0/1
        },
        "factors": {
            "sleep_disturbance": 0~3,
            ...
        }
    }

    반환 예:
    ["depression", "sleep"]
    """

    if not isinstance(analysis_result, dict):
        return []

    context_keys = []

    classification = analysis_result.get("classification", {})
    factors = analysis_result.get("factors", {})

    # 혹시 과거 테스트용 구조가 들어와도 처리
    legacy_symptom_factor = analysis_result.get("symptom_factor", {})

    # 1. 우울/불안 판별 결과 기준
    if classification.get("depression", 0) > 0 or analysis_result.get("depression", 0) > 0:
        context_keys.append("depression")

    if classification.get("anxiety", 0) > 0 or analysis_result.get("anxiety", 0) > 0:
        context_keys.append("anxiety")

    # 2. 중독은 현재 HIRA 주요 정신질환 통계의 질환명과 직접 매칭하지 않음
    # addiction > 0이어도 HIRA context_key에는 넣지 않는다.

    # 3. 수면 문제는 HIRA의 불면증 통계와 연결
    if factors.get("sleep_disturbance", 0) > 0:
        context_keys.append("sleep")

    if legacy_symptom_factor.get("sleep_disturbance", 0) > 0:
        context_keys.append("sleep")

    if analysis_result.get("sleep_disturbance", 0) > 0:
        context_keys.append("sleep")

    # 4. 중복 제거, 순서 유지
    return list(dict.fromkeys(context_keys))