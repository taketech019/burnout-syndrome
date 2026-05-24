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

def get_hira_context(
    age: int,
    gender: str,
    sido: str,
    sigungu: str,
    context_keys: list[str],
    year: int = 2024,
):
    """
    내담자 메타데이터와 상담 분석 결과를 기준으로 HIRA 통계를 찾는다.

    Parameters
    ----------
    age : int
        내담자 나이
    gender : str
        성별. 예: "남", "여"
    sido : str
        시도. 예: "서울"
    sigungu : str
        시군구. 예: "강남구"
    context_keys : list[str]
        찾고 싶은 질환 키. 예: ["depression", "sleep"]
    year : int
        기준 연도. 기본값 2024

    Returns
    -------
    list[dict]
        HIRA 매칭 결과 목록
    """

    df = pd.read_csv(HIRA_PATH, encoding="utf-8-sig")

    age_group = age_to_age_group(age)

    results = []

    for key in context_keys:
        disease = CONTEXT_TO_DISEASE.get(key)

        if disease is None:
            continue

        matched = df[
            (df["year"] == year)
            & (df["sido"] == sido)
            & (df["sigungu"] == sigungu)
            & (df["gender"] == gender)
            & (df["age_group"] == age_group)
            & (df["context_key"] == key)
        ]

        if matched.empty:
            results.append({
                "context_key": key,
                "disease": disease,
                "matched": False,
                "message": "일치하는 HIRA 통계를 찾지 못했습니다.",
            })
            continue

        row = matched.iloc[0]

        results.append({
            "context_key": key,
            "disease": row["disease"],
            "matched": True,
            "year": int(row["year"]),
            "sido": row["sido"],
            "sigungu": row["sigungu"],
            "gender": row["gender"],
            "age_group": row["age_group"],
            "patients": int(row["patients"]),
            "visit_days": int(row["visit_days"]),
            "cost": int(row["cost"]),
            "visit_days_per_patient": row["visit_days_per_patient"],
            "cost_per_patient": row["cost_per_patient"],
            "is_suppressed_or_zero": bool(row["is_suppressed_or_zero"]),
        })

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

# =========================
# 6. 테스트 실행 (앱배포 hira_utils.py에선 제거)
# =========================

if __name__ == "__main__":
    test_result = get_hira_context(
        age=32,
        gender="여",
        sido="서울",
        sigungu="강남구",
        context_keys=["depression", "sleep"],
    )

    print("\n[원본 매칭 결과]")
    for item in test_result:
        print(item)

    print("\n[보고서용 문장]")
    report_text = format_hira_report_text(test_result)
    print(report_text)