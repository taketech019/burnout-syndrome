from pathlib import Path
import zipfile
import pandas as pd
import json
from datetime import datetime


# =========================
# 1. 경로 설정
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "hira"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "hira"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# 2. 질환명 매핑
# =========================

DISEASE_MAP = {
    "우울증": "depression",
    "불안장애": "anxiety",
    "불면증": "sleep",
    "ADHD": "adhd",
    "조울증": "bipolar",
    "조현병": "schizophrenia",
}


# =========================
# 3. 유틸 함수
# =========================

def clean_number(value):
    """
    환자수, 입내원일수, 요양급여비용을 숫자로 변환한다.
    쉼표가 있거나 공백이 있어도 처리한다.
    """
    if pd.isna(value):
        return 0

    value = str(value).replace(",", "").strip()

    if value == "":
        return 0

    try:
        return int(float(value))
    except ValueError:
        return 0


def normalize_gender(value):
    """
    성별 값을 통일한다.
    """
    if pd.isna(value):
        return "unknown"

    value = str(value).strip()

    if value in ["남", "남자", "M", "m"]:
        return "남"
    if value in ["여", "여자", "F", "f"]:
        return "여"

    return value


def rename_columns(df):
    """
    HIRA 원본 컬럼명을 프로젝트용 영어 컬럼명으로 통일한다.
    파일에 따라 '상별구분' 또는 '상병구분'이 섞여 있을 수 있으므로 둘 다 처리한다.
    """
    column_map = {
        "진료년도": "year",
        "상별구분": "disease",
        "상병구분": "disease",
        "시도": "sido",
        "시군구": "sigungu",
        "성별": "gender",
        "연령구분": "age_group",
        "환자수": "patients",
        "입내원일수": "visit_days",
        "요양급여비용": "cost",
    }

    df = df.rename(columns=column_map)
    return df


def fix_zip_filename(name):
    """
    ZIP 내부 파일명이 한글인데 깨져 보이는 경우 복구한다.
    ZIP 파일명은 CP437로 잘못 해석된 뒤 CP949/EUC-KR 한글이 깨지는 경우가 많다.
    """
    try:
        return name.encode("cp437").decode("cp949")
    except Exception:
        return name


def read_csv_from_zip(zip_path):
    """
    ZIP 내부 CSV 파일을 읽는다.
    HIRA 공공데이터는 보통 euc-kr 또는 cp949 계열이다.
    ZIP 내부 파일명은 별도로 복구해서 source_file에 저장한다.
    """
    dataframes = []
    log_rows = []

    with zipfile.ZipFile(zip_path, "r") as zf:
        file_names = zf.namelist()

        for internal_file_name in file_names:
            if not internal_file_name.lower().endswith(".csv"):
                continue

            fixed_file_name = fix_zip_filename(internal_file_name)

            print(f"[읽는 중] {fixed_file_name}")

            with zf.open(internal_file_name) as f:
                try:
                    df = pd.read_csv(f, encoding="euc-kr")
                    encoding_used = "euc-kr"
                except UnicodeDecodeError:
                    f.seek(0)
                    df = pd.read_csv(f, encoding="cp949")
                    encoding_used = "cp949"

            df["source_file"] = fixed_file_name
            df["source_zip"] = zip_path.name

            dataframes.append(df)

            log_rows.append({
                "source_zip": zip_path.name,
                "source_file": fixed_file_name,
                "encoding": encoding_used,
                "rows": len(df),
                "columns": ", ".join(df.columns.astype(str)),
                "status": "success",
            })

    return dataframes, log_rows


# =========================
# 4. 메인 전처리
# =========================

def main():
    zip_files = list(RAW_DIR.glob("*.zip"))

    if not zip_files:
        raise FileNotFoundError(
            f"HIRA ZIP 파일을 찾지 못했습니다. 다음 폴더에 ZIP 파일을 넣어주세요: {RAW_DIR}"
        )

    all_dfs = []
    all_logs = []

    for zip_path in zip_files:
        print(f"\n[ZIP 처리 시작] {zip_path.name}")
        dfs, logs = read_csv_from_zip(zip_path)
        all_dfs.extend(dfs)
        all_logs.extend(logs)

    if not all_dfs:
        raise ValueError("ZIP 내부에서 CSV 파일을 찾지 못했습니다.")

    cleaned_tables = []

    for df in all_dfs:
        df = rename_columns(df)

        required_base_cols = [
            "year",
            "disease",
            "sido",
            "sigungu",
            "gender",
            "patients",
            "visit_days",
            "cost",
        ]

        missing_cols = [col for col in required_base_cols if col not in df.columns]

        if missing_cols:
            print(f"[건너뜀] 필수 컬럼 없음: {missing_cols}")
            continue

        # age_group이 없는 CSV도 있으므로 없으면 'all'로 채운다.
        if "age_group" not in df.columns:
            df["age_group"] = "all"

        keep_cols = [
            "year",
            "disease",
            "sido",
            "sigungu",
            "gender",
            "age_group",
            "patients",
            "visit_days",
            "cost",
            "source_file",
            "source_zip",
        ]

        df = df[keep_cols].copy()

        # 문자열 정리
        for col in ["disease", "sido", "sigungu", "gender", "age_group"]:
            df[col] = df[col].astype(str).str.strip()

        # 숫자형 정리
        df["year"] = df["year"].apply(clean_number)
        df["patients"] = df["patients"].apply(clean_number)
        df["visit_days"] = df["visit_days"].apply(clean_number)
        df["cost"] = df["cost"].apply(clean_number)

        # 성별 정규화
        df["gender"] = df["gender"].apply(normalize_gender)

        # 질환 매핑
        df["context_key"] = df["disease"].map(DISEASE_MAP).fillna("other")

        # 0명 처리 주의 플래그
        df["is_suppressed_or_zero"] = df["patients"].eq(0)

        # 1인당 지표
        df["visit_days_per_patient"] = df.apply(
            lambda row: row["visit_days"] / row["patients"] if row["patients"] > 0 else None,
            axis=1,
        )

        df["cost_per_patient"] = df.apply(
            lambda row: row["cost"] / row["patients"] if row["patients"] > 0 else None,
            axis=1,
        )

        # 매칭 키
        df["match_key"] = (
            df["year"].astype(str) + "_" +
            df["sido"] + "_" +
            df["sigungu"] + "_" +
            df["gender"] + "_" +
            df["age_group"] + "_" +
            df["context_key"]
        )

        cleaned_tables.append(df)

    if not cleaned_tables:
        raise ValueError("전처리 가능한 CSV 테이블이 없습니다.")

    hira_all = pd.concat(cleaned_tables, ignore_index=True)

    # =========================
    # 5. 파일 저장
    # =========================

    age_stats = hira_all[hira_all["age_group"] != "all"].copy()
    region_stats = hira_all[hira_all["age_group"] == "all"].copy()

    age_stats_path = PROCESSED_DIR / "hira_age_region_stats.csv"
    region_stats_path = PROCESSED_DIR / "hira_region_stats.csv"
    model_context_path = PROCESSED_DIR / "hira_model_context.csv"
    log_path = PROCESSED_DIR / "hira_preprocessing_log.csv"
    cards_path = PROCESSED_DIR / "hira_summary_cards.jsonl"

    age_stats.to_csv(age_stats_path, index=False, encoding="utf-8-sig")
    region_stats.to_csv(region_stats_path, index=False, encoding="utf-8-sig")
    hira_all.to_csv(model_context_path, index=False, encoding="utf-8-sig")

    log_df = pd.DataFrame(all_logs)
    log_df["processed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_df.to_csv(log_path, index=False, encoding="utf-8-sig")

    # =========================
    # 6. RAG용 요약 카드 생성
    # =========================

    with open(cards_path, "w", encoding="utf-8") as f:
        for _, row in hira_all.iterrows():
            text = (
                f"{row['year']}년 HIRA 정신질환 진료 통계에서 "
                f"{row['sido']} {row['sigungu']} 소재 요양기관 기준 "
                f"{row['age_group']} {row['gender']}의 {row['disease']} 진료 환자수는 "
                f"{row['patients']}명, 입내원일수는 {row['visit_days']}일, "
                f"요양급여비용은 {row['cost']}원으로 집계되었다. "
                f"이 통계는 개인의 진단이나 위험도 판단 근거가 아니라 "
                f"지역·성별·연령대별 참고 통계로만 사용해야 한다."
            )

            record = {
                "id": row["match_key"],
                "text": text,
                "metadata": {
                    "year": int(row["year"]),
                    "disease": row["disease"],
                    "context_key": row["context_key"],
                    "sido": row["sido"],
                    "sigungu": row["sigungu"],
                    "gender": row["gender"],
                    "age_group": row["age_group"],
                    "source": "HIRA 시군구별 주요 정신질환 진료 통계",
                    "is_suppressed_or_zero": bool(row["is_suppressed_or_zero"]),
                },
            }

            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("\n[전처리 완료]")
    print(f"- 연령별 통계: {age_stats_path}")
    print(f"- 지역별 통계: {region_stats_path}")
    print(f"- 모델 매칭용 통합 테이블: {model_context_path}")
    print(f"- RAG 요약 카드: {cards_path}")
    print(f"- 전처리 로그: {log_path}")

    print("\n[데이터 크기]")
    print(f"- 전체 행 수: {len(hira_all):,}")
    print(f"- 연령별 행 수: {len(age_stats):,}")
    print(f"- 지역별 행 수: {len(region_stats):,}")

    print("\n[질환 목록]")
    print(hira_all["disease"].drop_duplicates().tolist())


if __name__ == "__main__":
    main()