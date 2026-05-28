from pathlib import Path
import zipfile
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_HIRA_DIR = PROJECT_ROOT / "data" / "raw" / "hira"
PROCESSED_HIRA_DIR = PROJECT_ROOT / "data" / "processed" / "hira"
PROCESSED_HIRA_DIR.mkdir(parents=True, exist_ok=True)


def find_file(keyword: str):
    files = list(RAW_HIRA_DIR.glob("*"))
    matched = [p for p in files if keyword in p.name]
    if not matched:
        raise FileNotFoundError(f"{keyword} 포함 파일을 찾지 못했습니다: {RAW_HIRA_DIR}")
    return matched[0]


def read_csv_safely(path: Path):
    for enc in ["utf-8-sig", "cp949", "euc-kr"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    raise ValueError(f"CSV 인코딩을 확인할 수 없습니다: {path}")


def build_mental_disease_context():
    zip_path = find_file("시군구별 주요 정신질환 진료 통계")

    extract_dir = RAW_HIRA_DIR / "_mental_disease_unzipped"
    extract_dir.mkdir(parents=True, exist_ok=True)

    # 기존 압축해제 파일 삭제
    for old_file in extract_dir.rglob("*"):
        if old_file.is_file():
            old_file.unlink()

    with zipfile.ZipFile(zip_path, "r") as z:
        for info in z.infolist():
            # ZIP 내부 한글 파일명 깨짐 보정
            try:
                decoded_name = info.filename.encode("cp437").decode("cp949")
            except Exception:
                decoded_name = info.filename

            target_path = extract_dir / decoded_name
            target_path.parent.mkdir(parents=True, exist_ok=True)

            if not info.is_dir():
                with z.open(info) as src, open(target_path, "wb") as dst:
                    dst.write(src.read())

    csv_files = list(extract_dir.rglob("*.csv"))

    if not csv_files:
        raise FileNotFoundError("ZIP 안에서 CSV 파일을 찾지 못했습니다.")

    frames = []

    for csv_file in csv_files:
        df = read_csv_safely(csv_file)

        # 연령구분이 있는 CSV만 대시보드 핵심 데이터로 사용
        if "연령구분" not in df.columns:
            continue

        frames.append(df)

    if not frames:
        raise FileNotFoundError("연령구분이 포함된 HIRA CSV를 찾지 못했습니다.")

    raw = pd.concat(frames, ignore_index=True)

    rename_map = {
        "진료년도": "year",
        "상병구분": "disease",
        "상별구분": "disease",
        "시도": "sido",
        "시군구": "sigungu",
        "성별": "gender",
        "연령구분": "age_group",
        "환자수": "patients",
        "입내원일수": "visit_days",
        "요양급여비용": "cost",
    }

    raw = raw.rename(columns={k: v for k, v in rename_map.items() if k in raw.columns})

    required = [
        "year",
        "disease",
        "sido",
        "sigungu",
        "gender",
        "age_group",
        "patients",
        "visit_days",
        "cost",
    ]

    missing = [c for c in required if c not in raw.columns]

    if missing:
        raise ValueError(f"필수 컬럼이 없습니다: {missing}\n현재 컬럼: {raw.columns.tolist()}")

    # 문자열 정리
    for col in ["disease", "sido", "sigungu", "gender", "age_group"]:
        raw[col] = raw[col].astype(str).str.strip()

    # 성별 정규화
    raw["gender"] = raw["gender"].replace(
        {
            "남자": "남",
            "남성": "남",
            "M": "남",
            "여자": "여",
            "여성": "여",
            "F": "여",
        }
    )

    # 연령대 정규화
    age_group_map = {
        "0대": "0~9세",
        "10대": "10~19세",
        "20대": "20~29세",
        "30대": "30~39세",
        "40대": "40~49세",
        "50대": "50~59세",
        "60대": "60~69세",
        "70대": "70~79세",
        "80대": "80~89세",
        "90대": "90~99세",
        "100대": "100세이상",
        "100세 이상": "100세이상",
    }

    raw["age_group"] = raw["age_group"].replace(age_group_map)

    # 질환명 정규화
    disease_map = {
        "우울 장애": "우울증",
        "우울장애": "우울증",
        "우울증": "우울증",
        "불안 장애": "불안장애",
        "불안장애": "불안장애",
        "불면 장애": "불면증",
        "불면장애": "불면증",
        "불면증": "불면증",
        "주의력결핍과잉행동장애": "ADHD",
        "ADHD": "ADHD",
        "조울증": "조울증",
        "양극성장애": "조울증",
        "조현병": "조현병",
    }

    raw["disease"] = raw["disease"].replace(disease_map)

    # 숫자 컬럼 정리
    for col in ["patients", "visit_days", "cost"]:
        raw[col] = (
            raw[col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("-", "0", regex=False)
        )
        raw[col] = pd.to_numeric(raw[col], errors="coerce").fillna(0).astype(int)

    disease_to_context_key = {
        "우울증": "depression",
        "불안장애": "anxiety",
        "불면증": "sleep",
        "ADHD": "adhd",
        "조울증": "bipolar",
        "조현병": "schizophrenia",
    }

    raw["context_key"] = raw["disease"].map(disease_to_context_key).fillna("etc")

    raw["visit_days_per_patient"] = raw.apply(
        lambda r: r["visit_days"] / r["patients"] if r["patients"] > 0 else 0,
        axis=1,
    )

    raw["cost_per_patient"] = raw.apply(
        lambda r: r["cost"] / r["patients"] if r["patients"] > 0 else 0,
        axis=1,
    )

    print("[HIRA 확인] 컬럼:", raw.columns.tolist())
    print("[HIRA 확인] 질환:", sorted(raw["disease"].dropna().unique().tolist()))
    print("[HIRA 확인] 성별:", sorted(raw["gender"].dropna().unique().tolist()))
    print("[HIRA 확인] 연령:", sorted(raw["age_group"].dropna().unique().tolist()))
    print("[HIRA 확인] context_key:", sorted(raw["context_key"].dropna().unique().tolist()))
    print("[HIRA 확인] 행 수:", len(raw))

    out_path = PROCESSED_HIRA_DIR / "hira_model_context.csv"
    raw.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"[OK] 저장 완료: {out_path}")


def build_depression_screening_context():
    csv_path = find_file("일반건강검진_정신건강검사")
    df = read_csv_safely(csv_path)

    out_path = PROCESSED_HIRA_DIR / "depression_screening_context.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"[OK] 저장 완료: {out_path}")
    print(df.head())


def build_medical_access_context():
    clinic_path = find_file("의원표시과목별건수")
    hospital_type_path = find_file("종별진료인원")

    clinic = pd.read_excel(clinic_path)
    hospital_type = pd.read_excel(hospital_type_path)

    clinic_out = PROCESSED_HIRA_DIR / "clinic_department_counts_2024.csv"
    type_out = PROCESSED_HIRA_DIR / "hospital_type_patients_2024.csv"

    clinic.to_csv(clinic_out, index=False, encoding="utf-8-sig")
    hospital_type.to_csv(type_out, index=False, encoding="utf-8-sig")

    print(f"[OK] 저장 완료: {clinic_out}")
    print(f"[OK] 저장 완료: {type_out}")


if __name__ == "__main__":
    build_mental_disease_context()
    build_depression_screening_context()
    build_medical_access_context()