"""src/hira.py — HIRA 인구통계 매칭 (F2 대시보드).

`data/raw/건강보험심사평가원_시군구별 성별 연령별 주요 정실질환 통계 2024.csv` 등
HIRA 공개 CSV를 cp949로 로드해서 내담자(성별/연령/지역)와 매칭한 진료율 컨텍스트 반환.

CSV 구조 (예상):
- 진료년도 / 상병명 (또는 질환명) / 시도 / 시군구 / 성별 / 연령구분 / 환자수 / 내원일수 / 급여비용
- 다른 CSV (시군구별 성별)는 연령 없음

이 모듈은 두 CSV를 둘 다 로드 시도. 첫 매칭 우선. 정확 일치 실패 시 시도(시군구 무시)로 폴백.
"""
import logging
from functools import lru_cache
from typing import Optional

import pandas as pd

from config import RAW_DIR

log = logging.getLogger(__name__)

# CSV 이름 (인코딩 cp949 — 헤더도 한글 깨짐 가능)
_CSV_AGE = "건강보험심사평가원_시군구별 성별 연령별 주요 정실질환 통계 2024.csv"
_CSV_NO_AGE = "건강보험심사평가원_시군구별 성별 주요 정실질환 통계 2024.csv"

# 우리가 다루는 질환 → HIRA "상병명" 매핑 (cp949 디코딩 후 한글)
_DISEASE_KEYWORD = {
    "depression": ["우울증", "우울", "조울증", "기분장애"],
    "anxiety": ["불안장애", "불안"],
    "addiction": ["알코올", "물질", "중독", "남용"],
}

_GENDER_MAP = {"여성": "여", "여": "여", "F": "여", "남성": "남", "남": "남", "M": "남"}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """헤더 공백 제거 + 한글 강제 캐스팅."""
    df.columns = [str(c).strip() for c in df.columns]
    return df


@lru_cache(maxsize=2)
def _load_csv(name: str) -> Optional[pd.DataFrame]:
    path = RAW_DIR / name
    if not path.exists():
        log.warning("HIRA CSV 부재: %s", path)
        return None
    for enc in ("cp949", "utf-8-sig", "utf-8"):
        try:
            df = pd.read_csv(path, encoding=enc, low_memory=False)
            df = _normalize_columns(df)
            return df
        except (UnicodeDecodeError, UnicodeError):
            continue
        except Exception as e:
            log.warning("HIRA CSV 로드 실패 %s (%s): %s", name, enc, e)
            return None
    return None


def _age_bucket(age: int) -> str:
    """HIRA 연령구분 컬럼 값과 매칭 (예: '20~29세' 또는 '20대')."""
    decade = (int(age) // 10) * 10
    return f"{decade}~{decade+9}세"


def _find_column(df: pd.DataFrame, *keys: str) -> Optional[str]:
    """헤더에서 키워드를 포함한 첫 컬럼 반환."""
    for col in df.columns:
        for k in keys:
            if k in str(col):
                return col
    return None


def _filter_by_disease(df: pd.DataFrame, disease_col: str, disease: str) -> pd.DataFrame:
    keywords = _DISEASE_KEYWORD.get(disease, [disease])
    mask = df[disease_col].astype(str).str.contains("|".join(keywords), na=False, regex=True)
    return df[mask]


def lookup(patient: dict, disease: str = "depression") -> dict:
    """내담자 메타에 대응하는 HIRA 통계.

    반환:
      {
        "available": bool,
        "matched": {"region_rate": float?, "national_rate": float?, "patients": int?, "n_rows": int},
        "summary_text": str,
        "error"?: str,
      }
    실제로는 진료율(%)이 아니라 환자수만 있는 CSV — 진료율 계산은 인구수 없이 어려움.
    여기서는 "지역 환자수 vs 전국 환자수 합계" 표시로 대체.
    """
    df = _load_csv(_CSV_AGE)
    if df is None or df.empty:
        df = _load_csv(_CSV_NO_AGE)
        if df is None or df.empty:
            return {
                "available": False,
                "summary_text": "(HIRA CSV 로드 실패 — `data/raw/`에 파일 배치 필요)",
                "error": "HIRA CSV 부재",
            }

    disease_col = _find_column(df, "상병", "상별", "질환", "정신", "병명")
    region_col = _find_column(df, "시도", "지역")
    sub_region_col = _find_column(df, "시군구")
    gender_col = _find_column(df, "성별")
    age_col = _find_column(df, "연령")
    patient_col = _find_column(df, "환자수", "환자 수", "환자")
    if not (disease_col and gender_col and patient_col):
        return {
            "available": False,
            "summary_text": f"(HIRA 컬럼 자동 인식 실패: {list(df.columns)[:8]}...)",
        }

    filtered = _filter_by_disease(df, disease_col, disease)
    if filtered.empty:
        return {
            "available": False,
            "summary_text": f"(HIRA에 '{disease}' 관련 데이터 없음)",
        }

    gender = _GENDER_MAP.get(patient.get("gender", ""), patient.get("gender", ""))
    if gender:
        filtered = filtered[filtered[gender_col].astype(str).str.startswith(gender)]
    if age_col and patient.get("age"):
        bucket = _age_bucket(int(patient["age"]))
        age_filtered = filtered[filtered[age_col].astype(str).str.contains(bucket, na=False)]
        if not age_filtered.empty:
            filtered = age_filtered

    region = patient.get("region", "").strip()
    region_total = None
    if region and (region_col or sub_region_col):
        for col in (region_col, sub_region_col):
            if col is None:
                continue
            r = filtered[filtered[col].astype(str).str.contains(region, na=False)]
            if not r.empty:
                region_total = float(pd.to_numeric(r[patient_col], errors="coerce").fillna(0).sum())
                break

    national_total = float(pd.to_numeric(filtered[patient_col], errors="coerce").fillna(0).sum())

    parts = [
        f"**HIRA 2024 통계** ({disease}, 한국상병코드 매칭):",
    ]
    if region_total is not None:
        parts.append(f"- {region} 환자수 합계: {int(region_total):,}명")
    parts.append(f"- 전국(필터 조건 합계) 환자수: {int(national_total):,}명")
    if patient.get("age"):
        decade = (int(patient["age"]) // 10) * 10
        parts.append(f"- 필터: {decade}대 {patient.get('gender', '')}")
    parts.append("- 출처: 건강보험심사평가원 시군구·성별·연령별 정신질환 통계")

    return {
        "available": True,
        "matched": {
            "region_patients": region_total,
            "national_patients": national_total,
            "n_rows": int(len(filtered)),
        },
        "summary_text": "\n".join(parts),
    }
