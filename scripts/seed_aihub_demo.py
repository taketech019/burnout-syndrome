"""scripts/seed_aihub_demo.py — AI Hub 심리상담 라벨링 데이터 시드.

PRD §2 타겟: 사설 상담센터 상담심리사 / 성인 내담자.
임의 시나리오 생성 대신 AI Hub 공개 라벨링 데이터에서 환자 6명을 결정적으로 추출.

데이터 위치: data/references/심리상담데이터/Training/02.라벨링데이터/*.zip.part0
- TL_001 우울증 (ID prefix D)
- TL_002 불안장애 (ID prefix X)
- TL_003 중독       (ID prefix A)
- TL_004 일반군     (ID prefix N)

실행:
    python scripts/seed_aihub_demo.py [--clean] [--with-labels]
    --clean        : 기존 환자 전체 삭제 후 시드
    --with-labels  : classifier/summary/factors analyses 까지 함께 시드 (기본 off)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import db
from src.factor_extractor import FACTOR_KEYS


LABEL_DIR = ROOT / "data" / "references" / "심리상담데이터" / "Training" / "02.라벨링데이터"

CATEGORY_META = {
    "DEPRESSION": {"folder_prefix": "TL_001._우울증",   "scope": "우울"},
    "ANXIETY":    {"folder_prefix": "TL_002._불안장애", "scope": "불안"},
    "ADDICTION":  {"folder_prefix": "TL_003._중독",     "scope": "중독"},
    "NORMAL":     {"folder_prefix": "TL_004._일반군",   "scope": "일반"},
}

# 영문 라벨 → FACTOR_KEYS(한글) 매핑 — 카테고리별로 다름 (loss_of_control 등은 의미가 다름)
EN_TO_KR_BY_CATEGORY: dict[str, dict[str, str]] = {
    "DEPRESSION": {
        "depressive_mood":     "우울한 기분",
        "worthlessness":       "무가치감",
        "guilt":               "죄책감",
        "impaired_cognition":  "사고력 저하",
        "anhedonia":           "흥미감소",
        "psychomotor_changes": "정신운동변화",
        "weight_appetite":     "체중/식욕변화",
        "sleep_disturbance":   "수면문제",
        "fatigue":             "피로감",
        "suicidal":            "자살생각",
    },
    "ANXIETY": {
        "anxiety_mood":              "불안감",
        "derealization":             "비현실감",
        "perceived_loss_of_control": "통제력상실감",
        "anxiety_control":           "불안조절곤란",
        "concentration":             "집중력저하",
        "avoidance":                 "사회적상황회피",
        "physical_symptoms":         "신체증상",
        "irritability":              "과민성",
        # 보너스 (불안 카테고리 라벨에 있음 — 우울/위험·우울 일부 채움)
        "sleep_disturbance":         "수면문제",
        "fatigue":                   "피로감",
    },
    "ADDICTION": {
        "loss_of_control":     "조절실패",
        "craving":             "갈망",
        "lying":               "거짓말",
        "tolerance":           "내성",
        "withdrawal":          "금단",
        "salience":            "현저성",
        "resource_investment": "자원투자",
        "daily_functioning":   "자기관리",
        "social_problems":     "사회적문제발생",
        "negative_consequences": "부정적 결과",
    },
    "NORMAL": {},
}

# 빈도 → 0~3 점수 매핑
def _freq_to_score(freq: int) -> int:
    if freq <= 0:
        return 0
    if freq <= 2:
        return 1
    if freq <= 5:
        return 2
    return 3


REGIONS = ["서울", "경기", "부산", "인천", "대구"]


def _region_for(patient_id: str) -> str:
    """ID 해시로 결정적 지역 매핑."""
    h = int(hashlib.md5(patient_id.encode()).hexdigest(), 16)
    return REGIONS[h % len(REGIONS)]


def _gender_kr(g: str) -> str:
    return {"남": "남성", "여": "여성"}.get(g, g)


# ── zip 헬퍼 ──────────────────────────────────────────────────────────────────


_ID_RE = re.compile(r"_check_([DXAN]\d+)\.json$")


def _list_zips_for(category: str) -> list[Path]:
    """카테고리의 모든 회기 zip 정렬 반환."""
    prefix = CATEGORY_META[category]["folder_prefix"]
    files = sorted(LABEL_DIR.glob(f"{prefix}_*.zip.part0"))

    def _session_no(p: Path) -> int:
        m = re.search(r"_(\d+)회기\.zip", p.name)
        return int(m.group(1)) if m else 0

    return sorted(files, key=_session_no)


def _ids_in_zip(zip_path: Path) -> list[str]:
    with zipfile.ZipFile(zip_path) as z:
        return sorted({m.group(1) for n in z.namelist() if (m := _ID_RE.search(n))})


def _read_patient_json(zip_path: Path, patient_id: str) -> Optional[dict]:
    """zip 안에서 해당 환자 ID의 JSON 한 개를 찾아 파싱. 손상된 JSON은 None."""
    with zipfile.ZipFile(zip_path) as z:
        for n in z.namelist():
            m = _ID_RE.search(n)
            if m and m.group(1) == patient_id:
                try:
                    return json.loads(z.read(n).decode("utf-8-sig"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    return None
    return None


# ── transcript / summary / factors 변환 ─────────────────────────────────────


def _join_transcript(paragraphs: list[dict]) -> str:
    lines = []
    for p in paragraphs:
        sp = str(p.get("paragraph_speaker") or "").strip()
        tx = str(p.get("paragraph_text") or "").strip()
        if sp and tx:
            lines.append(f"{sp}: {tx}")
    return "\n".join(lines)


_SECTION_RE = re.compile(
    r"(주요\s*증상\s*:|위험\s*요인\s*:|개선\s*요인\s*:|상담사의\s*개입\s*요인\s*:)"
)
_SECTION_TO_KEY = {
    "주요 증상": "symptoms",
    "위험 요인": "risk_factors",
    "개선 요인": "improvement_factors",
    "상담사의 개입 요인": "intervention_factors",
}


def _split_summary(summary: str) -> dict[str, str]:
    out = {"symptoms": "", "risk_factors": "", "improvement_factors": "", "intervention_factors": ""}
    if not summary:
        return out
    # 헤더 위치 수집
    parts = _SECTION_RE.split(summary)
    # parts = [pre, header1, body1, header2, body2, ...]
    for i in range(1, len(parts) - 1, 2):
        header_raw = parts[i].strip().rstrip(":").strip()
        # 공백 정규화 — "주요  증상" → "주요 증상"
        header_norm = re.sub(r"\s+", " ", header_raw)
        body = parts[i + 1].strip()
        key = _SECTION_TO_KEY.get(header_norm)
        if key:
            out[key] = body
    return out


def _build_brief(sections: dict[str, str]) -> str:
    """4섹션 첫 문장만 모아 한 단락(최대 350자)."""
    pieces = []
    for k in ("symptoms", "risk_factors", "improvement_factors", "intervention_factors"):
        body = sections.get(k, "")
        if not body:
            continue
        first_sentence = body.split("다.")[0]
        if first_sentence:
            pieces.append(first_sentence.strip() + "다.")
    text = " ".join(pieces)
    return text[:350]


def _aggregate_factors(
    paragraphs: list[dict],
    category: str,
) -> dict[str, int]:
    """카테고리별 영문 라벨 → 한글 28요인 0~3 점수.

    paragraph에서 영문 라벨 == 1인 횟수를 세고 _freq_to_score로 0~3 변환.
    """
    out = {k: 0 for k in FACTOR_KEYS}
    en_to_kr = EN_TO_KR_BY_CATEGORY.get(category, {})
    if not en_to_kr:
        return out

    counter = defaultdict(int)
    for p in paragraphs:
        for en_key, kr_key in en_to_kr.items():
            if p.get(en_key) == 1:
                counter[kr_key] += 1

    for kr_key, freq in counter.items():
        if kr_key in out:
            out[kr_key] = _freq_to_score(freq)
    return out


def _classification_from(d: dict) -> dict:
    """JSON의 depression/anxiety/addiction (0~3) → boolean 분류."""
    dep = int(d.get("depression") or 0)
    anx = int(d.get("anxiety") or 0)
    add = int(d.get("addiction") or 0)
    return {
        "classification": {
            "depression": dep >= 1,
            "anxiety":    anx >= 1,
            "addiction":  add >= 1,
        },
        "scores": {"depression": dep, "anxiety": anx, "addiction": add},
        "class_label": d.get("class") or "",
    }


# ── 환자 선정 ────────────────────────────────────────────────────────────────


def _select_for_category(
    category: str,
    n: int,
    *,
    exclude: set[str],
    prefer_suicidal: bool = False,
) -> list[tuple[str, dict]]:
    """카테고리에서 n명 결정적 선정.

    반환: [(patient_id, first_session_record), ...]
    조건:
        - age 18~55
        - summary 길이 ≥ 300
    """
    zips = _list_zips_for(category)
    if not zips:
        return []
    first_zip = zips[0]
    candidate_ids = _ids_in_zip(first_zip)
    require_summary = category != "NORMAL"  # 일반군은 summary 미제공

    scored: list[tuple[int, str, dict]] = []  # (priority, id, record)
    for pid in candidate_ids:
        if pid in exclude:
            continue
        rec = _read_patient_json(first_zip, pid)
        if not rec:
            continue
        age = int(rec.get("age") or 0)
        if not (18 <= age <= 55):
            continue
        summary = rec.get("summary") or ""
        if require_summary and len(summary) < 300:
            continue

        # 자살 사고 빈도 (위기성 우선순위용)
        suicidal_freq = sum(1 for p in rec.get("paragraph", []) if p.get("suicidal") == 1)
        # priority: prefer_suicidal일 때 suicidal_freq 높을수록 우선 → 음수로 정렬
        priority = -suicidal_freq if prefer_suicidal else 0
        scored.append((priority, pid, rec))
        if len(scored) >= n * 4 and not prefer_suicidal:
            # 후보 충분히 확보되면 조기 종료 (속도)
            break

    scored.sort(key=lambda x: (x[0], x[1]))  # priority asc, id asc
    return [(pid, rec) for _, pid, rec in scored[:n]]


def _collect_sessions_for(
    category: str,
    patient_id: str,
    *,
    max_sessions: int = 3,
    first_session_rec: Optional[dict] = None,
) -> list[tuple[int, dict]]:
    """환자의 회기 데이터 수집 (회기 번호, JSON 레코드)."""
    out: list[tuple[int, dict]] = []
    zips = _list_zips_for(category)
    for zp in zips:
        m = re.search(r"_(\d+)회기\.zip", zp.name)
        if not m:
            continue
        sess_no = int(m.group(1))
        if sess_no == 1 and first_session_rec is not None:
            rec = first_session_rec
        else:
            rec = _read_patient_json(zp, patient_id)
        if rec:
            out.append((sess_no, rec))
        if len(out) >= max_sessions:
            break
    return out


# ── 시드 ─────────────────────────────────────────────────────────────────────


def _seed_patient(
    category: str,
    patient_id: str,
    sessions: list[tuple[int, dict]],
    *,
    with_labels: bool,
    quiet: bool = False,
) -> None:
    """환자 + 회기 + (옵션) analyses 시드."""
    if not sessions:
        return
    first_rec = sessions[0][1]
    age = int(first_rec.get("age") or 30)
    gender = _gender_kr(first_rec.get("gender") or "")
    region = _region_for(patient_id)
    dep, anx, add = int(first_rec.get("depression") or 0), int(first_rec.get("anxiety") or 0), int(first_rec.get("addiction") or 0)
    scope_kr = CATEGORY_META[category]["scope"]
    note = f"{scope_kr} 호소 · 라벨 점수 {dep}/{anx}/{add}"

    p = db.add_patient(patient_id, gender, age, region, note)
    if not quiet:
        print(f"  + {patient_id} ({p['id']}) {gender}/{age}/{region} · {note}")

    for sess_no, rec in sessions:
        transcript = _join_transcript(rec.get("paragraph", []))
        sections = _split_summary(rec.get("summary") or "")
        topic = sections.get("symptoms", "")[:120].split("\n")[0] or f"{scope_kr} 호소"
        s = db.add_session(
            p["id"],
            session_date=f"2026-{(2 + sess_no):02d}-15",  # 임의 데모 날짜 — 회기 간격 1개월
            transcript=transcript,
            session_no=f"{sess_no}회기",
            scope=scope_kr,
            topic=topic,
        )
        if not quiet:
            print(f"    · {sess_no}회기 ({s['id']}) — transcript {len(transcript)}자")

        if not with_labels:
            continue

        # classifier
        cls_payload = _classification_from(rec)
        cls_payload["backend"] = "aihub_label"
        db.add_analysis(s["id"], "classifier", "aihub_label", cls_payload)

        # summary
        brief = _build_brief(sections)
        body_text = "\n\n".join(
            f"### {label}\n{sections.get(key, '')}"
            for label, key in [
                ("주요 증상", "symptoms"),
                ("위험 요인", "risk_factors"),
                ("개선 요인", "improvement_factors"),
                ("개입 요인", "intervention_factors"),
            ]
        )
        summary_payload = {
            "ok": True,
            "status": "success",
            "text": body_text,
            "brief": brief,
            "sections": sections,
            "source": "aihub_label",
            "koalpaca_sections_filled": 0,
            "gemma_sections_filled": 4,
            "backend": "aihub_label",
        }
        db.add_analysis(s["id"], "summary", "aihub_label", summary_payload)

        # factors
        factors = _aggregate_factors(rec.get("paragraph", []), category)
        db.add_analysis(s["id"], "factors", "aihub_label", {
            "factors": factors,
            "backend": "aihub_label",
        })


def _check_label_dir() -> None:
    if not LABEL_DIR.exists():
        print(f"[ERROR] 라벨링 데이터 경로 없음: {LABEL_DIR}")
        print("AI Hub '심리상담 데이터' (Training/02.라벨링데이터)를 해당 경로에 배치하세요.")
        sys.exit(2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", action="store_true", help="기존 환자 전체 삭제 후 시드")
    parser.add_argument("--with-labels", action="store_true", help="classifier/summary/factors까지 함께 시드")
    parser.add_argument("--quiet", action="store_true", help="진행 로그 최소화")
    args = parser.parse_args()

    _check_label_dir()
    db.init_db()

    if args.clean:
        existing = db.list_patients()
        if not args.quiet:
            print(f"기존 환자 {len(existing)}명 삭제 중...")
        for p in existing:
            db.delete_patient(p["id"])

    # 환자 선정: D2 (1명 위기성 우선) + X1 (위기성 우선) + A1 + N1 + D1 추가 = 6명
    plan: list[tuple[str, int, bool]] = [
        ("DEPRESSION", 2, True),   # 우울 2명 (suicidal 우선)
        ("ANXIETY",    1, True),   # 불안 1명 (위기성 우선)
        ("ADDICTION",  1, False),  # 중독 1명
        ("NORMAL",     1, False),  # 일반 1명
        ("DEPRESSION", 1, False),  # 우울 1명 (기본 선정)
    ]

    selected_ids: set[str] = set()
    total = 0
    for category, n, prefer_suicidal in plan:
        if not args.quiet:
            print(f"\n[{category}] {n}명 선정...")
        candidates = _select_for_category(
            category, n,
            exclude=selected_ids,
            prefer_suicidal=prefer_suicidal,
        )
        for pid, first_rec in candidates:
            selected_ids.add(pid)
            sessions = _collect_sessions_for(
                category, pid, max_sessions=3, first_session_rec=first_rec
            )
            _seed_patient(
                category, pid, sessions,
                with_labels=args.with_labels, quiet=args.quiet,
            )
            total += 1

    print(f"\n완료 — 환자 {total}명 시드. (DB: {len(db.list_patients())}명)")
    if not args.with_labels:
        print("팁: --with-labels 옵션으로 classifier/summary/factors까지 함께 시드 가능.")


if __name__ == "__main__":
    main()
