from __future__ import annotations

import csv
import json
import re
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional


RAW_ROOT = Path("data/raw/counseling")
PROCESSED_ROOT = Path("data/processed")

OUTPUT_SESSIONS = PROCESSED_ROOT / "sessions.jsonl"
OUTPUT_SCORES = PROCESSED_ROOT / "session_scores.csv"
OUTPUT_SUMMARIES = PROCESSED_ROOT / "session_summaries.jsonl"
OUTPUT_CHUNKS = PROCESSED_ROOT / "counseling_chunks.jsonl"
OUTPUT_LOG = PROCESSED_ROOT / "preprocessing_log.csv"


SYMPTOM_FIELDS = [
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
]

RISK_FIELDS = [
    "trauma_experience",
    "negative_self-image",
    "emotional_requlation",
    "motivation_for_change",
    "belief",
    "unrealistic_recovery_expectations",
    "loss_of_control",
    "coping",
    "lifestyle",
    "family_history",
    "underlying_physical_condition",
    "history_of_mental_illness",
    "stressful_event",
    "social_support",
    "social_resources",
]

CHANGE_FIELDS = [
    "emotional_change",
    "cognitive_change",
    "behavioral_change",
    "acceptance_change",
    "enhancement_of_motivation",
]

INTERVENTION_FIELDS = [
    "sympathy_support",
    "clarification_reflection",
    "cognitive_restructuring",
    "information_provision",
    "goal_setting",
    "process_feedback",
    "behavioral_intervention",
    "task_assignment",
    "training_of_coping_skills",
    "emotional_regulation_education_training",
    "structuring",
]

ALL_LABEL_FIELDS = SYMPTOM_FIELDS + RISK_FIELDS + CHANGE_FIELDS + INTERVENTION_FIELDS


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("\r\n", "\n").replace("\r", "\n").strip()


def infer_session_from_name(name: str) -> str:
    """
    파일명 또는 zip명에서 '1회기', '2회기' 같은 회기 번호를 추출한다.
    실패하면 'unknown'을 반환한다.
    """
    match = re.search(r"(\d+)\s*회기", name)
    if match:
        return match.group(1)

    match = re.search(r"_(\d+)[^\d]*$", name)
    if match:
        return match.group(1)

    return "unknown"


def build_script(paragraphs: List[Dict[str, Any]]) -> str:
    """
    paragraph 배열을 상담사/내담자 대화 스크립트 문자열로 변환한다.
    """
    ordered = sorted(paragraphs, key=lambda row: safe_int(row.get("index", 0)))
    lines = []

    for row in ordered:
        speaker = normalize_text(row.get("paragraph_speaker", ""))
        text = normalize_text(row.get("paragraph_text", ""))

        if not text:
            continue

        if speaker:
            lines.append(f"{speaker}: {text}")
        else:
            lines.append(text)

    return "\n".join(lines)


def label_max(paragraphs: List[Dict[str, Any]], field: str) -> int:
    values = [safe_int(row.get(field, 0)) for row in paragraphs]
    return max(values) if values else 0


def label_sum(paragraphs: List[Dict[str, Any]], field: str) -> int:
    return sum(safe_int(row.get(field, 0)) for row in paragraphs)


def make_session_record(
    split: str,
    zip_path: Path,
    inner_json_name: str,
    data: Dict[str, Any],
) -> Dict[str, Any]:
    filename = normalize_text(data.get("filename")) or Path(inner_json_name).stem
    client_id = normalize_text(data.get("id")) or filename
    session = infer_session_from_name(zip_path.name + " " + inner_json_name)

    paragraphs = data.get("paragraph", [])
    if not isinstance(paragraphs, list):
        paragraphs = []

    script = build_script(paragraphs)

    label_max_values = {
        f"{field}_max": label_max(paragraphs, field)
        for field in ALL_LABEL_FIELDS
    }

    label_sum_values = {
        f"{field}_sum": label_sum(paragraphs, field)
        for field in ALL_LABEL_FIELDS
    }

    return {
        "split": split,
        "client_id": client_id,
        "session": session,
        "filename": filename,
        "source_zip": str(zip_path).replace("\\", "/"),
        "source_json": inner_json_name,
        "class": normalize_text(data.get("class")),
        "age": safe_int(data.get("age", 0)),
        "gender": normalize_text(data.get("gender")),
        "depression": safe_int(data.get("depression", 0)),
        "anxiety": safe_int(data.get("anxiety", 0)),
        "addiction": safe_int(data.get("addiction", 0)),
        "summary": normalize_text(data.get("summary")),
        "silence": safe_float(data.get("silence", 0)),
        "total_time": safe_float(data.get("total_time", 0)),
        "paragraph_count": len(paragraphs),
        "script": script,
        "label_max": label_max_values,
        "label_sum": label_sum_values,
    }


def make_summary_record(session_record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "doc_id": f"summary::{session_record['split']}::{session_record['client_id']}::{session_record['session']}::{session_record['filename']}",
        "split": session_record["split"],
        "client_id": session_record["client_id"],
        "session": session_record["session"],
        "class": session_record["class"],
        "age": session_record["age"],
        "gender": session_record["gender"],
        "depression": session_record["depression"],
        "anxiety": session_record["anxiety"],
        "addiction": session_record["addiction"],
        "text": session_record["summary"],
        "source_zip": session_record["source_zip"],
        "source_json": session_record["source_json"],
    }


def make_chunks(session_record: Dict[str, Any], chunk_size: int = 5, overlap: int = 2) -> List[Dict[str, Any]]:
    """
    sessions.jsonl의 script를 줄 단위로 다시 나누어 chunk 생성.
    초기 구현에서는 paragraph 원본 대신 script line 기준으로 chunk를 만든다.
    """
    lines = [line.strip() for line in session_record["script"].split("\n") if line.strip()]

    if not lines:
        return []

    step = max(1, chunk_size - overlap)
    chunks = []

    for start in range(0, len(lines), step):
        end = min(start + chunk_size, len(lines))
        selected = lines[start:end]

        if not selected:
            continue

        chunk_id = (
            f"chunk::{session_record['split']}::{session_record['client_id']}::"
            f"{session_record['session']}::{session_record['filename']}::{start}_{end - 1}"
        )

        chunks.append(
            {
                "chunk_id": chunk_id,
                "split": session_record["split"],
                "client_id": session_record["client_id"],
                "session": session_record["session"],
                "class": session_record["class"],
                "age": session_record["age"],
                "gender": session_record["gender"],
                "depression": session_record["depression"],
                "anxiety": session_record["anxiety"],
                "addiction": session_record["addiction"],
                "chunk_type": "dialogue",
                "start_line": start,
                "end_line": end - 1,
                "text": "\n".join(selected),
                "source_zip": session_record["source_zip"],
                "source_json": session_record["source_json"],
                "label_max": session_record["label_max"],
            }
        )

        if end == len(lines):
            break

    return chunks


def read_json_from_zip(zip_path: Path) -> List[Dict[str, Any]]:
    records = []

    with zipfile.ZipFile(zip_path, "r") as zf:
        for inner_name in zf.namelist():
            if not inner_name.lower().endswith(".json"):
                continue

            with zf.open(inner_name) as file:
                raw = file.read()

            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                text = raw.decode("cp949")

            data = json.loads(text)

            if isinstance(data, dict):
                records.append(
                    {
                        "inner_json_name": inner_name,
                        "data": data,
                    }
                )

    return records


def find_label_zip_files() -> List[Dict[str, Any]]:
    targets = []

    for split in ["Training", "Validation"]:
        label_dir = RAW_ROOT / split / "02.라벨링데이터"

        if not label_dir.exists():
            continue

        for zip_path in sorted(label_dir.rglob("*.zip")):
            targets.append(
                {
                    "split": split,
                    "zip_path": zip_path,
                }
            )

    return targets


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_scores_csv(path: Path, sessions: List[Dict[str, Any]]) -> None:
    base_fields = [
        "split",
        "client_id",
        "session",
        "filename",
        "class",
        "age",
        "gender",
        "depression",
        "anxiety",
        "addiction",
        "paragraph_count",
        "total_time",
        "source_zip",
        "source_json",
    ]

    max_fields = [f"{field}_max" for field in ALL_LABEL_FIELDS]
    sum_fields = [f"{field}_sum" for field in ALL_LABEL_FIELDS]

    fieldnames = base_fields + max_fields + sum_fields

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for session in sessions:
            row = {field: session.get(field, "") for field in base_fields}
            row.update(session.get("label_max", {}))
            row.update(session.get("label_sum", {}))
            writer.writerow(row)


def write_log_csv(path: Path, logs: List[Dict[str, Any]]) -> None:
    fieldnames = ["split", "zip_path", "status", "json_count", "message"]

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(logs)


def main() -> None:
    PROCESSED_ROOT.mkdir(parents=True, exist_ok=True)

    sessions: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []
    chunks: List[Dict[str, Any]] = []
    logs: List[Dict[str, Any]] = []

    targets = find_label_zip_files()

    print(f"[INFO] Found label zip files: {len(targets)}")

    for target in targets:
        split = target["split"]
        zip_path = target["zip_path"]

        try:
            json_items = read_json_from_zip(zip_path)

            for item in json_items:
                session_record = make_session_record(
                    split=split,
                    zip_path=zip_path,
                    inner_json_name=item["inner_json_name"],
                    data=item["data"],
                )

                sessions.append(session_record)

                if session_record["summary"]:
                    summaries.append(make_summary_record(session_record))

                chunks.extend(make_chunks(session_record))

            logs.append(
                {
                    "split": split,
                    "zip_path": str(zip_path).replace("\\", "/"),
                    "status": "success",
                    "json_count": len(json_items),
                    "message": "",
                }
            )

            print(f"[OK] {split} | {zip_path.name} | json={len(json_items)}")

        except Exception as error:
            logs.append(
                {
                    "split": split,
                    "zip_path": str(zip_path).replace("\\", "/"),
                    "status": "error",
                    "json_count": 0,
                    "message": str(error),
                }
            )

            print(f"[ERROR] {split} | {zip_path.name} | {error}")

    write_jsonl(OUTPUT_SESSIONS, sessions)
    write_jsonl(OUTPUT_SUMMARIES, summaries)
    write_jsonl(OUTPUT_CHUNKS, chunks)
    write_scores_csv(OUTPUT_SCORES, sessions)
    write_log_csv(OUTPUT_LOG, logs)

    print("")
    print("[DONE] Preprocessing completed.")
    print(f"  sessions: {len(sessions)} -> {OUTPUT_SESSIONS}")
    print(f"  summaries: {len(summaries)} -> {OUTPUT_SUMMARIES}")
    print(f"  chunks: {len(chunks)} -> {OUTPUT_CHUNKS}")
    print(f"  scores -> {OUTPUT_SCORES}")
    print(f"  log -> {OUTPUT_LOG}")


if __name__ == "__main__":
    main()