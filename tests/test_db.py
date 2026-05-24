import importlib
import sys
import tempfile
from pathlib import Path

import pytest

# 루트 경로 등록
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def temp_db(monkeypatch):
    """각 테스트마다 fresh DB."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = Path(f.name)
    import config
    monkeypatch.setattr(config, "DB_PATH", path)
    import src.db
    importlib.reload(src.db)
    src.db.init_db()
    yield src.db
    try:
        path.unlink()
    except OSError:
        pass


def test_add_and_list_patient(temp_db):
    p = temp_db.add_patient(alias="P001", gender="여성", age=30, region="서울", note="")
    assert p["id"]
    assert p["alias"] == "P001"
    assert len(temp_db.list_patients()) == 1


def test_session_with_analyses(temp_db):
    p = temp_db.add_patient(alias="P002", gender="남성", age=40, region="부산")
    s = temp_db.add_session(p["id"], "2026-05-24", "상담사: 안녕\n내담자: 우울해요")
    temp_db.add_analysis(s["id"], stage="classifier", backend="gemma_fallback",
                         payload={"classification": {"depression": 1}})
    temp_db.add_analysis(s["id"], stage="summary", backend="koalpaca",
                         payload={"text": "요약 본문"})
    rows = temp_db.list_analyses(s["id"])
    assert len(rows) == 2
    assert {r["stage"] for r in rows} == {"classifier", "summary"}


def test_cascade_delete(temp_db):
    """FK CASCADE: patient 삭제 시 sessions·analyses도 삭제."""
    p = temp_db.add_patient(alias="P003", gender="여성", age=25, region="대구")
    s = temp_db.add_session(p["id"], "2026-05-24", "테스트")
    temp_db.add_analysis(s["id"], stage="classifier", backend="x", payload={})
    assert temp_db.delete_patient(p["id"]) is True
    assert temp_db.list_sessions(p["id"]) == []
    assert temp_db.list_analyses(s["id"]) == []


def test_get_latest_analysis(temp_db):
    p = temp_db.add_patient(alias="P004", gender="남성", age=50, region="인천")
    s = temp_db.add_session(p["id"], "2026-05-24", "텍스트")
    temp_db.add_analysis(s["id"], stage="summary", backend="koalpaca", payload={"text": "old"})
    temp_db.add_analysis(s["id"], stage="summary", backend="gemma_fallback", payload={"text": "new"})
    latest = temp_db.get_latest_analysis(s["id"], stage="summary")
    assert latest["payload"]["text"] == "new"
